"""Controlled-factorial encoder training for the BCCT causal experiment.

One knob at a time: --arch {vit_s,resnet18} --objective {supervised,simclr,mae,dino}
--data {d1,d2,d3} --seed S --width W (W = representation/bitrate knob).

Two modes:
  train:    python train_controlled.py --arch vit_s --objective mae --data d1 --seed 0 --width 512 --out ckpt.pt
  extract:  python train_controlled.py --extract ckpt.pt --probe cifar100_test --out feat.pt
            -> saves {features:(N,W), model_name, model_config} matching data/features/*.pt
  smoke:    python train_controlled.py --arch vit_s --objective dino --smoke
            -> builds model, runs ONE step on random tensors (no data/disk), prints losses.

Deps: torch, torchvision. ~300 LOC. Small enough to train on CIFAR-100 (32x32) in ~1-3 GPU-h.
"""

from __future__ import annotations

import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------- backbones -------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img=32, patch=4, in_ch=3, dim=192):
        super().__init__()
        self.n = (img // patch) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch, patch)

    def forward(self, x):  # (B,3,32,32) -> (B, n, dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim)
        )

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class TinyViT(nn.Module):
    """Compact ViT for 32x32. Exposes patch tokens (for MAE) and a pooled embedding."""

    def __init__(self, img=32, patch=4, dim=192, depth=6, heads=3):
        super().__init__()
        self.patch, self.img, self.dim = patch, img, dim
        self.embed = PatchEmbed(img, patch, 3, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.embed.n + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.embed_dim = dim

    def tokens(self, x, mask=None):  # returns (B, n+1, dim); mask zeros patch tokens
        t = self.embed(x)  # (B,n,dim)
        if mask is not None:
            t = t * (~mask).unsqueeze(-1)
        t = torch.cat([self.cls.expand(x.size(0), -1, -1), t], 1) + self.pos
        for b in self.blocks:
            t = b(t)
        return self.norm(t)

    def forward(self, x):  # pooled embedding = CLS
        return self.tokens(x)[:, 0]


class CifarResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18

        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()  # CIFAR stem
        m.fc = nn.Identity()
        self.net = m
        self.embed_dim = 512

    def forward(self, x):
        return self.net(x)


def make_backbone(arch):
    if arch == "vit_s":
        return TinyViT(dim=192, depth=6, heads=3)
    if arch == "vit_b":
        return TinyViT(dim=384, depth=8, heads=6)
    return CifarResNet18()


# ------------------------- encoder (backbone + representation head) -------------------------
class Encoder(nn.Module):
    """Phi: image -> width-dim representation. `width` is the bitrate knob."""

    def __init__(self, arch, width):
        super().__init__()
        self.backbone = make_backbone(arch)
        self.repr = nn.Linear(self.backbone.embed_dim, width)
        self.width, self.arch = width, arch

    def forward(self, x):
        return self.repr(self.backbone(x))  # penultimate features (B, width)


# ------------------------- objective heads / losses -------------------------
def nt_xent(z1, z2, t=0.2):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    B = z1.size(0)
    s = z @ z.T / t
    s.fill_diagonal_(-1e4)  # -1e4 fits fp16 (amp); still masks self after softmax
    tgt = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(z.device)
    return F.cross_entropy(s, tgt)


class DinoHead(nn.Module):
    def __init__(self, win, K=4096):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(win, 256), nn.GELU(), nn.Linear(256, 256))
        self.last = nn.Linear(
            256, K, bias=False
        )  # (DINO uses weight_norm here; omitted for minimality)

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=1))


def build_objective(obj, enc, width, device):
    """Returns (params_extra, step_fn). step_fn(batch)->loss. batch shapes documented inline."""
    if obj == "supervised":
        head = nn.Linear(width, 100).to(device)

        def step(b):  # b=(x,y)
            x, y = b
            return F.cross_entropy(head(enc(x)), y)

        return head.parameters(), step
    if obj == "simclr":
        proj = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 128)).to(device)

        def step(b):  # b=(v1,v2)
            v1, v2 = b
            return nt_xent(proj(enc(v1)), proj(enc(v2)))

        return proj.parameters(), step
    if obj == "mae":
        assert enc.arch.startswith("vit"), "MAE requires ViT backbone"
        vit = enc.backbone
        p = vit.patch
        pix = 3 * p * p
        dec = nn.Sequential(nn.LayerNorm(vit.dim), nn.Linear(vit.dim, pix)).to(device)

        def patchify(x):
            B = x.size(0)
            n = vit.img // p
            return x.reshape(B, 3, n, p, n, p).permute(0, 2, 4, 1, 3, 5).reshape(B, n * n, pix)

        def step(b):  # b=(x,)
            x = b[0]
            B, N = x.size(0), vit.embed.n
            mask = torch.rand(B, N, device=x.device) < 0.75
            tok = vit.tokens(x, mask=mask)[:, 1:]  # (B,N,dim)
            pred = dec(tok)
            tgt = patchify(x)
            return (((pred - tgt) ** 2).mean(-1) * mask).sum() / mask.sum().clamp(min=1)

        return dec.parameters(), step
    if obj == "dino":
        K = 4096
        sh = DinoHead(width, K).to(device)
        teacher = copy.deepcopy(enc).to(device)
        thead = copy.deepcopy(sh).to(device)
        for q in list(teacher.parameters()) + list(thead.parameters()):
            q.requires_grad_(False)
        center = torch.zeros(1, K, device=device)
        st = {"m": 0.996, "tc": 0.04, "ts": 0.07}

        def step(b):  # b=(v1,v2)
            nonlocal center
            v1, v2 = b
            s = [F.log_softmax(sh(enc(v)) / st["ts"], dim=1) for v in (v1, v2)]
            with torch.no_grad():
                t = [F.softmax((thead(teacher(v)) - center) / st["tc"], dim=1) for v in (v1, v2)]
                center = 0.9 * center + 0.1 * torch.cat([thead(teacher(v)) for v in (v1, v2)]).mean(
                    0, keepdim=True
                )
            loss = -(t[0] * s[1]).sum(1).mean() - (t[1] * s[0]).sum(1).mean()
            with torch.no_grad():  # EMA teacher update
                for pe, pt in zip(enc.parameters(), teacher.parameters(), strict=True):
                    pt.mul_(st["m"]).add_(pe, alpha=1 - st["m"])
                for pe, pt in zip(sh.parameters(), thead.parameters(), strict=True):
                    pt.mul_(st["m"]).add_(pe, alpha=1 - st["m"])
            return loss / 2

        return sh.parameters(), step
    if obj == "simsiam":
        proj = nn.Sequential(
            nn.Linear(width, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512, affine=False),
        ).to(device)
        pred = nn.Sequential(
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 512),
        ).to(device)

        def step(b):  # b=(v1,v2); stop-grad prevents collapse (Chen & He 2021)
            v1, v2 = b
            z1, z2 = proj(enc(v1)), proj(enc(v2))
            p1, p2 = pred(z1), pred(z2)

            def d(p, z):
                return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

            return 0.5 * (d(p1, z2) + d(p2, z1))

        return list(proj.parameters()) + list(pred.parameters()), step
    raise ValueError(obj)


# ------------------------- data -------------------------
def two_crop():
    from torchvision import transforms as T

    return T.Compose(
        [
            T.RandomResizedCrop(32, (0.4, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.ToTensor(),
        ]
    )


def get_loader(data, obj, batch, train=True):
    from torchvision import datasets
    from torchvision import transforms as T

    root = "./data/torchvision"
    plain = T.Compose([T.ToTensor()])
    ds = datasets.CIFAR100(root, train=train, download=True, transform=plain)
    # data splits: d1=classes 0-49, d2=classes 50-99, d3=all (stand-in; swap to STL/TinyIN for true D3)
    if data in ("d1", "d2") and train:
        lo, hi = (0, 50) if data == "d1" else (50, 100)
        idx = [i for i, t in enumerate(ds.targets) if lo <= t < hi]
        ds = torch.utils.data.Subset(ds, idx)
    ssl = obj in ("simclr", "dino")
    aug = two_crop()

    class Wrap(torch.utils.data.Dataset):
        def __init__(s, base):
            s.base = base

        def __len__(s):
            return len(s.base)

        def __getitem__(s, i):
            img, y = s.base[i]
            if ssl:  # need PIL for two-crop; re-open via base dataset transform
                return aug(_to_pil(img)), aug(_to_pil(img)), y
            return img, y

    return torch.utils.data.DataLoader(
        Wrap(ds), batch_size=batch, shuffle=train, num_workers=4, drop_last=True
    )


def _to_pil(x):
    from torchvision.transforms.functional import to_pil_image

    return to_pil_image(x) if torch.is_tensor(x) else x


# ------------------------- run -------------------------
def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def smoke(arch, obj, width, device):
    set_seed(0)
    enc = Encoder(arch, width).to(device)
    extra, step = build_objective(obj, enc, width, device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(extra), lr=1e-3)
    x = torch.randn(8, 3, 32, 32, device=device)
    if obj == "supervised":
        batch = (x, torch.randint(0, 100, (8,), device=device))
    elif obj == "mae":
        batch = (x,)
    else:
        batch = (x, torch.randn(8, 3, 32, 32, device=device))
    for _it in range(3):
        opt.zero_grad()
        loss = step(batch)
        loss.backward()
        opt.step()
    feat = enc(x)
    print(
        f"  {arch:8s}/{obj:11s} width={width}: loss={loss.item():.4f}  feat={tuple(feat.shape)}  OK"
    )


def train(a, device):
    set_seed(a.seed)
    enc = Encoder(a.arch, a.width).to(device)
    extra, step = build_objective(a.objective, enc, a.width, device)
    loader = get_loader(a.data, a.objective, a.batch, train=True)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(extra), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(loader))
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    for ep in range(a.epochs):
        for b in loader:
            b = (
                tuple(t.to(device) for t in b[:-1])
                if a.objective in ("simclr", "dino")
                else (
                    (b[0].to(device),)
                    if a.objective == "mae"
                    else (b[0].to(device), b[1].to(device))
                )
            )
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                loss = step(b)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        print(f"epoch {ep} loss {loss.item():.4f}")
    torch.save({"state": enc.state_dict(), "config": vars(a)}, a.out)
    print("saved", a.out)


@torch.no_grad()
def extract(a, device):
    ck = torch.load(a.extract, map_location=device, weights_only=False)
    cfg = ck["config"]
    enc = Encoder(cfg["arch"], cfg["width"]).to(device)
    enc.load_state_dict(ck["state"])
    enc.eval()
    from torchvision import datasets
    from torchvision import transforms as T

    ds = datasets.CIFAR100("./data/torchvision", train=False, download=True, transform=T.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=4)
    feats = []
    for x, _ in loader:
        feats.append(enc(x.to(device)).cpu())
    F_ = torch.cat(feats)[: a.n_samples]
    name = f"{cfg['arch']}_{cfg['objective']}_{cfg['data']}_s{cfg['seed']}_w{cfg['width']}"
    torch.save(
        {
            "features": F_,
            "model_name": name,
            "model_config": {
                "arch": cfg["arch"],
                "objective": cfg["objective"],
                "data": cfg["data"],
                "seed": cfg["seed"],
                "width": cfg["width"],
                "family": f"{cfg['arch']}_{cfg['objective']}",
                "modality": "vision",
            },
        },
        a.out,
    )
    print(f"saved {a.out}: {name} feats={tuple(F_.shape)}")


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--arch", choices=["vit_s", "resnet18"], default="vit_s")
    P.add_argument(
        "--objective",
        choices=["supervised", "simclr", "mae", "dino", "simsiam"],
        default="supervised",
    )
    P.add_argument("--data", choices=["d1", "d2", "d3"], default="d1")
    P.add_argument("--seed", type=int, default=0)
    P.add_argument("--width", type=int, default=512)
    P.add_argument("--epochs", type=int, default=100)
    P.add_argument("--batch", type=int, default=256)
    P.add_argument("--lr", type=float, default=1e-3)
    P.add_argument("--out", type=str, default="ckpt.pt")
    P.add_argument("--extract", type=str, default=None)
    P.add_argument("--n-samples", type=int, default=5000)
    P.add_argument("--smoke", action="store_true")
    a = P.parse_args()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    if a.smoke:
        for arch in ["vit_s", "resnet18"]:
            for obj in ["supervised", "simclr", "mae", "dino"]:
                if obj == "mae" and arch != "vit_s":
                    continue
                smoke(arch, obj, a.width, dev)
    elif a.extract:
        extract(a, dev)
    else:
        train(a, dev)
