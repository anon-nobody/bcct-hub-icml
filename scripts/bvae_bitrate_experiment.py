# pyright: reportMissingTypeStubs=none, reportUnknownVariableType=none, reportUnknownMemberType=none, reportUnknownParameterType=none, reportUnknownArgumentType=none, reportMissingTypeArgument=none, reportArgumentType=none
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from latent.metrics.alignment import centered_kernel_alignment, null_calibrated_mutual_knn
from latent.metrics.bitrate import bottleneck_mismatch, representation_bitrate
from latent.metrics.regime import classify_regime
from latent.metrics.transport import transport_asymmetry_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class BetaVAE(nn.Module):
    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc_flat_dim = 256 * 2 * 2
        self.fc_mu = nn.Linear(self.enc_flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.enc_flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h = self.encoder(x)
        h_flat = h.flatten(start_dim=1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return h_flat, mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        h = self.fc_decode(z).view(z.shape[0], 256, 2, 2)
        return self.decoder(h)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h_flat, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, h_flat, mu, logvar


def vae_loss(
    recon: Tensor, x: Tensor, mu: Tensor, logvar: Tensor, beta: float
) -> tuple[Tensor, Tensor, Tensor]:
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum") / x.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    total = recon_loss + beta * kl
    return total, recon_loss, kl


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cifar10_loaders(
    data_root: Path, batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def train_single_beta(
    beta: float,
    latent_dim: int,
    epochs: int,
    train_loader: DataLoader,
    device: str,
    checkpoint_path: Path,
) -> dict[str, float]:
    model = BetaVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history: dict[str, float] = {"loss": 0.0, "recon": 0.0, "kl": 0.0}

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        recon_sum = 0.0
        kl_sum = 0.0
        n_batches = 0

        for images, _targets in tqdm(train_loader, desc=f"beta={beta} epoch={epoch + 1}/{epochs}"):
            images = images.to(device)
            recon, _h_flat, mu, logvar = model(images)
            loss, recon_loss, kl = vae_loss(recon, images, mu, logvar, beta)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
            recon_sum += float(recon_loss.item())
            kl_sum += float(kl.item())
            n_batches += 1

        history = {
            "loss": loss_sum / max(n_batches, 1),
            "recon": recon_sum / max(n_batches, 1),
            "kl": kl_sum / max(n_batches, 1),
        }
        logger.info(
            "beta=%s epoch=%d loss=%.4f recon=%.4f kl=%.4f",
            beta,
            epoch + 1,
            history["loss"],
            history["recon"],
            history["kl"],
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "beta": beta,
            "latent_dim": latent_dim,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "final_train_metrics": history,
        },
        checkpoint_path,
    )
    logger.info("Saved checkpoint: %s", checkpoint_path)
    return history


@torch.no_grad()
def extract_latent_features(
    checkpoint_path: Path,
    latent_dim: int,
    test_loader: DataLoader,
    max_samples: int,
    device: str,
) -> Tensor:
    model = BetaVAE(latent_dim=latent_dim).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    feats: list[Tensor] = []
    collected = 0
    for images, _targets in tqdm(test_loader, desc=f"features {checkpoint_path.stem}"):
        if collected >= max_samples:
            break
        images = images.to(device)
        _h_flat, mu, _logvar = model.encode(images)
        feats.append(mu.cpu())
        collected += images.shape[0]

    return torch.cat(feats, dim=0)[:max_samples]


def compute_pairwise_bcct(
    features_a: Tensor, features_b: Tensor, device: str
) -> dict[str, float | str]:
    n = min(features_a.shape[0], features_b.shape[0])
    fa = features_a[:n].to(device)
    fb = features_b[:n].to(device)

    s_local, _raw_overlap, _null_mean = null_calibrated_mutual_knn(
        fa, fb, k=10, n_permutations=100, seed=42
    )
    s_global = centered_kernel_alignment(fa, fb)
    delta = bottleneck_mismatch(fa, fb)
    transport = transport_asymmetry_index(fa, fb, mlp_steps=500, seed=42)
    tau = min(transport.a_to_b.tau, transport.b_to_a.tau)
    lambda_val = s_local / s_global if s_global > 1e-10 else float("inf")

    r_a = representation_bitrate(fa).bitrate
    r_b = representation_bitrate(fb).bitrate
    regime = classify_regime(
        bitrate=(r_a + r_b) / 2,
        s_local=s_local,
        tau=tau,
        s_global=s_global,
        lambda_val=lambda_val,
    ).regime.value

    return {
        "s_local": round(float(s_local), 4),
        "s_global": round(float(s_global), 4),
        "lambda": round(float(lambda_val), 4),
        "tau": round(float(tau), 4),
        "tau_a_to_b": round(float(transport.a_to_b.tau), 4),
        "tau_b_to_a": round(float(transport.b_to_a.tau), 4),
        "tai": round(float(transport.tai), 4),
        "delta": round(float(delta), 4),
        "regime": regime,
    }


def load_existing_model_features(features_dir: Path) -> dict[str, Tensor]:
    targets = [
        "vit_base_patch16_224.mae",
        "vit_large_patch16_224.mae",
        "beit_base_patch16_224",
        "beit_large_patch16_224",
    ]
    loaded: dict[str, Tensor] = {}
    for name in targets:
        pt_file = features_dir / f"{name}.pt"
        if not pt_file.exists():
            continue
        data = torch.load(pt_file, weights_only=True)
        feats = data.get("features")
        if isinstance(feats, torch.Tensor):
            loaded[name] = feats
    return loaded


def analyze_beta_trend(
    beta_to_bitrate: dict[float, float], pairwise: dict[str, dict[str, float | str]]
) -> str:
    betas = np.array(sorted(beta_to_bitrate.keys()), dtype=np.float32)
    bitrates = np.array([beta_to_bitrate[b] for b in betas], dtype=np.float32)

    corr = float(np.corrcoef(betas, bitrates)[0, 1]) if len(betas) > 1 else float("nan")
    divergent_pairs = sum(1 for v in pairwise.values() if v["regime"] == "Divergent")
    total_pairs = max(len(pairwise), 1)

    if np.isnan(corr):
        trend_part = "beta-bitrate trend undefined"
    elif corr < -0.2:
        trend_part = f"higher beta lowers bitrate (corr={corr:.3f})"
    elif corr > 0.2:
        trend_part = f"higher beta raises bitrate (corr={corr:.3f})"
    else:
        trend_part = f"beta-bitrate trend weak (corr={corr:.3f})"

    divergent_ratio = divergent_pairs / total_pairs
    if divergent_ratio >= 0.5:
        divergent_part = f"Divergent pairs are frequent ({divergent_pairs}/{total_pairs})"
    else:
        divergent_part = f"Divergent pairs are limited ({divergent_pairs}/{total_pairs})"

    return f"{trend_part}; {divergent_part}"


def main() -> None:
    parser = argparse.ArgumentParser(description="beta-VAE bitrate experiment on CIFAR-10")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--n-test-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="data/cifar10")
    parser.add_argument("--output", default="data/bvae_experiment.json")
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--betas", nargs="*", type=float, default=[1, 2, 4, 8, 16])
    args = parser.parse_args()

    set_seed(args.seed)
    root = Path(__file__).resolve().parent.parent
    data_root = root / args.data_root
    output_path = root / args.output
    checkpoint_dir = root / "data" / "bvae_checkpoints"
    existing_features_dir = root / args.features_dir

    train_loader, test_loader = get_cifar10_loaders(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=4,
    )

    beta_models: dict[str, dict[str, float | str]] = {}
    beta_features: dict[str, Tensor] = {}

    for beta in args.betas:
        beta_name = f"beta_{int(beta)}" if float(beta).is_integer() else f"beta_{beta}"
        ckpt_path = checkpoint_dir / f"bvae_beta_{beta_name.split('_')[1]}.pt"
        final_train = train_single_beta(
            beta=beta,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            train_loader=train_loader,
            device=args.device,
            checkpoint_path=ckpt_path,
        )
        feats = extract_latent_features(
            checkpoint_path=ckpt_path,
            latent_dim=args.latent_dim,
            test_loader=test_loader,
            max_samples=args.n_test_samples,
            device=args.device,
        )
        bitrate_result = representation_bitrate(feats.to(args.device))

        beta_models[beta_name] = {
            "beta": beta,
            "checkpoint": str(ckpt_path),
            "bitrate": round(float(bitrate_result.bitrate), 4),
            "effective_rank": round(float(bitrate_result.effective_rank_value), 2),
            "n_samples": int(feats.shape[0]),
            "dim": int(feats.shape[1]),
            "final_train_loss": round(float(final_train["loss"]), 4),
            "final_recon_loss": round(float(final_train["recon"]), 4),
            "final_kl": round(float(final_train["kl"]), 4),
            "feature_representation": "post-bottleneck mean (mu)",
        }
        beta_features[beta_name] = feats

    beta_names = sorted(beta_models.keys(), key=lambda x: float(beta_models[x]["beta"]))
    pairwise: dict[str, dict[str, float | str]] = {}
    for i, name_a in enumerate(beta_names):
        for j, name_b in enumerate(beta_names):
            if j <= i:
                continue
            pair_key = f"{name_a} <-> {name_b}"
            metrics = compute_pairwise_bcct(
                beta_features[name_a], beta_features[name_b], args.device
            )
            pairwise[pair_key] = metrics

    existing_features = load_existing_model_features(existing_features_dir)
    cross_compare: dict[str, dict[str, dict[str, float | str]]] = {}
    for beta_name in beta_names:
        cross_compare[beta_name] = {}
        for ext_name, ext_feats in existing_features.items():
            pair_key = f"{beta_name} <-> {ext_name}"
            cross_compare[beta_name][pair_key] = compute_pairwise_bcct(
                beta_features[beta_name], ext_feats, args.device
            )

    beta_to_bitrate = {
        float(beta_models[name]["beta"]): float(beta_models[name]["bitrate"]) for name in beta_names
    }
    trend_statement = analyze_beta_trend(beta_to_bitrate, pairwise)

    results = {
        "experiment": {
            "dataset": "CIFAR-10",
            "image_shape": [32, 32, 3],
            "betas": args.betas,
            "latent_dim": args.latent_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "lr": 1e-3,
            "loss": "BCE + beta * KL",
        },
        "models": beta_models,
        "pairwise": pairwise,
        "cross_compare": cross_compare,
        "trend_statement": trend_statement,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved results to %s", output_path)

    print("\n" + "=" * 72)
    print("BETA-VAE BITRATE EXPERIMENT SUMMARY")
    print("=" * 72)
    print("Bitrate vs beta:")
    for name in beta_names:
        model = beta_models[name]
        print(
            f"  beta={model['beta']}: bitrate={model['bitrate']:.4f}, "
            f"effective_rank={model['effective_rank']:.2f}"
        )

    print("\nRegime classification for beta-VAE pairs:")
    for pair_key, metrics in pairwise.items():
        print(
            f"  {pair_key}: s_local={metrics['s_local']:.4f}, tau={metrics['tau']:.4f}, "
            f"delta={metrics['delta']:.4f} -> {metrics['regime']}"
        )

    print("\nBCCT prediction check (higher beta -> lower bitrate -> more Divergent):")
    print(f"  {trend_statement}")

    if existing_features:
        print("\nCross-comparison available with existing atlas models:")
        print(f"  {', '.join(sorted(existing_features.keys()))}")
    else:
        print("\nNo existing atlas feature files found for MAE/BEiT cross-comparison.")


if __name__ == "__main__":
    main()
