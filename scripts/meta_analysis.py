"""
Phase 2: Meta-Analysis for BCCT Survey Paper.

Generates:
1. Structured extraction table (CSV + LaTeX)
2. Literature landscape figure (thread × domain heatmap)
3. Temporal evolution figure (publications by year and thread)
"""

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# 1. EXTRACTION SCHEMA & DATA
# ============================================================================
# Each row: (cite_key, short_name, year, thread, domain, scale, metric_type,
#            metric_name, reported_value, bcct_regime_inferred, key_finding)
#
# thread: convergence | latent_design | transport | theory
# domain: vision | language | multimodal | cross_modal | neuroscience | theory
# scale: S (<5 models), M (5-20), L (20-100), XL (100+)
# metric_type: alignment | transport | quality | bitrate | theoretical
# bcct_regime_inferred: Convergent | Local-Only | Divergent | Collapsed | Mixed | N/A

EXTRACTION_DATA: list[dict[str, str]] = [
    # ========================================================================
    # CONVERGENCE THREAD
    # ========================================================================
    # --- Core convergence papers ---
    {
        "cite_key": "prh2024",
        "short_name": "PRH",
        "year": "2024",
        "thread": "convergence",
        "domain": "vision",
        "scale": "XL",
        "metric_type": "alignment",
        "metric_name": "mutual k-NN",
        "reported_value": "Increasing with scale",
        "bcct_regime": "Mixed",
        "key_finding": (
            "78 vision models converge toward shared statistical structure;"
            " alignment increases with model/data scale"
        ),
    },
    {
        "cite_key": "groger2026aristotelian",
        "short_name": "Aristotelian",
        "year": "2026",
        "thread": "convergence",
        "domain": "vision",
        "scale": "XL",
        "metric_type": "alignment",
        "metric_name": "null-calibrated k-NN",
        "reported_value": "Global vanishes after calibration",
        "bcct_regime": "Local-Only",
        "key_finding": (
            "After null calibration, global convergence disappears;"
            " only local neighborhood structure survives"
        ),
    },
    {
        "cite_key": "crh2025",
        "short_name": "CRH",
        "year": "2025",
        "thread": "convergence",
        "domain": "vision",
        "scale": "L",
        "metric_type": "alignment",
        "metric_name": "6 canonical relations",
        "reported_value": "Universal alignment relations",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Six alignment relations among representations, weights,"
            " and gradients universally govern feature formation"
        ),
    },
    {
        "cite_key": "park2026convergent",
        "short_name": "Convergent Reps",
        "year": "2026",
        "thread": "convergence",
        "domain": "multimodal",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "CKA, k-NN",
        "reported_value": "Multi-task drives convergence",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Multi-task training drives convergence in controlled"
            " city-coordinate worlds; divergent tasks actively harm it"
        ),
    },
    {
        "cite_key": "chawla2026metaothello",
        "short_name": "Meta-Othello",
        "year": "2026",
        "thread": "convergence",
        "domain": "language",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "board probe accuracy",
        "reported_value": "Task-dependent convergence",
        "bcct_regime": "Mixed",
        "key_finding": (
            "Mixed-variant board games confirm convergence is task-dependent, not universal"
        ),
    },
    {
        "cite_key": "kapoor2025convergent",
        "short_name": "Convergent early",
        "year": "2025",
        "thread": "convergence",
        "domain": "vision",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "epoch-wise k-NN",
        "reported_value": "Crystallizes in epoch 1",
        "bcct_regime": "Convergent",
        "key_finding": ("Nearly all alignment crystallizes within the first training epoch"),
    },
    {
        "cite_key": "dunlap2025divergent",
        "short_name": "Divergent Reps",
        "year": "2025",
        "thread": "convergence",
        "domain": "vision",
        "scale": "L",
        "metric_type": "alignment",
        "metric_name": "representation divergence",
        "reported_value": "Task-specific divergence",
        "bcct_regime": "Divergent",
        "key_finding": (
            "Identifies conditions under which representations actively"
            " diverge rather than converge"
        ),
    },
    {
        "cite_key": "calvo2026flatdino",
        "short_name": "FlatDINO",
        "year": "2026",
        "thread": "convergence",
        "domain": "vision",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "flatness + alignment",
        "reported_value": "Flat minima align better",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Compressing to 32 tokens improves alignment; flat minima"
            " produce more convergent representations"
        ),
    },
    # --- Similarity metrics papers ---
    {
        "cite_key": "kornblith2019similarity",
        "short_name": "CKA",
        "year": "2019",
        "thread": "convergence",
        "domain": "vision",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "CKA",
        "reported_value": "CKA > invariance tests",
        "bcct_regime": "N/A",
        "key_finding": (
            "CKA is more reliable than CCA-based metrics for comparing"
            " representations across architectures"
        ),
    },
    {
        "cite_key": "raghu2017svcca",
        "short_name": "SVCCA",
        "year": "2017",
        "thread": "convergence",
        "domain": "vision",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "SVCCA",
        "reported_value": "Layer-wise similarity",
        "bcct_regime": "N/A",
        "key_finding": ("SVCCA reveals layer-wise representation dynamics during training"),
    },
    {
        "cite_key": "morcos2018pwcca",
        "short_name": "PWCCA",
        "year": "2018",
        "thread": "convergence",
        "domain": "vision",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "PWCCA",
        "reported_value": "Projection-weighted CCA",
        "bcct_regime": "N/A",
        "key_finding": (
            "Projection weighting removes sensitivity to noise dimensions in CCA-based comparisons"
        ),
    },
    {
        "cite_key": "klabunde2023similarity",
        "short_name": "Similarity survey",
        "year": "2023",
        "thread": "convergence",
        "domain": "vision",
        "scale": "XL",
        "metric_type": "alignment",
        "metric_name": "multiple",
        "reported_value": "Metrics disagree",
        "bcct_regime": "N/A",
        "key_finding": (
            "Comprehensive survey showing similarity metrics often"
            " disagree on representation comparisons"
        ),
    },
    {
        "cite_key": "barannikov2022rtd",
        "short_name": "RTD",
        "year": "2022",
        "thread": "convergence",
        "domain": "vision",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "RTD (topological)",
        "reported_value": "Topological distance",
        "bcct_regime": "N/A",
        "key_finding": (
            "Representation Topology Divergence captures structural"
            " differences missed by kernel-based metrics"
        ),
    },
    {
        "cite_key": "williams2021generalized",
        "short_name": "Gen. Shape",
        "year": "2021",
        "thread": "convergence",
        "domain": "vision",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "shape metrics",
        "reported_value": "Generalized shape framework",
        "bcct_regime": "N/A",
        "key_finding": (
            "Unifies Procrustes, CKA, and other metrics within a"
            " generalized shape analysis framework"
        ),
    },
    # --- SSL / Foundation model convergence ---
    {
        "cite_key": "chen2020simclr",
        "short_name": "SimCLR",
        "year": "2020",
        "thread": "convergence",
        "domain": "vision",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "linear probe",
        "reported_value": "76.5% ImageNet top-1",
        "bcct_regime": "Local-Only",
        "key_finding": (
            "Contrastive SSL produces representations with high linear"
            " probe accuracy; temperature controls implicit bitrate"
        ),
    },
    {
        "cite_key": "he2020moco",
        "short_name": "MoCo",
        "year": "2020",
        "thread": "convergence",
        "domain": "vision",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "linear probe",
        "reported_value": "Momentum contrast",
        "bcct_regime": "Local-Only",
        "key_finding": (
            "Momentum-based contrastive learning matches supervised"
            " pre-training in transfer quality"
        ),
    },
    {
        "cite_key": "grill2020byol",
        "short_name": "BYOL",
        "year": "2020",
        "thread": "convergence",
        "domain": "vision",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "linear probe",
        "reported_value": "74.3% without negatives",
        "bcct_regime": "Local-Only",
        "key_finding": (
            "Self-supervised learning without negative pairs; predictor"
            " network acts as implicit spectral filter"
        ),
    },
    {
        "cite_key": "caron2021dino",
        "short_name": "DINO",
        "year": "2021",
        "thread": "convergence",
        "domain": "vision",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "k-NN, attention maps",
        "reported_value": "Emergent segmentation",
        "bcct_regime": "Local-Only",
        "key_finding": (
            "Self-distillation produces features with emergent semantic segmentation properties"
        ),
    },
    {
        "cite_key": "oquab2024dinov2",
        "short_name": "DINOv2",
        "year": "2024",
        "thread": "convergence",
        "domain": "vision",
        "scale": "L",
        "metric_type": "alignment",
        "metric_name": "multiple benchmarks",
        "reported_value": "Universal visual features",
        "bcct_regime": "Convergent",
        "key_finding": (
            "All-purpose visual features that approach universal"
            " representations; highest information density among SSL"
            " encoders"
        ),
    },
    {
        "cite_key": "radford2021clip",
        "short_name": "CLIP",
        "year": "2021",
        "thread": "convergence",
        "domain": "cross_modal",
        "scale": "XL",
        "metric_type": "alignment",
        "metric_name": "zero-shot transfer",
        "reported_value": "76.2% ImageNet zero-shot",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Contrastive vision-language pre-training enables zero-shot"
            " transfer; highest transport linearity in BCCT atlas"
        ),
    },
    {
        "cite_key": "sun2024evaclip",
        "short_name": "EVA-CLIP",
        "year": "2024",
        "thread": "convergence",
        "domain": "cross_modal",
        "scale": "L",
        "metric_type": "alignment",
        "metric_name": "zero-shot, linear probe",
        "reported_value": "Improved CLIP at scale",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Improved training techniques push CLIP further toward universal representations"
        ),
    },
    {
        "cite_key": "zhai2023siglip",
        "short_name": "SigLIP",
        "year": "2023",
        "thread": "convergence",
        "domain": "cross_modal",
        "scale": "L",
        "metric_type": "alignment",
        "metric_name": "sigmoid loss",
        "reported_value": "Simpler loss, same quality",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Sigmoid loss achieves SOTA alignment, suggesting"
            " cross-modal convergence is robust to loss choice"
        ),
    },
    # --- LLM convergence ---
    {
        "cite_key": "elhage2022toymodels",
        "short_name": "Superposition",
        "year": "2022",
        "thread": "convergence",
        "domain": "language",
        "scale": "S",
        "metric_type": "alignment",
        "metric_name": "feature geometry",
        "reported_value": "Superposition mechanism",
        "bcct_regime": "N/A",
        "key_finding": (
            "Networks develop superposition to represent more features"
            " than dimensions; mechanistic account of bitrate"
            " maximization"
        ),
    },
    {
        "cite_key": "bricken2023monosemanticity",
        "short_name": "Monosemantic",
        "year": "2023",
        "thread": "convergence",
        "domain": "language",
        "scale": "S",
        "metric_type": "alignment",
        "metric_name": "SAE features",
        "reported_value": "Interpretable features",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Sparse autoencoders extract monosemantic features consistent across models"
        ),
    },
    {
        "cite_key": "templeton2024scaling",
        "short_name": "Scaling SAEs",
        "year": "2024",
        "thread": "convergence",
        "domain": "language",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "SAE features at scale",
        "reported_value": "Features scale with model",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Monosemantic features scale predictably with model size;"
            " more features discovered at larger scale"
        ),
    },
    {
        "cite_key": "hernandez2024linearity",
        "short_name": "Linear relations",
        "year": "2024",
        "thread": "convergence",
        "domain": "language",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "linearity score",
        "reported_value": "Linear encoding of relations",
        "bcct_regime": "Convergent",
        "key_finding": (
            "LLMs develop linear representations of semantic relations,"
            " supporting cross-model transport"
        ),
    },
    {
        "cite_key": "gurnee2024language",
        "short_name": "Space & Time",
        "year": "2024",
        "thread": "convergence",
        "domain": "language",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "probing accuracy",
        "reported_value": "Spatial/temporal representations",
        "bcct_regime": "Convergent",
        "key_finding": (
            "LLMs develop shared representations of space and time across model families"
        ),
    },
    {
        "cite_key": "chen2025transferring",
        "short_name": "Transfer structs",
        "year": "2025",
        "thread": "convergence",
        "domain": "language",
        "scale": "M",
        "metric_type": "transport",
        "metric_name": "cross-model transfer",
        "reported_value": "Near-lossless transfer",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Internal structures can be transferred across LLMs with near-lossless quality"
        ),
    },
    {
        "cite_key": "li2024othello",
        "short_name": "Othello-GPT",
        "year": "2024",
        "thread": "convergence",
        "domain": "language",
        "scale": "S",
        "metric_type": "alignment",
        "metric_name": "world model probes",
        "reported_value": "Emergent world model",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Sequence models trained on game transcripts develop"
            " emergent internal world representations"
        ),
    },
    # --- Cross-lingual convergence ---
    {
        "cite_key": "conneau2020xlmr",
        "short_name": "XLM-R",
        "year": "2020",
        "thread": "convergence",
        "domain": "language",
        "scale": "XL",
        "metric_type": "alignment",
        "metric_name": "cross-lingual transfer",
        "reported_value": "100+ languages aligned",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Multilingual models develop shared representations across 100+ languages at scale"
        ),
    },
    {
        "cite_key": "pires2019multilingual",
        "short_name": "mBERT",
        "year": "2019",
        "thread": "convergence",
        "domain": "language",
        "scale": "L",
        "metric_type": "alignment",
        "metric_name": "cross-lingual zero-shot",
        "reported_value": "Surprising cross-lingual transfer",
        "bcct_regime": "Local-Only",
        "key_finding": (
            "Multilingual BERT enables zero-shot cross-lingual transfer"
            " without explicit alignment objective"
        ),
    },
    {
        "cite_key": "wu2019beto",
        "short_name": "Beto/Bentz/Becas",
        "year": "2019",
        "thread": "convergence",
        "domain": "language",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "cross-lingual probing",
        "reported_value": "Similar internal structure",
        "bcct_regime": "Local-Only",
        "key_finding": (
            "BERT models trained on different languages develop similar internal representations"
        ),
    },
    # --- Neuroscience convergence ---
    {
        "cite_key": "yamins2014hierarchical",
        "short_name": "DNN-Brain RSA",
        "year": "2014",
        "thread": "convergence",
        "domain": "neuroscience",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "RSA",
        "reported_value": "DNN-ventral stream match",
        "bcct_regime": "Local-Only",
        "key_finding": (
            "DNNs trained on object recognition converge toward primate"
            " ventral stream representations"
        ),
    },
    {
        "cite_key": "conwell2023what",
        "short_name": "DNN-Brain survey",
        "year": "2023",
        "thread": "convergence",
        "domain": "neuroscience",
        "scale": "XL",
        "metric_type": "alignment",
        "metric_name": "multiple",
        "reported_value": "Alignment is metric-sensitive",
        "bcct_regime": "Mixed",
        "key_finding": (
            "DNN-brain alignment is significant but sensitive to similarity metric choice"
        ),
    },
    {
        "cite_key": "soni2024metric",
        "short_name": "Metric sensitivity",
        "year": "2024",
        "thread": "convergence",
        "domain": "neuroscience",
        "scale": "L",
        "metric_type": "alignment",
        "metric_name": "metric comparison",
        "reported_value": "Metric choice alters conclusions",
        "bcct_regime": "N/A",
        "key_finding": (
            "Metric choice can fundamentally alter conclusions about DNN-brain alignment"
        ),
    },
    {
        "cite_key": "wakhloo2026manifold",
        "short_name": "Manifold capacity",
        "year": "2026",
        "thread": "convergence",
        "domain": "neuroscience",
        "scale": "M",
        "metric_type": "bitrate",
        "metric_name": "manifold capacity",
        "reported_value": "Information density metric",
        "bcct_regime": "N/A",
        "key_finding": ("Manifold capacity is the neuroscience analogue of representation bitrate"),
    },
    # ========================================================================
    # LATENT DESIGN THREAD
    # ========================================================================
    {
        "cite_key": "kingma2014vae",
        "short_name": "VAE",
        "year": "2014",
        "thread": "latent_design",
        "domain": "vision",
        "scale": "S",
        "metric_type": "quality",
        "metric_name": "ELBO",
        "reported_value": "Principled latent framework",
        "bcct_regime": "N/A",
        "key_finding": (
            "First principled framework for continuous latent spaces"
            " with explicit bitrate control via KL divergence"
        ),
    },
    {
        "cite_key": "higgins2017betavae",
        "short_name": "beta-VAE",
        "year": "2017",
        "thread": "latent_design",
        "domain": "vision",
        "scale": "S",
        "metric_type": "quality",
        "metric_name": "disentanglement",
        "reported_value": "beta controls disentanglement",
        "bcct_regime": "N/A",
        "key_finding": (
            "Increasing beta decouples reconstruction from bottleneck"
            " constraint, encouraging disentangled factors"
        ),
    },
    {
        "cite_key": "oord2017vqvae",
        "short_name": "VQ-VAE",
        "year": "2017",
        "thread": "latent_design",
        "domain": "vision",
        "scale": "S",
        "metric_type": "quality",
        "metric_name": "reconstruction",
        "reported_value": "Discrete codebook latents",
        "bcct_regime": "N/A",
        "key_finding": (
            "Discrete codebooks provide hard bitrate cap; strict R control via codebook size"
        ),
    },
    {
        "cite_key": "razavi2019vqvae2",
        "short_name": "VQ-VAE-2",
        "year": "2019",
        "thread": "latent_design",
        "domain": "vision",
        "scale": "M",
        "metric_type": "quality",
        "metric_name": "FID",
        "reported_value": "Hierarchical discrete latents",
        "bcct_regime": "N/A",
        "key_finding": (
            "Hierarchical discrete latents separate global from local"
            " structure at different bitrate levels"
        ),
    },
    {
        "cite_key": "esser2021taming",
        "short_name": "VQGAN",
        "year": "2021",
        "thread": "latent_design",
        "domain": "vision",
        "scale": "M",
        "metric_type": "quality",
        "metric_name": "FID, LPIPS",
        "reported_value": "FID ~7.9 on ImageNet",
        "bcct_regime": "N/A",
        "key_finding": (
            "Adversarial + perceptual losses decouple perceptual quality from raw bitrate"
        ),
    },
    {
        "cite_key": "rombach2022ldm",
        "short_name": "LDM/SD",
        "year": "2022",
        "thread": "latent_design",
        "domain": "vision",
        "scale": "L",
        "metric_type": "quality",
        "metric_name": "FID",
        "reported_value": "FID ~3.6 on ImageNet-256",
        "bcct_regime": "N/A",
        "key_finding": (
            "Latent diffusion in 8x-compressed VAE space enables"
            " high-quality generation at reduced compute"
        ),
    },
    {
        "cite_key": "ho2020ddpm",
        "short_name": "DDPM",
        "year": "2020",
        "thread": "latent_design",
        "domain": "vision",
        "scale": "S",
        "metric_type": "quality",
        "metric_name": "FID, IS",
        "reported_value": "FID 3.17 on CIFAR-10",
        "bcct_regime": "N/A",
        "key_finding": (
            "Denoising diffusion establishes progressive bottleneck;"
            " reverse process crystallizes structure"
        ),
    },
    {
        "cite_key": "song2021score",
        "short_name": "Score SDE",
        "year": "2021",
        "thread": "latent_design",
        "domain": "vision",
        "scale": "M",
        "metric_type": "quality",
        "metric_name": "likelihood, FID",
        "reported_value": "Unified SDE framework",
        "bcct_regime": "N/A",
        "key_finding": (
            "Unifies SMLD and DDPM under continuous-time SDE; enables exact likelihood computation"
        ),
    },
    {
        "cite_key": "dhariwal2021diffusion",
        "short_name": "Diff. beats GANs",
        "year": "2021",
        "thread": "latent_design",
        "domain": "vision",
        "scale": "M",
        "metric_type": "quality",
        "metric_name": "FID",
        "reported_value": "FID 2.97 on ImageNet-256",
        "bcct_regime": "N/A",
        "key_finding": (
            "Diffusion models surpass GANs in image quality; classifier"
            " guidance provides dynamic bitrate boost"
        ),
    },
    {
        "cite_key": "lipman2023flow",
        "short_name": "Flow Matching",
        "year": "2023",
        "thread": "latent_design",
        "domain": "vision",
        "scale": "M",
        "metric_type": "quality",
        "metric_name": "FID, likelihood",
        "reported_value": "Deterministic vector field",
        "bcct_regime": "N/A",
        "key_finding": (
            "Conditional flow matching learns deterministic transport"
            " from noise to data without SDE simulation"
        ),
    },
    {
        "cite_key": "liu2023rectified",
        "short_name": "Rectified Flow",
        "year": "2023",
        "thread": "latent_design",
        "domain": "vision",
        "scale": "M",
        "metric_type": "quality",
        "metric_name": "FID, straightness",
        "reported_value": "Straighter transport paths",
        "bcct_regime": "N/A",
        "key_finding": (
            "Rectifying flow trajectories enables few-step generation;"
            " connects OT to generative modeling"
        ),
    },
    {
        "cite_key": "esser2024sd3",
        "short_name": "SD3",
        "year": "2024",
        "thread": "latent_design",
        "domain": "vision",
        "scale": "L",
        "metric_type": "quality",
        "metric_name": "FID, human eval",
        "reported_value": "SOTA text-to-image",
        "bcct_regime": "N/A",
        "key_finding": (
            "Scaling rectified flow transformers achieves new SOTA in text-to-image generation"
        ),
    },
    {
        "cite_key": "ul2026",
        "short_name": "Unified Latents",
        "year": "2026",
        "thread": "latent_design",
        "domain": "vision",
        "scale": "L",
        "metric_type": "bitrate",
        "metric_name": "bpd, FID, FVD",
        "reported_value": "FID 1.4, FVD 1.3",
        "bcct_regime": "N/A",
        "key_finding": (
            "First system with continuous bitrate control via diffusion"
            " prior; enables causal bitrate intervention"
        ),
    },
    {
        "cite_key": "rae2025",
        "short_name": "RAE",
        "year": "2025",
        "thread": "latent_design",
        "domain": "vision",
        "scale": "M",
        "metric_type": "quality",
        "metric_name": "FID, rFID",
        "reported_value": "Semantic latents from DINOv2",
        "bcct_regime": "N/A",
        "key_finding": (
            "Using foundation model features as latent space accelerates"
            " diffusion training and improves quality"
        ),
    },
    {
        "cite_key": "dcae2025",
        "short_name": "DC-AE",
        "year": "2025",
        "thread": "latent_design",
        "domain": "vision",
        "scale": "M",
        "metric_type": "quality",
        "metric_name": "rFID, PSNR",
        "reported_value": "32x spatial compression",
        "bcct_regime": "N/A",
        "key_finding": (
            "Deep compression at 32x preserves fidelity; tests limits of extreme bitrate reduction"
        ),
    },
    {
        "cite_key": "bao2022beit",
        "short_name": "BEiT",
        "year": "2022",
        "thread": "latent_design",
        "domain": "vision",
        "scale": "M",
        "metric_type": "quality",
        "metric_name": "top-1 accuracy",
        "reported_value": "Discrete tokens for MIM",
        "bcct_regime": "N/A",
        "key_finding": (
            "Hard bitrate limit via discrete tokens serves as powerful"
            " pretext for learning transportable features"
        ),
    },
    {
        "cite_key": "tolstikhin2018wasserstein",
        "short_name": "WAE",
        "year": "2018",
        "thread": "latent_design",
        "domain": "vision",
        "scale": "S",
        "metric_type": "quality",
        "metric_name": "FID, reconstruction",
        "reported_value": "OT-based regularization",
        "bcct_regime": "N/A",
        "key_finding": (
            "Replacing KL with Wasserstein distance produces sharper"
            " reconstructions and better latent geometry"
        ),
    },
    # ========================================================================
    # TRANSPORT THREAD
    # ========================================================================
    {
        "cite_key": "bansal2021stitching",
        "short_name": "Stitching",
        "year": "2021",
        "thread": "transport",
        "domain": "vision",
        "scale": "L",
        "metric_type": "transport",
        "metric_name": "stitching accuracy",
        "reported_value": "tau ~0.95 same-seed",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Thin linear layer bridges representations; Anna Karenina effect: good models are alike"
        ),
    },
    {
        "cite_key": "moschella2023relative",
        "short_name": "Relative Reps",
        "year": "2023",
        "thread": "transport",
        "domain": "cross_modal",
        "scale": "M",
        "metric_type": "transport",
        "metric_name": "zero-shot stitching",
        "reported_value": "Zero-shot cross-model communication",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Kernel-based relative representations enable zero-shot"
            " latent communication without paired training"
        ),
    },
    {
        "cite_key": "theseus2026",
        "short_name": "Theseus",
        "year": "2026",
        "thread": "transport",
        "domain": "vision",
        "scale": "M",
        "metric_type": "transport",
        "metric_name": "cross-arch transfer",
        "reported_value": "Task vector transport",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Task vectors transport across heterogeneous architectures via Procrustes alignment"
        ),
    },
    {
        "cite_key": "fula2025",
        "short_name": "FuLA",
        "year": "2025",
        "thread": "transport",
        "domain": "vision",
        "scale": "M",
        "metric_type": "transport",
        "metric_name": "functional alignment",
        "reported_value": "Nonlinear alignment",
        "bcct_regime": "Local-Only",
        "key_finding": (
            "Functional latent alignment extends transport to"
            " cross-objective pairs where linear maps fail"
        ),
    },
    {
        "cite_key": "ifula2026",
        "short_name": "iFuLA",
        "year": "2026",
        "thread": "transport",
        "domain": "vision",
        "scale": "M",
        "metric_type": "transport",
        "metric_name": "invariance-aware transport",
        "reported_value": "Geometric constraints",
        "bcct_regime": "Local-Only",
        "key_finding": (
            "Incorporating geometric constraints improves nonlinear transport robustness"
        ),
    },
    {
        "cite_key": "aligned_sae2026",
        "short_name": "Aligned SAE",
        "year": "2026",
        "thread": "transport",
        "domain": "cross_modal",
        "scale": "S",
        "metric_type": "transport",
        "metric_name": "bimodal atoms",
        "reported_value": "Bimodal atoms carry alignment",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Specific bimodal subspaces carry the entirety of CLIP's cross-modal alignment signal"
        ),
    },
    # --- Model merging ---
    {
        "cite_key": "ilharco2023editing",
        "short_name": "Task Arithmetic",
        "year": "2023",
        "thread": "transport",
        "domain": "vision",
        "scale": "M",
        "metric_type": "transport",
        "metric_name": "task accuracy",
        "reported_value": "Additive task vectors",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Fine-tuning directions are linearly composable in weight"
            " space for task addition/removal"
        ),
    },
    {
        "cite_key": "wortsman2022soups",
        "short_name": "Model Soups",
        "year": "2022",
        "thread": "transport",
        "domain": "vision",
        "scale": "M",
        "metric_type": "transport",
        "metric_name": "OOD accuracy",
        "reported_value": "Averaging improves OOD",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Weight averaging of fine-tuned models improves out-of-distribution generalization"
        ),
    },
    {
        "cite_key": "ainsworth2023rebasin",
        "short_name": "Git Re-Basin",
        "year": "2023",
        "thread": "transport",
        "domain": "vision",
        "scale": "M",
        "metric_type": "transport",
        "metric_name": "barrier height",
        "reported_value": "Permutation removes barrier",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Loss barrier between models is largely a permutation"
            " symmetry artifact; re-basin enables merging"
        ),
    },
    {
        "cite_key": "yadav2023ties",
        "short_name": "TIES-Merging",
        "year": "2023",
        "thread": "transport",
        "domain": "vision",
        "scale": "M",
        "metric_type": "transport",
        "metric_name": "multi-task accuracy",
        "reported_value": "Resolves sign conflicts",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Trimming + sign election + averaging resolves interference in multi-task merging"
        ),
    },
    {
        "cite_key": "yu2024dare",
        "short_name": "DARE",
        "year": "2024",
        "thread": "transport",
        "domain": "language",
        "scale": "M",
        "metric_type": "transport",
        "metric_name": "LLM benchmark accuracy",
        "reported_value": "90% dropout, rescale",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Random pruning of 90% of task vector entries + rescaling"
            " effectively reduces merge interference"
        ),
    },
    {
        "cite_key": "matena2022fisher",
        "short_name": "Fisher Merging",
        "year": "2022",
        "thread": "transport",
        "domain": "vision",
        "scale": "M",
        "metric_type": "transport",
        "metric_name": "Fisher-weighted acc.",
        "reported_value": "Curvature-aware merging",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Fisher information weighting preserves critical parameters during model merging"
        ),
    },
    {
        "cite_key": "stoica2024zipit",
        "short_name": "ZipIt",
        "year": "2024",
        "thread": "transport",
        "domain": "vision",
        "scale": "M",
        "metric_type": "transport",
        "metric_name": "merged accuracy",
        "reported_value": "Feature-based zipping",
        "bcct_regime": "Local-Only",
        "key_finding": (
            "Merging layers by feature similarity enables fusion of"
            " models with different widths/architectures"
        ),
    },
    {
        "cite_key": "frankle2020linear",
        "short_name": "Linear connectivity",
        "year": "2020",
        "thread": "transport",
        "domain": "vision",
        "scale": "M",
        "metric_type": "transport",
        "metric_name": "loss barrier",
        "reported_value": "Linear connectivity from same init",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Models from same initialization exhibit linear mode"
            " connectivity; different inits separated by barriers"
        ),
    },
    # --- Optimal transport ---
    {
        "cite_key": "cuturi2013sinkhorn",
        "short_name": "Sinkhorn",
        "year": "2013",
        "thread": "transport",
        "domain": "theory",
        "scale": "N/A",
        "metric_type": "theoretical",
        "metric_name": "entropic OT",
        "reported_value": "Efficient OT solver",
        "bcct_regime": "N/A",
        "key_finding": (
            "Entropic regularization makes OT differentiable and scalable via Sinkhorn iterations"
        ),
    },
    {
        "cite_key": "peyre2019computational",
        "short_name": "Comp. OT",
        "year": "2019",
        "thread": "transport",
        "domain": "theory",
        "scale": "N/A",
        "metric_type": "theoretical",
        "metric_name": "OT survey",
        "reported_value": "Comprehensive OT framework",
        "bcct_regime": "N/A",
        "key_finding": (
            "Comprehensive computational OT framework including GW"
            " distance for heterogeneous space alignment"
        ),
    },
    {
        "cite_key": "courty2017ot",
        "short_name": "OT Domain Adapt.",
        "year": "2017",
        "thread": "transport",
        "domain": "vision",
        "scale": "M",
        "metric_type": "transport",
        "metric_name": "domain adaptation acc.",
        "reported_value": "OT preserves semantics",
        "bcct_regime": "Local-Only",
        "key_finding": (
            "OT-based domain adaptation preserves semantic structure under distribution shift"
        ),
    },
    {
        "cite_key": "moreau2026probing",
        "short_name": "OT String Method",
        "year": "2026",
        "thread": "transport",
        "domain": "vision",
        "scale": "M",
        "metric_type": "transport",
        "metric_name": "path energy",
        "reported_value": "Sharp regime transitions",
        "bcct_regime": "Mixed",
        "key_finding": (
            "Minimum energy paths between models reveal sharp phase"
            " transitions in alignment regimes"
        ),
    },
    {
        "cite_key": "sotalign2026",
        "short_name": "SOTAlign",
        "year": "2026",
        "thread": "transport",
        "domain": "vision",
        "scale": "M",
        "metric_type": "transport",
        "metric_name": "semi-supervised OT",
        "reported_value": "Anchor-constrained OT",
        "bcct_regime": "Local-Only",
        "key_finding": (
            "Semi-supervised OT with anchor samples significantly improves alignment under high TAI"
        ),
    },
    {
        "cite_key": "achara2026multiway",
        "short_name": "Multi-Way OT",
        "year": "2026",
        "thread": "transport",
        "domain": "vision",
        "scale": "L",
        "metric_type": "transport",
        "metric_name": "Wasserstein barycenter",
        "reported_value": "Multi-model alignment",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Wasserstein barycenter enables simultaneous alignment of M"
            " models into a unified latent space"
        ),
    },
    {
        "cite_key": "tong2024flow",
        "short_name": "OT-CFM",
        "year": "2024",
        "thread": "transport",
        "domain": "vision",
        "scale": "M",
        "metric_type": "quality",
        "metric_name": "FID with OT coupling",
        "reported_value": "Minibatch OT improves flow",
        "bcct_regime": "N/A",
        "key_finding": (
            "Minibatch OT couplings improve flow matching; connects"
            " generative modeling to transport theory"
        ),
    },
    # ========================================================================
    # THEORY THREAD
    # ========================================================================
    {
        "cite_key": "tishby1999information",
        "short_name": "IB Principle",
        "year": "1999",
        "thread": "theory",
        "domain": "theory",
        "scale": "N/A",
        "metric_type": "theoretical",
        "metric_name": "I(X;Z), I(Z;Y)",
        "reported_value": "IB framework",
        "bcct_regime": "N/A",
        "key_finding": (
            "Information Bottleneck principle: optimal representations"
            " compress X while preserving information about Y"
        ),
    },
    {
        "cite_key": "alemi2017deep",
        "short_name": "Deep VIB",
        "year": "2017",
        "thread": "theory",
        "domain": "theory",
        "scale": "M",
        "metric_type": "theoretical",
        "metric_name": "variational IB",
        "reported_value": "Tractable deep IB",
        "bcct_regime": "N/A",
        "key_finding": (
            "Variational approximation makes IB tractable for deep"
            " networks; connects compression to generalization"
        ),
    },
    {
        "cite_key": "shwartzziv2017opening",
        "short_name": "Opening Black Box",
        "year": "2017",
        "thread": "theory",
        "domain": "theory",
        "scale": "S",
        "metric_type": "theoretical",
        "metric_name": "information plane",
        "reported_value": "Compression phase observed",
        "bcct_regime": "N/A",
        "key_finding": (
            "DNNs exhibit a compression phase in the information plane; sparked IB-in-DL debate"
        ),
    },
    {
        "cite_key": "saxe2018information",
        "short_name": "IB critique",
        "year": "2018",
        "thread": "theory",
        "domain": "theory",
        "scale": "S",
        "metric_type": "theoretical",
        "metric_name": "information plane",
        "reported_value": "Compression is activation-dependent",
        "bcct_regime": "N/A",
        "key_finding": (
            "Compression in DNNs depends on activation function, not a universal training dynamic"
        ),
    },
    {
        "cite_key": "geoib2026",
        "short_name": "GeoIB",
        "year": "2026",
        "thread": "theory",
        "domain": "theory",
        "scale": "M",
        "metric_type": "theoretical",
        "metric_name": "Fisher-Rao, Jacobian",
        "reported_value": "Geometric IB decomposition",
        "bcct_regime": "N/A",
        "key_finding": (
            "I(X;Z) and I(Z;Y) decompose into Fisher-Rao (distribution)"
            " and Jacobian-Frobenius (geometry) terms"
        ),
    },
    {
        "cite_key": "pereg2026representation",
        "short_name": "Rep. Rate",
        "year": "2026",
        "thread": "theory",
        "domain": "theory",
        "scale": "N/A",
        "metric_type": "theoretical",
        "metric_name": "representation rate",
        "reported_value": "Fundamental limits",
        "bcct_regime": "N/A",
        "key_finding": (
            "Defines representation rate and derives fundamental limits"
            " determined by source entropy"
        ),
    },
    {
        "cite_key": "papyan2020neural",
        "short_name": "Neural Collapse",
        "year": "2020",
        "thread": "theory",
        "domain": "vision",
        "scale": "M",
        "metric_type": "alignment",
        "metric_name": "NC metrics",
        "reported_value": "Simplex ETF geometry",
        "bcct_regime": "Convergent",
        "key_finding": (
            "Terminal phase training produces simplex ETF geometry;"
            " ultimate convergence for classifiers"
        ),
    },
    {
        "cite_key": "kaplan2020scaling",
        "short_name": "Scaling Laws",
        "year": "2020",
        "thread": "theory",
        "domain": "language",
        "scale": "XL",
        "metric_type": "theoretical",
        "metric_name": "power-law fits",
        "reported_value": "Loss ~ params^{-alpha}",
        "bcct_regime": "N/A",
        "key_finding": (
            "Power-law scaling of loss with compute, data, and"
            " parameters; bitrate implications for convergence"
        ),
    },
    {
        "cite_key": "hoffmann2022chinchilla",
        "short_name": "Chinchilla",
        "year": "2022",
        "thread": "theory",
        "domain": "language",
        "scale": "XL",
        "metric_type": "theoretical",
        "metric_name": "compute-optimal scaling",
        "reported_value": "Data and params scale equally",
        "bcct_regime": "N/A",
        "key_finding": (
            "Compute-optimal training scales data and parameters"
            " equally; implications for representation quality"
        ),
    },
    {
        "cite_key": "wei2022emergent",
        "short_name": "Emergent abilities",
        "year": "2022",
        "thread": "theory",
        "domain": "language",
        "scale": "XL",
        "metric_type": "theoretical",
        "metric_name": "capability emergence",
        "reported_value": "Abilities emerge at scale",
        "bcct_regime": "N/A",
        "key_finding": (
            "Certain capabilities emerge only at sufficient scale;"
            " possible phase transition in representation quality"
        ),
    },
    {
        "cite_key": "schaeffer2023mirage",
        "short_name": "Mirage",
        "year": "2023",
        "thread": "theory",
        "domain": "language",
        "scale": "XL",
        "metric_type": "theoretical",
        "metric_name": "metric choice",
        "reported_value": "Emergence is metric-dependent",
        "bcct_regime": "N/A",
        "key_finding": (
            "Apparent emergent abilities may be artifacts of nonlinear"
            " metric choice; echoes Aristotelian critique"
        ),
    },
]


def write_csv(data: list[dict[str, str]], path: Path) -> None:
    """Write extraction data to CSV."""
    fields = [
        "cite_key",
        "short_name",
        "year",
        "thread",
        "domain",
        "scale",
        "metric_type",
        "metric_name",
        "reported_value",
        "bcct_regime",
        "key_finding",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)
    print(f"Wrote {len(data)} rows to {path}")


def generate_landscape_heatmap(data: list[dict[str, str]], out_path: Path) -> None:
    """Generate thread × domain heatmap showing literature coverage."""
    threads = ["convergence", "latent_design", "transport", "theory"]
    domains = ["vision", "language", "cross_modal", "multimodal", "neuroscience", "theory"]
    thread_labels = ["Convergence", "Latent Design", "Transport", "Theory"]
    domain_labels = ["Vision", "Language", "Cross-Modal", "Multimodal", "Neuroscience", "Theory"]

    matrix = np.zeros((len(threads), len(domains)), dtype=int)
    for row in data:
        if row["thread"] in threads and row["domain"] in domains:
            ti = threads.index(row["thread"])
            di = domains.index(row["domain"])
            matrix[ti, di] += 1

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domain_labels, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(threads)))
    ax.set_yticklabels(thread_labels, fontsize=10)

    for i in range(len(threads)):
        for j in range(len(domains)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=color,
            )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Number of papers", fontsize=10)
    ax.set_title("Literature Coverage: Thread × Domain", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved landscape heatmap to {out_path}")


def generate_temporal_plot(data: list[dict[str, str]], out_path: Path) -> None:
    """Generate temporal evolution plot: papers by year, colored by thread."""
    thread_colors = {
        "convergence": "#2196F3",
        "latent_design": "#4CAF50",
        "transport": "#FF9800",
        "theory": "#9C27B0",
    }
    thread_labels = {
        "convergence": "Convergence",
        "latent_design": "Latent Design",
        "transport": "Transport",
        "theory": "Theory",
    }

    years = sorted(set(int(r["year"]) for r in data))
    year_range = list(range(min(years), max(years) + 1))

    thread_counts: dict[str, list[int]] = {}
    for t in thread_colors:
        thread_counts[t] = [
            sum(1 for r in data if int(r["year"]) == y and r["thread"] == t) for y in year_range
        ]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bottom = np.zeros(len(year_range))
    for t in ["theory", "latent_design", "convergence", "transport"]:
        counts = np.array(thread_counts[t])
        ax.bar(
            year_range,
            counts,
            bottom=bottom,
            color=thread_colors[t],
            label=thread_labels[t],
            width=0.8,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += counts

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Number of papers", fontsize=11)
    ax.set_title("Temporal Distribution of Surveyed Literature", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    # Show every other year to avoid crowding
    ax.set_xticks(year_range)
    ax.set_xticklabels([str(y) if i % 2 == 0 else "" for i, y in enumerate(year_range)], fontsize=9)
    ax.set_ylim(0, max(bottom) + 2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved temporal plot to {out_path}")


def generate_regime_distribution(
    data: list[dict[str, str]], atlas_path: Path, out_path: Path
) -> None:
    """Generate combined regime distribution: literature + atlas."""
    # Literature regime counts
    lit_regimes = Counter(r["bcct_regime"] for r in data if r["bcct_regime"] not in ("N/A", ""))
    # Atlas regime counts
    with open(atlas_path) as f:
        atlas = json.load(f)
    atlas_regimes = Counter(p["regime"] for p in atlas["pairwise"].values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8))

    # Literature-inferred regimes
    regime_order = ["Convergent", "Local-Only", "Divergent", "Mixed"]
    regime_colors = {
        "Convergent": "#4CAF50",
        "Local-Only": "#FFC107",
        "Divergent": "#F44336",
        "Mixed": "#9E9E9E",
        "Collapsed": "#607D8B",
    }
    lit_vals = [lit_regimes.get(r, 0) for r in regime_order]
    bars1 = ax1.bar(
        regime_order,
        lit_vals,
        color=[regime_colors[r] for r in regime_order],
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, val in zip(bars1, lit_vals, strict=False):
        if val > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                str(val),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
    ax1.set_title("Literature: Inferred BCCT Regime", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of papers", fontsize=10)
    ax1.tick_params(axis="x", rotation=20)
    ymax1 = max(lit_vals) * 1.35 if lit_vals else 10
    ax1.set_ylim(0, ymax1)

    # Atlas regime distribution
    atlas_order = ["Convergent", "Local-Only", "Divergent", "Collapsed"]
    atlas_vals = [atlas_regimes.get(r, 0) for r in atlas_order]
    bars2 = ax2.bar(
        atlas_order,
        atlas_vals,
        color=[regime_colors[r] for r in atlas_order],
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, val in zip(bars2, atlas_vals, strict=False):
        if val > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                str(val),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
    ax2.set_title("Atlas: 190-Pair Regime Distribution", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Number of pairs", fontsize=10)
    ax2.tick_params(axis="x", rotation=20)
    ymax2 = max(atlas_vals) * 1.25 if atlas_vals else 10
    ax2.set_ylim(0, ymax2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved regime distribution to {out_path}")


def print_stats(data: list[dict[str, str]]) -> None:
    """Print aggregate statistics for the meta-analysis text."""
    print(f"\n{'=' * 60}")
    print(f"META-ANALYSIS STATISTICS ({len(data)} papers)")
    print(f"{'=' * 60}")

    print("\nThread distribution:")
    for t, c in sorted(Counter(r["thread"] for r in data).items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} ({100 * c / len(data):.0f}%)")

    print("\nDomain distribution:")
    for d, c in sorted(Counter(r["domain"] for r in data).items(), key=lambda x: -x[1]):
        print(f"  {d}: {c} ({100 * c / len(data):.0f}%)")

    print("\nScale distribution:")
    for s, c in sorted(Counter(r["scale"] for r in data).items()):
        print(f"  {s}: {c}")

    print(f"\nYear range: {min(int(r['year']) for r in data)}-{max(int(r['year']) for r in data)}")

    print("\nInferred BCCT regime (where applicable):")
    regime_data = [r for r in data if r["bcct_regime"] not in ("N/A", "")]
    for regime, c in sorted(
        Counter(r["bcct_regime"] for r in regime_data).items(), key=lambda x: -x[1]
    ):
        print(f"  {regime}: {c} ({100 * c / len(regime_data):.0f}%)")

    print("\nMetric types:")
    for mt, c in sorted(Counter(r["metric_type"] for r in data).items(), key=lambda x: -x[1]):
        print(f"  {mt}: {c}")


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    data_dir = root / "data"
    fig_dir = root / "paper" / "figures"

    # Write CSV
    write_csv(EXTRACTION_DATA, data_dir / "meta_analysis.csv")

    # Generate figures
    generate_landscape_heatmap(EXTRACTION_DATA, fig_dir / "fig09_literature_landscape.pdf")
    generate_temporal_plot(EXTRACTION_DATA, fig_dir / "fig10_temporal_evolution.pdf")
    generate_regime_distribution(
        EXTRACTION_DATA,
        data_dir / "atlas.json",
        fig_dir / "fig11_regime_distribution.pdf",
    )

    # Print stats
    print_stats(EXTRACTION_DATA)
