"""Visualize training results: loss curve, sampled Ramachandran plot, and diffusion trajectory."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from idpdiff.models.denoiser import TorsionDenoiser, ModelConfig
from idpdiff.diffusion.wrapped import WrappedDiffusion, DiffusionConfig
from idpdiff.data.embeddings import MockEmbedder, ESM_WIDTHS
from idpdiff.utils.angles import features_to_angles

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "full")
VIZ_DIR = os.path.join(OUT_DIR, "viz")
os.makedirs(VIZ_DIR, exist_ok=True)

# ── 1. Loss curve ──────────────────────────────────────────────────────
log_path = os.path.join(OUT_DIR, "train_log.jsonl")
steps, losses = [], []
with open(log_path) as f:
    for line in f:
        d = json.loads(line)
        steps.append(d["step"])
        losses.append(d["loss"])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(steps, losses, color="#2563eb", linewidth=1.5)
ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Reconstruction Loss", fontsize=12)
ax.set_title("Training Loss Curve", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_xlim(steps[0], steps[-1])
fig.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, "loss_curve.png"), dpi=150)
print(f"Saved loss_curve.png")
plt.close()

# ── 2. Load model and sample ──────────────────────────────────────────
ckpt = torch.load(os.path.join(OUT_DIR, "ema_latest.pt"), map_location="cpu", weights_only=True)
model_cfg = ModelConfig(**ckpt["model_cfg"])
model = TorsionDenoiser(model_cfg)
model.load_state_dict(ckpt["model"])
model.eval()

diffusion = WrappedDiffusion(DiffusionConfig(num_steps=200), device="cpu")
embedder = MockEmbedder(dim=model_cfg.seq_embed_dim)

test_seqs = ["AGSTYKNLDEFWPQR", "GPKDEAVLMFHRNWQ", "STAGNPDEKYWLQRFM"]
all_phi, all_psi = [], []

print("Sampling conformations (this may take a minute on CPU)...")
for seq in test_seqs:
    L = len(seq)
    emb = torch.from_numpy(embedder.embed(seq)).float().unsqueeze(0)
    mask = torch.ones(1, L, dtype=torch.bool)

    sampled = diffusion.sample(model, emb, mask, L)
    phi = sampled[0, :, 0].numpy()
    psi = sampled[0, :, 1].numpy()
    all_phi.extend(phi)
    all_psi.extend(psi)

all_phi = np.degrees(np.array(all_phi))
all_psi = np.degrees(np.array(all_psi))

# ── 3. Ramachandran plot ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
scatter = ax.scatter(all_phi, all_psi, s=20, alpha=0.6, c="#e11d48", edgecolors="white", linewidths=0.3)
ax.set_xlabel("$\\phi$ (degrees)", fontsize=13)
ax.set_ylabel("$\\psi$ (degrees)", fontsize=13)
ax.set_title("Ramachandran Plot — Sampled Conformations", fontsize=14, fontweight="bold")
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.set_xticks([-180, -90, 0, 90, 180])
ax.set_yticks([-180, -90, 0, 90, 180])
ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, "ramachandran.png"), dpi=150)
print(f"Saved ramachandran.png")
plt.close()

# ── 4. Ramachandran density heatmap ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
h = ax.hist2d(all_phi, all_psi, bins=36, range=[[-180, 180], [-180, 180]],
              cmap="magma_r", cmin=0.5)
plt.colorbar(h[3], ax=ax, label="Count", shrink=0.8)
ax.set_xlabel("$\\phi$ (degrees)", fontsize=13)
ax.set_ylabel("$\\psi$ (degrees)", fontsize=13)
ax.set_title("Ramachandran Density — Sampled Conformations", fontsize=14, fontweight="bold")
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, "ramachandran_density.png"), dpi=150)
print(f"Saved ramachandran_density.png")
plt.close()

# ── 5. Diffusion trajectory (one protein, snapshots) ──────────────────
seq = "AGSTYKNLDEFWPQR"
L = len(seq)
emb = torch.from_numpy(embedder.embed(seq)).float().unsqueeze(0)
mask = torch.ones(1, L, dtype=torch.bool)

diffusion_traj = WrappedDiffusion(DiffusionConfig(num_steps=200), device="cpu")
final_angles, trajectory = diffusion_traj.sample(model, emb, mask, L, return_trajectory=True)

n_snapshots = min(6, len(trajectory))
indices = np.linspace(0, len(trajectory) - 1, n_snapshots, dtype=int)

fig, axes = plt.subplots(1, n_snapshots, figsize=(4 * n_snapshots, 4))
for i, idx in enumerate(indices):
    snap = trajectory[idx][0].numpy()
    phi_d = np.degrees(snap[:, 0])
    psi_d = np.degrees(snap[:, 1])
    t_approx = 200 - int(idx * 200 / len(trajectory))
    axes[i].scatter(phi_d, psi_d, s=40, c="#2563eb", alpha=0.7, edgecolors="white", linewidths=0.4)
    axes[i].set_xlim(-180, 180)
    axes[i].set_ylim(-180, 180)
    axes[i].set_aspect("equal")
    axes[i].set_title(f"t ~ {t_approx}", fontsize=11)
    axes[i].set_xlabel("$\\phi$")
    if i == 0:
        axes[i].set_ylabel("$\\psi$")
    axes[i].grid(True, alpha=0.2)

fig.suptitle("Reverse Diffusion Trajectory — Noise to Structure", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(VIZ_DIR, "diffusion_trajectory.png"), dpi=150, bbox_inches="tight")
print(f"Saved diffusion_trajectory.png")
plt.close()

print(f"\nAll visualizations saved to {VIZ_DIR}")
