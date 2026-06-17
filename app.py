"""Streamlit dashboard for the IDP torsion-space diffusion model."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
from idpdiff.models.denoiser import TorsionDenoiser, ModelConfig
from idpdiff.diffusion.wrapped import WrappedDiffusion, DiffusionConfig
from idpdiff.data.embeddings import MockEmbedder
from idpdiff.utils.angles import features_to_angles
from idpdiff.validation.reconstruct import (
    reconstruct_backbone, radius_of_gyration, end_to_end_distance, clash_count,
)
from idpdiff.eval.metrics import ensemble_observables
from idpdiff.data.torsions import THREE_TO_ONE

ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}

PRESET_SEQUENCES = [
    "AGSTYKNLDEFWPQR",
    "GPKDEAVLMFHRNWQ",
    "STAGNPDEKYWLQRFM",
]

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**ckpt["model_cfg"])
    model = TorsionDenoiser(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt["model_cfg"]


@st.cache_resource
def get_diffusion(num_steps: int):
    return WrappedDiffusion(DiffusionConfig(num_steps=num_steps), device="cpu")


@st.cache_resource
def get_embedder(dim: int):
    return MockEmbedder(dim=dim)


@st.cache_data
def load_training_log(log_path: str):
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@st.cache_data
def load_config(config_path: str):
    with open(config_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def coords_to_pdb_string(coords: np.ndarray, sequence: str) -> str:
    atom_names = ["N", "CA", "C"]
    lines = []
    serial = 1
    for i in range(coords.shape[0]):
        resname = ONE_TO_THREE.get(sequence[i], "GLY") if i < len(sequence) else "GLY"
        for a, name in enumerate(atom_names):
            x, y, z = coords[i, a]
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s} {resname} A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
            )
            serial += 1
    lines.append("END")
    return "\n".join(lines)


def find_run_dirs():
    runs_dir = os.path.join(PROJECT_ROOT, "runs")
    if not os.path.isdir(runs_dir):
        return []
    dirs = []
    for name in sorted(os.listdir(runs_dir)):
        p = os.path.join(runs_dir, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "train_log.jsonl")):
            dirs.append(name)
    return dirs


def validate_sequence(seq: str) -> str | None:
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    seq = seq.upper().strip()
    if not seq:
        return None
    bad = set(seq) - valid
    if bad:
        st.error(f"Invalid amino acids: {', '.join(sorted(bad))}")
        return None
    if len(seq) > 50:
        st.warning("Sequence capped at 50 residues for CPU performance.")
        seq = seq[:50]
    return seq


# ---------------------------------------------------------------------------
# Page: Training Monitor
# ---------------------------------------------------------------------------

def render_training_monitor():
    st.header("Training Monitor")

    run_dirs = find_run_dirs()
    if not run_dirs:
        st.warning("No training runs found in `runs/`. Run training first.")
        return

    selected = st.sidebar.selectbox("Select run", run_dirs)
    run_path = os.path.join(PROJECT_ROOT, "runs", selected)
    log_path = os.path.join(run_path, "train_log.jsonl")
    records = load_training_log(log_path)

    if not records:
        st.info("Training log is empty.")
        return

    steps = [r["step"] for r in records]
    losses = [r["loss"] for r in records]
    recons = [r["recon"] for r in records]
    regs = [r["reg"] for r in records]
    lrs = [r["lr"] for r in records]
    lambdas = [r["lambda"] for r in records]

    # Loss curves
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=losses, name="Total Loss", line=dict(color="#2563eb", width=2)))
    fig.add_trace(go.Scatter(x=steps, y=recons, name="Reconstruction", line=dict(color="#16a34a", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=steps, y=regs, name="Regularizer", line=dict(color="#dc2626", width=1.5, dash="dot")))
    fig.update_layout(title="Loss Curves", xaxis_title="Step", yaxis_title="Loss", height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # LR and Lambda
    col1, col2 = st.columns(2)
    with col1:
        fig_lr = go.Figure()
        fig_lr.add_trace(go.Scatter(x=steps, y=lrs, line=dict(color="#7c3aed", width=2)))
        fig_lr.update_layout(title="Learning Rate", xaxis_title="Step", yaxis_title="LR", height=300, template="plotly_white")
        st.plotly_chart(fig_lr, use_container_width=True)

    with col2:
        fig_lam = go.Figure()
        fig_lam.add_trace(go.Scatter(x=steps, y=lambdas, line=dict(color="#ea580c", width=2)))
        fig_lam.update_layout(title="Regularizer Weight (λ)", xaxis_title="Step", yaxis_title="λ", height=300, template="plotly_white")
        st.plotly_chart(fig_lam, use_container_width=True)

    # Config
    config_path = os.path.join(run_path, "config.json")
    if os.path.exists(config_path):
        with st.expander("Run Configuration"):
            st.json(load_config(config_path))


# ---------------------------------------------------------------------------
# Page: Sample & Visualize
# ---------------------------------------------------------------------------

def render_sample_visualize():
    st.header("Sample & Visualize")

    # Sidebar controls
    st.sidebar.subheader("Sampling Controls")
    preset = st.sidebar.selectbox("Preset sequence", ["Custom"] + PRESET_SEQUENCES)
    if preset == "Custom":
        seq_input = st.sidebar.text_input("Protein sequence", value="AGSTYKNLDEFWPQR")
    else:
        seq_input = preset

    n_conf = st.sidebar.slider("Conformations", 1, 50, 10)
    num_steps = st.sidebar.slider("Diffusion steps", 50, 500, 200, step=50)
    generate = st.sidebar.button("Generate", type="primary")

    # Find checkpoint
    ckpt_path = os.path.join(PROJECT_ROOT, "runs", "full", "ema_latest.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "runs", "smoke", "ema_latest.pt")
    if not os.path.exists(ckpt_path):
        st.error("No checkpoint found. Run training first.")
        return

    if generate:
        seq = validate_sequence(seq_input)
        if seq is None:
            return

        model, model_cfg, _ = load_model(ckpt_path)
        diffusion = get_diffusion(num_steps)
        embedder = get_embedder(model_cfg.seq_embed_dim)

        L = len(seq)
        emb = torch.from_numpy(embedder.embed(seq)).float().unsqueeze(0).repeat(n_conf, 1, 1)
        mask = torch.ones(n_conf, L, dtype=torch.bool)

        with st.spinner(f"Sampling {n_conf} conformations ({num_steps} steps)..."):
            with torch.no_grad():
                angles = diffusion.sample(model, emb, mask, L)
            angles_np = angles.cpu().numpy()

        st.session_state["sampled_angles"] = angles_np
        st.session_state["sampled_seq"] = seq

    if "sampled_angles" not in st.session_state:
        st.info("Configure parameters in the sidebar and click **Generate**.")
        return

    angles_np = st.session_state["sampled_angles"]
    seq = st.session_state["sampled_seq"]
    n_conf_actual = angles_np.shape[0]

    # Ramachandran plots
    phi_all = np.degrees(angles_np[:, :, 0].flatten())
    psi_all = np.degrees(angles_np[:, :, 1].flatten())

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Scattergl(
            x=phi_all, y=psi_all, mode="markers",
            marker=dict(size=4, color="#e11d48", opacity=0.5),
        ))
        fig.update_layout(
            title="Ramachandran Scatter", xaxis_title="φ (degrees)", yaxis_title="ψ (degrees)",
            xaxis=dict(range=[-180, 180], dtick=90), yaxis=dict(range=[-180, 180], dtick=90),
            width=500, height=500, template="plotly_white",
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        hist, xedges, yedges = np.histogram2d(phi_all, psi_all, bins=36, range=[[-180, 180], [-180, 180]])
        fig = go.Figure(go.Heatmap(
            z=hist.T, x=xedges[:-1], y=yedges[:-1], colorscale="Magma_r",
        ))
        fig.update_layout(
            title="Ramachandran Density", xaxis_title="φ (degrees)", yaxis_title="ψ (degrees)",
            xaxis=dict(range=[-180, 180]), yaxis=dict(range=[-180, 180]),
            width=500, height=500, template="plotly_white",
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)

    # 3D structure viewer
    st.subheader("3D Backbone Structure")
    conf_idx = st.slider("Conformer", 0, n_conf_actual - 1, 0) if n_conf_actual > 1 else 0
    show_overlay = st.checkbox("Overlay all conformers", value=False) if n_conf_actual > 1 else False

    try:
        import py3Dmol
        from stmol import showmol

        view = py3Dmol.view(width=700, height=500)
        if show_overlay:
            colors = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed",
                      "#0891b2", "#be185d", "#4d7c0f", "#9333ea", "#b45309"]
            for k in range(min(n_conf_actual, 10)):
                coords = reconstruct_backbone(angles_np[k, :, 0], angles_np[k, :, 1])
                pdb_str = coords_to_pdb_string(coords, seq)
                view.addModel(pdb_str, "pdb")
                view.setStyle({"model": k}, {"cartoon": {"color": colors[k % len(colors)]}})
        else:
            coords = reconstruct_backbone(angles_np[conf_idx, :, 0], angles_np[conf_idx, :, 1])
            pdb_str = coords_to_pdb_string(coords, seq)
            view.addModel(pdb_str, "pdb")
            view.setStyle({"cartoon": {"color": "spectrum"}})
        view.zoomTo()
        showmol(view, height=500, width=700)
    except Exception as e:
        st.warning(f"3D viewer unavailable: {e}")

    # Observables
    st.subheader("Physical Observables")
    with st.spinner("Computing observables..."):
        obs = ensemble_observables(angles_np)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Rg", f"{obs['rg'].mean():.2f} Å")
    with col2:
        st.metric("Mean Ree", f"{obs['ree'].mean():.2f} Å")
    with col3:
        clash_counts = []
        for k in range(n_conf_actual):
            coords = reconstruct_backbone(angles_np[k, :, 0], angles_np[k, :, 1])
            clash_counts.append(clash_count(coords[:, 1, :]))
        st.metric("Mean Clashes", f"{np.mean(clash_counts):.1f}")

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = go.Figure(go.Histogram(x=obs["rg"], nbinsx=20, marker_color="#2563eb"))
        fig.update_layout(title="Radius of Gyration", xaxis_title="Rg (Å)", yaxis_title="Count", height=300, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure(go.Histogram(x=obs["ree"], nbinsx=20, marker_color="#16a34a"))
        fig.update_layout(title="End-to-End Distance", xaxis_title="Ree (Å)", yaxis_title="Count", height=300, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = go.Figure(go.Histogram(x=obs["asphericity"], nbinsx=20, marker_color="#7c3aed"))
        fig.update_layout(title="Asphericity", xaxis_title="Asphericity", yaxis_title="Count", height=300, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Diffusion Process
# ---------------------------------------------------------------------------

def render_diffusion_process():
    st.header("Diffusion Process")

    st.sidebar.subheader("Diffusion Controls")
    seq_input = st.sidebar.text_input("Sequence", value="AGSTYKNLDEFWPQR", key="diff_seq")
    num_steps = st.sidebar.slider("Diffusion steps", 50, 500, 200, step=50, key="diff_steps")
    run_diff = st.sidebar.button("Run Diffusion", type="primary")

    ckpt_path = os.path.join(PROJECT_ROOT, "runs", "full", "ema_latest.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "runs", "smoke", "ema_latest.pt")
    if not os.path.exists(ckpt_path):
        st.error("No checkpoint found.")
        return

    if run_diff:
        seq = validate_sequence(seq_input)
        if seq is None:
            return

        model, model_cfg, _ = load_model(ckpt_path)
        diffusion = get_diffusion(num_steps)
        embedder = get_embedder(model_cfg.seq_embed_dim)

        L = len(seq)
        emb = torch.from_numpy(embedder.embed(seq)).float().unsqueeze(0)
        mask = torch.ones(1, L, dtype=torch.bool)

        with st.spinner(f"Running reverse diffusion ({num_steps} steps)..."):
            with torch.no_grad():
                final, trajectory = diffusion.sample(model, emb, mask, L, return_trajectory=True)

        st.session_state["trajectory"] = [t[0].numpy() for t in trajectory]
        st.session_state["traj_num_steps"] = num_steps

    if "trajectory" not in st.session_state:
        st.info("Click **Run Diffusion** to visualize the denoising process.")
        return

    trajectory = st.session_state["trajectory"]
    num_steps = st.session_state["traj_num_steps"]

    # Animated playback
    st.subheader("Animated Denoising")
    frames = []
    for idx, snap in enumerate(trajectory):
        phi_d = np.degrees(snap[:, 0])
        psi_d = np.degrees(snap[:, 1])
        t_approx = num_steps - int(idx * num_steps / max(1, len(trajectory) - 1))
        frames.append(go.Frame(
            data=[go.Scatter(x=phi_d, y=psi_d, mode="markers",
                             marker=dict(size=8, color="#2563eb", opacity=0.7))],
            name=str(idx),
            layout=go.Layout(title_text=f"Reverse Diffusion — t ≈ {t_approx}")
        ))

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            xaxis=dict(range=[-180, 180], title="φ (degrees)", dtick=90),
            yaxis=dict(range=[-180, 180], title="ψ (degrees)", dtick=90, scaleanchor="x", scaleratio=1),
            height=550, width=600, template="plotly_white",
            title="Reverse Diffusion — t ≈ " + str(num_steps),
            updatemenus=[dict(
                type="buttons", showactive=False, y=1.12, x=0.5, xanchor="center",
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                ]
            )]
        ),
        frames=frames,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interactive slider
    st.subheader("Step-Through")
    snap_idx = st.slider("Snapshot", 0, len(trajectory) - 1, len(trajectory) - 1)
    snap = trajectory[snap_idx]
    t_approx = num_steps - int(snap_idx * num_steps / max(1, len(trajectory) - 1))

    fig = go.Figure(go.Scatter(
        x=np.degrees(snap[:, 0]), y=np.degrees(snap[:, 1]),
        mode="markers", marker=dict(size=10, color="#e11d48", opacity=0.7,
                                     line=dict(width=1, color="white")),
    ))
    fig.update_layout(
        title=f"Ramachandran at t ≈ {t_approx}",
        xaxis=dict(range=[-180, 180], title="φ (degrees)", dtick=90),
        yaxis=dict(range=[-180, 180], title="ψ (degrees)", dtick=90, scaleanchor="x", scaleratio=1),
        height=500, width=550, template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Snapshot grid
    st.subheader("Denoising Snapshots")
    n_show = min(8, len(trajectory))
    indices = np.linspace(0, len(trajectory) - 1, n_show, dtype=int)
    rows = 2 if n_show > 4 else 1
    cols = min(n_show, 4)

    titles = []
    for idx in indices:
        t_approx = num_steps - int(idx * num_steps / max(1, len(trajectory) - 1))
        titles.append(f"t ≈ {t_approx}")

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)
    for i, idx in enumerate(indices):
        r = i // cols + 1
        c = i % cols + 1
        snap = trajectory[idx]
        fig.add_trace(go.Scatter(
            x=np.degrees(snap[:, 0]), y=np.degrees(snap[:, 1]),
            mode="markers", marker=dict(size=5, color="#2563eb", opacity=0.7),
            showlegend=False,
        ), row=r, col=c)
        fig.update_xaxes(range=[-180, 180], row=r, col=c)
        fig.update_yaxes(range=[-180, 180], row=r, col=c)

    fig.update_layout(height=300 * rows, template="plotly_white", title_text="Noise → Structure")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Model Info
# ---------------------------------------------------------------------------

def render_model_info():
    st.header("Model Information")

    ckpt_path = os.path.join(PROJECT_ROOT, "runs", "full", "ema_latest.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "runs", "smoke", "ema_latest.pt")
    if not os.path.exists(ckpt_path):
        st.error("No checkpoint found.")
        return

    model, model_cfg, cfg_dict = load_model(ckpt_path)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Architecture")
        st.metric("Total Parameters", f"{model.num_parameters():,}")
        st.table({k: str(v) for k, v in cfg_dict.items()})

    with col2:
        st.subheader("Checkpoint")
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        st.metric("Checkpoint Size", f"{size_mb:.1f} MB")
        st.text(f"Path: {ckpt_path}")

    # Run config
    for run_name in ["full", "smoke"]:
        config_path = os.path.join(PROJECT_ROOT, "runs", run_name, "config.json")
        if os.path.exists(config_path):
            with st.expander(f"Run config: {run_name}"):
                st.json(load_config(config_path))

    st.subheader("Diffusion Config (defaults)")
    dc = DiffusionConfig()
    st.table({
        "num_steps": dc.num_steps,
        "sigma_max": dc.sigma_max,
        "sigma_min": dc.sigma_min,
        "schedule_s": dc.schedule_s,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="IDP Diffusion Explorer",
        page_icon="🧬",
        layout="wide",
    )

    st.sidebar.title("IDP Diffusion")
    page = st.sidebar.radio("Navigation", [
        "Training Monitor",
        "Sample & Visualize",
        "Diffusion Process",
        "Model Info",
    ])

    if page == "Training Monitor":
        render_training_monitor()
    elif page == "Sample & Visualize":
        render_sample_visualize()
    elif page == "Diffusion Process":
        render_diffusion_process()
    elif page == "Model Info":
        render_model_info()


if __name__ == "__main__":
    main()
