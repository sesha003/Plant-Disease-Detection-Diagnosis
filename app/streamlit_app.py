"""Plant Disease Detection - Streamlit Web Application

Run with automatic latest checkpoint discovery:
    streamlit run app/streamlit_app.py

Or specify a checkpoint explicitly:
    streamlit run app/streamlit_app.py -- --checkpoint runs/resnet18_20261106-224351/best.pt
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.cm as cm  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from plantdx.models import gradcam_layer  # noqa: E402
from plantdx.predict import Predictor  # noqa: E402
from plantdx.visualize import GradCAM  # noqa: E402


def find_available_checkpoints() -> list[Path]:
    runs_dir = PROJECT_ROOT / "runs"
    if not runs_dir.exists():
        return []
    return sorted(runs_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_resource
def load_predictor(checkpoint_path: str) -> Predictor:
    return Predictor(checkpoint_path)


def compute_gradcam_overlay(predictor: Predictor, image: Image.Image, class_idx: int) -> Image.Image:
    """Generate a Grad-CAM overlay heatmap on the original image."""
    x = predictor.transform(image.convert("RGB"))[None].to(predictor.device)
    target_layer = gradcam_layer(predictor.model, predictor.model_name)
    cam = GradCAM(predictor.model, target_layer)

    # Enable gradients temporarily for Grad-CAM
    with torch.set_grad_enabled(True):
        heatmap, _, _ = cam(x, class_idx=class_idx)
    cam.remove()

    # Apply colormap (jet)
    import matplotlib as mpl

    colormap = mpl.colormaps["jet"]
    colored_heatmap = colormap(heatmap)[:, :, :3]  # drop alpha -> (H, W, 3)

    # Resize original image to match heatmap dimensions (typically 224x224)
    orig_np = np.asarray(image.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0])), dtype=np.float32) / 255.0

    # Overlay heatmap on original image
    overlay = 0.55 * orig_np + 0.45 * colored_heatmap
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def main():
    st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")

    # Parse optional CLI argument
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="Path to best.pt checkpoint")
    args, _ = ap.parse_known_args()

    # Discover checkpoints
    checkpoints = find_available_checkpoints()

    st.sidebar.title("🌿 Settings")

    if args.checkpoint:
        selected_checkpoint = args.checkpoint
        st.sidebar.info(f"Using CLI checkpoint:\n`{Path(selected_checkpoint).name}`")
    elif checkpoints:
        checkpoint_options = {str(p): f"{p.parent.name} ({p.name})" for p in checkpoints}
        selected_checkpoint = st.sidebar.selectbox(
            "Select Model Checkpoint",
            options=list(checkpoint_options.keys()),
            format_func=lambda k: checkpoint_options[k],
        )
    else:
        st.error("No model checkpoints found in `runs/`. Please train a model first.")
        st.stop()

    predictor = load_predictor(selected_checkpoint)

    st.sidebar.divider()
    show_cam = st.sidebar.checkbox("Show Grad-CAM Explanation", value=True, help="Highlights leaf regions that influenced the prediction.")

    # Main UI
    st.title("🌿 Plant Disease Detection & Diagnosis")
    st.caption(
        f"Model: **{predictor.model_name}** | Classes: **{len(predictor.class_names)}** | Device: **{predictor.device}**"
    )

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.subheader("1. Provide a Leaf Photo")
        tab_upload, tab_sample = st.tabs(["Upload Photo", "Choose Sample"])

        image: Image.Image | None = None

        with tab_upload:
            uploaded = st.file_uploader(
                "Upload a leaf image (JPG, PNG)", type=["jpg", "jpeg", "png", "bmp", "webp"]
            )
            if uploaded:
                image = Image.open(uploaded)

        with tab_sample:
            sample_dir = PROJECT_ROOT / "data" / "plantvillage" / "images" / "raw" / "color"
            if sample_dir.exists():
                sample_classes = [
                    "Apple___Apple_scab",
                    "Corn_(maize)___Common_rust_",
                    "Grape___Black_rot",
                    "Potato___Late_blight",
                    "Strawberry___healthy",
                    "Tomato___Early_blight",
                    "Tomato___healthy",
                ]
                chosen_cls = st.selectbox("Sample category", sample_classes)
                cls_path = sample_dir / chosen_cls
                if cls_path.exists():
                    sample_files = list(cls_path.glob("*.JPG"))[:5]
                    if sample_files:
                        chosen_file = st.selectbox("Sample file", sample_files, format_func=lambda p: p.name)
                        if st.button("Load this sample", use_container_width=True):
                            st.session_state["sample_img"] = str(chosen_file)

            if "sample_img" in st.session_state and Path(st.session_state["sample_img"]).exists():
                image = Image.open(st.session_state["sample_img"])

        if image is not None:
            st.image(image, caption="Input Leaf Photo", use_container_width=True)

    with col2:
        st.subheader("2. Diagnosis & Predictions")
        if image is None:
            st.info("👈 Upload a leaf photo or pick a sample to view diagnosis.")
        else:
            with st.spinner("Analyzing leaf..."):
                results = predictor(image, topk=5)

            best = results[0]
            is_healthy = best["disease"].lower() == "healthy"
            status_text = "HEALTHY LEAF ✅" if is_healthy else "DISEASE DETECTED ⚠️"

            if is_healthy:
                st.success(f"### {best['crop']} — Healthy")
            else:
                st.error(f"### {best['crop']} — {best['disease']}")

            m1, m2 = st.columns(2)
            m1.metric("Top Diagnosis", f"{best['crop']} ({best['disease']})")
            m2.metric("Confidence", f"{best['prob']:.1%}", delta=status_text, delta_color="normal" if is_healthy else "inverse")

            # Grad-CAM visualization
            if show_cam:
                st.divider()
                st.markdown("**Grad-CAM Heatmap (What the model looks at):**")
                try:
                    best_class_idx = predictor.class_names.index(best["label"])
                    overlay_img = compute_gradcam_overlay(predictor, image, best_class_idx)
                    st.image(overlay_img, caption="Grad-CAM Attention Map (Warm colors = regions driving the prediction)", use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not compute Grad-CAM overlay: {e}")

            # Top-5 probabilities
            st.divider()
            st.markdown("**Top-5 Predictions:**")
            for r in results:
                label = f"{r['crop']} — {r['disease']}"
                st.progress(float(r["prob"]), text=f"{label} ({r['prob']:.1%})")


if __name__ == "__main__":
    main()
