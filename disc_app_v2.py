import streamlit as st
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import io
import zipfile
import os
import re
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Register HEIC opener if the package is installed
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# ----------------------------
# Page Config & Session State
# ----------------------------
st.set_page_config(page_title="DiscAnalyzer-2.0", layout="wide")

if 'hsv_params' not in st.session_state:
    st.session_state.hsv_params = {'h_min': 110, 'h_max': 165, 's_min': 20, 's_max': 255, 'v_min': 30, 'v_max': 255}
if 'clip_params' not in st.session_state:
    st.session_state.clip_params = {'nonoverlap_gap': 0.0, 'edge_shave_pct': 0.0, 'refine_edges': True}
if 'm1_all_detections' not in st.session_state:
    st.session_state.m1_all_detections = {}
if 'm1_preview_idx' not in st.session_state:
    st.session_state.m1_preview_idx = 0
if 'm1_out_dir' not in st.session_state:
    st.session_state.m1_out_dir = ''
if 'm1_out_dir_text' not in st.session_state:
    st.session_state.m1_out_dir_text = st.session_state.m1_out_dir
if 'm1_output_mode' not in st.session_state:
    st.session_state.m1_output_mode = "Download ZIP (Deployment-safe)"
if 'm1_zip_bytes' not in st.session_state:
    st.session_state.m1_zip_bytes = None
if 'm1_zip_name' not in st.session_state:
    st.session_state.m1_zip_name = "clipped_discs.zip"
if 'm1_latest_discs' not in st.session_state:
    st.session_state.m1_latest_discs = []
if 'm2_preview_idx' not in st.session_state:
    st.session_state.m2_preview_idx = 0
if 'm3_in_dir' not in st.session_state:
    st.session_state.m3_in_dir = ''
if 'm3_out_dir' not in st.session_state:
    st.session_state.m3_out_dir = ''
if 'm3_in_text' not in st.session_state:
    st.session_state.m3_in_text = st.session_state.m3_in_dir
if 'm3_out_text' not in st.session_state:
    st.session_state.m3_out_text = st.session_state.m3_out_dir
if 'm3_output_mode' not in st.session_state:
    st.session_state.m3_output_mode = "Download ZIP (Deployment-safe)"
if 'm3_zip_bytes' not in st.session_state:
    st.session_state.m3_zip_bytes = None
if 'm3_zip_name' not in st.session_state:
    st.session_state.m3_zip_name = "module3_results.zip"
# Slider state bound directly to widget keys for smooth dragging
if 'm2_h_range' not in st.session_state:
    _p0 = st.session_state.hsv_params
    st.session_state.m2_h_range = (_p0['h_min'], _p0['h_max'])
    st.session_state.m2_s_range = (_p0['s_min'], _p0['s_max'])
    st.session_state.m2_v_range = (_p0['v_min'], _p0['v_max'])

# Allowed image types for Streamlit uploaders
ALLOWED_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "heic", "jfif"]

# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize_dir_input(path_text: str) -> str:
    """Trim surrounding whitespace/quotes from a manually pasted directory path."""
    if not path_text:
        return ""
    return path_text.strip().strip('"').strip("'")

def looks_like_windows_drive_path(path_text: str) -> bool:
    """Detect paths like C:\\... so we can prevent misleading saves on non-Windows hosts."""
    return bool(re.match(r"^[a-zA-Z]:[\\/]", path_text or ""))

def encode_png_bytes(img_bgra):
    ok, enc = cv2.imencode('.png', img_bgra)
    if not ok:
        return None
    return enc.tobytes()

def write_bytes_to_path(file_path: Path, payload: bytes) -> bool:
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open('wb') as f:
            f.write(payload)
        return True
    except Exception:
        return False

def extract_images_from_zip_bytes(zip_bytes: bytes):
    """Extract supported image files from ZIP bytes as [{'name': str, 'bytes': bytes}, ...]."""
    items = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), mode='r') as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower().lstrip('.')
            if ext in ALLOWED_TYPES:
                items.append({
                    "name": Path(info.filename).name,
                    "bytes": zf.read(info.filename),
                })
    return items

def validate_local_output_dir(dir_text: str):
    clean = sanitize_dir_input(dir_text)
    if not clean:
        return False, None, "Please provide an output directory path."
    if (os.name != 'nt') and looks_like_windows_drive_path(clean):
        return False, None, "This app instance is not running on Windows, so a Windows path cannot be used here. Use 'Download ZIP (Deployment-safe)'."
    out_path = Path(clean).expanduser()
    try:
        ensure_dir(out_path)
        probe = out_path / ".write_probe.tmp"
        write_ok = write_bytes_to_path(probe, b"ok")
        if write_ok and probe.exists():
            probe.unlink(missing_ok=True)
        if not write_ok:
            return False, None, "Output directory is not writable. Check permissions and path validity."
        return True, out_path, ""
    except Exception as e:
        return False, None, f"Output directory is invalid or not writable: {e}"

def _convert_pil_to_bgr_alpha(pil_img):
    pil_img = ImageOps.exif_transpose(pil_img)
    img = np.array(pil_img)
    
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    elif img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    elif img.shape[2] == 4:
        bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
        alpha = img[:, :, 3]
    else:
        raise ValueError("Unsupported image channels.")
    return bgr, alpha

def load_bgr_alpha_bytes(file_bytes):
    try:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is not None:
            return _parse_channels(img)
    except Exception:
        pass
    
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        return _convert_pil_to_bgr_alpha(pil_img)
    except Exception as e:
        raise ValueError(f"Failed to load image. Error: {e}")

def load_bgr_alpha_path(path):
    try:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            return _parse_channels(img)
    except Exception:
        pass
    
    try:
        pil_img = Image.open(path)
        return _convert_pil_to_bgr_alpha(pil_img)
    except Exception as e:
        raise ValueError(f"Failed to load image: {path}. Error: {e}")

def _parse_channels(img):
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    elif img.shape[2] == 3:
        bgr = img
        alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    elif img.shape[2] == 4:
        bgr, alpha = img[:, :, :3], img[:, :, 3]
    else:
        raise ValueError("Unsupported image channels.")
    return bgr, alpha

# ----------------------------
# Folder Browser (local Windows)
# ----------------------------
def browse_folder(initial_dir: str = '') -> str:
    """Open a Windows Explorer folder-selection dialog and return the chosen path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        folder = filedialog.askdirectory(
            master=root,
            initialdir=initial_dir if initial_dir else '/',
            title='Select Folder'
        )
        root.destroy()
        return folder or ''
    except Exception:
        return ''

# ----------------------------
# Grid Sort — column-major order
# ----------------------------
def sort_circles_column_major(circles):
    """Sort circles top-left first, then top→bottom within each column, columns left→right."""
    if not circles:
        return []
    median_r = float(np.median([c['r'] for c in circles]))
    col_threshold = median_r * 0.75          # x-spread tolerance to be in the same column

    sorted_x = sorted(circles, key=lambda c: c['x'])
    columns = []
    for c in sorted_x:
        placed = False
        for col in columns:
            if abs(c['x'] - float(np.mean([d['x'] for d in col]))) < col_threshold:
                col.append(c)
                placed = True
                break
        if not placed:
            columns.append([c])

    for col in columns:
        col.sort(key=lambda c: c['y'])
    columns.sort(key=lambda col: float(np.mean([c['x'] for c in col])))
    return [c for col in columns for c in col]

# ----------------------------
# Auto-Detect Purple HSV Params
# ----------------------------
def auto_detect_purple_params(bgr, alpha):
    """Estimate HSV bounds for purple colony detection from a clipped disc image."""
    vis_mask = alpha > 128
    if not np.any(vis_mask):
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Isolate colored (non-white, non-dark) pixels inside the disc
    colored = vis_mask & (s_ch > 30) & (v_ch > 30) & (v_ch < 248)
    if not np.any(colored):
        return {'h_min': 110, 'h_max': 165, 's_min': 20, 's_max': 255, 'v_min': 30, 'v_max': 255}
    hues = h_ch[colored].astype(np.float32)
    sats = s_ch[colored]
    vals = v_ch[colored]
    # Purple hue in OpenCV 0-179: ~105-175 (represents ~210-350°)
    sel = (hues >= 105) & (hues <= 175)
    ph, ps, pv = hues[sel], sats[sel], vals[sel]
    if len(ph) < 20:
        return {'h_min': 110, 'h_max': 165, 's_min': 20, 's_max': 255, 'v_min': 30, 'v_max': 255}
    h_min = max(0,   int(np.percentile(ph, 3))  - 8)
    h_max = min(179, int(np.percentile(ph, 97)) + 8)
    s_min = max(10,  int(np.percentile(ps, 10)) - 5)
    v_min = max(20,  int(np.percentile(pv, 5))  - 10)
    return {'h_min': h_min, 'h_max': h_max, 's_min': s_min, 's_max': 255, 'v_min': v_min, 'v_max': 255}

def compute_green_residual_mask(bgr_img, hsv_img, vis_mask, sat_min=38, val_min=20):
    """Detect residual green pixels only near the disc border for conservative exclusion."""
    vis_u8 = vis_mask.astype(np.uint8) * 255
    vis_area = int(np.count_nonzero(vis_mask))
    if vis_area == 0:
        return np.zeros_like(vis_mask, dtype=bool)

    # Build a thin edge annulus from the visible disc mask so removal cannot reach deep interior.
    est_radius = max(1, int(np.sqrt(vis_area / np.pi)))
    edge_band_px = max(2, int(est_radius * 0.05))
    erode_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (edge_band_px * 2 + 1, edge_band_px * 2 + 1)
    )
    inner = cv2.erode(vis_u8, erode_k, iterations=1)
    edge_band = (cv2.bitwise_and(vis_u8, cv2.bitwise_not(inner)) > 0)

    # Green hue gate plus channel-dominance gate to avoid brown/purple false positives.
    green_hsv = cv2.inRange(hsv_img, np.array([40, sat_min, val_min]), np.array([82, 255, 255])) > 0
    b_ch, g_ch, r_ch = cv2.split(bgr_img)
    green_dom = (g_ch.astype(np.float32) > r_ch.astype(np.float32) + 6) & \
                (g_ch.astype(np.float32) > b_ch.astype(np.float32) + 6)

    green_mask = edge_band & green_hsv & green_dom
    if not np.any(green_mask):
        return green_mask

    # Keep excluded region tight to the detected edge residue.
    return cv2.dilate(green_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0

# ----------------------------
# Cached Image Prep — Module 2
# ----------------------------
@st.cache_data(show_spinner=False)
def prepare_m2_image(file_bytes):
    """Load, optionally downscale, and pre-convert to HSV. Cached so slider drags don't re-load."""
    bgr, alpha = load_bgr_alpha_bytes(file_bytes)
    h, w = bgr.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        bgr   = cv2.resize(bgr,   (new_w, new_h), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    hsv_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return bgr, alpha, hsv_img

def save_comparison_plot(original_bgr, classified_bgr, white_pct, blue_pct, out_path):
    plot_bytes = build_comparison_plot_bytes(original_bgr, classified_bgr, white_pct, blue_pct)
    write_bytes_to_path(Path(out_path), plot_bytes)

def build_comparison_plot_bytes(original_bgr, classified_bgr, white_pct, blue_pct):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

    axes[0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(classified_bgr, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Classified")
    axes[1].axis('off')

    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label=f'Classified: {blue_pct:.1f}%'),
        Patch(facecolor='white', edgecolor='black', label=f'Non-Classified: {white_pct:.1f}%'),
        Patch(facecolor='lightgray', edgecolor='black', label='Background')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=True)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ----------------------------
# Chroma Key Disc Detection
# ----------------------------
def auto_robust_disc_detection(bgr, nonoverlap_gap=0.0):
    h, w = bgr.shape[:2]
    img_area = h * w

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    
    blurred = cv2.GaussianBlur(a_channel, (9, 9), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_candidates = []
    for cnt in contours:
        if cv2.contourArea(cnt) < (img_area * 0.001): 
            continue

        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull, True)
        if perimeter == 0: continue

        circularity = 4 * np.pi * (area / (perimeter ** 2))
        (cx, cy), cr = cv2.minEnclosingCircle(hull)

        if circularity > 0.80:
            raw_candidates.append({
                'x': int(cx), 'y': int(cy), 'r': int(cr),
                'score': circularity
            })

    if not raw_candidates:
        return []

    radii = [c['r'] for c in raw_candidates]
    median_r = np.median(radii)

    size_filtered = [c for c in raw_candidates if (0.85 * median_r) <= c['r'] <= (1.15 * median_r)]
    size_filtered.sort(key=lambda c: c['score'], reverse=True)
    
    chosen = []
    for c in size_filtered:
        overlap = False
        for d in chosen:
            dist = np.hypot(c['x'] - d['x'], c['y'] - d['y'])
            if dist < (c['r'] + d['r'] + nonoverlap_gap):
                overlap = True
                break
        if not overlap:
            chosen.append(c)

    return chosen

def refine_alpha_mask(roi_bgr, geometric_mask):
    """Refines using Global A-Channel Histogram Splitting to neutralize lighting gradients."""
    # 1. Convert to LAB and extract A Channel (Green vs. Magenta axis)
    # Green table = Low Values (~80-110). White/Purple Paper = High Values (~128-170).
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    
    # 2. Otsu Threshold on the ENTIRE square. 
    # The geometry guarantees ~21% of pixels are green background, creating a perfect bimodal peak
    # regardless of shadows or gradients in the specific crop.
    thresh_val, _ = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Nudge the threshold UP by 4 points. 
    # This acts as an anti-aliasing filter, forcing transitional "halo" pixels 
    # on the very edge of the paper down into the background class.
    adjusted_thresh = min(int(thresh_val + 4), 255)
    
    # 4. Generate the foreground mask (Keeping values higher than the threshold)
    _, fg_mask = cv2.threshold(a_channel, adjusted_thresh, 255, cv2.THRESH_BINARY)
    
    # Confine it strictly to the geometric circle
    intersected = cv2.bitwise_and(fg_mask, geometric_mask)
    
    # 5. Clean up the mask (Fill internal holes, drop tiny green noise)
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(intersected, cv2.MORPH_CLOSE, clean_kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, clean_kernel, iterations=1)
    
    # 6. Extract the final single continuous shape
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return geometric_mask
        
    largest_cnt = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(geometric_mask)
    cv2.drawContours(final_mask, [largest_cnt], -1, 255, thickness=-1)

    # 7. Green-only veto in a thin edge annulus — targets cardinal-point (12/3/6/9) residuals.
    # Keep this strictly color-gated (HSV green) so interior non-green pixels are preserved.
    roi_h, roi_w = geometric_mask.shape
    est_radius = min(roi_h, roi_w) // 2
    edge_band_px = max(2, int(est_radius * 0.05))   # ~5% of radius to limit action to the border rim

    erode_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (edge_band_px * 2 + 1, edge_band_px * 2 + 1)
    )
    inner_geo = cv2.erode(geometric_mask, erode_k, iterations=1)
    edge_band = cv2.bitwise_and(geometric_mask, cv2.bitwise_not(inner_geo))

    # HSV hue veto: stable under brightness shifts; separates green (H 35-85) from
    # white (low S) and purple (H ~135-150).
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    green_hsv = cv2.inRange(hsv, np.array([35, 28, 20]), np.array([85, 255, 255]))

    # Apply veto only where pixels are both in the edge band and confidently green.
    green_in_band = cv2.bitwise_and(edge_band, green_hsv)
    final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(green_in_band))

    # Re-close any small holes the veto may have punched in the interior boundary
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                  iterations=1)

    # 8. Surgical 1-Pixel Erode to guarantee absolute background isolation
    final_mask = cv2.erode(final_mask, np.ones((3, 3), np.uint8), iterations=1)
    
    return final_mask


# ----------------------------
# UI Layout
# ----------------------------
st.title("DiscAnalyzer-2.0")
tab1, tab2, tab3 = st.tabs(["Module 1: Auto-Detect & Clip", "Module 2: HSV Binary Tuning", "Module 3: Batch Processing"])

# ═══════════════════════════
# MODULE 1
# ═══════════════════════════
with tab1:
    st.header("Module 1: Batch Auto-Detect & Clip Discs")

    m1_files = st.file_uploader("Choose Input Image(s)", type=ALLOWED_TYPES, key="m1_files", accept_multiple_files=True)
    st.caption("Batch output is generated as ZIP and can be consumed directly by Modules 2 and 3 from session.")

    st.subheader("Processing Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.clip_params['nonoverlap_gap'] = st.number_input(
            "Non-overlap gap (px)",
            value=float(st.session_state.clip_params.get('nonoverlap_gap', 0.0)),
            step=1.0
        )
    with col2:
        st.session_state.clip_params['edge_shave_pct'] = st.number_input(
            "Edge Shave (%)",
            value=float(st.session_state.clip_params.get('edge_shave_pct', 0.0)),
            step=0.5,
            help="Shrinks the rough detection circle. Keep at 0.0 if using Chroma Key Refinement."
        )
    with col3:
        st.session_state.clip_params['refine_edges'] = st.checkbox(
            "Chroma Key Refinement (A-Channel Histogram)",
            value=st.session_state.clip_params.get('refine_edges', True),
            help="Uses global A-Channel Otsu thresholding to flawlessly separate green tables."
        )

    st.write("")
    b1, b2, b3 = st.columns([3, 3, 4])
    preview_clicked = b1.button("Preview Detection (All Images)", use_container_width=True)
    save_clicked    = b2.button("Batch Save Clipped Discs (All Images)", type="primary", use_container_width=True)
    download_slot = b3.empty()
    with b3:
        if st.session_state.m1_zip_bytes:
            download_slot.download_button(
                "Download Clipped Discs ZIP",
                data=st.session_state.m1_zip_bytes,
                file_name=st.session_state.m1_zip_name,
                mime="application/zip",
                use_container_width=True,
                key="m1_download_zip_inline",
            )

    # Run detection on every uploaded image and cache results
    if preview_clicked and m1_files:
        with st.spinner("Running detection on all images…"):
            detections = {}
            prog = st.progress(0)
            for i, f in enumerate(m1_files):
                bgr_i, _ = load_bgr_alpha_bytes(f.getvalue())
                detections[f.name] = auto_robust_disc_detection(
                    bgr_i, st.session_state.clip_params['nonoverlap_gap']
                )
                prog.progress((i + 1) / len(m1_files))
            prog.empty()
        st.session_state.m1_all_detections = detections
        st.session_state.m1_preview_idx    = 0
        total_d = sum(len(v) for v in detections.values())
        st.success(f"Detection complete — {total_d} disc(s) found across {len(m1_files)} image(s).")

    # Image navigation and preview
    if m1_files and st.session_state.m1_all_detections:
        n_files = len(m1_files)
        # Clamp index in case file list changed
        st.session_state.m1_preview_idx = min(st.session_state.m1_preview_idx, n_files - 1)
        idx = st.session_state.m1_preview_idx

        nav_l, nav_c, nav_r = st.columns([1, 8, 1])
        with nav_l:
            if st.button("◄", key='m1_prev', use_container_width=True, disabled=(idx == 0)):
                st.session_state.m1_preview_idx = max(0, idx - 1)
        with nav_r:
            if st.button("►", key='m1_next', use_container_width=True, disabled=(idx == n_files - 1)):
                st.session_state.m1_preview_idx = min(n_files - 1, idx + 1)

        # Re-read after potential button update
        idx        = st.session_state.m1_preview_idx
        curr_file  = m1_files[idx]
        curr_circles = st.session_state.m1_all_detections.get(curr_file.name, [])

        with nav_c:
            st.markdown(
                f"<div style='text-align:center;padding-top:10px'>"
                f"Image <b>{idx + 1}</b> / {n_files} &nbsp;|&nbsp; "
                f"<i>{curr_file.name}</i> &nbsp;|&nbsp; "
                f"<b>{len(curr_circles)}</b> disc(s) detected</div>",
                unsafe_allow_html=True
            )

        if curr_circles:
            bgr_c, _ = load_bgr_alpha_bytes(curr_file.getvalue())
            overlay = bgr_c.copy()
            shave_factor   = 1.0 - (st.session_state.clip_params['edge_shave_pct'] / 100.0)
            sorted_circles = sort_circles_column_major(curr_circles)
            for disc_num, c in enumerate(sorted_circles, start=1):
                display_r = int(c['r'] * shave_factor)
                cv2.circle(overlay, (c['x'], c['y']), display_r, (36, 255, 12), 2)
                cv2.circle(overlay, (c['x'], c['y']), 3, (0, 0, 255), -1)
                cv2.putText(overlay, str(disc_num), (c['x'] - 12, c['y'] + 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                     caption="Numbers show column-major save order (top-left first, then down each column)",
                     use_container_width=True)
        else:
            st.warning(f"No discs detected in {curr_file.name}.")

    # Batch save
    if save_clicked:
        st.session_state.m1_zip_bytes = None
        st.session_state.m1_latest_discs = []

        if not m1_files:
            st.warning("Please upload at least one image to process.")
        else:
            zip_buffer = io.BytesIO()
            zip_ref = zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_DEFLATED)
            latest_discs = []
            total_saved  = 0
            failed_saved = 0
            progress_bar = st.progress(0)
            status_text  = st.empty()
            shave_factor = 1.0 - (st.session_state.clip_params['edge_shave_pct'] / 100.0)

            try:
                for img_idx, file_obj in enumerate(m1_files):
                    status_text.text(f"Processing image {img_idx + 1}/{len(m1_files)}: {file_obj.name}")
                    img_bgr, img_alpha = load_bgr_alpha_bytes(file_obj.getvalue())
                    h, w = img_bgr.shape[:2]

                    current_circles = auto_robust_disc_detection(
                        img_bgr, st.session_state.clip_params['nonoverlap_gap']
                    )
                    if not current_circles:
                        progress_bar.progress((img_idx + 1) / len(m1_files))
                        continue

                    circles_to_save = sort_circles_column_major(current_circles)

                    for disc_idx, c in enumerate(circles_to_save, start=1):
                        x, y = c['x'], c['y']
                        r    = int(c['r'] * shave_factor)
                        x0, y0 = max(0, x - r), max(0, y - r)
                        x1, y1 = min(w, x + r), min(h, y + r)
                        roi_bgr   = img_bgr[y0:y1, x0:x1].copy()
                        roi_alpha = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                        cv2.circle(roi_alpha, (x - x0, y - y0), r, 255, thickness=-1)
                        if st.session_state.clip_params['refine_edges']:
                            roi_alpha = refine_alpha_mask(roi_bgr, roi_alpha)
                        if img_alpha is not None:
                            roi_alpha = cv2.bitwise_and(roi_alpha, img_alpha[y0:y1, x0:x1])
                        bgra      = np.dstack([roi_bgr, roi_alpha])
                        base_name = Path(file_obj.name).stem
                        out_name = f"{base_name}_disc_{disc_idx:03d}.png"

                        png_bytes = encode_png_bytes(bgra)
                        if png_bytes is not None:
                            zip_ref.writestr(out_name, png_bytes)
                            latest_discs.append({"name": out_name, "bytes": png_bytes})
                            total_saved += 1
                        else:
                            failed_saved += 1

                    progress_bar.progress((img_idx + 1) / len(m1_files))
            finally:
                zip_ref.close()

            status_text.text("Batch processing complete!")
            if total_saved > 0:
                zip_buffer.seek(0)
                st.session_state.m1_zip_bytes = zip_buffer.getvalue()
                st.session_state.m1_zip_name = "clipped_discs.zip"
                st.session_state.m1_latest_discs = latest_discs
                # Update the inline download button in the same run (no tab switching needed).
                download_slot.download_button(
                    "Download Clipped Discs ZIP",
                    data=st.session_state.m1_zip_bytes,
                    file_name=st.session_state.m1_zip_name,
                    mime="application/zip",
                    use_container_width=True,
                    key="m1_download_zip_inline_after_save",
                )
                st.success(
                    f"Prepared {total_saved} disc(s) from {len(m1_files)} image(s). "
                    "ZIP download is available beside the save button and these files are now available to Modules 2 and 3."
                )
            else:
                st.warning("No discs were generated.")
            if failed_saved:
                st.warning(f"{failed_saved} disc(s) could not be encoded as PNG.")


# ═══════════════════════════
# MODULE 2
# ═══════════════════════════
with tab2:
    st.header("Module 2: HSV Binary Tuning — Purple Detection")

    m2_source_options = ["Upload Image(s)"]
    if st.session_state.m1_latest_discs:
        m2_source_options.insert(0, "Use Latest Module 1 Discs (Session)")
    if 'm2_input_source' in st.session_state and st.session_state.m2_input_source not in m2_source_options:
        del st.session_state['m2_input_source']
    m2_input_source = st.radio(
        "Input Source",
        m2_source_options,
        key="m2_input_source",
        horizontal=True,
    )

    if m2_input_source == "Use Latest Module 1 Discs (Session)":
        m2_items = list(st.session_state.m1_latest_discs)
        st.caption(f"Using {len(m2_items)} disc(s) from the latest Module 1 batch.")
    else:
        m2_uploads = st.file_uploader(
            "Choose Image(s) (clipped disc recommended)",
            type=ALLOWED_TYPES,
            key="m2_file",
            accept_multiple_files=True,
        )
        m2_items = [{"name": f.name, "bytes": f.getvalue()} for f in (m2_uploads or [])]

    col_json_upload, col_json_dl = st.columns(2)
    with col_json_upload:
        json_file = st.file_uploader("Load Params (JSON)", type=["json"])
        if json_file is not None:
            loaded_params = json.load(json_file)
            st.session_state.hsv_params.update(loaded_params)
            st.session_state.m2_h_range = (loaded_params['h_min'], loaded_params['h_max'])
            st.session_state.m2_s_range = (loaded_params['s_min'], loaded_params['s_max'])
            st.session_state.m2_v_range = (loaded_params['v_min'], loaded_params['v_max'])
            st.success("Parameters loaded!")

    with col_json_dl:
        # Always export the live slider values
        _cur = {
            'h_min': st.session_state.m2_h_range[0], 'h_max': st.session_state.m2_h_range[1],
            's_min': st.session_state.m2_s_range[0], 's_max': st.session_state.m2_s_range[1],
            'v_min': st.session_state.m2_v_range[0], 'v_max': st.session_state.m2_v_range[1],
        }
        st.download_button(
            label="Download Current Params (JSON)",
            data=json.dumps(_cur, indent=2),
            file_name="hsv_params.json",
            mime="application/json"
        )

    ignore_green = st.checkbox(
        "Ignore residual green pixels from Module 1",
        value=True,
        help="Excludes leftover green edge pixels from both classification percentages and display.",
    )

    # Multi-image navigation (same behavior as Module 1)
    active_m2_file = None
    m2_data = None
    n_files = 0
    if m2_items:
        n_files = len(m2_items)
        st.session_state.m2_preview_idx = min(st.session_state.m2_preview_idx, n_files - 1)
        idx = st.session_state.m2_preview_idx
        active_m2_file = m2_items[idx]
        m2_data = prepare_m2_image(active_m2_file["bytes"])

    # Auto-detect button MUST appear before slider widgets so that session-state
    # keys are updated before Streamlit creates the slider widgets on this same run.
    if m2_data is not None:
        bgr_m2, alpha_m2, _ = m2_data
    ctrl_col, preview_col = st.columns([2, 5], gap="large")

    with ctrl_col:
        if m2_data is not None:
            if st.button("Auto-detect Purple Parameters", use_container_width=True):
                params = auto_detect_purple_params(bgr_m2, alpha_m2)
                if params:
                    st.session_state.m2_h_range = (params['h_min'], params['h_max'])
                    st.session_state.m2_s_range = (params['s_min'], params['s_max'])
                    st.session_state.m2_v_range = (params['v_min'], params['v_max'])
                    st.session_state.hsv_params.update(params)
                    st.rerun()
                else:
                    st.warning("Not enough colored pixels detected. Please adjust sliders manually.")

        # Sliders — placed in the left control pane so they remain visible while viewing discs.
        st.subheader("HSV Sliders")
        h_range = st.slider("H Range", 0, 179, key='m2_h_range')
        s_range = st.slider("S Range", 0, 255, key='m2_s_range')
        v_range = st.slider("V Range", 0, 255, key='m2_v_range')

        # Keep hsv_params in sync so Module 3 picks up the latest values
        st.session_state.hsv_params.update({
            'h_min': h_range[0], 'h_max': h_range[1],
            's_min': s_range[0], 's_max': s_range[1],
            'v_min': v_range[0], 'v_max': v_range[1],
        })

    with preview_col:
        if m2_data is not None:
            bgr_m2, alpha_m2, hsv_m2 = m2_data
            vis_mask = alpha_m2 > 128

            if ignore_green:
                green_residual = compute_green_residual_mask(bgr_m2, hsv_m2, vis_mask)
                effective_vis_mask = vis_mask & (~green_residual)
            else:
                green_residual = np.zeros_like(vis_mask, dtype=bool)
                effective_vis_mask = vis_mask

            lower = np.array([h_range[0], s_range[0], v_range[0]])
            upper = np.array([h_range[1], s_range[1], v_range[1]])
            hsv_match_mask = cv2.inRange(hsv_m2, lower, upper)

            display_bgr = np.full((*bgr_m2.shape[:2], 3), (200, 200, 200), dtype=np.uint8)
            display_bgr[vis_mask] = bgr_m2[vis_mask]
            display_bgr[green_residual] = [180, 180, 180]

            output_bgr = np.full_like(bgr_m2, (200, 200, 200), dtype=np.uint8)
            blue_mask  = effective_vis_mask & (hsv_match_mask > 0)
            white_mask = effective_vis_mask & (hsv_match_mask == 0)
            output_bgr[white_mask] = [255, 255, 255]
            output_bgr[blue_mask]  = [255, 0, 0]

            roi_total = np.count_nonzero(effective_vis_mask)
            blue_pct  = (np.count_nonzero(blue_mask)  / roi_total * 100) if roi_total else 0
            white_pct = (np.count_nonzero(white_mask) / roi_total * 100) if roi_total else 0
            ignored_green_pct = (np.count_nonzero(green_residual) / np.count_nonzero(vis_mask) * 100) if np.count_nonzero(vis_mask) else 0

            st.markdown(
                f"**Blue (Classified):** {blue_pct:.1f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**White (Non-Classified):** {white_pct:.1f}%"
            )
            if ignore_green:
                st.caption(f"Ignored residual green pixels: {ignored_green_pct:.1f}% of visible disc area")

            # Keep navigation right above images so users can switch without scrolling up.
            if n_files > 1 and m2_items:
                idx = st.session_state.m2_preview_idx
                nav_l, nav_c, nav_r = st.columns([1, 8, 1])
                with nav_l:
                    if st.button("◄", key='m2_prev', use_container_width=True, disabled=(idx == 0)):
                        st.session_state.m2_preview_idx = max(0, idx - 1)
                        st.rerun()
                with nav_r:
                    if st.button("►", key='m2_next', use_container_width=True, disabled=(idx == n_files - 1)):
                        st.session_state.m2_preview_idx = min(n_files - 1, idx + 1)
                        st.rerun()
                with nav_c:
                    st.markdown(
                        f"<div style='text-align:center;padding-top:2px;padding-bottom:8px'>"
                        f"Image <b>{idx + 1}</b> / {n_files} &nbsp;|&nbsp; "
                        f"<i>{m2_items[idx]['name']}</i></div>",
                        unsafe_allow_html=True
                    )

            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB),
                         caption="Original (ROI)", use_container_width=True)
            with img_col2:
                st.image(cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB),
                         caption="Processed Mask", use_container_width=True)


# ═══════════════════════════
# MODULE 3
# ═══════════════════════════
with tab3:
    st.header("Module 3: Batch Processing")

    st.info(f"Current HSV Parameters (from Module 2): {st.session_state.hsv_params}")
    m3_source_options = ["Upload ZIP Input"]
    if st.session_state.m1_latest_discs:
        m3_source_options.insert(0, "Use Latest Module 1 Discs (Session)")

    if 'm3_input_mode' in st.session_state and st.session_state.m3_input_mode not in m3_source_options:
        del st.session_state['m3_input_mode']

    m3_input_mode = st.radio(
        "Input Source",
        m3_source_options,
        key="m3_input_mode",
        horizontal=True,
    )

    uploaded_m3_zip = None
    if m3_input_mode == "Upload ZIP Input":
        uploaded_m3_zip = st.file_uploader(
            "Upload ZIP (Module 1 output or any ZIP containing clipped disc images)",
            type=["zip"],
            key="m3_zip_input",
        )
    else:
        st.caption(f"Using {len(st.session_state.m1_latest_discs)} disc(s) from the latest Module 1 batch.")

    m3_diameter = st.number_input("Disk Diameter (cm):", value=2.2, step=0.1)

    run_col, dl_col = st.columns([3, 4])

    if run_col.button("Run Batch", use_container_width=True):
        st.session_state.m3_zip_bytes = None

        if m3_input_mode == "Use Latest Module 1 Discs (Session)" and not st.session_state.m1_latest_discs:
            st.warning("No Module 1 session output is available yet. Run Module 1 batch save first or upload a ZIP.")
        elif m3_input_mode == "Upload ZIP Input" and uploaded_m3_zip is None:
            st.warning("Please upload a ZIP file.")
        else:
            if m3_input_mode == "Use Latest Module 1 Discs (Session)":
                input_items = list(st.session_state.m1_latest_discs)
            else:
                try:
                    input_items = extract_images_from_zip_bytes(uploaded_m3_zip.getvalue())
                except Exception as e:
                    input_items = []
                    st.error(f"Could not read ZIP file: {e}")

            if not input_items:
                st.error("No valid input images found.")
            else:
                zip_buffer = io.BytesIO()
                zip_ref = zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_DEFLATED)

                radius_cm       = m3_diameter / 2.0
                total_area_cm2  = np.pi * (radius_cm ** 2)

                records      = []
                progress_bar = st.progress(0)
                status_text  = st.empty()
                log_area     = st.empty()
                logs = [f"Assuming total disk area: {total_area_cm2:.3f} cm² (diameter {m3_diameter} cm)"]

                params = st.session_state.hsv_params
                lower  = np.array([params['h_min'], params['s_min'], params['v_min']])
                upper  = np.array([params['h_max'], params['s_max'], params['v_max']])

                try:
                    for i, item in enumerate(input_items):
                        item_name = item["name"]
                        try:
                            bgr, alpha = load_bgr_alpha_bytes(item["bytes"])

                            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                            hsv_match_mask = cv2.inRange(hsv, lower, upper)
                            visible_mask   = (alpha > 128)

                            output_bgr = np.full(bgr.shape, (200, 200, 200), dtype=np.uint8)
                            blue_mask  = visible_mask & (hsv_match_mask > 0)
                            white_mask = visible_mask & (hsv_match_mask == 0)
                            output_bgr[white_mask] = [255, 255, 255]
                            output_bgr[blue_mask]  = [255, 0, 0]

                            roi_total  = np.count_nonzero(visible_mask)
                            blue_pct   = (np.count_nonzero(blue_mask)  / roi_total * 100) if roi_total else 0
                            white_pct  = (np.count_nonzero(white_mask) / roi_total * 100) if roi_total else 0

                            classified_area_cm2     = (blue_pct  / 100.0) * total_area_cm2
                            non_classified_area_cm2 = (white_pct / 100.0) * total_area_cm2

                            item_stem = Path(item_name).stem
                            out_name = f"{item_stem}_classified.png"
                            plot_bytes = build_comparison_plot_bytes(bgr, output_bgr, white_pct, blue_pct)
                            zip_ref.writestr(f"classified_images/{out_name}", plot_bytes)

                            records.append({
                                "image": item_name,
                                "classified_blue_pct": blue_pct,
                                "classified_area_cm2": classified_area_cm2,
                                "non_classified_white_pct": white_pct,
                                "non_classified_area_cm2": non_classified_area_cm2
                            })
                            logs.append(
                                f"OK: {item_name} | Blue {blue_pct:.2f}% ({classified_area_cm2:.3f} cm²)"
                            )
                        except Exception as e:
                            logs.append(f"ERROR: {item_name}: {e}")

                        progress_bar.progress((i + 1) / len(input_items))
                        status_text.text(f"Processed {i + 1}/{len(input_items)}")
                        log_area.text("\n".join(logs[-10:]))

                    if records:
                        df = pd.DataFrame(records)[[
                            "image", "classified_blue_pct", "classified_area_cm2",
                            "non_classified_white_pct", "non_classified_area_cm2"
                        ]]
                        excel_buffer = io.BytesIO()
                        df.to_excel(excel_buffer, index=False, float_format="%.3f")
                        excel_buffer.seek(0)
                        zip_ref.writestr("summary.xlsx", excel_buffer.getvalue())

                        zip_ref.close()
                        zip_ref = None
                        zip_buffer.seek(0)
                        st.session_state.m3_zip_bytes = zip_buffer.getvalue()
                        st.session_state.m3_zip_name = "module3_results.zip"
                        st.success(f"Processed {len(records)} image(s). Download the ZIP using the button beside Run Batch.")
                    else:
                        st.warning("Batch finished, but no valid images were processed.")
                finally:
                    if zip_ref is not None:
                        zip_ref.close()

    with dl_col:
        if st.session_state.m3_zip_bytes:
            st.download_button(
                "Download Module 3 Results ZIP",
                data=st.session_state.m3_zip_bytes,
                file_name=st.session_state.m3_zip_name,
                mime="application/zip",
                use_container_width=True,
                key="m3_download_zip",
            )
