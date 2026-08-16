# ============================================================
# Multi-Model Comparison & Evaluation Script
# Evaluates trained models on unseen datasets (e.g., Ra_1e8)
# ============================================================

import sys
from pathlib import Path

# Add repository root to sys.path for config import
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import config

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ============================================================
# Paths & Settings Setup
# ============================================================

# Target test dataset resolution
data_file_sub = config.PROCESSED_DATA_DIR / "Ra_1e8" / "Ra_1e8_fields_Q.npz"
data_file_flat = config.PROCESSED_DATA_DIR / "Ra_1e8_fields_Q.npz"
data_file = data_file_sub if data_file_sub.exists() else data_file_flat

n_start = 2  # Set to 2 to accommodate (t-2) for past windowing models
n_end = 322
frame_step = 1
batch_size = 32
active_percentile = 95
SUMMARY_TXT_OUT = config.OUTPUT_DIR / "model_comparison_summary.txt"

# Model configurations
models_config = [
    {"name": "Model v001", "folder": "v001", "model_file": "v001.keras", "norm_file": "normalization_constants_v001.npz"},
    {"name": "Model v002", "folder": "v002", "model_file": "v002.keras", "norm_file": "normalization_constants_v002.npz"},
    {"name": "Model v003", "folder": "v003", "model_file": "v003.keras", "norm_file": "normalization_constants_v003.npz"},
    {"name": "Model v004", "folder": "v004", "model_file": "v004.keras", "norm_file": "normalization_constants_v004.npz"},
    {"name": "Model v005", "folder": "v005", "model_file": "v005.keras", "norm_file": "normalization_constants_v005.npz"},
    {"name": "Model v006", "folder": "v006", "model_file": "v006.keras", "norm_file": "normalization_constants_v006.npz"},
    {"name": "Model v007", "folder": "v007", "model_file": "v007.keras", "norm_file": "normalization_constants_v007.npz"},
    {"name": "Model v008", "folder": "v008", "model_file": "v008.keras", "norm_file": "normalization_constants_v008.npz"},
    {"name": "Model v009", "folder": "v009", "model_file": "v009.keras", "norm_file": "normalization_constants_v009.npz"},
    {"name": "Model v010 (14-Ch Centered)", "folder": "v010", "model_file": "v010_3frame.keras", "norm_file": "normalization_constants_v010.npz", "window_type": "centered"},
    {"name": "Model v011 (14-Ch Past)",     "folder": "v011", "model_file": "v011_3frame_past.keras", "norm_file": "normalization_constants_v011.npz", "window_type": "past"},
    {"name": "Model v012 (14-Ch Past Mixed)", "folder": "v012", "model_file": "v012_3frame_past.keras", "norm_file": "normalization_constants_v012.npz", "window_type": "past"},
    {"name": "Model v013 (14-Ch Past Modern)", "folder": "v013", "model_file": "v013_3frame_past.keras", "norm_file": "normalization_constants_v013.npz", "window_type": "past"},
]

def extract_ra_pr(npz_data, label):
    """Extracts Ra and Pr from NPZ metadata or falls back to string parsing."""
    if "Ra" in npz_data and "Pr" in npz_data:
        return float(npz_data["Ra"]), float(npz_data["Pr"])

    pr_match = re.search(r"Pr_(\d+)", label)
    if pr_match:
        pr_str = pr_match.group(1)
        pr_val = float(pr_str) if len(pr_str) == 1 else float(pr_str) / 10.0
    else:
        pr_val = 1.0

    ra_match = re.search(r"Ra_([0-9eE\+\-]+)", label)
    ra_val = float(ra_match.group(1)) if ra_match else 1e6

    return ra_val, pr_val

# ============================================================
# Custom Losses / Metrics
# ============================================================
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def gradient_loss(y_true, y_pred):
    dy_true = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
    dy_pred = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    dz_true = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
    dz_pred = y_pred[:, :-1, :, :]
    return tf.reduce_mean(tf.square(dy_true - dy_pred)) + tf.reduce_mean(tf.square(dz_true - dz_pred))

def combined_loss(y_true, y_pred):
    return mse(y_true, y_pred) + 0.05 * gradient_loss(y_true, y_pred)

def correlation(y_true, y_pred):
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    y_true_flat -= tf.reduce_mean(y_true_flat, axis=1, keepdims=True)
    y_pred_flat -= tf.reduce_mean(y_pred_flat, axis=1, keepdims=True)
    numerator = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
    denominator = tf.sqrt(
        tf.reduce_sum(tf.square(y_true_flat), axis=1) * tf.reduce_sum(tf.square(y_pred_flat), axis=1)
    )
    return tf.reduce_mean(numerator / (denominator + 1e-8))

custom_objects = {
    "combined_loss": combined_loss,
    "mse": mse,
    "mae": mae,
    "correlation": correlation,
    "gradient_loss": gradient_loss,
    "masked_mse": mse,
    "masked_mae": mae,
    "masked_combined_loss": combined_loss,
    "masked_correlation": correlation,
    "masked_gradient_loss": gradient_loss,
}

# ============================================================
# Metric Evaluation Helpers (Section 1, 2, 3)
# ============================================================
def calculate_frame_metrics(Q_pred, Q_true, percentile=95):
    # --- Section 1: Unmasked / Full Field ---
    err_full = Q_pred - Q_true
    full_rmse = np.sqrt(np.mean(err_full**2))
    full_mae = np.mean(np.abs(err_full))
    rms_true_full = np.sqrt(np.mean(Q_true**2))
    full_qrmse = full_rmse / (rms_true_full + 1e-12)

    # --- Section 2: Domain-Averaged Values ---
    domain_mean_pred = np.mean(Q_pred)
    domain_mean_true = np.mean(Q_true)

    # --- Section 3: Active Q Region (>= 95th Percentile) ---
    thresh = np.percentile(np.abs(Q_true), percentile)
    mask = np.abs(Q_true) >= thresh
    
    if np.sum(mask) > 0:
        err_active = err_full[mask]
        active_rmse = np.sqrt(np.mean(err_active**2))
        active_mae = np.mean(np.abs(err_active))
        rms_true_active = np.sqrt(np.mean(Q_true[mask]**2))
        active_qrmse = active_rmse / (rms_true_active + 1e-12)
    else:
        active_rmse, active_qrmse, active_mae = np.nan, np.nan, np.nan

    return {
        "full_qrmse": full_qrmse,
        "full_rmse": full_rmse,
        "full_mae": full_mae,
        "domain_mean_pred": domain_mean_pred,
        "domain_mean_true": domain_mean_true,
        "active_qrmse": active_qrmse,
        "active_rmse": active_rmse,
        "active_mae": active_mae
    }


# ============================================================
# Load Ground Truth Data
# ============================================================
print(f"--> Loading ground truth data from: {data_file}")
if not data_file.exists():
    raise FileNotFoundError(f"Evaluation dataset not found at {data_file}")

data = np.load(data_file)

# Support alternate key names for relative humidity / q field
q_key = "q" if "q" in data else ("rel_q" if "rel_q" in data else None)

if q_key is None and "_fields_Q.npz" in str(data_file):
    # Try loading paired _q.npz file
    q_filename = str(data_file).replace("_fields_Q.npz", "_q.npz")
    if os.path.exists(q_filename):
        data_q = np.load(q_filename)
        rel_q = data_q["q"]
    else:
        raise KeyError("Could not resolve relative humidity field ('q' or 'rel_q') in NPZ.")
else:
    rel_q = data[q_key]

u, v, theta, Q_true_all = data["u"], data["v"], data["theta"], data["Q"]

frames = list(range(n_start, n_end, frame_step))
num_frames = len(frames)


# ============================================================
# Loop Over Models and Evaluate
# ============================================================
results = {}

for mcfg in models_config:
    model_name = mcfg["name"]
    
    # Search inside config.MODEL_DIR across root and candidate subfolders
    m_search_paths = [
        config.MODEL_DIR / mcfg["model_file"],
        config.MODEL_DIR / mcfg["folder"] / mcfg["model_file"],
        config.MODEL_DIR / "v004 MODEL DIFFERENT RUNS" / mcfg["folder"] / mcfg["model_file"]
    ]
    n_search_paths = [
        config.MODEL_DIR / mcfg["norm_file"],
        config.MODEL_DIR / mcfg["folder"] / mcfg["norm_file"],
        config.MODEL_DIR / "v004 MODEL DIFFERENT RUNS" / mcfg["folder"] / mcfg["norm_file"]
    ]

    m_path = next((p for p in m_search_paths if p.exists()), None)
    n_path = next((p for p in n_search_paths if p.exists()), None)

    if n_path is None:
        # Check alternative underscore filenames
        for base_p in n_search_paths:
            alt1 = base_p.parent / (base_p.stem + "__.npz")
            alt2 = base_p.parent / (base_p.stem + "_.npz")
            if alt1.exists():
                n_path = alt1
                break
            elif alt2.exists():
                n_path = alt2
                break

    if m_path is None or n_path is None:
        print(f"[WARNING] Skipping {model_name}: File not found ({mcfg['model_file']} or {mcfg['norm_file']})")
        continue

    print(f"--> Evaluating: {model_name}")
    model = tf.keras.models.load_model(m_path, custom_objects=custom_objects)
    
    norm = np.load(n_path)
    X_mean, X_std = norm["X_mean"], norm["X_std"]
    Y_mean, Y_std = norm["Y_mean"], norm["Y_std"]
    
    n_channels = X_mean.shape[-1]
    window_type = mcfg.get("window_type", "centered")
    
    m_metrics = {
        "full_qrmse": [], "full_rmse": [], "full_mae": [],
        "domain_mean_pred": [], "domain_mean_true": [],
        "active_qrmse": [], "active_rmse": [], "active_mae": []
    }

    # Batched Inference
    for i in range(0, num_frames, batch_size):
        batch_frames = frames[i : i + batch_size]

        if n_channels == 14:
            ra_val, pr_val = extract_ra_pr(data, "Ra_1e8")
            nz, ny = u.shape[0], u.shape[1]
            nt_batch = len(batch_frames)

            ra_map = np.full((nz, ny, nt_batch), np.log10(ra_val), dtype=np.float32)
            pr_map = np.full((nz, ny, nt_batch), pr_val, dtype=np.float32)

            if window_type == "past":
                # Past temporal windowing: (t-2, t-1, t) -> Q(t)
                f_tm2 = [f - 2 for f in batch_frames]
                f_tm1 = [f - 1 for f in batch_frames]
                f_t   = batch_frames

                feature_list = [
                    u[:, :, f_tm2],     u[:, :, f_tm1],     u[:, :, f_t],
                    v[:, :, f_tm2],     v[:, :, f_tm1],     v[:, :, f_t],
                    theta[:, :, f_tm2], theta[:, :, f_tm1], theta[:, :, f_t],
                    rel_q[:, :, f_tm2], rel_q[:, :, f_tm1], rel_q[:, :, f_t],
                    ra_map, pr_map
                ]
            else:
                # Centered temporal windowing: (t-1, t, t+1) -> Q(t)
                prev_f = [f - 1 for f in batch_frames]
                curr_f = batch_frames
                next_f = [f + 1 for f in batch_frames]

                feature_list = [
                    u[:, :, prev_f],     u[:, :, curr_f],     u[:, :, next_f],
                    v[:, :, prev_f],     v[:, :, curr_f],     v[:, :, next_f],
                    theta[:, :, prev_f], theta[:, :, curr_f], theta[:, :, next_f],
                    rel_q[:, :, prev_f], rel_q[:, :, curr_f], rel_q[:, :, next_f],
                    ra_map, pr_map
                ]
        elif n_channels == 6:
            ra_val, pr_val = extract_ra_pr(data, "Ra_1e8")
            nz, ny = u.shape[0], u.shape[1]
            nt_batch = len(batch_frames)
            
            ra_map = np.full((nz, ny, nt_batch), np.log10(ra_val), dtype=np.float32)
            pr_map = np.full((nz, ny, nt_batch), pr_val, dtype=np.float32)
            
            feature_list = [
                u[:, :, batch_frames], v[:, :, batch_frames],
                theta[:, :, batch_frames], rel_q[:, :, batch_frames],
                ra_map, pr_map
            ]
        elif n_channels == 4:
            feature_list = [u[:, :, batch_frames], v[:, :, batch_frames], theta[:, :, batch_frames], rel_q[:, :, batch_frames]]
        elif n_channels == 2:
            feature_list = [theta[:, :, batch_frames], rel_q[:, :, batch_frames]]
        elif n_channels == 1:
            feature_list = [rel_q[:, :, batch_frames]]
        else:
            feature_list = [u[:, :, batch_frames], v[:, :, batch_frames], theta[:, :, batch_frames], rel_q[:, :, batch_frames]][:n_channels]

        X_batch = np.stack(feature_list, axis=-1)
        X_batch = np.transpose(X_batch, (2, 0, 1, 3)).astype(np.float32)

        # Normalize & Predict
        X_batch_norm = (X_batch - X_mean) / X_std
        Q_pred_norm_batch = model.predict(X_batch_norm, batch_size=len(batch_frames), verbose=0)
        
        # Denormalize
        Q_pred_batch = (Q_pred_norm_batch * Y_std + Y_mean)[:, :, :, 0]

        for idx, n in enumerate(batch_frames):
            Q_true = Q_true_all[:, :, n]
            Q_pred = Q_pred_batch[idx]
            
            res = calculate_frame_metrics(Q_pred, Q_true, percentile=active_percentile)
            for k in m_metrics:
                m_metrics[k].append(res[k])

    results[model_name] = {k: np.array(v) for k, v in m_metrics.items()}


# ============================================================
# Compute Section Summaries & Export to Text File
# ============================================================
print(f"\n--> Exporting metrics to text file: {SUMMARY_TXT_OUT}")

with open(SUMMARY_TXT_OUT, "w") as f:
    f.write("=====================================================================================\n")
    f.write("                          MODEL PERFORMANCE COMPARISON SUMMARY                        \n")
    f.write("=====================================================================================\n\n")

    # SECTION 1: UNMASKED FULL FIELD
    f.write("-------------------------------------------------------------------------------------\n")
    f.write("SECTION 1: UNMASKED / FULL-FIELD METRICS (Entire Domain)\n")
    f.write("-------------------------------------------------------------------------------------\n")
    f.write(f"{'Model Name':<35} | {'QRMSE (Rel RMSE)':<18} | {'RMSE':<15} | {'MAE':<15}\n")
    f.write("-" * 90 + "\n")
    for m_name, m_data in results.items():
        qrmse = np.nanmean(m_data["full_qrmse"])
        rmse = np.nanmean(m_data["full_rmse"])
        mae_v = np.nanmean(m_data["full_mae"])
        f.write(f"{m_name:<35} | {qrmse:<18.6e} | {rmse:<15.6e} | {mae_v:<15.6e}\n")
    f.write("\n\n")

    # SECTION 2: DOMAIN-AVERAGED FIELD
    f.write("-------------------------------------------------------------------------------------\n")
    f.write("SECTION 2: DOMAIN-AVERAGED FIELD METRICS (<Q>(t) Time Series)\n")
    f.write("-------------------------------------------------------------------------------------\n")
    f.write(f"{'Model Name':<35} | {'QRMSE (Rel RMSE)':<18} | {'RMSE':<15} | {'MAE':<15}\n")
    f.write("-" * 90 + "\n")
    
    # Ground truth mean time series reference
    ref_model = list(results.keys())[0] if results else None
    if ref_model:
        true_mean_ts = results[ref_model]["domain_mean_true"]
        rms_true_mean_ts = np.sqrt(np.mean(true_mean_ts**2))

        for m_name, m_data in results.items():
            pred_mean_ts = m_data["domain_mean_pred"]
            err_ts = pred_mean_ts - true_mean_ts
            
            avg_rmse = np.sqrt(np.mean(err_ts**2))
            avg_mae = np.mean(np.abs(err_ts))
            avg_qrmse = avg_rmse / (rms_true_mean_ts + 1e-12)
            
            f.write(f"{m_name:<35} | {avg_qrmse:<18.6e} | {avg_rmse:<15.6e} | {avg_mae:<15.6e}\n")
    f.write("\n\n")

    # SECTION 3: ACTIVE Q REGION
    f.write("-------------------------------------------------------------------------------------\n")
    f.write(f"SECTION 3: ACTIVE Q REGION METRICS (>= {active_percentile}th Percentile of |Q_true|)\n")
    f.write("-------------------------------------------------------------------------------------\n")
    f.write(f"{'Model Name':<35} | {'QRMSE (Rel RMSE)':<18} | {'RMSE':<15} | {'MAE':<15}\n")
    f.write("-" * 90 + "\n")
    for m_name, m_data in results.items():
        act_qrmse = np.nanmean(m_data["active_qrmse"])
        act_rmse = np.nanmean(m_data["active_rmse"])
        act_mae = np.nanmean(m_data["active_mae"])
        f.write(f"{m_name:<35} | {act_qrmse:<18.6e} | {act_rmse:<15.6e} | {act_mae:<15.6e}\n")

print(f"✅ Text summary successfully written to {SUMMARY_TXT_OUT}")


# ============================================================
# Plotting Performance Comparisons (6 Cases x 3 Sections = 18 Figures)
# ============================================================
print("\n--> Generating evaluation plots for specific comparison cases...")

comparison_cases = [
    {
        "title": "Epoch influence",
        "prefix": "Case1_Epoch_Influence",
        "folders": ["v002", "v003", "v004"]
    },
    {
        "title": "Dataset influence",
        "prefix": "Case2_Dataset_Influence",
        "folders": ["v001", "v002", "v004", "v005"]
    },
    {
        "title": "Ablation influence",
        "prefix": "Case3_Ablation_Influence",
        "folders": ["v005", "v006"]
    },
    {
        "title": "Pooling influence",
        "prefix": "Case4_Pooling_Influence",
        "folders": ["v005", "v007"]
    },
    {
        "title": "Pr and Ra maps influence",
        "prefix": "Case5_Pr_Ra_Maps_Influence",
        "folders": ["v008", "v009"]
    },
    {
        "title": "Temporal window & pooling influence",
        "prefix": "Case6_Temporal_Window_Pooling_Influence",
        "folders": ["v010", "v011", "v012", "v013"]
    },
    {
        "title": "q vs q-qs Testing",
        "prefix": "Case7_q_or_qs",
        "folders": ["v012", "v013"]
    },
    {
        "title": "True rel_q ablation",
        "prefix": "Case8_true_ablation",
        "folders": ["v012", "v014"]
    },
]

folder_to_name = {mcfg["folder"]: mcfg["name"] for mcfg in models_config}
generated_plot_count = 0

for case in comparison_cases:
    case_models = [folder_to_name[f] for f in case["folders"] if folder_to_name.get(f) in results]

    if not case_models:
        print(f"[WARNING] Skipping {case['title']}: None of the specified models were found in evaluation results.")
        continue

    # ------------------------------------------------------------
    # SECTION 1 PLOT: Full-Field Unmasked
    # ------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    for m_name in case_models:
        m_data = results[m_name]
        axes[0].plot(frames, m_data["full_qrmse"], label=m_name)
        axes[1].plot(frames, m_data["full_rmse"], label=m_name)
        axes[2].plot(frames, m_data["full_mae"], label=m_name)

    axes[0].set_ylabel("Full-Field QRMSE", fontsize=10)
    axes[0].set_yscale("log")
    axes[0].set_title(f"{case['title']} - Section 1: Unmasked Full-Field Metrics", fontsize=12, fontweight="bold")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].set_ylabel("Full-Field RMSE", fontsize=10)
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", linestyle=":", alpha=0.6)

    axes[2].set_ylabel("Full-Field MAE", fontsize=10)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Time Index (n)", fontsize=11)
    axes[2].grid(True, which="both", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_name = config.OUTPUT_DIR / f"{case['prefix']}_Section1_Full_Field_LOG_SCALE.png"
    plt.savefig(out_name, dpi=300)
    plt.close()
    generated_plot_count += 1

    # ------------------------------------------------------------
    # SECTION 2 PLOT: Domain-Averaged Field
    # ------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(frames, true_mean_ts, color="black", linestyle="-", linewidth=2.0, label="Ground Truth")

    for m_name in case_models:
        m_data = results[m_name]
        pred_ts = m_data["domain_mean_pred"]
        err_ts = np.abs(pred_ts - true_mean_ts)
        rel_err_ts = err_ts / (rms_true_mean_ts + 1e-12)

        axes[0].plot(frames, pred_ts, label=m_name)
        axes[1].plot(frames, rel_err_ts, label=m_name)
        axes[2].plot(frames, err_ts, label=m_name)

    axes[0].set_ylabel(r"Domain Mean $\langle Q \rangle$", fontsize=10)
    axes[0].set_title(f"{case['title']} - Section 2: Domain-Averaged Field Metrics", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].set_ylabel(r"Domain Mean QRMSE", fontsize=10)
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", linestyle=":", alpha=0.6)

    axes[2].set_ylabel(r"Domain Mean MAE / Error", fontsize=10)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Time Index (n)", fontsize=11)
    axes[2].grid(True, which="both", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_name = config.OUTPUT_DIR / f"{case['prefix']}_Section2_Domain_Averaged_LOG_SCALE.png"
    plt.savefig(out_name, dpi=300)
    plt.close()
    generated_plot_count += 1

    # ------------------------------------------------------------
    # SECTION 3 PLOT: Active Q Region
    # ------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    for m_name in case_models:
        m_data = results[m_name]
        axes[0].plot(frames, m_data["active_qrmse"], label=m_name)
        axes[1].plot(frames, m_data["active_rmse"], label=m_name)
        axes[2].plot(frames, m_data["active_mae"], label=m_name)

    axes[0].set_ylabel("Active Region QRMSE", fontsize=10)
    axes[0].set_yscale("log")
    axes[0].set_title(f"{case['title']} - Section 3: Active Q Region (≥ {active_percentile}th Percentile) Metrics", fontsize=12, fontweight="bold")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].set_ylabel("Active Region RMSE", fontsize=10)
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", linestyle=":", alpha=0.6)

    axes[2].set_ylabel("Active Region MAE", fontsize=10)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Time Index (n)", fontsize=11)
    axes[2].grid(True, which="both", linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_name = config.OUTPUT_DIR / f"{case['prefix']}_Section3_Active_Region_LOG_SCALE.png"
    plt.savefig(out_name, dpi=300)
    plt.close()
    generated_plot_count += 1

print(f"\nFinished. Successfully generated {generated_plot_count} plots in {config.OUTPUT_DIR}.")
