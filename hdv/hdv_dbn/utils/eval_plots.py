import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def plot_entropy_heatmaps(H_joint, H_style, H_action, lengths, out_dir, prefix="entropy", 
                          sort_by="joint_median",   # "joint_median", "length", or None 
                          vmax=1.0,               # entropy is normalized in [0,1]
                          ):
    """
    Save three entropy heatmaps (joint/style/action) as PNG files.

    Parameters
    H_joint, H_style, H_action
        Arrays shaped (N, Tmax) with NaN padding.
    lengths
        Array shaped (N,) holding valid lengths for each row.
    out_dir
        Folder to save PNGs.
    prefix
        File prefix for outputs.
    sort_by
        Row ordering strategy:
          - "joint_median": sort by median joint entropy per vehicle (ascending)
          - "length": sort by sequence length (descending)
          - None: keep original order
    vmax
        Upper limit for color scaling (use 1.0 for normalized entropy).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lengths = np.asarray(lengths, dtype=np.int64)
    N = int(H_joint.shape[0])
    if N == 0:
        return {}

    # Sorting (stable across runs if you keep same input order)
    if sort_by == "joint_median":
        row_score = np.nanmedian(H_joint, axis=1)
        order = np.argsort(row_score)  # low -> high uncertainty
    elif sort_by == "length":
        order = np.argsort(-lengths)   # long -> short
    else:
        order = np.arange(N)

    Hj = H_joint[order]
    Hs = H_style[order]
    Ha = H_action[order]

    def _save_one(H, title, fname):
        plt.figure(figsize=(10, 6))
        plt.imshow(H, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=vmax)
        plt.colorbar(label="Normalized posterior entropy")
        plt.xlabel("Time step (window index)")
        plt.ylabel("Vehicle (sorted)" if sort_by else "Vehicle")
        plt.title(title)
        path = out_dir / fname
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        return str(path)

    paths = {}
    paths["entropy_joint_png"]  = _save_one(Hj, f"{prefix}: Joint entropy H(S,A)",  f"{prefix}_joint.png")
    paths["entropy_style_png"]  = _save_one(Hs, f"{prefix}: Style entropy H(S)",    f"{prefix}_style.png")
    paths["entropy_action_png"] = _save_one(Ha, f"{prefix}: Action entropy H(A)",   f"{prefix}_action.png")

    return paths


def plot_T_vs_avg_nll(per_seq, out_dir, fname="T_vs_avg_nll.png", title="Online predictive: T vs avg NLL"):
    """
    Scatter plot of trajectory length T (windows) vs avg NLL per window.

    Parameters
    per_seq : dict
        key -> {"T": int, "avg_nll": float}
    out_dir : str or Path
        Where to save the PNG.
    fname : str
        File name for saved figure.
    title : str
        Plot title.

    Returns
    str : path to saved PNG
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    T_list = []
    nll_list = []

    for _, rec in per_seq.items():
        T = rec.get("T", None)
        avg_nll = rec.get("avg_nll", None)
        if T is None or avg_nll is None:
            continue
        if np.isfinite(T) and np.isfinite(avg_nll) and int(T) > 0:
            T_list.append(int(T))
            nll_list.append(float(avg_nll))

    if len(T_list) == 0:
        # nothing to plot
        return ""

    T_arr = np.asarray(T_list, dtype=np.int64)
    nll_arr = np.asarray(nll_list, dtype=np.float64)

    plt.figure(figsize=(8, 5))
    plt.scatter(T_arr, nll_arr, s=10, alpha=0.6)
    plt.xlabel("Trajectory length T (windows)")
    plt.ylabel("Average NLL per window")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    path = out_dir / fname
    plt.savefig(path, dpi=200)
    plt.close()
    return str(path)


def plot_ap_nll_vs_horizon(ap_nll_by_h, out_dir, fname="ap_nll_vs_horizon.png", title="Anticipatory predictive NLL vs horizon"):
    """
    Line plot of mean NLL at each horizon h=1..H.

    ap_nll_by_h : array-like, shape (H,)
        ap_nll_by_h[h-1] = - mean_t log p(o_{t+h} | o_{0:t})
    """
    ap_nll_by_h = np.asarray(ap_nll_by_h, dtype=np.float64)
    if ap_nll_by_h.size == 0 or not np.isfinite(ap_nll_by_h).any():
        return ""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h = np.arange(1, ap_nll_by_h.size + 1)

    plt.figure(figsize=(7, 4.5))
    plt.plot(h, ap_nll_by_h, marker="o", linewidth=1.5)
    plt.xlabel("Prediction horizon h (windows ahead)")
    plt.ylabel("NLL:  −log p(o_{t+h} | o_{0:t})")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    path = out_dir / fname
    plt.savefig(path, dpi=200)
    plt.close()
    return str(path)
