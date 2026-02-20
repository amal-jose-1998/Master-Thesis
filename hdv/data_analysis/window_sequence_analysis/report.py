from pathlib import Path 
import sys
import json
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Ensure project root (containing 'hdv/') is on PYTHONPATH
# ---------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
for parent in [_THIS_FILE] + list(_THIS_FILE.parents):
    if (parent / "hdv").is_dir():
        sys.path.insert(0, str(parent))
        break
else:
    raise ImportError("Could not locate project root containing 'hdv/' directory.")

from hdv.hdv_dbn.datasets.highd_loader import load_highd_folder
from hdv.hdv_dbn.config import WINDOW_FEATURE_COLS, WINDOW_CONFIG, SEMANTIC_CONFIG, TRAINING_CONFIG, BERNOULLI_FEATURES
from hdv.hdv_dbn.datasets.highd.sequences import load_or_build_windowized


# =============================================================================
# USER SETTINGS 
# =============================================================================
EXP_DIR = Path("/home/RUS_CIP/st184634/implementation/hdv/models/5-actions-hierarchical-sticky_S2_A5_hierarchical").resolve()
SPLIT = "train" # Which split to analyze: "train" | "val" | "test" | "all"

# Histogram settings
BINS = 200
XLIM_Q = (0.01, 0.99)  # robust x-limits using quantiles
_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]

# Output folder
OUT_DIR = EXP_DIR / "data_analysis" / "window_feature_distributions" / SPLIT

# =============================================================================
# Helpers
# =============================================================================
def _is_binary_feature(name):
    return name.lower() in BERNOULLI_FEATURES

def _is_fraction_feature(name):
    n = name.lower()
    # fractions in [0,1] that are often spiky at 0/1
    return (
        n.endswith("_frac")
        or n.endswith("_vfrac")
        or n.endswith("_exists_frac")
        or n.endswith("_neg_frac")
        or n.endswith("_pos_frac")
        or n.endswith("_zero_frac")
    )

def _draw_lines(ax, vf):
    # add summary lines 
    if vf.size == 0:
        return
    mean = float(vf.mean())
    med = float(np.median(vf))
    p10 = float(np.quantile(vf, 0.10))
    p90 = float(np.quantile(vf, 0.90))

    ax.axvline(mean, linestyle="-.", linewidth=2.0, color="red", label="mean")
    ax.axvline(med,  linestyle="-",  linewidth=2.2, color="green", label="median")
    ax.axvline(p10,  linestyle="--", linewidth=2.0, color="orange", label="p10")
    ax.axvline(p90,  linestyle="--", linewidth=2.0, color="purple", label="p90")

def load_split_json(exp_dir):
    p = exp_dir / "split.json"
    if not p.exists():
        raise FileNotFoundError(f"split.json not found: {p}")
    return json.loads(p.read_text())

def get_split_keys(split_obj, split_name):
    sn = str(split_name).lower().strip()
    try:
        result = split_obj["keys"][sn]
        if isinstance(result, list):
            return [str(x) for x in result]
    except (KeyError, TypeError):
        pass
    raise KeyError(f"Split '{sn}' not found in split.json.")

def parse_vehicle_key(key):
    s = str(key).strip()
    if ":" not in s:
        raise ValueError(f"Bad key '{key}', expected 'recording_id:vehicle_id'")

    a, b = s.split(":", 1)

    # tolerate float-formatted strings like "29.0"
    rid = int(float(a))
    vid = int(float(b))
    return rid, vid

def filter_df_to_keys(df, keys):
    pairs = [parse_vehicle_key(k) for k in keys]
    kdf = pd.DataFrame(pairs, columns=["recording_id", "vehicle_id"]).astype({"recording_id": "int64", "vehicle_id": "int64"})
    return df.merge(kdf, on=["recording_id", "vehicle_id"], how="inner")

def stack_windows(win_sequences, feature_names):
    if not win_sequences:
        return np.zeros((0, len(feature_names)), dtype=np.float64)

    # sequences.WindowSequence has obs (Nw, F) and obs_names
    ref = list(win_sequences[0].obs_names)
    if ref != list(feature_names):
        raise RuntimeError("WINDOW_FEATURE_COLS mismatch with windowized sequence schema.")

    blocks = []
    for s in win_sequences:
        X = np.asarray(s.obs, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(feature_names):
            raise RuntimeError(f"Bad obs shape {X.shape}")
        blocks.append(X)

    return np.vstack(blocks) if blocks else np.zeros((0, len(feature_names)), dtype=np.float64)

def _to_numeric_finite_1d(x):
    v = pd.to_numeric(pd.Series(np.asarray(x).reshape(-1)), errors="coerce").to_numpy()
    v = v[np.isfinite(v)]
    return v

def compute_stats(v, feature_name=None):
    vf = _to_numeric_finite_1d(v)
    if vf.size == 0:
        out = {f"p{int(q*100):02d}": np.nan for q in _QUANTILES}
        out.update({"mean": np.nan, "std": np.nan, "count": 0})
        out.update({"mass_at_0": np.nan, "mass_at_1": np.nan, "rate_1": np.nan, "mid_count": 0})
        return out

    out = {f"p{int(q*100):02d}": float(np.quantile(vf, q)) for q in _QUANTILES}
    out["mean"] = float(vf.mean())
    out["std"] = float(vf.std(ddof=0))
    out["count"] = int(vf.size)
    
    eps = 1e-12
    if feature_name is not None and _is_binary_feature(feature_name):
        xb = (vf > 0.5).astype(np.int64)
        out["rate_1"] = float(xb.mean())
        out["mass_at_0"] = float((xb == 0).mean())
        out["mass_at_1"] = float((xb == 1).mean())
        out["mid_count"] = 0
    elif feature_name is not None and _is_fraction_feature(feature_name):
        x = np.clip(vf, 0.0, 1.0)
        out["mass_at_0"] = float((x <= eps).mean())
        out["mass_at_1"] = float((x >= 1.0 - eps).mean())
        mid = x[(x > eps) & (x < 1.0 - eps)]
        out["mid_count"] = int(mid.size)
        out["rate_1"] = np.nan
    else:
        out["mass_at_0"] = np.nan
        out["mass_at_1"] = np.nan
        out["rate_1"] = np.nan
        out["mid_count"] = 0
    
    return out


def robust_xlim(v, qlo, qhi):
    vf = _to_numeric_finite_1d(v)
    if vf.size == 0:
        return None
    lo = float(np.quantile(vf, qlo))
    hi = float(np.quantile(vf, qhi))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return None
    return (lo, hi)

def save_hist(v, feature, out_png):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    vf = _to_numeric_finite_1d(v)
    if vf.size == 0:
        return

    fig, ax = plt.subplots()

    if feature == "jerk_x_p95" or feature == "jerk_y_p95":
        # Keep a robust view but preserve discreteness
        lim = robust_xlim(vf, XLIM_Q[0], XLIM_Q[1])
        if lim is not None:
            lo, hi = lim
            vp = vf[(vf >= lo) & (vf <= hi)]
        else:
            vp = vf

        # Count exact values 
        s = pd.Series(vp).value_counts().sort_index()
        xs = s.index.to_numpy(dtype=np.float64)
        ys = s.to_numpy(dtype=np.int64)

        # Choose a sensible bar width
        if xs.size >= 2:
            dx = np.diff(xs)
            dx = dx[np.isfinite(dx) & (dx > 0)]
            width = 0.9 * (float(np.median(dx)) if dx.size else 0.02)
        else:
            width = 0.02

        ax.bar(xs, ys, width=width, edgecolor="black", alpha=0.85)
        _draw_lines(ax, vf)  # lines on full vf

        if lim is not None:
            ax.set_xlim(lo, hi)

        ax.set_title(feature, fontsize=14)
        ax.set_xlabel(feature, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    # ---------------------------
    # 1) Binary features: plot counts (0 vs 1)
    # ---------------------------
    if _is_binary_feature(feature):
        xb = (vf > 0.5).astype(np.int64)
        n0 = int((xb == 0).sum())
        n1 = int((xb == 1).sum())
        rate = (n1 / (n0 + n1)) if (n0 + n1) else 0.0

        ax.bar([0, 1], [n0, n1], edgecolor="black", alpha=0.85)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{feature}  (rate=1: {rate:.6f})", fontsize=14)

        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    
    # ---------------------------
    # 2) Fraction features in [0,1] with spikes: show mass at 0/1 explicitly
    # ---------------------------
    if _is_fraction_feature(feature):
        x = np.clip(vf, 0.0, 1.0)
        eps = 1e-12

        n0 = int((x <= eps).sum())
        n1 = int((x >= 1.0 - eps).sum())
        mid = x[(x > eps) & (x < 1.0 - eps)]

        # Plot interior histogram (counts)
        if mid.size:
            ax.hist(
                mid,
                bins=np.linspace(0.0, 1.0, int(BINS) + 1),
                density=False,
                alpha=0.60,
                edgecolor="black",
                linewidth=0.25,
                label="(0,1) interior"
            )

        # Add explicit bars at 0 and 1
        ax.bar([0.0, 1.0], [n0, n1], width=0.035, edgecolor="black",
               alpha=0.90, label="mass at {0,1}")

        # Lines computed on full x (including 0/1)
        _draw_lines(ax, x)

        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel(feature, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        m0 = n0 / x.size
        m1 = n1 / x.size
        ax.set_title(f"{feature}  (mass@0={m0:.3f}, mass@1={m1:.3f}, mid={mid.size})", fontsize=14)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    # ---------------------------
    # 3) Continuous-ish features: robust clipping + counts histogram
    # ---------------------------
    # robust x-limits using quantiles for display only
    lim = robust_xlim(vf, XLIM_Q[0], XLIM_Q[1])
    if lim is not None:
        lo, hi = lim
        # plot clipped view so outliers don't flatten the figure
        vp = vf[(vf >= lo) & (vf <= hi)]
    else:
        vp = vf

    ax.hist(vp, bins=int(BINS), density=False, alpha=0.85, edgecolor="black", linewidth=0.25)
    _draw_lines(ax, vf)  # compute lines on full vf

    if lim is not None:
        ax.set_xlim(lo, hi)

    ax.set_title(feature, fontsize=14)
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    split_obj = load_split_json(EXP_DIR)
    if SPLIT.lower() == "all":
        keys = (
            get_split_keys(split_obj, "train")
            + get_split_keys(split_obj, "val")
            + get_split_keys(split_obj, "test")
        )
    else:
        keys = get_split_keys(split_obj, SPLIT)

    W = int(split_obj.get("W", WINDOW_CONFIG.W))
    stride = int(split_obj.get("stride", WINDOW_CONFIG.stride))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hist_dir = OUT_DIR / "hists"
    hist_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load canonical highD dataframe 
    data_root = Path(SEMANTIC_CONFIG.data_root).resolve()
    df = load_highd_folder(
        root=data_root,
        cache_path=None,
        force_rebuild=False,
        max_recordings=TRAINING_CONFIG.max_highd_recordings,
        apply_vehicle_centric=True,
        flip_lateral=True,
        flip_positions=False,
    )

    # 2) Filter by split keys
    df_split = filter_df_to_keys(df, keys)

    # 3) Windowize
    cache_dir = data_root / "cache"
    win_sequences = load_or_build_windowized(
        df_split,
        cache_dir=cache_dir,
        W=W,
        stride=stride,
        force_rebuild=False,
    )

    feature_names = list(WINDOW_FEATURE_COLS)
    X = stack_windows(win_sequences, feature_names)  # (N_windows_total, F)

    # 4) Stats + plots
    stats_rows = []
    for j, feat in enumerate(feature_names):
        v = X[:, j] if X.size else np.array([], dtype=np.float64)
        st = compute_stats(v, feat)
        stats_rows.append({"feature": feat, **st})
        save_hist(v, feat, hist_dir / f"{feat}.png")

    stats_df = pd.DataFrame(stats_rows)

    stats_df.to_csv(OUT_DIR / "window_feature_stats.csv", index=False)

    meta = {
        "exp_dir": str(EXP_DIR),
        "split": SPLIT,
        "num_vehicle_keys": int(len(keys)),
        "num_sequences": int(len(win_sequences)),
        "num_windows_total": int(X.shape[0]),
        "num_features": int(X.shape[1]) if X.ndim == 2 else int(len(feature_names)),
        "W": W,
        "stride": stride,
        "bins": int(BINS),
        "xlim_q": [float(XLIM_Q[0]), float(XLIM_Q[1])]
    }
    
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    print("==== Window feature distribution report ====")
    print(f"exp_dir:   {EXP_DIR}")
    print(f"split:     {SPLIT}  (vehicles={len(keys)})")
    print(f"windows:   {X.shape[0]}  features={len(feature_names)}")
    print(f"out_dir:   {OUT_DIR}")
    print(f"stats:     {OUT_DIR / 'window_feature_stats.csv'}")
    print(f"hists:     {hist_dir}")



if __name__ == "__main__":
    main()