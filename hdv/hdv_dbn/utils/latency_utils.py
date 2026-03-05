import numpy as np
import torch
from pathlib import Path
import sys

try:
    from ..trainer import HDVTrainer
    from ..config import FRAME_FEATURE_COLS, WINDOW_FEATURE_COLS, TRAINING_CONFIG
except ImportError:
    project_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(project_root))
    from hdv.hdv_dbn.trainer import HDVTrainer
    from hdv.hdv_dbn.config import FRAME_FEATURE_COLS, WINDOW_FEATURE_COLS, TRAINING_CONFIG


def audit_device_dtype(trainer: HDVTrainer, label):
    print(f"\n[audit:{label}] trainer.device/dtype:", getattr(trainer, "device", None), getattr(trainer, "dtype", None))
    for name in ("pi_s0", "pi_a0_given_s0", "A_s", "A_a"):
        x = getattr(trainer, name, None)
        if torch.is_tensor(x):
            print(f"[audit:{label}] {name}: {x.device} {x.dtype}")
        else:
            print(f"[audit:{label}] {name}: (not a tensor)")

    em = getattr(trainer, "emissions", None)
    if em is None:
        print(f"[audit:{label}] trainer.emissions: None")
        return
    print(f"[audit:{label}] emissions._device/_dtype:", getattr(em, "_device", None), getattr(em, "_dtype", None))

    if hasattr(em, "gauss") and hasattr(em.gauss, "_mean_t"):
        t = em.gauss._mean_t
        print(f"[audit:{label}] gauss._mean_t:", None if t is None else t.device)
    if hasattr(em, "bern") and hasattr(em.bern, "_p_t"):
        t = em.bern._p_t
        print(f"[audit:{label}] bern._p_t:", None if t is None else t.device)


def summary_stats(arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {"mean": np.nan, "p50": np.nan, "p90": np.nan, "p99": np.nan}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def sync(dev: torch.device):
    """Synchronize CUDA operations for accurate timing."""
    if dev.type == "cuda":
        torch.cuda.synchronize()


def summarize_arr(times_ms: np.ndarray, label):
    times_ms = np.asarray(times_ms, dtype=np.float64)
    if times_ms.size == 0:
        print(f"{label}: (empty)")
        return
    print(
        f"{label}: mean={times_ms.mean():.4f} ms | "
        f"p50={np.percentile(times_ms, 50):.4f} | "
        f"p90={np.percentile(times_ms, 90):.4f} | "
        f"p99={np.percentile(times_ms, 99):.4f}"
    )


def print_benchmark_env():
    import platform
    import json
    import subprocess
    import shutil

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }

    cpu_model = None
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    info["cpu_model"] = cpu_model or platform.processor()

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_total_mem_gb"] = round(props.total_memory / (1024**3), 2)
        info["gpu_cc"] = f"{props.major}.{props.minor}"

        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                    text=True,
                ).strip()
                info["nvidia_smi"] = out
            except Exception:
                info["nvidia_smi"] = None

    print("\n[benchmark-env]")
    print(json.dumps(info, indent=2))


def _infer_obs_dim_from_trainer(trainer: HDVTrainer):
    em = getattr(trainer, "emissions", None)
    if em is not None and hasattr(em, "obs_names"):
        n = len(getattr(em, "obs_names"))
        if n > 0:
            return int(n)
    if hasattr(trainer, "obs_names"):
        n = len(getattr(trainer, "obs_names"))
        if n > 0:
            return int(n)
    raise RuntimeError("Could not infer observation dimension (obs_names missing).")


def force_trainer_device(trainer: HDVTrainer, dev: torch.device, dtype: torch.dtype = torch.float32):
    trainer.device = torch.device(dev)
    trainer.dtype = dtype

    for name in ("pi_s0", "pi_a0_given_s0", "A_s", "A_a"):
        x = getattr(trainer, name, None)
        if torch.is_tensor(x):
            setattr(trainer, name, x.to(device=trainer.device, dtype=trainer.dtype))

    if hasattr(trainer, "emissions") and hasattr(trainer.emissions, "to_device"):
        trainer.emissions.to_device(device=trainer.device, dtype=trainer.dtype)


def build_obs_vectors(n_veh):
    """
    Build synthetic observation vectors (dtype float32) for the model-only benchmarks, 
    compatible with the model's expected input features (WINDOW_FEATURE_COLS). 
    """
    obs_dim = len(WINDOW_FEATURE_COLS)
    seed = int(getattr(TRAINING_CONFIG, "seed", 123))
    rng = np.random.default_rng(seed)
    obs_vecs = [rng.standard_normal(size=(obs_dim,)).astype(np.float32) for _ in range(int(n_veh))]
    print(f"[benchmark] Using {n_veh} synthetic RNG *window* obs vectors, F={obs_dim}, seed={seed}")
    return obs_vecs

def build_frame_vectors(n_veh):
    """
    Build synthetic frame feature vectors (dtype float64) for the end-to-end benchmarks, 
    compatible with the LiveWindowizer's expected input features (FRAME_FEATURE_COLS). 
    """
    D = len(FRAME_FEATURE_COLS)
    seed = int(getattr(TRAINING_CONFIG, "seed", 123)+1) # use a different seed than obs_vecs to ensure different synthetic values
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(size=(D,)).astype(np.float64) for _ in range(int(n_veh))]

def get_window_scaler(trainer_obj):
    """
    Extract window-feature scaler mean/std from a loaded trainer.
    Falls back to (zeros, ones) if unavailable.
    """
    F = len(WINDOW_FEATURE_COLS)
    mean = None
    std = None

    if trainer_obj is not None:
        mean = getattr(trainer_obj, "scaler_mean", None)
        std = getattr(trainer_obj, "scaler_std", None)

    # Handle dict (classwise) by choosing the first entry
    if isinstance(mean, dict):
        mean = next(iter(mean.values()), None)
    if isinstance(std, dict):
        std = next(iter(std.values()), None)

    if mean is None or std is None:
        mean = np.zeros((F,), dtype=np.float64)
        std = np.ones((F,), dtype=np.float64)
        return mean, std

    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.asarray(std, dtype=np.float64).reshape(-1)
    if mean.size != F or std.size != F:
        mean = np.zeros((F,), dtype=np.float64)
        std = np.ones((F,), dtype=np.float64)
    return mean, std


