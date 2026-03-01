"""
How long does one online prediction step take after the system is running?

Measures per-step latency (compute-only):
    predictor.update(obs_t) + predictor.predict_next()
where obs_t is a single observation vector already on the target device.

CPU and GPU are benchmarked in separate runs of a freshly loaded trainer/model
to avoid cross-device cache contamination.
"""
import time
import numpy as np
import torch
from pathlib import Path
import sys

try:
    from .online_predictor import OnlinePredictor
    from .model_interface import HDVDbnModel
    from .data_loader import load_test_data_for_prediction
    from ..trainer import HDVTrainer
except ImportError:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from hdv.hdv_dbn.prediction.online_predictor import OnlinePredictor
    from hdv.hdv_dbn.prediction.model_interface import HDVDbnModel
    from hdv.hdv_dbn.prediction.data_loader import load_test_data_for_prediction
    from hdv.hdv_dbn.trainer import HDVTrainer

def audit_device_dtype(trainer, label):
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

    # Hierarchical caches
    if hasattr(em, "gauss") and hasattr(em.gauss, "_mean_t"):
        t = em.gauss._mean_t
        print(f"[audit:{label}] gauss._mean_t:", None if t is None else t.device)
    if hasattr(em, "bern") and hasattr(em.bern, "_p_t"):
        t = em.bern._p_t
        print(f"[audit:{label}] bern._p_t:", None if t is None else t.device)


def _sync(dev: torch.device):
    """Helper to synchronize GPU for accurate timing."""
    if dev.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def bench_compute_only(predictor: OnlinePredictor, obs_vec_np, device, iters=2000, warmup=200):
    """
    Benchmark the compute time of the OnlinePredictor's update + predict steps for a single observation vector.
    This isolates the model inference time without data loading or preprocessing overhead.
    
    Parameters
    predictor: OnlinePredictor
        An instance of the OnlinePredictor to benchmark.
    obs_vec_np: np.ndarray
        A single observation vector (F,) in numpy format to feed into the predictor.
    device: str or torch.device
        Device to run the benchmark on ("cpu" or "cuda").
    iters: int
        Number of timed iterations to run after warmup.
    warmup: int
        Number of initial iterations to run before timing (to allow for JIT warmup, GPU clock stabilization, etc.).
    
    Returns
    np.ndarray
        Array of timing measurements in milliseconds.
    """    
    dev = torch.device(device)
    predictor.reset()

    obs_vec_np = np.asarray(obs_vec_np, dtype=np.float32).reshape(-1) # Ensure obs_vec is 1D and float32 for consistency.
    obs_t = torch.as_tensor(obs_vec_np, dtype=torch.float32, device=dev) # Convert the observation vector to a torch tensor on the target device.

    # warmup
    for _ in range(warmup):
        predictor.update(obs_t)
        if predictor.is_ready:
            _ = predictor.predict_next().pred_logprob.reshape(-1)[0].item()
    _sync(dev)

    # timed
    times_ms = []
    while len(times_ms) < iters:
        _sync(dev)
        t0 = time.perf_counter() # Start timing just before the update/predict steps.

        predictor.update(obs_t)
        if not predictor.is_ready:
            continue

        out = predictor.predict_next()
        _ = float(out.pred_logprob.reshape(-1)[0].item())  # force materialization of the output logits to include it in the timing.

        _sync(dev)
        t1 = time.perf_counter() # End timing immediately after the output is fully materialized.
        times_ms.append((t1 - t0) * 1000.0) # Convert to milliseconds.

    return np.asarray(times_ms, dtype=np.float64)


def summarize(times_ms, label):
    print(
        f"{label}: mean={times_ms.mean():.4f} ms | "
        f"p50={np.percentile(times_ms,50):.4f} | "
        f"p90={np.percentile(times_ms,90):.4f} | "
        f"p99={np.percentile(times_ms,99):.4f}"
    )

def _infer_obs_dim_from_trainer(trainer: HDVTrainer):
    em = getattr(trainer, "emissions", None)
    if em is not None and hasattr(em, "obs_names"):
        n = len(getattr(em, "obs_names"))
        if n > 0:
            return int(n)

    # Fallback: trainer may store obs_names
    if hasattr(trainer, "obs_names"):
        n = len(getattr(trainer, "obs_names"))
        if n > 0:
            return int(n)

    raise RuntimeError("Could not infer observation dimension (obs_names missing).")

def force_trainer_device(trainer: HDVTrainer, dev: torch.device, dtype: torch.dtype = torch.float32):
    # Make trainer consistent (so audits and any downstream uses are clean)
    trainer.device = torch.device(dev)
    trainer.dtype = dtype

    for name in ("pi_s0", "pi_a0_given_s0", "A_s", "A_a"):
        x = getattr(trainer, name, None)
        if torch.is_tensor(x):
            setattr(trainer, name, x.to(device=trainer.device, dtype=trainer.dtype))

    # Move emission caches/params using the correct API
    if hasattr(trainer, "emissions") and hasattr(trainer.emissions, "to_device"):
        trainer.emissions.to_device(device=trainer.device, dtype=trainer.dtype)

def print_benchmark_env():
    import platform, json, os, subprocess
    import torch

    info = {}
    info["platform"] = platform.platform()
    info["python"] = platform.python_version()
    info["pytorch"] = torch.__version__
    info["torch_num_threads"] = torch.get_num_threads()

    # CUDA / GPU
    info["cuda_available"] = torch.cuda.is_available()
    info["torch_cuda_version"] = torch.version.cuda
    info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_total_mem_gb"] = round(props.total_memory / (1024**3), 2)

        # Try driver + detailed GPU info (best-effort)
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"],
                text=True
            ).strip()
            info["nvidia_smi"] = out
        except Exception:
            info["nvidia_smi"] = None

    # CPU model (Linux best-effort)
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        info["cpu_model"] = platform.processor()

    print("\n[benchmark-env]")
    print(json.dumps(info, indent=2))


def main():
    print_benchmark_env()
    torch.set_grad_enabled(False)

    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent.parent

    data_root = workspace_root / "hdv" / "data" / "highd"
    exp_dir = workspace_root / "hdv" / "models" / "main-model-sticky_S2_A4_hierarchical"
    checkpoint_path = exp_dir / "final.npz"

    # ----------------------------
    # Get one observation vector (CPU-side numpy) for benchmarking
    # ----------------------------
    obs_vec = None
    try:
        trainer_tmp, test = load_test_data_for_prediction(
            exp_dir=exp_dir,
            data_root=data_root,
            checkpoint_name="final.npz",
        )
        obs_seq = test.scaled_obs[0]   # (T,F)
        obs_vec = np.asarray(obs_seq[0], dtype=np.float32)  # (F,)
        print(f"[benchmark] Using real test obs vector dim F={obs_vec.shape[0]}")
    except Exception as e:
        print(f"[benchmark] Data loading failed ({type(e).__name__}: {e}).")
        print(f"[benchmark] Falling back to checkpoint-only benchmark: {checkpoint_path}")
        trainer_tmp = HDVTrainer.load(checkpoint_path)
        obs_dim = _infer_obs_dim_from_trainer(trainer_tmp)
        obs_vec = np.zeros((obs_dim,), dtype=np.float32)
        print(f"[benchmark] Using synthetic zero obs vector dim F={obs_vec.shape[0]}")

    warmup_steps = 5

    # ----------------------------
    # CPU run (fresh trainer)
    # ----------------------------
    cpu_dev = torch.device("cpu")
    trainer_cpu = HDVTrainer.load(checkpoint_path)
    force_trainer_device(trainer_cpu, cpu_dev, torch.float32)
    model_cpu = HDVDbnModel(trainer_cpu, device=cpu_dev, dtype=torch.float32)
    audit_device_dtype(trainer_cpu, "cpu")

    pred_cpu = OnlinePredictor(model_cpu, warmup_steps=warmup_steps, device=cpu_dev, dtype=torch.float32)
    t_cpu = bench_compute_only(pred_cpu, obs_vec, device=cpu_dev)
    summarize(t_cpu, "CPU compute-only (1 vector)")

    # ----------------------------
    # GPU run (fresh trainer)
    # ----------------------------
    if torch.cuda.is_available():
        gpu_dev = torch.device("cuda")
        trainer_gpu = HDVTrainer.load(checkpoint_path)
        force_trainer_device(trainer_gpu, gpu_dev, torch.float32)
        model_gpu = HDVDbnModel(trainer_gpu, device=gpu_dev, dtype=torch.float32)
        audit_device_dtype(trainer_gpu, "gpu")

        pred_gpu = OnlinePredictor(model_gpu, warmup_steps=warmup_steps, device=gpu_dev, dtype=torch.float32)
        t_gpu = bench_compute_only(pred_gpu, obs_vec, device=gpu_dev)
        summarize(t_gpu, "GPU compute-only (1 vector)")
    else:
        print("[benchmark] CUDA not available; GPU benchmark skipped.")


if __name__ == "__main__":
    main()