"""
Latency benchmark for online DBN inference:
  - single: OnlinePredictor (1 stream)
  - loop_multi: N OnlinePredictor objects stepped sequentially in Python
  - batched: BatchedOnlinePredictor (true (B,F) batch)

Measures per-step wall-clock compute:
    update(obs_t) + predict_next()
"""
import time
import numpy as np
import torch
from pathlib import Path
import sys
import csv

try:
    from .online_predictor import OnlinePredictor, BatchedOnlinePredictor
    from .model_interface import HDVDbnModel
    from .data_loader import load_test_data_for_prediction
    from ..trainer import HDVTrainer
except ImportError:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from hdv.hdv_dbn.prediction.online_predictor import OnlinePredictor, BatchedOnlinePredictor
    from hdv.hdv_dbn.prediction.model_interface import HDVDbnModel
    from hdv.hdv_dbn.prediction.data_loader import load_test_data_for_prediction
    from hdv.hdv_dbn.trainer import HDVTrainer

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


def _summary_stats(arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {"mean": np.nan, "p50": np.nan, "p90": np.nan, "p99": np.nan}
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def _sync(dev: torch.device):
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



def _build_obs_vectors(exp_dir, data_root, checkpoint_path, n_veh):
    """Load n_veh observation vectors; fallback to zero vectors if test data unavailable."""
    try:
        _, test = load_test_data_for_prediction(exp_dir=exp_dir, data_root=data_root, checkpoint_name="final.npz")
        obs_vecs = []
        for i in range(min(n_veh, len(test.scaled_obs))):
            obs_seq = test.scaled_obs[i]
            obs_vecs.append(np.asarray(obs_seq[0], dtype=np.float32))

        if len(obs_vecs) == 0:
            raise RuntimeError("No sequences in test split.")

        while len(obs_vecs) < n_veh:
            obs_vecs.append(obs_vecs[-1].copy())

        print(f"[benchmark] Using {n_veh} real test obs vectors, F={obs_vecs[0].shape[0]}")
        return obs_vecs
    except Exception as e:
        print(f"[benchmark] Data loading failed ({type(e).__name__}: {e}).")
        trainer_tmp = HDVTrainer.load(checkpoint_path)
        obs_dim = _infer_obs_dim_from_trainer(trainer_tmp)
        obs_vecs = [np.zeros((obs_dim,), dtype=np.float32) for _ in range(n_veh)]
        print(f"[benchmark] Using {n_veh} synthetic zero obs vectors, F={obs_dim}")
        return obs_vecs



@torch.no_grad()
def bench_single_vehicle(predictor: OnlinePredictor, obs_vec_np, device, subsequent_iters=2000, prewarm_cuda=True, max_pre_ready_updates=100000):
    """Benchmark one predictor: first prediction latency and subsequent per-step latency."""
    dev = torch.device(device)
    predictor.reset()

    obs_vec_np = np.asarray(obs_vec_np, dtype=np.float32).reshape(-1)
    obs_t = torch.as_tensor(obs_vec_np, dtype=torch.float32, device=dev)

    if dev.type == "cuda" and prewarm_cuda:
        _ = torch.zeros(1, device=dev)
        _sync(dev)

    pre_ready_update_ms = []
    _sync(dev)
    t0_total = time.perf_counter()

    while not predictor.is_ready:
        _sync(dev)
        t0 = time.perf_counter()
        predictor.update(obs_t)
        _sync(dev)
        t1 = time.perf_counter()
        pre_ready_update_ms.append((t1 - t0) * 1000.0)

    _sync(dev)
    t0_pred = time.perf_counter()
    _ = predictor.predict_next()
    _sync(dev)
    t1_pred = time.perf_counter()

    first_predict_call_ms = (t1_pred - t0_pred) * 1000.0 # time for the predict_next() call itself
    first_prediction_total_ms = (t1_pred - t0_total) * 1000.0 # total time from start until first prediction ready

    subsequent_total_ms = np.empty((subsequent_iters,), dtype=np.float64)
    for i in range(subsequent_iters):
        _sync(dev)
        t0 = time.perf_counter()
        predictor.update(obs_t)
        _ = predictor.predict_next()
        _sync(dev)
        t1 = time.perf_counter()
        subsequent_total_ms[i] = (t1 - t0) * 1000.0 # time for update+predict for subsequent steps

    return {
        "warmup_updates": int(len(pre_ready_update_ms)),
        "first_prediction_total_ms": float(first_prediction_total_ms),
        "first_predict_call_ms": float(first_predict_call_ms),
        "subsequent_total_ms": subsequent_total_ms,
    }


@torch.no_grad()
def bench_loop_multi(predictors, obs_vecs_np, device, subsequent_iters=2000, prewarm_cuda=True, max_total_updates=100000):
    """
    N independent OnlinePredictor objects stepped sequentially in Python.
    Measures wall time per timestep for the whole fleet (N streams).
    """
    dev = torch.device(device)
    n_veh = len(predictors)

    for p in predictors:
        p.reset()

    obs_ts = [
        torch.as_tensor(np.asarray(v, dtype=np.float32).reshape(-1), dtype=torch.float32, device=dev)
        for v in obs_vecs_np
    ]

    if dev.type == "cuda" and prewarm_cuda:
        _ = torch.zeros(1, device=dev)
        _sync(dev)

    per_ready = [False] * n_veh
    warmup_steps_used = [0] * n_veh
    total_updates = 0

    _sync(dev)
    t0_total = time.perf_counter()
    while not all(per_ready):
        if total_updates >= max_total_updates:
            raise RuntimeError("Not all predictors became ready within max_total_updates.")
        for i, (p, o) in enumerate(zip(predictors, obs_ts)):
            if per_ready[i]:
                continue
            p.update(o)
            total_updates += 1
            warmup_steps_used[i] += 1
            if p.is_ready:
                _ = p.predict_next()
                per_ready[i] = True
    _sync(dev)
    t1_total = time.perf_counter()

    first_prediction_total_ms = (t1_total - t0_total) * 1000.0

    subsequent_total_ms = np.empty((subsequent_iters,), dtype=np.float64)
    for k in range(subsequent_iters):
        _sync(dev)
        t0 = time.perf_counter()
        for p, o in zip(predictors, obs_ts):
            p.update(o)
            _ = p.predict_next()
        _sync(dev)
        t1 = time.perf_counter()
        subsequent_total_ms[k] = (t1 - t0) * 1000.0

    return {
        "n_veh": int(n_veh),
        "warmup_updates": int(np.max(np.asarray(warmup_steps_used, dtype=np.int64))),
        "first_prediction_total_ms": float(first_prediction_total_ms),
        "first_predict_call_ms": np.nan,
        "subsequent_total_ms": subsequent_total_ms,
        "subsequent_per_vehicle_ms": subsequent_total_ms / float(n_veh),
        "warmup_steps_per_vehicle": np.asarray(warmup_steps_used, dtype=np.int64),
    }


@torch.no_grad()
def bench_batched(predictor: BatchedOnlinePredictor, obs_vecs_np, device, subsequent_iters=2000, prewarm_cuda=True):
    """
    One BatchedOnlinePredictor processing a (B,F) observation batch.
    """
    dev = torch.device(device)
    B = int(len(obs_vecs_np))
    predictor.reset(B)

    obs_batch = torch.stack(
        [torch.as_tensor(np.asarray(v, dtype=np.float32).reshape(-1), device=dev) for v in obs_vecs_np],
        dim=0,
    ).to(dtype=torch.float32)

    if dev.type == "cuda" and prewarm_cuda:
        _ = torch.zeros(1, device=dev)
        _sync(dev)

    # Warmup until all streams ready
    _sync(dev)
    t0_total = time.perf_counter()
    warmup_updates = 0
    while True:
        ready = predictor.is_ready  # (B,) bool
        if ready is not None and bool(ready.all()):
            break
        predictor.update(obs_batch)
        warmup_updates += 1

    _sync(dev)
    t0_pred = time.perf_counter()
    _ = predictor.predict_next(strict_ready=True)
    _sync(dev)
    t1_pred = time.perf_counter()

    first_predict_call_ms = (t1_pred - t0_pred) * 1000.0
    first_prediction_total_ms = (t1_pred - t0_total) * 1000.0

    subsequent_total_ms = np.empty((subsequent_iters,), dtype=np.float64)
    for i in range(subsequent_iters):
        _sync(dev)
        t0 = time.perf_counter()
        predictor.update(obs_batch)
        _ = predictor.predict_next(strict_ready=True)
        _sync(dev)
        t1 = time.perf_counter()
        subsequent_total_ms[i] = (t1 - t0) * 1000.0

    return {
        "n_veh": int(B),
        "warmup_updates": int(warmup_updates),
        "first_prediction_total_ms": float(first_prediction_total_ms),
        "first_predict_call_ms": float(first_predict_call_ms),
        "subsequent_total_ms": subsequent_total_ms,
        "subsequent_per_vehicle_ms": subsequent_total_ms / float(B),
    }


def _write_results_csv(csv_path: Path, rows):
    fieldnames = [
        "device",
        "scenario",
        "n_veh",
        "warmup_updates",
        "first_prediction_total_ms",
        "first_predict_call_ms",
        "first_prediction_per_vehicle_ms",
        "subsequent_mean_ms",
        "subsequent_p50_ms",
        "subsequent_p90_ms",
        "subsequent_p99_ms",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[benchmark] Wrote CSV summary: {csv_path}")


def _print_block(title: str, res: dict):
    print(f"\n{title}")
    for k in ("warmup_updates", "first_prediction_total_ms", "first_predict_call_ms"):
        if k in res:
            print(f"  {k}: {res[k]}")
    if "subsequent_total_ms" in res:
        summarize_arr(res["subsequent_total_ms"], "  subsequent (update+predict)")


def main():
    print_benchmark_env()
    torch.set_grad_enabled(False)

    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent.parent

    data_root = workspace_root / "hdv" / "data" / "highd"
    exp_dir = workspace_root / "hdv" / "models" / "main-model-sticky_S2_A4_hierarchical"
    checkpoint_path = exp_dir / "final.npz"

    warmup_steps = 5
    N_VEH = 5
    subsequent_iters = 2000
    csv_rows = []

    obs_vecs = _build_obs_vectors(exp_dir=exp_dir, data_root=data_root, checkpoint_path=checkpoint_path, n_veh=N_VEH)
    obs_vec = obs_vecs[0]

    def run_one_device(dev: torch.device, label: str):
        trainer = HDVTrainer.load(checkpoint_path)
        force_trainer_device(trainer, dev, torch.float32)
        model = HDVDbnModel(trainer, device=dev, dtype=torch.float32)
        audit_device_dtype(trainer, label)

        # 1) single
        p_single = OnlinePredictor(model, warmup_steps=warmup_steps, device=dev, dtype=torch.float32)
        res_single = bench_single_vehicle(p_single, obs_vec, device=dev, subsequent_iters=subsequent_iters)

        # 2) loop_multi
        ps = [OnlinePredictor(model, warmup_steps=warmup_steps, device=dev, dtype=torch.float32) for _ in range(N_VEH)]
        res_loop = bench_loop_multi(ps, obs_vecs, device=dev, subsequent_iters=subsequent_iters)

        # 3) batched
        p_batched = BatchedOnlinePredictor(model, warmup_steps=warmup_steps, device=dev, dtype=torch.float32)
        res_batched = bench_batched(p_batched, obs_vecs, device=dev, subsequent_iters=subsequent_iters)

        _print_block(f"[{label}] single (OnlinePredictor)", res_single)
        _print_block(f"[{label}] loop_multi (N={N_VEH}, OnlinePredictor x N)", res_loop)
        _print_block(f"[{label}] batched (B={N_VEH}, BatchedOnlinePredictor)", res_batched)

        # CSV rows
        for scenario, res, n_veh, first_predict in [
            ("single", res_single, 1, res_single["first_predict_call_ms"]),
            ("loop_multi", res_loop, N_VEH, np.nan),
            ("batched", res_batched, N_VEH, res_batched["first_predict_call_ms"]),
        ]:
            stats = _summary_stats(res["subsequent_total_ms"])
            csv_rows.append(
                {
                    "device": label.lower(),
                    "scenario": scenario,
                    "n_veh": int(n_veh),
                    "warmup_updates": int(res["warmup_updates"]),
                    "first_prediction_total_ms": float(res["first_prediction_total_ms"]),
                    "first_predict_call_ms": float(first_predict) if np.isfinite(first_predict) else np.nan,
                    "first_prediction_per_vehicle_ms": float(res["first_prediction_total_ms"]) / float(n_veh),
                    "subsequent_mean_ms": stats["mean"],
                    "subsequent_p50_ms": stats["p50"],
                    "subsequent_p90_ms": stats["p90"],
                    "subsequent_p99_ms": stats["p99"],
                }
            )

    # CPU
    run_one_device(torch.device("cpu"), "CPU")

    # GPU
    if torch.cuda.is_available():
        run_one_device(torch.device("cuda"), "GPU")
    else:
        print("[benchmark] CUDA not available; skipping GPU.")

    csv_path = Path(__file__).resolve().parent / "benchmark_latency_results.csv"
    _write_results_csv(csv_path, csv_rows)


if __name__ == "__main__":
    main()