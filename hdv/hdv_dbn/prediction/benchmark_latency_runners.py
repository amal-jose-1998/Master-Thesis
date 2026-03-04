"""
Benchmark runner functions for latency evaluation.

This module contains execution paths for:
- Inference-only benchmarks (window observations already prepared)
- End-to-end benchmarks (frame -> window -> inference)
"""

import time
import numpy as np
import torch

from ..config import FRAME_FEATURE_COLS, WINDOW_FEATURE_COLS, CONTINUOUS_FEATURES

try:
    from ....simulation.live_windowizer import LiveWindowizer
except Exception:
    LiveWindowizer = None

from ..utils.latency_utils import _sync
from .online_predictor import OnlinePredictor, BatchedOnlinePredictor


def _require_windowizer():
    if LiveWindowizer is None:
        raise ImportError(
            "LiveWindowizer could not be imported. Ensure `live_windowizer.py` is importable "
            "(e.g., located in your workspace root or on PYTHONPATH)."
        )


def _normalize_scaler_inputs(scaler_mean: np.ndarray, scaler_std: np.ndarray):
    scale_idx = [i for i, n in enumerate(WINDOW_FEATURE_COLS) if n in CONTINUOUS_FEATURES]
    mean = np.asarray(scaler_mean, dtype=np.float64).reshape(-1)
    std = np.asarray(scaler_std, dtype=np.float64).reshape(-1)
    return scale_idx, mean, std


def _scale_window_inplace(win: np.ndarray, scale_idx, scaler_mean: np.ndarray, scaler_std: np.ndarray):
    if scale_idx and scaler_mean.size == win.size and scaler_std.size == win.size:
        win[scale_idx] = (win[scale_idx] - scaler_mean[scale_idx]) / (scaler_std[scale_idx] + 1e-12)


def _window_to_obs_tensor(win: np.ndarray, scale_idx, scaler_mean: np.ndarray, scaler_std: np.ndarray, dev: torch.device):
    win_copy = win.copy()
    _scale_window_inplace(win_copy, scale_idx, scaler_mean, scaler_std)
    return torch.as_tensor(win_copy, dtype=torch.float32, device=dev)


def _windows_to_obs_batch(emitted_windows, scale_idx, scaler_mean: np.ndarray, scaler_std: np.ndarray, dev: torch.device):
    win_stack = np.stack([w.copy() for w in emitted_windows], axis=0)
    if scale_idx and scaler_mean.size == win_stack.shape[1] and scaler_std.size == win_stack.shape[1]:
        win_stack[:, scale_idx] = (win_stack[:, scale_idx] - scaler_mean[scale_idx]) / (scaler_std[scale_idx] + 1e-12)
    return torch.as_tensor(win_stack, dtype=torch.float32, device=dev)


def _make_windowizers(num_streams: int):
    windowizers = [LiveWindowizer() for _ in range(num_streams)]
    for w in windowizers:
        w.reset()
    return windowizers


def _validate_frame_vectors(frame_vecs_np):
    frame_vecs = [np.asarray(v, dtype=np.float64).reshape(-1) for v in frame_vecs_np]
    for v in frame_vecs:
        if v.shape[0] != len(FRAME_FEATURE_COLS):
            raise ValueError("One of the frame_vecs has wrong length for FRAME_FEATURE_COLS.")
    return frame_vecs


@torch.no_grad()
def bench_single_vehicle(predictor: OnlinePredictor, obs_vec_np, device, subsequent_iters=2000, prewarm_cuda=True):
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

    first_predict_call_ms = (t1_pred - t0_pred) * 1000.0
    first_prediction_total_ms = (t1_pred - t0_total) * 1000.0

    subsequent_total_ms = np.empty((subsequent_iters,), dtype=np.float64)
    for i in range(subsequent_iters):
        _sync(dev)
        t0 = time.perf_counter()
        predictor.update(obs_t)
        _ = predictor.predict_next()
        _sync(dev)
        t1 = time.perf_counter()
        subsequent_total_ms[i] = (t1 - t0) * 1000.0

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

    _sync(dev)
    t0_total = time.perf_counter()
    warmup_updates = 0
    while True:
        ready = predictor.is_ready
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


@torch.no_grad()
def bench_single_vehicle_e2e(
    predictor: OnlinePredictor,
    frame_vec_np,
    device,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    subsequent_windows=2000,
    prewarm_cuda=True,
    max_frames=5_000_000,
):
    """
    Single-stream end-to-end benchmark:
      frame feature vector -> LiveWindowizer -> (scale+transfer) -> update+predict.
    """
    _require_windowizer()
    dev = torch.device(device)

    predictor.reset()
    windowizer = LiveWindowizer()
    windowizer.reset()

    frame_vec = np.asarray(frame_vec_np, dtype=np.float64).reshape(-1)
    if frame_vec.shape[0] != len(FRAME_FEATURE_COLS):
        raise ValueError(
            f"frame_vec length {frame_vec.shape[0]} != len(FRAME_FEATURE_COLS)={len(FRAME_FEATURE_COLS)}"
        )

    scale_idx, scaler_mean, scaler_std = _normalize_scaler_inputs(scaler_mean, scaler_std)

    if dev.type == "cuda" and prewarm_cuda:
        _ = torch.zeros(1, device=dev)
        _sync(dev)

    per_frame_windowize_ms = []
    frames_used = 0
    n_updates = 0

    _sync(dev)
    t0_total = time.perf_counter()

    while not predictor.is_ready:
        frames_used += 1
        if frames_used >= max_frames:
            raise RuntimeError("Exceeded max_frames during warmup (predictor never became ready).")

        t0w = time.perf_counter()
        windows = windowizer.add_frame(frame_vec)
        t1w = time.perf_counter()
        per_frame_windowize_ms.append((t1w - t0w) * 1000.0)

        if not windows:
            continue

        obs_t = _window_to_obs_tensor(windows[-1], scale_idx, scaler_mean, scaler_std, dev)
        _sync(dev)
        predictor.update(obs_t)
        n_updates += 1

    _sync(dev)
    t0_pred = time.perf_counter()
    _ = predictor.predict_next()
    _sync(dev)
    t1_pred = time.perf_counter()

    first_predict_call_ms = (t1_pred - t0_pred) * 1000.0
    first_prediction_total_ms = (t1_pred - t0_total) * 1000.0

    per_pred_total_ms = np.empty((subsequent_windows,), dtype=np.float64)
    per_pred_windowize_ms = np.empty((subsequent_windows,), dtype=np.float64)
    per_pred_prep_ms = np.empty((subsequent_windows,), dtype=np.float64)
    per_pred_infer_ms = np.empty((subsequent_windows,), dtype=np.float64)

    produced = 0
    while produced < subsequent_windows:
        frames_used += 1
        if frames_used >= max_frames:
            raise RuntimeError("Exceeded max_frames while collecting subsequent windows.")

        t0w = time.perf_counter()
        windows = windowizer.add_frame(frame_vec)
        t1w = time.perf_counter()
        win_ms = (t1w - t0w) * 1000.0
        per_frame_windowize_ms.append(win_ms)

        if not windows:
            continue

        t0_total_step = t0w

        t0_prep = time.perf_counter()
        obs_t = _window_to_obs_tensor(windows[-1], scale_idx, scaler_mean, scaler_std, dev)
        _sync(dev)
        t1_prep = time.perf_counter()

        _sync(dev)
        t0_inf = time.perf_counter()
        predictor.update(obs_t)
        _ = predictor.predict_next()
        _sync(dev)
        t1_inf = time.perf_counter()

        t1_total_step = t1_inf

        per_pred_windowize_ms[produced] = win_ms
        per_pred_prep_ms[produced] = (t1_prep - t0_prep) * 1000.0
        per_pred_infer_ms[produced] = (t1_inf - t0_inf) * 1000.0
        per_pred_total_ms[produced] = (t1_total_step - t0_total_step) * 1000.0
        produced += 1

    return {
        "warmup_updates": int(n_updates),
        "warmup_frames": int(frames_used),
        "window_W": int(getattr(windowizer, "W", -1)),
        "window_stride": int(getattr(windowizer, "stride", -1)),
        "first_prediction_total_ms": float(first_prediction_total_ms),
        "first_predict_call_ms": float(first_predict_call_ms),
        "per_frame_windowize_ms": np.asarray(per_frame_windowize_ms, dtype=np.float64),
        "subsequent_total_ms": per_pred_total_ms,
        "subsequent_windowize_ms": per_pred_windowize_ms,
        "subsequent_prep_ms": per_pred_prep_ms,
        "subsequent_infer_ms": per_pred_infer_ms,
    }


@torch.no_grad()
def bench_loop_multi_e2e(
    predictors,
    frame_vecs_np,
    device,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    subsequent_windows=2000,
    prewarm_cuda=True,
    max_frames=5_000_000,
):
    """
    N independent OnlinePredictor objects stepped sequentially in Python, end-to-end:
      frame features -> windowize -> update+predict (when a window is emitted).
    """
    _require_windowizer()
    dev = torch.device(device)
    n_veh = len(predictors)

    for p in predictors:
        p.reset()

    windowizers = _make_windowizers(n_veh)
    frame_vecs = _validate_frame_vectors(frame_vecs_np)
    scale_idx, scaler_mean, scaler_std = _normalize_scaler_inputs(scaler_mean, scaler_std)

    if dev.type == "cuda" and prewarm_cuda:
        _ = torch.zeros(1, device=dev)
        _sync(dev)

    per_ready = [False] * n_veh
    warmup_updates_used = [0] * n_veh
    warmup_frames = 0
    per_frame_windowize_ms = []

    _sync(dev)
    t0_total = time.perf_counter()
    while not all(per_ready):
        warmup_frames += 1
        if warmup_frames >= max_frames:
            raise RuntimeError("Exceeded max_frames during warmup in loop_multi_e2e.")

        t0w = time.perf_counter()
        emitted = [None] * n_veh
        for i in range(n_veh):
            windows = windowizers[i].add_frame(frame_vecs[i])
            if windows:
                emitted[i] = windows[-1]
        t1w = time.perf_counter()
        per_frame_windowize_ms.append((t1w - t0w) * 1000.0)

        for i in range(n_veh):
            if per_ready[i] or emitted[i] is None:
                continue
            obs_t = _window_to_obs_tensor(emitted[i], scale_idx, scaler_mean, scaler_std, dev)
            _sync(dev)
            predictors[i].update(obs_t)
            warmup_updates_used[i] += 1
            if predictors[i].is_ready:
                _ = predictors[i].predict_next()
                per_ready[i] = True

    _sync(dev)
    t1_total = time.perf_counter()
    first_prediction_total_ms = (t1_total - t0_total) * 1000.0

    per_event_total_ms = np.empty((subsequent_windows,), dtype=np.float64)
    per_event_infer_ms = np.empty((subsequent_windows,), dtype=np.float64)
    per_event_windowize_ms = np.empty((subsequent_windows,), dtype=np.float64)

    produced = 0
    frames_used = warmup_frames
    while produced < subsequent_windows:
        frames_used += 1
        if frames_used >= max_frames:
            raise RuntimeError("Exceeded max_frames while collecting subsequent windows in loop_multi_e2e.")

        t0_event = time.perf_counter()
        emitted = [None] * n_veh

        t0w = time.perf_counter()
        for i in range(n_veh):
            windows = windowizers[i].add_frame(frame_vecs[i])
            if windows:
                emitted[i] = windows[-1]
        t1w = time.perf_counter()
        win_ms = (t1w - t0w) * 1000.0
        per_frame_windowize_ms.append(win_ms)

        if any(e is None for e in emitted):
            continue

        _sync(dev)
        t0_inf = time.perf_counter()
        for i in range(n_veh):
            obs_t = _window_to_obs_tensor(emitted[i], scale_idx, scaler_mean, scaler_std, dev)
            predictors[i].update(obs_t)
            _ = predictors[i].predict_next()
        _sync(dev)
        t1_inf = time.perf_counter()

        t1_event = t1_inf
        per_event_windowize_ms[produced] = win_ms
        per_event_infer_ms[produced] = (t1_inf - t0_inf) * 1000.0
        per_event_total_ms[produced] = (t1_event - t0_event) * 1000.0
        produced += 1

    return {
        "n_veh": int(n_veh),
        "warmup_updates": int(np.max(np.asarray(warmup_updates_used, dtype=np.int64))),
        "warmup_frames": int(warmup_frames),
        "window_W": int(getattr(windowizers[0], "W", -1)),
        "window_stride": int(getattr(windowizers[0], "stride", -1)),
        "first_prediction_total_ms": float(first_prediction_total_ms),
        "first_predict_call_ms": np.nan,
        "per_frame_windowize_ms": np.asarray(per_frame_windowize_ms, dtype=np.float64),
        "subsequent_total_ms": per_event_total_ms,
        "subsequent_windowize_ms": per_event_windowize_ms,
        "subsequent_prep_ms": np.full_like(per_event_total_ms, np.nan),
        "subsequent_infer_ms": per_event_infer_ms,
        "subsequent_per_vehicle_ms": per_event_total_ms / float(n_veh),
    }


@torch.no_grad()
def bench_batched_e2e(
    predictor: BatchedOnlinePredictor,
    frame_vecs_np,
    device,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    subsequent_windows=2000,
    prewarm_cuda=True,
    max_frames=5_000_000,
):
    """
    Batched end-to-end benchmark:
      B frame feature vectors -> B windowizers -> stack emitted windows -> batched update+predict.
    """
    _require_windowizer()
    dev = torch.device(device)
    B = int(len(frame_vecs_np))

    predictor.reset(B)

    windowizers = _make_windowizers(B)
    frame_vecs = _validate_frame_vectors(frame_vecs_np)
    scale_idx, scaler_mean, scaler_std = _normalize_scaler_inputs(scaler_mean, scaler_std)

    if dev.type == "cuda" and prewarm_cuda:
        _ = torch.zeros(1, device=dev)
        _sync(dev)

    warmup_frames = 0
    n_updates = 0
    per_frame_windowize_ms = []

    _sync(dev)
    t0_total = time.perf_counter()
    while True:
        ready = predictor.is_ready
        if ready is not None and bool(ready.all()):
            break

        warmup_frames += 1
        if warmup_frames >= max_frames:
            raise RuntimeError("Exceeded max_frames during warmup in batched_e2e.")

        t0w = time.perf_counter()
        emitted = []
        for i in range(B):
            windows = windowizers[i].add_frame(frame_vecs[i])
            emitted.append(windows[-1] if windows else None)
        t1w = time.perf_counter()
        per_frame_windowize_ms.append((t1w - t0w) * 1000.0)

        if any(e is None for e in emitted):
            continue

        obs_batch = _windows_to_obs_batch(emitted, scale_idx, scaler_mean, scaler_std, dev)
        _sync(dev)
        predictor.update(obs_batch)
        n_updates += 1

    _sync(dev)
    t0_pred = time.perf_counter()
    _ = predictor.predict_next(strict_ready=True)
    _sync(dev)
    t1_pred = time.perf_counter()

    first_predict_call_ms = (t1_pred - t0_pred) * 1000.0
    first_prediction_total_ms = (t1_pred - t0_total) * 1000.0

    per_event_total_ms = np.empty((subsequent_windows,), dtype=np.float64)
    per_event_windowize_ms = np.empty((subsequent_windows,), dtype=np.float64)
    per_event_prep_ms = np.empty((subsequent_windows,), dtype=np.float64)
    per_event_infer_ms = np.empty((subsequent_windows,), dtype=np.float64)

    produced = 0
    frames_used = warmup_frames
    while produced < subsequent_windows:
        frames_used += 1
        if frames_used >= max_frames:
            raise RuntimeError("Exceeded max_frames while collecting subsequent windows in batched_e2e.")

        t0_event = time.perf_counter()

        t0w = time.perf_counter()
        emitted = []
        for i in range(B):
            windows = windowizers[i].add_frame(frame_vecs[i])
            emitted.append(windows[-1] if windows else None)
        t1w = time.perf_counter()
        win_ms = (t1w - t0w) * 1000.0
        per_frame_windowize_ms.append(win_ms)

        if any(e is None for e in emitted):
            continue

        t0_prep = time.perf_counter()
        obs_batch = _windows_to_obs_batch(emitted, scale_idx, scaler_mean, scaler_std, dev)
        _sync(dev)
        t1_prep = time.perf_counter()

        _sync(dev)
        t0_inf = time.perf_counter()
        predictor.update(obs_batch)
        _ = predictor.predict_next(strict_ready=True)
        _sync(dev)
        t1_inf = time.perf_counter()

        t1_event = t1_inf

        per_event_windowize_ms[produced] = win_ms
        per_event_prep_ms[produced] = (t1_prep - t0_prep) * 1000.0
        per_event_infer_ms[produced] = (t1_inf - t0_inf) * 1000.0
        per_event_total_ms[produced] = (t1_event - t0_event) * 1000.0
        produced += 1

    return {
        "n_veh": int(B),
        "warmup_updates": int(n_updates),
        "warmup_frames": int(warmup_frames),
        "window_W": int(getattr(windowizers[0], "W", -1)),
        "window_stride": int(getattr(windowizers[0], "stride", -1)),
        "first_prediction_total_ms": float(first_prediction_total_ms),
        "first_predict_call_ms": float(first_predict_call_ms),
        "per_frame_windowize_ms": np.asarray(per_frame_windowize_ms, dtype=np.float64),
        "subsequent_total_ms": per_event_total_ms,
        "subsequent_windowize_ms": per_event_windowize_ms,
        "subsequent_prep_ms": per_event_prep_ms,
        "subsequent_infer_ms": per_event_infer_ms,
        "subsequent_per_vehicle_ms": per_event_total_ms / float(B),
    }
