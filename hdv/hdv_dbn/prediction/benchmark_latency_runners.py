"""
Benchmark runner functions for latency evaluation.

This module contains execution paths for:
- Inference-only benchmarks (window observations already prepared)
- End-to-end benchmarks (frame -> window -> inference)
"""

import time
from typing import List
import numpy as np
import torch

from ..config import FRAME_FEATURE_COLS, WINDOW_FEATURE_COLS, CONTINUOUS_FEATURES
from simulation.live_windowizer import LiveWindowizer
from ..utils.latency_utils import sync
from .online_predictor import OnlinePredictor, BatchedOnlinePredictor


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
def bench_single_vehicle(predictor: OnlinePredictor, obs_vec_np, device, subsequent_iters=200, prewarm_cuda=True):
    """
    Benchmark one predictor: first prediction latency and subsequent per-step latency.
    
    Measures:
    - Time taken by update() calls before the predictor becomes ready (warmup time).
    - Time taken by the first predict() call alone, excluding warmup.
    - Total time from the start of the first update() call to the completion of the first predict() call, including warmup.
    - Time taken by subsequent update()+predict() calls for a number of iterations, to capture steady-state performance after warmup.
    
    returns a dict containing these latency measurements and warmup info.
    """
    dev = torch.device(device) # ensure predictor and model are on the correct device and dtype
    predictor.reset() # reset predictor state before benchmarking

    obs_vec_np = np.asarray(obs_vec_np, dtype=np.float32).reshape(-1) # ensure obs_vec is a 1D float32 numpy array
    obs_t = torch.as_tensor(obs_vec_np, dtype=torch.float32, device=dev) # convert obs_vec to a torch tensor on the correct device and dtype

    if dev.type == "cuda" and prewarm_cuda: # pre-warm CUDA context to avoid including context init time in measurements
        _ = torch.zeros(1, device=dev)

    sync(dev)
    t0_total = time.perf_counter() # start total timer from the beginning of the first update call, to capture warmup time as well

    warmup_updates = 0
    while not predictor.is_ready: # keep calling update() until the predictor reports it's ready
        predictor.update(obs_t) # perform an update step with the observation tensor
        warmup_updates += 1

    _ = predictor.predict_next() # perform the first prediction
    sync(dev)
    t1_pred = time.perf_counter() # end timer for the first predict call

    first_prediction_total_ms = (t1_pred - t0_total) * 1000.0 # total time from the start of the first update() call to the completion of the first predict() call, including warmup

    # subsequent latency: measure total time for update+predict for a number of iterations, to capture steady-state performance after warmup
    subsequent_total_ms = np.empty((subsequent_iters,), dtype=np.float64)
    for i in range(subsequent_iters):
        sync(dev)
        t0 = time.perf_counter() # start timer for this iteration's update+predict
        predictor.update(obs_t)
        _ = predictor.predict_next()
        sync(dev)
        t1 = time.perf_counter() # end timer for this iteration's update+predict
        subsequent_total_ms[i] = (t1 - t0) * 1000.0

    return {
        "warmup_updates": int(warmup_updates),
        "first_prediction_total_ms": float(first_prediction_total_ms),
        "subsequent_total_ms": subsequent_total_ms,
    }


@torch.no_grad()
def bench_loop_multi(predictors: list[OnlinePredictor], obs_vecs_np, device, subsequent_iters=200, prewarm_cuda=True, max_total_updates=100):
    """
    N independent OnlinePredictor objects stepped sequentially in Python.
    Measures wall time per timestep for the whole fleet (N streams).
    """
    dev = torch.device(device)
    n_veh = len(predictors)

    for p in predictors:
        p.reset() # reset all predictors before benchmarking

    obs_ts = [ # convert each observation vector to a torch tensor on the correct device and dtype
        torch.as_tensor(np.asarray(v, dtype=np.float32).reshape(-1), dtype=torch.float32, device=dev)
        for v in obs_vecs_np
    ]

    if dev.type == "cuda" and prewarm_cuda:
        _ = torch.zeros(1, device=dev)

    per_ready = [False] * n_veh # track which predictors have become ready, as they may require different numbers of update() calls due to warmup variability
    warmup_steps_used = [0] * n_veh # track how many update() calls were used for each predictor to become ready, for warmup analysis
    total_updates = 0 # track total number of update() calls across all predictors, to prevent infinite loops if some predictors never become ready

    sync(dev)
    t0_total = time.perf_counter() # start total timer from the beginning of the first update call
    while not all(per_ready): # keep calling update() on all predictors until they all report they're ready, tracking the time taken by each call and how many calls are needed for each predictor to become ready
        if total_updates >= max_total_updates: # safety check to prevent infinite loops if some predictors never become ready
            raise RuntimeError("Not all predictors became ready within max_total_updates.")
        for i, (p, o) in enumerate(zip(predictors, obs_ts)): # step through each predictor and its corresponding observation tensor
            if per_ready[i]: # skip predictors that are already ready, as we only want to track the time taken by update() calls before they become ready
                continue
            p.update(o) 
            total_updates += 1 
            warmup_steps_used[i] += 1
            if p.is_ready:
                _ = p.predict_next() 
                per_ready[i] = True # mark this predictor as ready so we can stop calling update() on it 
    sync(dev)
    t1_total = time.perf_counter() # end timer after all warmup updates and first predictions have fully completed

    first_prediction_total_ms = (t1_total - t0_total) * 1000.0 # total time for getting the first prediction from all predictors, including warmup

    subsequent_total_ms = np.empty((subsequent_iters,), dtype=np.float64) # to store the time for all the subsequent update+predict steps across all predictors
    for k in range(subsequent_iters):
        sync(dev)
        t0 = time.perf_counter() # start timer for this iteration's update+predict across all predictors
        for p, o in zip(predictors, obs_ts):
            p.update(o)
            _ = p.predict_next()
        sync(dev)
        t1 = time.perf_counter() # end timer for this iteration's update+predict across all predictors
        subsequent_total_ms[k] = (t1 - t0) * 1000.0 # total time for this iteration's update+predict across all predictors

    return {
        "n_veh": int(n_veh),
        "warmup_updates": int(np.max(np.asarray(warmup_steps_used, dtype=np.int64))),
        "first_prediction_total_ms": float(first_prediction_total_ms),
        "subsequent_total_ms": subsequent_total_ms,
        "warmup_steps_per_vehicle": np.asarray(warmup_steps_used, dtype=np.int64),
    }


@torch.no_grad()
def bench_batched(predictor: BatchedOnlinePredictor, obs_vecs_np, device, subsequent_iters=200, prewarm_cuda=True):
    """
    One BatchedOnlinePredictor processing a (B,F) observation batch.
    """
    dev = torch.device(device)
    B = int(len(obs_vecs_np)) # batch size = number of vehicles/streams
    predictor.reset(B) # reset predictor state before benchmarking, providing batch size for proper initialization

    obs_batch = torch.stack(
        [torch.as_tensor(np.asarray(v, dtype=np.float32).reshape(-1), device=dev) for v in obs_vecs_np],
        dim=0,
    ).to(dtype=torch.float32)

    if dev.type == "cuda" and prewarm_cuda: # pre-warm CUDA context to avoid including context init time in measurements
        _ = torch.zeros(1, device=dev)

    sync(dev)
    t0_total = time.perf_counter() # start total timer from the beginning of the first update call, to capture warmup time as well
    warmup_updates = 0
    while True: # keep calling update() until the predictor reports it's ready, tracking the time taken by each call
        ready = predictor.is_ready
        if ready is not None and bool(ready.all()): # check if all streams in the batch are ready; if so, break out of the loop to start the prediction timer
            break
        predictor.update(obs_batch) # perform an update step with the batch of observation tensors
        warmup_updates += 1

    _ = predictor.predict_next(strict_ready=True) # perform the first prediction, ensuring that the predictor is ready before starting the timer
    sync(dev)
    t1_pred = time.perf_counter() # end timer for the first predict call

    first_prediction_total_ms = (t1_pred - t0_total) * 1000.0 # total time for the first prediction, including warmup

    subsequent_total_ms = np.empty((subsequent_iters,), dtype=np.float64)
    for i in range(subsequent_iters): # loop through some iterations of update+predict to capture steady-state performance after warmup, tracking the time taken by each iteration
        sync(dev)
        t0 = time.perf_counter()
        predictor.update(obs_batch)
        _ = predictor.predict_next(strict_ready=True)
        sync(dev)
        t1 = time.perf_counter()
        subsequent_total_ms[i] = (t1 - t0) * 1000.0 # total time for this iteration's update+predict across the whole batch

    return {
        "n_veh": int(B),
        "warmup_updates": int(warmup_updates),
        "first_prediction_total_ms": float(first_prediction_total_ms),
        "subsequent_total_ms": subsequent_total_ms,
    }


@torch.no_grad()
def bench_single_vehicle_e2e(predictor: OnlinePredictor, frame_vec_np, device, scaler_mean, scaler_std, subsequent_windows=2000, prewarm_cuda=True, max_frames=5000000):
    """
    Single-stream end-to-end benchmark:
      frame feature vector -> LiveWindowizer -> (scale+transfer) -> update+predict.
    """
    dev = torch.device(device)

    predictor.reset() # reset predictor state before benchmarking
    windowizer = LiveWindowizer() # initialize a LiveWindowizer for this single stream; it will maintain its own internal state as we feed frames into it
    windowizer.reset() # ensure the windowizer is in a clean state before starting the benchmark

    frame_vec = np.asarray(frame_vec_np, dtype=np.float64).reshape(-1) # ensure frame_vec is a 1D float64 numpy array, as expected by the LiveWindowizer
    if frame_vec.shape[0] != len(FRAME_FEATURE_COLS): # sanity check to ensure the input frame vector has the correct number of features expected by the LiveWindowizer
        raise ValueError(
            f"frame_vec length {frame_vec.shape[0]} != len(FRAME_FEATURE_COLS)={len(FRAME_FEATURE_COLS)}"
        )

    # normalize and validate the scaler inputs, and determine which indices of the window features correspond to continuous features that need to be scaled
    scale_idx, scaler_mean, scaler_std = _normalize_scaler_inputs(scaler_mean, scaler_std)

    if dev.type == "cuda" and prewarm_cuda: # pre-warm CUDA context to avoid including context init time in measurements
        _ = torch.zeros(1, device=dev)

    frames_used = 0
    n_updates = 0

    sync(dev)
    t0_total = time.perf_counter() # start total timer from the beginning of the first frame processing, to capture warmup time as well

    while not predictor.is_ready: # keep feeding frames into the windowizer and calling update() on the predictor with the emitted windows until the predictor reports it's ready, tracking the time taken by each step and how many frames are needed for warmup
        frames_used += 1
        if frames_used >= max_frames: # safety check to prevent infinite loops if the predictor never becomes ready
            raise RuntimeError("Exceeded max_frames during warmup (predictor never became ready).")

        windows = windowizer.add_frame(frame_vec)
        if not windows: # if the windowizer did not emit any windows after processing this frame, continue to the next frame without calling update() on the predictor, as we only want to call update() when we have a new window to feed into the model
            continue

        obs_t = _window_to_obs_tensor(windows[-1], scale_idx, scaler_mean, scaler_std, dev) # take the most recent emitted window, scale it, and convert it to a torch tensor on the correct device and dtype, to feed into the predictor
        predictor.update(obs_t) # call update() on the predictor with the new observation tensor; this may eventually lead to the predictor becoming ready after enough updates, which will allow us to start the prediction timer
        n_updates += 1

    _ = predictor.predict_next() # perform the first prediction after the predictor reports it's ready, to capture the latency of the first prediction including all the warmup overhead of feeding frames and windows into the predictor
    sync(dev)
    t1_pred = time.perf_counter() # end timer for the first prediction, which includes the time taken to feed frames into the windowizer, process them, call update() on the predictor until it becomes ready, and then call predict_next() for the first time

    warmup_frames = frames_used
    first_prediction_total_ms = (t1_pred - t0_total) * 1000.0 # time for the first prediction, including warmup

    # subsequent latency: measure only end-to-end total per emitted window.
    per_pred_total_ms = np.empty((subsequent_windows,), dtype=np.float64)

    produced = 0
    while produced < subsequent_windows: # keep feeding frames into the windowizer and calling update()+predict() on the predictor with the emitted windows until we've collected enough subsequent predictions for steady-state analysis, tracking the time taken by each step
        frames_used += 1
        if frames_used >= max_frames:
            raise RuntimeError("Exceeded max_frames while collecting subsequent windows.")
        
        sync(dev)
        t0_total_step = time.perf_counter() # start timer for this step
        windows = windowizer.add_frame(frame_vec)
        if not windows: # if the windowizer did not emit any windows after processing this frame,
            continue

        obs_t = _window_to_obs_tensor(windows[-1], scale_idx, scaler_mean, scaler_std, dev)
        predictor.update(obs_t)
        _ = predictor.predict_next()
        sync(dev)
        t1_total_step = time.perf_counter() # end time for the whole step
        per_pred_total_ms[produced] = (t1_total_step - t0_total_step) * 1000.0 # time taken for the whole step of this new window
        produced += 1

    return {
        "warmup_updates": int(n_updates), # number of update() calls made during warmup until the predictor became ready
        "warmup_frames": int(warmup_frames), # number of frames fed into the windowizer during warmup until the predictor became ready
        "window_W": int(getattr(windowizer, "W", -1)), # window size used by the LiveWindowizer, if available
        "window_stride": int(getattr(windowizer, "stride", -1)),# window stride used by the LiveWindowizer, if available
        "first_prediction_total_ms": float(first_prediction_total_ms), # time taken for the first prediction after warmup
        "subsequent_total_ms": per_pred_total_ms, # total time taken for each subsequent window from the start of windowization to the completion of prediction
    }


@torch.no_grad()
def bench_loop_multi_e2e(predictors: List[OnlinePredictor], frame_vecs_np, device, scaler_mean, scaler_std, subsequent_windows=2000, prewarm_cuda=True, max_frames=5000000):
    """
    N independent OnlinePredictor objects stepped sequentially in Python, end-to-end:
      frame features -> windowize -> update+predict (when a window is emitted).
    """
    dev = torch.device(device)
    n_veh = len(predictors)

    for p in predictors:
        p.reset()

    windowizers = _make_windowizers(n_veh) # create a LiveWindowizer for each stream/vehicle, as they will maintain their own internal state as we feed frames into them; this allows us to benchmark the windowization step for each stream independently in the end-to-end loop
    frame_vecs = _validate_frame_vectors(frame_vecs_np) # validate and convert the input frame vectors to a list of 1D float64 numpy arrays, one for each stream/vehicle, as expected by the LiveWindowizers
    scale_idx, scaler_mean, scaler_std = _normalize_scaler_inputs(scaler_mean, scaler_std) # normalize and validate the scaler inputs, and determine which indices of the window features correspond to continuous features that need to be scaled, so we can apply the same scaling in the end-to-end loop when preparing the windows for the predictor

    if dev.type == "cuda" and prewarm_cuda:
        _ = torch.zeros(1, device=dev)

    per_ready = [False] * n_veh # track which predictors are ready
    warmup_updates_used = [0] * n_veh # track how many update() calls were used for each predictor to become ready, for warmup analysis
    warmup_frames = 0

    sync(dev)
    t0_total = time.perf_counter() # start total timer from the beginning of the first frame processing, to capture warmup time as well
    while not all(per_ready): # keep feeding frames into the windowizers and calling update() on the predictors with the emitted windows until all predictors report they're ready, tracking the time taken by each step and how many frames are needed for warmup
        warmup_frames += 1
        if warmup_frames >= max_frames:
            raise RuntimeError("Exceeded max_frames during warmup in loop_multi_e2e.")

        emitted = [None] * n_veh # to store the most recent emitted window for each stream/vehicle after processing this frame
        for i in range(n_veh): # loop through each stream/vehicle and feed the corresponding frame vector into its LiveWindowizer, which may emit a new window if enough frames have been processed
            windows = windowizers[i].add_frame(frame_vecs[i]) # feed the frame vector for this stream/vehicle into its LiveWindowizer
            if windows:
                emitted[i] = windows[-1] # take the most recent emitted window for this stream/vehicle, which we will prepare and feed into the predictor for this stream/vehicle if it's not None

        for i in range(n_veh): # loop through each stream/vehicle and call update() on the predictor with the new observation tensor if a new window was emitted and this predictor is not ready yet, as we only want to call update() during warmup until the predictor becomes ready; once a predictor becomes ready, we will stop calling update() on it and just let it be for the rest of the warmup phase, as we only want to measure the time taken by update() calls before the predictor becomes ready for warmup analysis
            if per_ready[i] or emitted[i] is None: # once a stream is ready, you stop updating it during the remaining warmup of other streams. If no new window was emitted for this stream, you also skip the update() call, as we only want to call update() when we have a new window to feed into the model during warmup
                continue
            obs_t = _window_to_obs_tensor(emitted[i], scale_idx, scaler_mean, scaler_std, dev)
            predictors[i].update(obs_t)
            warmup_updates_used[i] += 1
            if predictors[i].is_ready: # once this predictor reports it's ready, call predict_next() on it to mark it as ready so we can stop calling update() on it during the remaining warmup phase
                _ = predictors[i].predict_next()
                per_ready[i] = True

    sync(dev)
    t1_total = time.perf_counter() # end total timer after all predictors have become ready and we've called predict_next() on them, to capture warmup time as well
    first_prediction_total_ms = (t1_total - t0_total) * 1000.0 # total time for getting the first prediction from all predictors, including warmup

    per_event_total_ms = np.empty((subsequent_windows,), dtype=np.float64)

    produced = 0
    frames_used = warmup_frames # start counting frames used from the end of warmup, as we want to measure how many additional frames are needed to produce the subsequent windows after warmup, and to prevent infinite loops if we never emit enough windows for the subsequent analysis
    while produced < subsequent_windows:
        frames_used += 1
        if frames_used >= max_frames:
            raise RuntimeError("Exceeded max_frames while collecting subsequent windows in loop_multi_e2e.")

        sync(dev)
        t0_event = time.perf_counter() # start timer for this event, which includes the time taken by the windowizers to process this new frame and emit new windows, as well as the time taken by the predictors to perform inference for this new event
        emitted = [None] * n_veh # to store the most recent emitted window for each stream/vehicle after processing this frame

        for i in range(n_veh): # loop throght each stream/vehicle and feed the corresponding frame vector into its LiveWindowizer
            windows = windowizers[i].add_frame(frame_vecs[i])
            if windows:
                emitted[i] = windows[-1]

        if any(e is None for e in emitted): # if any stream/vehicle did not emit a new window after processing this frame, skip the update+predict step for this frame
            continue

        for i in range(n_veh): # loop through each stream/vehicle and call update() and predict_next() on the predictor with the new observation tensor 
            obs_t = _window_to_obs_tensor(emitted[i], scale_idx, scaler_mean, scaler_std, dev)
            predictors[i].update(obs_t)
            _ = predictors[i].predict_next()
        sync(dev)
        t1_event = time.perf_counter() # end time for this event, which includes both the windowization and inference steps for this new window
        per_event_total_ms[produced] = (t1_event - t0_event) * 1000.0 # total time for this event, including both the windowization and inference steps
        produced += 1

    return {
        "n_veh": int(n_veh),
        "warmup_updates": int(np.max(np.asarray(warmup_updates_used, dtype=np.int64))),
        "warmup_frames": int(warmup_frames),
        "window_W": int(getattr(windowizers[0], "W", -1)),
        "window_stride": int(getattr(windowizers[0], "stride", -1)),
        "first_prediction_total_ms": float(first_prediction_total_ms),
        "subsequent_total_ms": per_event_total_ms,
    }


@torch.no_grad()
def bench_batched_e2e(predictor: BatchedOnlinePredictor, frame_vecs_np, device, scaler_mean, scaler_std, subsequent_windows=2000, prewarm_cuda=True, max_frames=5_000_000):
    """
    Batched end-to-end benchmark:
      B frame feature vectors -> B windowizers -> stack emitted windows -> batched update+predict.
    """
    dev = torch.device(device)
    B = int(len(frame_vecs_np)) # batch size = number of vehicles/streams

    predictor.reset(B)

    windowizers = _make_windowizers(B) # create a LiveWindowizer for each stream/vehicle
    frame_vecs = _validate_frame_vectors(frame_vecs_np)
    scale_idx, scaler_mean, scaler_std = _normalize_scaler_inputs(scaler_mean, scaler_std)

    if dev.type == "cuda" and prewarm_cuda:
        _ = torch.zeros(1, device=dev)

    warmup_frames = 0
    n_updates = 0

    sync(dev)
    t0_total = time.perf_counter() # start total timer from the beginning of the first frame processing, to capture warmup time as well
    while True:
        ready = predictor.is_ready # check if all streams in the batch are ready; if so, break out of the loop to start the prediction timer
        if ready is not None and bool(ready.all()):
            break

        warmup_frames += 1
        if warmup_frames >= max_frames:
            raise RuntimeError("Exceeded max_frames during warmup in batched_e2e.")

        emitted = []
        for i in range(B): # loop through each stream/vehicle and feed the corresponding frame vector into its LiveWindowizer
            windows = windowizers[i].add_frame(frame_vecs[i])
            emitted.append(windows[-1] if windows else None) # take the most recent emitted window for this stream/vehicle, or None if no window was emitted

        if any(e is None for e in emitted): # if any stream/vehicle did not emit a new window after processing this frame, skip the update() call for this frame, as we only want to call update() when we have a new window to feed into the model during warmup
            continue

        obs_batch = _windows_to_obs_batch(emitted, scale_idx, scaler_mean, scaler_std, dev) # take the emitted windows, scale them, and convert them to a batch observation tensor on the correct device and dtype, to feed into the predictor
        predictor.update(obs_batch)
        n_updates += 1

    _ = predictor.predict_next(strict_ready=True)
    sync(dev)
    t1_pred = time.perf_counter() # end timer for the first prediction after warmup

    first_prediction_total_ms = (t1_pred - t0_total) * 1000.0 # total time for the first prediction, including warmup

    per_event_total_ms = np.empty((subsequent_windows,), dtype=np.float64)

    produced = 0
    frames_used = warmup_frames
    while produced < subsequent_windows:
        frames_used += 1
        if frames_used >= max_frames:
            raise RuntimeError("Exceeded max_frames while collecting subsequent windows in batched_e2e.")

        sync(dev)
        t0_event = time.perf_counter() # start timer for this event, which includes the time taken by the windowizers to process this new frame and emit new windows, as well as the time taken by the predictor to perform inference for this new event

        emitted = []
        for i in range(B): # loop throgh each stream and feed the corresponding frame vector into its LiveWindowizer, which may emit a new window if enough frames have been processed
            windows = windowizers[i].add_frame(frame_vecs[i])
            emitted.append(windows[-1] if windows else None)
        if any(e is None for e in emitted): # skip the update+predict step for this frame if any stream/vehicle did not emit a new window after processing this frame
            continue

        obs_batch = _windows_to_obs_batch(emitted, scale_idx, scaler_mean, scaler_std, dev)
        predictor.update(obs_batch)
        _ = predictor.predict_next(strict_ready=True)
        sync(dev)
        t1_event = time.perf_counter() # end time for this event, which includes both the windowization and inference steps for this new window
        per_event_total_ms[produced] = (t1_event - t0_event) * 1000.0 # total time for this event, including both the windowization and inference steps
        produced += 1

    return {
        "n_veh": int(B),
        "warmup_updates": int(n_updates),
        "warmup_frames": int(warmup_frames),
        "window_W": int(getattr(windowizers[0], "W", -1)),
        "window_stride": int(getattr(windowizers[0], "stride", -1)),
        "first_prediction_total_ms": float(first_prediction_total_ms),
        "subsequent_total_ms": per_event_total_ms,
    }
