"""Latency benchmark entrypoint for online DBN inference."""

from pathlib import Path
import sys
import numpy as np
import torch

try:
    from .model_interface import HDVDbnModel
    from ..trainer import HDVTrainer
    from .online_predictor import OnlinePredictor, BatchedOnlinePredictor
    from .benchmark_latency_runners import (
        LiveWindowizer,
        bench_single_vehicle,
        bench_loop_multi,
        bench_batched,
        bench_single_vehicle_e2e,
        bench_loop_multi_e2e,
        bench_batched_e2e,
    )
    from .benchmark_latency_reporting import print_block, write_results_csv, make_csv_row
    from ..utils.latency_utils import (
        print_benchmark_env,
        force_trainer_device,
        audit_device_dtype,
        build_obs_vectors,
        build_frame_vectors,
    )
except ImportError:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))

    from hdv.hdv_dbn.prediction.model_interface import HDVDbnModel
    from hdv.hdv_dbn.trainer import HDVTrainer
    from hdv.hdv_dbn.prediction.online_predictor import OnlinePredictor, BatchedOnlinePredictor
    from hdv.hdv_dbn.prediction.benchmark_latency_runners import (
        LiveWindowizer,
        bench_single_vehicle,
        bench_loop_multi,
        bench_batched,
        bench_single_vehicle_e2e,
        bench_loop_multi_e2e,
        bench_batched_e2e,
    )
    from hdv.hdv_dbn.prediction.benchmark_latency_reporting import print_block, write_results_csv, make_csv_row
    from hdv.hdv_dbn.utils.latency_utils import (
        print_benchmark_env,
        force_trainer_device,
        audit_device_dtype,
        build_obs_vectors,
        build_frame_vectors,
    )


def main():
    print_benchmark_env() 
    torch.set_grad_enabled(False)

    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent.parent

    exp_dir = workspace_root / "hdv" / "models" / "main-model-sticky_S2_A4_hierarchical"
    checkpoint_path = exp_dir / "final.npz"

    warmup_steps = 5
    n_veh = 5
    subsequent_iters = 100
    csv_rows = []

    trainer = HDVTrainer.load(checkpoint_path)

    obs_vecs = build_obs_vectors(n_veh) # randomly initialized observation vectors; shape (n_veh, window_feat_dim)
    obs_vec = obs_vecs[0] # single observation vector for the single-vehicle benchmark; shape (window_feat_dim,)

    # mean and std for scaling the windowized features before feeding into the model; using dummy values since these are synthetic vectors for benchmarking, not real data
    scaler_mean = np.zeros_like(obs_vec, dtype=np.float64)
    scaler_std = np.ones_like(obs_vec, dtype=np.float64)

    # For the end-to-end benchmarks, we need synthetic *frame* feature vectors (shape (n_veh, frame_feat_dim)) 
    # that can be windowized and scaled by the LiveWindowizer, before being fed into the model. 
    frame_vecs = build_frame_vectors(n_veh) # randomly initialized frame feature vectors; shape (n_veh, frame_feat_dim)

    def run_one_device(dev: torch.device, label):
        force_trainer_device(trainer, dev, torch.float32) # ensure trainer and model are on the correct device and dtype
        model = HDVDbnModel(trainer, device=dev, dtype=torch.float32) # initialize model interface with the trainer's parameters, on the correct device and dtype
        audit_device_dtype(trainer, label) # sanity check to ensure trainer tensors are on the expected device and dtype

        #-------------------------------------------------------
        # model-only benchmarks (using pre-built obs_vecs, no windowization or scaling)
        #-------------------------------------------------------
        # case 1: single-vehicle benchmark (update+predict)
        p_single = OnlinePredictor(model, warmup_steps=warmup_steps, device=dev, dtype=torch.float32)
        res_single = bench_single_vehicle(p_single, obs_vec, device=dev, subsequent_iters=subsequent_iters)

        # case 2: loop_multi benchmark (N independent OnlinePredictor objects stepped sequentially)
        ps = [OnlinePredictor(model, warmup_steps=warmup_steps, device=dev, dtype=torch.float32) for _ in range(n_veh)] # n independent predictors for the loop_multi benchmark
        res_loop = bench_loop_multi(ps, obs_vecs, device=dev, subsequent_iters=subsequent_iters)

        # case 3: batched benchmark (single BatchedOnlinePredictor object stepping all vehicles in parallel)
        p_batched = BatchedOnlinePredictor(model, warmup_steps=warmup_steps, device=dev, dtype=torch.float32) # single predictor for all vehicles in the batched benchmark
        res_batched = bench_batched(p_batched, obs_vecs, device=dev, subsequent_iters=subsequent_iters)

        print_block(f"[{label}] single (update+predict)", res_single, is_e2e=False)
        print_block(f"[{label}] loop_multi (N={n_veh}, update+predict)", res_loop, is_e2e=False)
        print_block(f"[{label}] batched (B={n_veh}, update+predict)", res_batched, is_e2e=False)

        csv_rows.append(make_csv_row(label, "single", res_single, 1))
        csv_rows.append(make_csv_row(label, "loop_multi", res_loop, n_veh))
        csv_rows.append(make_csv_row(label, "batched", res_batched, n_veh))

        #-------------------------------------------------------
        # end-to-end benchmarks (using frame_vecs that require windowization and scaling before being fed into the model)
        #-------------------------------------------------------
        # Case 1: single-vehicle end-to-end benchmark (frame->windowize->scale->predict)
        p_single_e2e = OnlinePredictor(model, warmup_steps=warmup_steps, device=dev, dtype=torch.float32) # predictor for the single-vehicle end-to-end benchmark
        res_single_e2e = bench_single_vehicle_e2e(p_single_e2e, frame_vecs[0], device=dev, scaler_mean=scaler_mean,
            scaler_std=scaler_std, subsequent_windows=subsequent_iters) # using the first frame_vec for the single-vehicle end-to-end benchmark

        # Case 2: loop_multi end-to-end benchmark (N independent OnlinePredictor objects stepped sequentially, each processing its own stream of frames through the windowizer and model)
        ps_e2e = [OnlinePredictor(model, warmup_steps=warmup_steps, device=dev, dtype=torch.float32) for _ in range(n_veh)]
        res_loop_e2e = bench_loop_multi_e2e(ps_e2e, frame_vecs, device=dev, scaler_mean=scaler_mean, scaler_std=scaler_std,
            subsequent_windows=subsequent_iters)

        # Case 3: batched end-to-end benchmark (single BatchedOnlinePredictor object stepping all vehicles in parallel, processing all streams of frames through the windowizer and model in parallel)
        p_batched_e2e = BatchedOnlinePredictor(model, warmup_steps=warmup_steps, device=dev, dtype=torch.float32)
        res_batched_e2e = bench_batched_e2e(p_batched_e2e, frame_vecs, device=dev, scaler_mean=scaler_mean, scaler_std=scaler_std,
            subsequent_windows=subsequent_iters)

        print_block(f"[{label}] single_e2e (frame->window->predict)", res_single_e2e, is_e2e=True)
        print_block(f"[{label}] loop_multi_e2e (N={n_veh}, frame->window->predict)", res_loop_e2e, is_e2e=True)
        print_block(f"[{label}] batched_e2e (B={n_veh}, frame->window->predict)", res_batched_e2e, is_e2e=True)

        csv_rows.append(make_csv_row(label, "single_e2e", res_single_e2e, 1))
        csv_rows.append(make_csv_row(label, "loop_multi_e2e", res_loop_e2e, n_veh))
        csv_rows.append(make_csv_row(label, "batched_e2e", res_batched_e2e, n_veh))

    run_one_device(torch.device("cpu"), "CPU")

    if torch.cuda.is_available():
        run_one_device(torch.device("cuda"), "GPU")
    else:
        print("[benchmark] CUDA not available; skipping GPU.")

    output_dir = exp_dir / "latency benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_latency_results.csv"
    write_results_csv(csv_path, csv_rows)


if __name__ == "__main__":
    main()
