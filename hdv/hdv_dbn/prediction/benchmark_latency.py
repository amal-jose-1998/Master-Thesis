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
    from .benchmark_latency_reporting import print_block, write_results_csv
    from ..utils.latency_utils import (
        print_benchmark_env,
        force_trainer_device,
        audit_device_dtype,
        _build_obs_vectors,
        _build_frame_vectors,
        _get_window_scaler,
        _summary_stats,
    )
except ImportError:
    project_root = Path(__file__).resolve().parents[4]
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
    from hdv.hdv_dbn.prediction.benchmark_latency_reporting import print_block, write_results_csv
    from hdv.hdv_dbn.utils.latency_utils import (
        print_benchmark_env,
        force_trainer_device,
        audit_device_dtype,
        _build_obs_vectors,
        _build_frame_vectors,
        _get_window_scaler,
        _summary_stats,
    )


def main():
    print_benchmark_env() 
    torch.set_grad_enabled(False)

    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent.parent.parent

    data_root = workspace_root / "hdv" / "data" / "highd"
    exp_dir = workspace_root / "hdv" / "models" / "main-model-sticky_S2_A4_hierarchical"
    checkpoint_path = exp_dir / "final.npz"

    warmup_steps = 5
    n_veh = 5
    subsequent_iters = 1000
    subsequent_windows = 100
    csv_rows = []

    obs_vecs, test_obj = _build_obs_vectors(exp_dir=exp_dir, data_root=data_root, checkpoint_path=checkpoint_path, n_veh=n_veh)
    obs_vec = obs_vecs[0]

    scaler_mean, scaler_std = _get_window_scaler(test_obj)
    frame_vecs = _build_frame_vectors(n_veh, kind="zeros")

    def add_csv_row(label: str, scenario: str, res: dict, n_streams: int):
        stats_total = _summary_stats(res.get("subsequent_total_ms", np.asarray([], dtype=np.float64)))
        stats_win = _summary_stats(res.get("subsequent_windowize_ms", np.asarray([], dtype=np.float64)))
        stats_prep = _summary_stats(res.get("subsequent_prep_ms", np.asarray([], dtype=np.float64)))
        stats_inf = _summary_stats(res.get("subsequent_infer_ms", np.asarray([], dtype=np.float64)))
        stats_pf = _summary_stats(res.get("per_frame_windowize_ms", np.asarray([], dtype=np.float64)))

        csv_rows.append(
            {
                "device": label.lower(),
                "scenario": scenario,
                "n_veh": int(n_streams),
                "warmup_updates": int(res.get("warmup_updates", -1)),
                "warmup_frames": int(res.get("warmup_frames", -1)),
                "window_W": int(res.get("window_W", -1)),
                "window_stride": int(res.get("window_stride", -1)),
                "first_prediction_total_ms": float(res.get("first_prediction_total_ms", np.nan)),
                "first_predict_call_ms": float(res.get("first_predict_call_ms", np.nan)),
                "first_prediction_per_vehicle_ms": float(res.get("first_prediction_total_ms", np.nan)) / float(n_streams),
                "subsequent_mean_ms": stats_total["mean"],
                "subsequent_p50_ms": stats_total["p50"],
                "subsequent_p90_ms": stats_total["p90"],
                "subsequent_p99_ms": stats_total["p99"],
                "subsequent_windowize_mean_ms": stats_win["mean"],
                "subsequent_windowize_p50_ms": stats_win["p50"],
                "subsequent_windowize_p90_ms": stats_win["p90"],
                "subsequent_windowize_p99_ms": stats_win["p99"],
                "subsequent_prep_mean_ms": stats_prep["mean"],
                "subsequent_prep_p50_ms": stats_prep["p50"],
                "subsequent_prep_p90_ms": stats_prep["p90"],
                "subsequent_prep_p99_ms": stats_prep["p99"],
                "subsequent_infer_mean_ms": stats_inf["mean"],
                "subsequent_infer_p50_ms": stats_inf["p50"],
                "subsequent_infer_p90_ms": stats_inf["p90"],
                "subsequent_infer_p99_ms": stats_inf["p99"],
                "per_frame_windowize_mean_ms": stats_pf["mean"],
                "per_frame_windowize_p50_ms": stats_pf["p50"],
                "per_frame_windowize_p90_ms": stats_pf["p90"],
                "per_frame_windowize_p99_ms": stats_pf["p99"],
            }
        )

    def run_one_device(dev: torch.device, label: str):
        trainer = HDVTrainer.load(checkpoint_path)
        force_trainer_device(trainer, dev, torch.float32)
        model = HDVDbnModel(trainer, device=dev, dtype=torch.float32)
        audit_device_dtype(trainer, label)

        p_single = OnlinePredictor(model, warmup_steps=warmup_steps, device=dev, dtype=torch.float32)
        res_single = bench_single_vehicle(p_single, obs_vec, device=dev, subsequent_iters=subsequent_iters)

        ps = [OnlinePredictor(model, warmup_steps=warmup_steps, device=dev, dtype=torch.float32) for _ in range(n_veh)]
        res_loop = bench_loop_multi(ps, obs_vecs, device=dev, subsequent_iters=subsequent_iters)

        p_batched = BatchedOnlinePredictor(model, warmup_steps=warmup_steps, device=dev, dtype=torch.float32)
        res_batched = bench_batched(p_batched, obs_vecs, device=dev, subsequent_iters=subsequent_iters)

        print_block(f"[{label}] single (update+predict)", res_single, is_e2e=False)
        print_block(f"[{label}] loop_multi (N={n_veh}, update+predict)", res_loop, is_e2e=False)
        print_block(f"[{label}] batched (B={n_veh}, update+predict)", res_batched, is_e2e=False)

        add_csv_row(label, "single", res_single, 1)
        add_csv_row(label, "loop_multi", res_loop, n_veh)
        add_csv_row(label, "batched", res_batched, n_veh)

        if LiveWindowizer is None:
            print(f"[{label}] LiveWindowizer not importable; skipping end-to-end benchmarks.")
            return

        p_single_e2e = OnlinePredictor(model, warmup_steps=warmup_steps, device=dev, dtype=torch.float32)
        res_single_e2e = bench_single_vehicle_e2e(
            p_single_e2e,
            frame_vecs[0],
            device=dev,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            subsequent_windows=subsequent_windows,
        )

        ps_e2e = [OnlinePredictor(model, warmup_steps=warmup_steps, device=dev, dtype=torch.float32) for _ in range(n_veh)]
        res_loop_e2e = bench_loop_multi_e2e(
            ps_e2e,
            frame_vecs,
            device=dev,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            subsequent_windows=subsequent_windows,
        )

        p_batched_e2e = BatchedOnlinePredictor(model, warmup_steps=warmup_steps, device=dev, dtype=torch.float32)
        res_batched_e2e = bench_batched_e2e(
            p_batched_e2e,
            frame_vecs,
            device=dev,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            subsequent_windows=subsequent_windows,
        )

        print_block(f"[{label}] single_e2e (frame->window->predict)", res_single_e2e, is_e2e=True)
        print_block(f"[{label}] loop_multi_e2e (N={n_veh}, frame->window->predict)", res_loop_e2e, is_e2e=True)
        print_block(f"[{label}] batched_e2e (B={n_veh}, frame->window->predict)", res_batched_e2e, is_e2e=True)

        add_csv_row(label, "single_e2e", res_single_e2e, 1)
        add_csv_row(label, "loop_multi_e2e", res_loop_e2e, n_veh)
        add_csv_row(label, "batched_e2e", res_batched_e2e, n_veh)

    run_one_device(torch.device("cpu"), "CPU")

    if torch.cuda.is_available():
        run_one_device(torch.device("cuda"), "GPU")
    else:
        print("[benchmark] CUDA not available; skipping GPU.")

    csv_path = Path(__file__).resolve().parent / "benchmark_latency_results.csv"
    write_results_csv(csv_path, csv_rows)


if __name__ == "__main__":
    main()
