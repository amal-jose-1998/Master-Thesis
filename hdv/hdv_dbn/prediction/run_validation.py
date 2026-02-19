"""
End-to-end validation script.

    1. Finds the model folder (exp_dir) and the checkpoint file (final.npz).
    2. Loads the test dataset (only the test vehicles from split.json).
    3. For each test trajectory, creates a clean evaluation object (TrajectoryData) that contains:
        a. scaled observations (what the model expects as input)
        b. ground-truth labels (style/action) computed from rules
    4. Runs validation (prediction + metric computation).
    5. Prints results and saves them.
"""

from pathlib import Path
import numpy as np
import sys
import json

try:
    from . import TrajectoryData, HDVDbnModel, ValidationStep, ValidationConfig
    from .data_loader import load_test_data_for_prediction
    from .apply_gt_labels import compute_gt_latents, z_to_sa
    from .visualize_metrics import visualize_all_metrics
    from .semantic_label_utils import load_semantic_labels_from_yaml

except ImportError:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))

    from hdv.hdv_dbn.prediction import (
        TrajectoryData, HDVDbnModel, ValidationStep, ValidationConfig
    )
    from hdv.hdv_dbn.prediction.data_loader import load_test_data_for_prediction
    from hdv.hdv_dbn.prediction.apply_gt_labels import compute_gt_latents, z_to_sa
    from hdv.hdv_dbn.prediction.visualize_metrics import visualize_all_metrics
    from hdv.hdv_dbn.prediction.semantic_label_utils import load_semantic_labels_from_yaml


def main():
    script_dir = Path(__file__).parent # Directory containing run_validation.py.
    workspace_root = script_dir.parent.parent.parent # “workspace root” that contains hdv/.
    
    # Construct paths
    data_root = workspace_root / "hdv" / "data" / "highd"
    exp_dir = workspace_root / "hdv" / "models" / "main-model-sticky_S2_A4_hierarchical"
    checkpoint_path = exp_dir / "final.npz"
    out_dir = exp_dir / "prediction"
    
    print(f"[run_validation] Workspace: {workspace_root}")
    print(f"[run_validation] Data root: {data_root}")
    print(f"[run_validation] Exp dir: {exp_dir}")
    print(f"[run_validation] Checkpoint: {checkpoint_path}\n")
    
    # Step 1: Load test data and trainer
    print("\n[run_validation] Loading test sequences from split.json...")
    trainer, test_data = load_test_data_for_prediction(
        exp_dir=exp_dir,
        data_root=data_root,
        checkpoint_name="final.npz"
    )
    print(f"\n[run_validation] Loaded {len(test_data.trajectory_ids)} test sequences")
    print(f"[run_validation] Feature columns: {test_data.feature_cols}")
    print(f"[run_validation] Scaler mode: {test_data.scaler_mode}\n")
    
    # Step 2: Convert each test sequence into a TrajectoryData object
    print("\n[run_validation] Converting sequences to TrajectoryData...")
    trajectories = []
    
    A = trainer.A  # Number of actions
    
    for i, (traj_id, scaled_obs, raw_obs) in enumerate(zip(test_data.trajectory_ids, test_data.scaled_obs, test_data.raw_obs)): # iterate over each test vehicle trajectory
        # Apply rule-based ground truth labels (use raw observations for physical-unit thresholds)
        z_labels = compute_gt_latents( # shape: (T,); array of joint labels z using raw features + thresholds.
            obs_seq=raw_obs,
            feature_cols=test_data.feature_cols,
            thr=None,  # Use default RuleThresholds
            A=A,
            debug=False, 
            fill_unknown="none" # UNKNOWN windows stay -1.
        )
        
        # Converts each joint label z to (s, a) pairs for metrics 
        latents_gt = np.array(
            [z_to_sa(z, A) for z in z_labels],
            dtype=np.int32
        ) # if z is unknown (-1), z_to_sa returns (-1,-1)
        
        # For each test trajectory, creates a clean evaluation object
        traj = TrajectoryData(
            obs=scaled_obs,  # scaled observations for model inference
            latents_gt=latents_gt,   # ground‑truth (style, action) pairs
            trajectory_id=traj_id
        )
        trajectories.append(traj)
        
        if (i + 1) % 10 == 0:
            print(f"  -> Processed {i + 1}/{len(test_data.trajectory_ids)} sequences")
    
    print(f"[run_validation] Created {len(trajectories)} TrajectoryData objects\n")
    
    # Step 3: Run validation (prediction + metric computation).
    print("[run_validation] Running validation...")
    config = ValidationConfig(
        warmup_steps=5,
        horizon=10,
        fps=25.0,
        stride_frames=10,
        skip_partial_horizons=True # don’t score near the end if we can’t see H future steps
    )
    
    model = HDVDbnModel(trainer) # Wrap the trainer as a generative model
    print(f"[run_validation] Model: S={model.num_styles}, A={model.num_actions}\n")
    
    validator = ValidationStep(model, config) # Creates an evaluator instance holding the model + config.
    metrics, all_predictions = validator.evaluate(trajectories) # actual filtering + prediction loop, returns metrics AND predictions
    print(f"[run_validation] Collected {len(all_predictions)} total predictions from single evaluation pass\n")
    
    # Step 4: Print summary results
    summary = metrics.summary(S=model.num_styles, A=model.num_actions)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Hit@H{config.horizon}:    {summary['hit_rate']:.4f}") # proportion of predictions where the predicted latent appears at least once within the next H frames
    print(f"Mean TTE (hits only):     {summary['mean_tte_sec']:.2f} sec")
    print(f"Median TTE (hits only):   {summary['median_tte_sec']:.2f} sec")
    print(f"Percentile 25 TTE:        {summary['p25_tte_sec']:.2f} sec")
    print(f"Percentile 75 TTE:        {summary['p75_tte_sec']:.2f} sec")
    print(f"Total predictions:        {summary['num_total']}")
    print(f"Hits:                     {summary['num_hits']}")
    print(f"Hit Rate:                 {summary['num_hits'] / summary['num_total']:.1%}")
    print(f"{'='*70}\n")
    
    # Step 5: Save summary and generate visualizations
    summary_path = out_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[run_validation] Saved summary to {summary_path}\n")
    
    print("[run_validation] Generating visualization plots...")
    # Load semantic labels from semantic_map.yaml for this experiment
    semantic_map_path = exp_dir / "semantic_map.yaml"
    S = model.num_styles
    A = model.num_actions
    semantic_labels = load_semantic_labels_from_yaml(semantic_map_path, S, A)

    # Pass metrics to visualization for correct confusion matrix
    figs = visualize_all_metrics(
        predictions=all_predictions,
        output_dir=out_dir,
        S=S,
        A=A,
        labels=semantic_labels,
        fps=config.fps,
        stride_frames=config.stride_frames,
        metrics=metrics
    )
    print(f"[run_validation] Generated {len(figs)} visualization plots\n")
    
    print(f"{'='*70}")
    print("VALIDATION COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
