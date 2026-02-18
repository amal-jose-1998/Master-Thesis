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

try:
    from . import TrajectoryData, HDVDbnModel, ValidationStep, ValidationConfig
    from .data_loader import load_test_data_for_prediction
    from .apply_gt_labels import compute_gt_latents, z_to_sa

except ImportError:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))

    from hdv.hdv_dbn.prediction import (
        TrajectoryData, HDVDbnModel, ValidationStep, ValidationConfig
    )
    from hdv.hdv_dbn.prediction.data_loader import load_test_data_for_prediction
    from hdv.hdv_dbn.prediction.apply_gt_labels import compute_gt_latents, z_to_sa


def main():
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent.parent.parent
    
    data_root = workspace_root / "hdv" / "data" / "highd"
    exp_dir = workspace_root / "hdv" / "models" / "main-model-sticky_S2_A4_hierarchical"
    checkpoint_path = exp_dir / "final.npz"
    out_dir = exp_dir / "prediction"
    
    print(f"[run_validation] Workspace: {workspace_root}")
    print(f"[run_validation] Data root: {data_root}")
    print(f"[run_validation] Exp dir: {exp_dir}")
    print(f"[run_validation] Checkpoint: {checkpoint_path}\n")
    
    # Step 1: Load test data and trainer
    print("[run_validation] Loading test sequences from split.json...")
    trainer, test_data = load_test_data_for_prediction(
        exp_dir=exp_dir,
        data_root=data_root,
        checkpoint_name="final.npz"
    )
    print(f"[run_validation] Loaded {len(test_data.raw_seqs)} test sequences")
    print(f"[run_validation] Feature columns: {test_data.feature_cols}")
    print(f"[run_validation] Scaler mode: {test_data.scaler_mode}\n")
    
    # Step 2: Convert to TrajectoryData objects
    print("[run_validation] Converting sequences to TrajectoryData...")
    trajectories = []
    
    A = trainer.A  # Number of actions
    
    for i, (raw_seq, scaled_obs) in enumerate(zip(test_data.raw_seqs, test_data.scaled_obs)): # Iterate over raw sequences and corresponding scaled observations
        traj_id = test_data.raw_seqs[i].__dict__.get("trajectory_id", None) or f"seq_{i:03d}" # Use trajectory_id if available, else fallback to seq_{i:03d}
        
        # Apply rule-based ground truth labels (use raw observations for physical-unit thresholds)
        raw_obs = test_data.raw_obs[i]
        z_labels = compute_gt_latents(
            obs_seq=raw_obs,
            feature_cols=test_data.feature_cols,
            thr=None,  # Use default RuleThresholds
            A=A
        )
        
        # Convert z to (s, a) pairs for metrics and confusion matrix indexing.
        latents_gt = np.array(
            [z_to_sa(z, A) for z in z_labels],
            dtype=np.int32
        )
        
        # For each test trajectory, creates a clean evaluation object
        traj = TrajectoryData(
            obs=scaled_obs,  # scaled observations for model inference
            latents_gt=latents_gt,   # ground‑truth (style, action) pairs
            trajectory_id=traj_id
        )
        trajectories.append(traj)
        
        if (i + 1) % 10 == 0:
            print(f"  → Processed {i + 1}/{len(test_data.raw_seqs)} sequences")
    
    print(f"[run_validation] Created {len(trajectories)} TrajectoryData objects\n")
    
    # Step 3: Run validation (prediction + metric computation).
    print("[run_validation] Running validation...")
    config = ValidationConfig(
        warmup_steps=10,
        horizon=10,
        fps=25.0,
        stride_frames=10,
        skip_partial_horizons=True
    )
    
    model = HDVDbnModel(trainer) # Wrap the trainer as a generative model
    print(f"[run_validation] Model: S={model.num_styles}, A={model.num_actions}\n")
    
    validator = ValidationStep(model, config) # stores handles and copies S and A from the model.
    metrics = validator.evaluate(trajectories) # actual filtering+prediction loop
    
    # Step 4: Print and save results
    summary = metrics.summary(S=model.num_styles, A=model.num_actions)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Exact Accuracy (1-step):  {summary['exact_accuracy']:.4f}") # proportion of times the predicted latent at time t+1 exactly equals the ground‑truth latent at time t+1.
    print(f"Hit@H{config.horizon}:          {summary['hit_rate']:.4f}") # proportion of predictions where the predicted latent appears at least once within the next H frames
    print(f"Mean TTE (hits only):     {summary['mean_tte_sec']:.2f} sec")
    print(f"Median TTE (hits only):   {summary['median_tte_sec']:.2f} sec")
    print(f"Total predictions:        {summary['num_total']}") # total number of predictions made.
    print(f"Hits:                     {summary['num_hits']}")  # number of predictions that were hits.
    print(f"{'='*70}\n")
    
    # Save results
    import json
    summary_path = out_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[run_validation] Saved summary to {summary_path}")
    
    cm = metrics.exact.confusion_matrix(model.num_styles, model.num_actions)
    cm_path = out_dir / "confusion_matrix.npy"
    np.save(cm_path, cm)
    print(f"[run_validation] Saved confusion matrix to {cm_path}")
    
    print("\n Validation complete")


if __name__ == "__main__":
    main()
