import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .wandb_plots import *


class WandbLogger:
    @staticmethod
    def _safe_close(fig):
        try:
            plt.close(fig)
        except Exception:
            pass

    # -----------------------------
    # Main logging entry point
    # -----------------------------
    @staticmethod
    def log_iteration(trainer, wandb_run, it, iter_start, total_train_loglik, total_val_loglik, improvement,
        criterion_for_stop, val_num_obs, train_num_obs, delta_pi, delta_A, state_weights_flat, total_responsibility_mass,
        state_weights_frac, val_obs_seqs, A_prev, A_new, switch_rates_train, run_lengths_train, runlen_median_per_traj,
        ent_all_train, ent_mean_per_traj, sem_feat_names=None, sem_means=None, sem_stds=None):
        """
        Log per-iteration metrics to Weights & Biases.

        Parameters
        trainer: HDVTrainer instance 
        wandb_run : wandb.sdk.wandb_run.Run | None
            WandB run object.
        it : int
            EM iteration index (0-based).
        iter_start : float
            Start time (perf_counter) of this EM iteration.
        total_train_loglik : float
            Total train log-likelihood for this iteration.
        total_val_loglik : float
            Total validation log-likelihood (np.nan if no validation).
        improvement : float
            Improvement in early-stop criterion (average log-likelihood per timestep) vs previous iteration.
        criterion_for_stop : float
            Early-stopping criterion value used in this EM iteration. Defined as the average log-likelihood per timestep, computed on the
            validation set if available, otherwise on the training set.
        val_num_obs : int | None
            Total number of timesteps in the validation dataset. 
        train_num_obs : int
            Total number of timesteps in the training dataset. 
        delta_pi, delta_A : float
            Transition parameter deltas.
        state_weights_flat : np.ndarray
            Responsibility mass per state.
        total_responsibility_mass : float
            Sum of responsibilities (diagnostic).
        state_weights_frac : np.ndarray
            Responsibility fractions per state.
        val_obs_seqs : list[np.ndarray] | None
            Validation set (used only to decide what to log).
        A_prev, A_new : np.ndarray | None
            Previous and updated A matrices for delta plotting.
        """
        if wandb_run is None:
            return

        import wandb

        iter_time = time.perf_counter() - iter_start

        # convert tensors to numpy for logging + plotting
        pi_np = trainer.pi_z.detach().cpu().numpy()
        A_np = trainer.A_zz.detach().cpu().numpy()

        # Emission stats 
        g_means, g_vars = trainer.emissions.gauss.to_arrays()
        cont_dim = int(trainer.emissions.cont_dim)
        means_2d = g_means.reshape(trainer.num_states, cont_dim) if cont_dim > 0 else np.zeros((trainer.num_states, 0))
        vars_2d  = g_vars.reshape(trainer.num_states, cont_dim) if cont_dim > 0 else np.zeros((trainer.num_states, 0))
        cov_traces = vars_2d.sum(axis=1) # "trace" analogue for diagonal covariance is sum of variances
        cov_logdets = np.sum(np.log(vars_2d + 1e-15), axis=1) 
        mean_norms = np.linalg.norm(means_2d, axis=1) 

        # π diagnostics
        pi_entropy = float(-np.sum(pi_np * np.log(pi_np + 1e-15))) # Shannon entropy
        pi_max = float(pi_np.max())
        pi_min = float(pi_np.min())

        metrics = {
            "em_iter": int(it + 1),
            "time/iter_seconds": float(iter_time),

            "train/loglik": float(total_train_loglik), # Sum of log-likelihood over all training trajectories for that EM iteration.
            "train/loglik_per_obs": float(total_train_loglik) / max(int(train_num_obs), 1),
            "train/delta_pi": float(delta_pi), # L1 change in initial state distribution
            "train/delta_A": float(delta_A), # Mean absolute difference between old and new transition matrices
            "train/log_delta_A": float(np.log10(float(delta_A) + 1e-15)),
            
            "pi/entropy": pi_entropy,
            "pi/max": pi_max,
            "pi/min": pi_min,

            "emissions/total_responsibility_mass": float(total_responsibility_mass), # Sum of γ over all trajectories, time steps, and states
            "emissions/cov_logdet_mean": float(np.mean(cov_logdets)),
            "emissions/cov_logdet_std": float(np.std(cov_logdets)),

            "early_stop/criterion": float(criterion_for_stop) if np.isfinite(criterion_for_stop) else np.nan,
            "early_stop/source": "val" if val_obs_seqs is not None else "train",
            "early_stop/improvement_per_obs": float(improvement) if np.isfinite(improvement) else np.nan,          
        }

        # Trajectory switch-rate stats
        sr = np.asarray(switch_rates_train, dtype=np.float64).ravel()
        sr = sr[np.isfinite(sr)]  # drop NaN/inf (e.g., T<=1)
        metrics["traj/num_traj_used_for_switch_rate"] = int(sr.size)
        if sr.size > 0:
            metrics["traj/switch_rate_mean"] = float(np.mean(sr))
            metrics["traj/switch_rate_median"] = float(np.median(sr))
            metrics["traj/switch_rate_p95"] = float(np.quantile(sr, 0.95))
        else:
            metrics["traj/switch_rate_mean"] = np.nan
            metrics["traj/switch_rate_median"] = np.nan
            metrics["traj/switch_rate_p95"] = np.nan

        # Run-length (state duration) summaries
        rl = np.asarray(run_lengths_train).ravel()
        rl = rl[np.isfinite(rl)]
        rl = rl[rl > 0]
        metrics["traj/num_segments_total"] = int(rl.size)
        if rl.size > 0:
            metrics["traj/runlen_median"] = float(np.median(rl))
            metrics["traj/runlen_p10"] = float(np.quantile(rl, 0.10))
            metrics["traj/runlen_p90"] = float(np.quantile(rl, 0.90))
            metrics["traj/runlen_mean"] = float(np.mean(rl))
        else:
            metrics["traj/runlen_median"] = np.nan
            metrics["traj/runlen_p10"] = np.nan
            metrics["traj/runlen_p90"] = np.nan
            metrics["traj/runlen_mean"] = np.nan

        if val_obs_seqs is not None:
            metrics["val/loglik"] = float(total_val_loglik) # Sum of log-likelihood on the validation set
            metrics["val/loglik_per_obs"] = float(total_val_loglik) / max(int(val_num_obs), 1)
        else:
            metrics["val/loglik"] = np.nan

        e = np.asarray(ent_all_train, dtype=np.float64).ravel()
        e = e[np.isfinite(e)]
        metrics["traj/entropy_num_timesteps"] = int(e.size)
        if e.size > 0:
            metrics["traj/entropy_mean"] = float(np.mean(e))
            metrics["traj/entropy_median"] = float(np.median(e))
            metrics["traj/entropy_p95"] = float(np.quantile(e, 0.95))
        else:
            metrics["traj/entropy_mean"] = np.nan
            metrics["traj/entropy_median"] = np.nan
            metrics["traj/entropy_p95"] = np.nan
        mt = np.asarray(runlen_median_per_traj).ravel()
        mt = mt[np.isfinite(mt)]
        metrics["traj/runlen_median_per_traj_mean"] = float(np.mean(mt)) if mt.size > 0 else np.nan
        metrics["traj/runlen_median_per_traj_p10"] = float(np.quantile(mt, 0.10)) if mt.size > 0 else np.nan

        et = np.asarray(ent_mean_per_traj, dtype=np.float64).ravel()
        et = et[np.isfinite(et)]
        metrics["traj/entropy_mean_per_traj_mean"] = float(np.mean(et)) if et.size > 0 else np.nan
        metrics["traj/entropy_mean_per_traj_p10"]  = float(np.quantile(et, 0.10)) if et.size > 0 else np.nan

        # Bernoulli/Lane scalar summaries 
        bern_mean_per_state = None
        if getattr(trainer.emissions, "bin_dim", 0) > 0 and getattr(trainer.emissions, "bern_p", None) is not None:
            bern_p = np.asarray(trainer.emissions.bern_p, dtype=np.float64) 
            bern_mean_overall = bern_p.mean()          # (N,B)
            bern_mean_per_state = bern_p.mean(axis=1)  # (N,)
            metrics["bern/mean_overall"] = float(bern_mean_overall)

        lane_p = getattr(trainer.emissions, "lane_p", None)
        if lane_p is not None:
            lp = np.asarray(lane_p, dtype=np.float64) # (N, K)
            lp = np.clip(lp, 1e-15, 1.0)
            lp = lp / lp.sum(axis=1, keepdims=True)
            lane_entropy = -np.sum(lp * np.log(lp), axis=1) # (N,)
            lp_max = lp.max(axis=1)
            lp_min = lp.min(axis=1)
            metrics["lane/entropy_mean"] = float(np.mean(lane_entropy))
            metrics["lane/entropy_std"]  = float(np.std(lane_entropy))   
            metrics["lane/p_max_mean"]   = float(np.mean(lp_max))     
            metrics["lane/p_min_mean"]   = float(np.mean(lp_min)) 

        # Figures 
        try:
            # π bar plot
            fig = plot_pi_bar(pi_np, trainer.num_states)
            metrics["pi/plot"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            # A_zz heatmap
            fig = plot_A_heatmap(A_np)
            metrics["A/heatmap"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            # A_zz diagonal (stay probabilities)
            fig = plot_A_diag(A_np)
            metrics["A/diag_plot"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            # ΔA heatmap (change in transition matrix)
            if A_prev is not None and A_new is not None:
                fig = plot_A_delta(A_prev, A_new)
                metrics["A/delta_heatmap"] = wandb.Image(fig)
                WandbLogger._safe_close(fig)

            # responsibility fraction per state 
            fig = plot_state_mass_bar(state_weights_frac, "State responsibility fraction", "fraction of total γ")
            metrics["emissions/state_responsibility_frac_plot"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            # responsibility mass per state 
            fig = plot_state_mass_bar(state_weights_flat, "State responsibility mass", "total γ mass")
            metrics["emissions/state_responsibility_mass_plot"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            # mean norms per state 
            fig = plot_state_line(mean_norms, "Emission mean norms ||μ_z||", "L2 norm")
            metrics["emissions/mean_norms_plot"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            # covariance trace per state 
            fig = plot_state_line(cov_traces, "Diagonal variance sum per state Σ_d var_zd", "sum of variances")
            metrics["emissions/var_sum_plot"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            # Bernoulli mean per state
            if getattr(trainer.emissions, "bin_dim", 0) > 0 and getattr(trainer.emissions, "bern_p", None) is not None:
                fig = plot_bernoulli_means_per_state(bern_mean_per_state, trainer.num_states)
                metrics["bern/mean_per_state_plot"] = wandb.Image(fig)
                WandbLogger._safe_close(fig)

            # Lane categorical heatmap
            if lane_p is not None:
                fig = plot_lane_heatmap(lane_p)
                metrics["lane/heatmap"] = wandb.Image(fig)
                WandbLogger._safe_close(fig)
            
            # Switch-rate distribution (per trajectory)
            fig = plot_switch_rate_distribution(
                switch_rates_train,
                title="Switch rate per trajectory (expected)",
                xlabel="1 - diag(xi_sum)/(T-1)",
            )
            metrics["traj/switch_rate_dist"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            # Run-length distribution (segment durations)
            fig = plot_run_length_distribution(
                run_lengths_train,
                title="Run-length distribution (argmax gamma)",
                xlabel="segment length (timesteps)",
            )
            metrics["traj/runlen_dist"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            fig = plot_entropy_distribution(ent_all_train)
            metrics["traj/entropy_dist"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            # Posterior-weighted key feature semantics (derived features)
            if sem_means is not None and sem_stds is not None and sem_feat_names is not None and len(sem_feat_names) > 0:
                figs = plot_key_feature_per_feature(sem_means, sem_stds, sem_feat_names, title_prefix="Posterior-weighted")
                for fname, fig in figs.items():
                    # safe key for W&B
                    key = fname.replace(" ", "_").replace("|", "").replace("=", "").replace("__", "_")
                    metrics[f"semantics/{key}"] = wandb.Image(fig)
                    WandbLogger._safe_close(fig)

        except Exception as e:
            if int(getattr(trainer, "verbose", 0)) >= 2:
                print(f"[WandbLogger] Plotting failed at iter {it+1}: {e}")

        wandb_run.log(metrics)

    