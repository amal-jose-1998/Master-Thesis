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

    @staticmethod
    def _joint_from_structured(pi_s0, pi_a0_given_s0, A_s, A_a):
        """
        Build joint-form diagnostics from structured parameters.

        z = (s, a) with flattening z = s*A + a.

        pi_z[z] = pi_s0[s] * pi_a0_given_s0[s,a]

        A_zz[z_prev, z_cur] = P(s_cur,a_cur | s_prev,a_prev)
                           = A_s[s_prev,s_cur] * A_a[s_cur,a_prev,a_cur]
        """
        pi_s0 = np.asarray(pi_s0, dtype=np.float64).ravel()
        pi_a0_given_s0 = np.asarray(pi_a0_given_s0, dtype=np.float64)
        A_s = np.asarray(A_s, dtype=np.float64)
        A_a = np.asarray(A_a, dtype=np.float64)

        S = int(pi_s0.size)
        A = int(pi_a0_given_s0.shape[1])

        pi_z = (pi_s0[:, None] * pi_a0_given_s0).reshape(S * A)
        
        # (s_prev, s_cur, a_prev, a_cur)
        A_joint = A_s[:, :, None, None] * A_a[None, :, :, :]
        A_zz = A_joint.reshape(S * A, S * A)
        return pi_z, A_zz

    # -----------------------------
    # Main logging entry point
    # -----------------------------
    @staticmethod
    def log_iteration(trainer, wandb_run, it, iter_start, total_train_loglik, total_val_loglik, improvement,
        criterion_for_stop, val_num_obs, train_num_obs, delta_pi, delta_A, state_weights_flat, total_responsibility_mass,
        state_weights_frac, val_obs_seqs, A_prev, A_new, run_lengths_train, runlen_median_per_traj,
        ent_all_train, ent_mean_per_traj, sem_feat_names=None, sem_means=None, sem_stds=None, sem_means_raw=None, sem_stds_raw=None):
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
        pi_s0 = trainer.pi_s0.detach().cpu().numpy()
        pi_a0_given_s0 = trainer.pi_a0_given_s0.detach().cpu().numpy()
        A_s = trainer.A_s.detach().cpu().numpy()
        A_a = trainer.A_a.detach().cpu().numpy()

        print("[debug] pi_s0", pi_s0.shape,
                "pi_a0_given_s0", pi_a0_given_s0.shape,
                "A_s", A_s.shape,
                "A_a", A_a.shape)

        # Joint-form diagnostics 
        pi_z_np, A_zz_np = WandbLogger._joint_from_structured(pi_s0, pi_a0_given_s0, A_s, A_a)

        # Emission stats 
        g_means, g_vars = trainer.emissions.gauss.to_arrays()
        cont_dim = int(trainer.emissions.cont_dim)
        means_2d = g_means.reshape(trainer.num_states, cont_dim) if cont_dim > 0 else np.zeros((trainer.num_states, 0))
        vars_2d  = g_vars.reshape(trainer.num_states, cont_dim) if cont_dim > 0 else np.zeros((trainer.num_states, 0))
        cov_traces = vars_2d.sum(axis=1) # "trace" analogue for diagonal covariance is sum of variances
        mean_norms = np.linalg.norm(means_2d, axis=1) 
        
        # π diagnostics
        pi_s_entropy = float(-np.sum(pi_s0 * np.log(pi_s0 + 1e-15)))
        pi_z_entropy = float(-np.sum(pi_z_np * np.log(pi_z_np + 1e-15))) # Shannon entropy
        pi_z_max = float(pi_z_np.max())
        pi_z_min = float(pi_z_np.min())

        metrics = {
            "em_iter": int(it + 1),
            "time/iter_seconds": float(iter_time),

            "train/loglik": float(total_train_loglik), # Sum of log-likelihood over all training trajectories for that EM iteration.
            "train/loglik_per_obs": float(total_train_loglik) / max(int(train_num_obs), 1),
            #"train/delta_pi": float(delta_pi), # L1 change in initial state distribution
            "train/delta_A": float(delta_A), # Mean absolute difference between old and new transition matrices
            #"train/log_delta_A": float(np.log10(float(delta_A) + 1e-15)),
            
            "pi_s0/entropy": pi_s_entropy,

            "early_stop/criterion": float(criterion_for_stop) if np.isfinite(criterion_for_stop) else np.nan,
            "early_stop/source": "val" if val_obs_seqs is not None else "train",
            "early_stop/improvement_per_obs": float(improvement) if np.isfinite(improvement) else np.nan,          
        }

        if val_obs_seqs is not None:
            metrics["val/loglik"] = float(total_val_loglik) # Sum of log-likelihood on the validation set
            metrics["val/loglik_per_obs"] = float(total_val_loglik) / max(int(val_num_obs), 1)
        else:
            metrics["val/loglik"] = np.nan
        
        # Bernoulli features (window-level; e.g., lc_left_present / lc_right_present)
        # - `bern_p` is (K, B) with B binary dims.
        bern_p = None
        bern_names = getattr(getattr(trainer, "emissions", None), "bernoulli_names", None)
        if getattr(trainer.emissions, "bin_dim", 0) > 0 and getattr(trainer.emissions, "bern_p", None) is not None:
            bern_p = trainer.emissions.bern_p
            if hasattr(bern_p, "detach"):
                bern_p = bern_p.detach().cpu().numpy()
            bern_p = np.asarray(bern_p, dtype=np.float64)


        # Figures 
        try:
            # Joint diagnostics (derived)
            fig = plot_state_mass_bar(pi_z_np, "Derived π_z (diagnostic)", "probability")
            metrics["pi_z/plot"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            fig = plot_A_heatmap(A_zz_np, title="Joint transition A_zz (diagnostic)")
            metrics["A_zz/heatmap"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            fig = plot_A_diag(A_zz_np, title="diag(A_zz): stay probability per joint state")
            metrics["A_zz/diag_plot"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            # Structured π plots
            fig = plot_pi_s0_bar(pi_s0)
            metrics["pi_s0/plot"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            fig = plot_pi_a0_given_s0_heatmap(pi_a0_given_s0)
            metrics["pi_a0_given_s0/heatmap"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            # Structured transition plots
            fig = plot_A_s_heatmap(A_s)
            metrics["A_s/heatmap"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            fig = plot_A_s_diag(A_s)
            metrics["A_s/diag_plot"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            figs = plot_A_a_heatmaps(A_a)
            for key, fig in figs.items():
                metrics[f"A_a/heatmap_{key}"] = wandb.Image(fig)
                WandbLogger._safe_close(fig)

            # Joint state interpretability grid (S × A)
            # Show which (style,action) cells carry probability/mass.
            fig = plot_joint_sa_grid(
                state_weights_frac,
                S=int(trainer.S),
                A=int(trainer.A),
                title="State responsibility fraction on (Style × Action) grid",
            )
            metrics["emissions/state_frac_sa_grid"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            fig = plot_joint_index_grid(
                S=int(trainer.S),
                A=int(trainer.A),
                title="Joint state index z = s*A + a (Style × Action)",
            )
            metrics["debug/joint_index_grid"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            # Structured deltas 
            if isinstance(A_prev, dict) and isinstance(A_new, dict) and ("A_s" in A_prev) and ("A_s" in A_new):
                fig = plot_A_s_delta(A_prev["A_s"], A_new["A_s"])
                metrics["A_s/delta_heatmap"] = wandb.Image(fig)
                WandbLogger._safe_close(fig)

                if ("A_a" in A_prev) and ("A_a" in A_new):
                    figs = plot_A_a_delta_per_style(A_prev["A_a"], A_new["A_a"])
                    for key, fig in figs.items():
                        metrics[f"A_a/delta_{key}"] = wandb.Image(fig)
                        WandbLogger._safe_close(fig)

            # responsibility fraction per joint state
            fig = plot_state_mass_bar(state_weights_frac, "State responsibility fraction", "fraction of total γ")
            metrics["emissions/state_responsibility_frac_plot"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            # responsibility mass per joint state
            fig = plot_state_mass_bar(state_weights_flat, "State responsibility mass", "total γ mass")
            metrics["emissions/state_responsibility_mass_plot"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            # mean norms per joint state
            fig = plot_state_line(mean_norms, "Emission mean norms ||μ_{s,a}||", "L2 norm")
            metrics["emissions/mean_norms_plot"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            # variance sum per joint state
            fig = plot_state_line(cov_traces, "Diagonal variance sum per joint state Σ var", "sum of variances")
            metrics["emissions/var_sum_plot"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            if bern_p is not None:
                try:
                    fig = plot_bernoulli_feature_means_per_state(bern_p, bern_names=bern_names)
                    metrics["bern/features_per_state_plot"] = wandb.Image(fig)
                    WandbLogger._safe_close(fig)
                except Exception:
                    # Fallback: plot mean across Bernoulli dims if labels mismatch
                    bern_mean_per_state = np.asarray(bern_p, dtype=np.float64).mean(axis=1)
                    fig = plot_bernoulli_means_per_state(
                        bern_mean_per_state,
                        trainer.num_states,
                        title="Bernoulli mean p(x=1) per joint state",
                        ylabel="mean p(x=1) across Bernoulli dims",
                    )
                    metrics["bern/mean_per_state_plot"] = wandb.Image(fig)
                    WandbLogger._safe_close(fig)

            # Run-length + entropy distributions
            fig = plot_run_length_distribution(
                run_lengths_train,
                title="Run-length distribution (argmax gamma)",
                xlabel="segment length (timesteps)",
            )
            metrics["traj/runlen_dist"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            # Raw (physical-unit) semantics -> log numeric tables (values), split by style
            if sem_means_raw is not None and sem_feat_names is not None and len(sem_feat_names) > 0:
                figs = plot_semantics_table_by_style(
                    sem_means_raw,
                    sem_feat_names,
                    S=int(trainer.S),
                    A=int(trainer.A),
                    stds=sem_stds_raw,                 # optional: shows mean±std if available
                    title_prefix="Semantics (raw) table",
                    max_cols=12,                       # adjust if you want wider/narrower tables
                    fmt="{:.2f}",                      # change precision here
                )
                for k, fig in figs.items():
                    metrics[f"semantics_raw/table_by_style/{k}"] = wandb.Image(fig)
                    WandbLogger._safe_close(fig)

            # Scaled semantics -> log numeric tables (values), split by style
            if sem_means is not None and sem_feat_names is not None and len(sem_feat_names) > 0:
                figs = plot_semantics_table_by_style(
                    sem_means,
                    sem_feat_names,
                    S=int(trainer.S),
                    A=int(trainer.A),
                    stds=sem_stds,                    # optional: shows mean±std if available
                    title_prefix="Semantics (scaled) table",
                    max_cols=12,                       # adjust if you want wider/narrower tables
                    fmt="{:.2f}",                      # change precision here
                )
                for k, fig in figs.items():
                    metrics[f"semantics_scaled/table_by_style/{k}"] = wandb.Image(fig)
                    WandbLogger._safe_close(fig)

        except Exception as e:
            if int(getattr(trainer, "verbose", 0)) >= 0:
                print(f"[WandbLogger] Plotting failed at iter {it+1}: {e}")

        wandb_run.log(metrics)

    