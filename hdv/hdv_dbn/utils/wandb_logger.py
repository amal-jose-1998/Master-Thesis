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

        # (s_prev, a_prev, s_cur, a_cur)
        A_joint = A_s[:, None, :, None] * A_a[None, :, :, :]
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
            
            "pi_s/entropy": pi_s_entropy,
            #"pi_s/max": float(np.max(pi_s0)),
            #"pi_s/min": float(np.min(pi_s0)),

            #"pi_z/entropy_diagnostic": pi_z_entropy,
            #"pi_z/max": pi_z_max,
            #"pi_z/min": pi_z_min,

            #"emissions/total_responsibility_mass": float(total_responsibility_mass), # Sum of γ over all trajectories, time steps, and states

            "early_stop/criterion": float(criterion_for_stop) if np.isfinite(criterion_for_stop) else np.nan,
            "early_stop/source": "val" if val_obs_seqs is not None else "train",
            "early_stop/improvement_per_obs": float(improvement) if np.isfinite(improvement) else np.nan,          
        }

        #if vars_2d.shape[1] > 0:
        #    cov_logdets = np.sum(np.log(vars_2d + 1e-15), axis=1)
        #    metrics["emissions/cov_logdet_mean"] = float(np.mean(cov_logdets))
        #    metrics["emissions/cov_logdet_std"] = float(np.std(cov_logdets))
        #else:
        #    cov_logdets = np.zeros((trainer.num_states,), dtype=np.float64)
        #    metrics["emissions/cov_logdet_mean"] = np.nan
        #    metrics["emissions/cov_logdet_std"] = np.nan

        # Run-length (state duration) summaries
        #rl = np.asarray(run_lengths_train).ravel()
        #rl = rl[np.isfinite(rl)]
        #rl = rl[rl > 0]
        #metrics["traj/num_segments_total"] = int(rl.size)
        #if rl.size > 0:
        #    metrics["traj/runlen_median"] = float(np.median(rl))
        #    metrics["traj/runlen_p10"] = float(np.quantile(rl, 0.10))
        #    metrics["traj/runlen_p90"] = float(np.quantile(rl, 0.90))
        #    #metrics["traj/runlen_mean"] = float(np.mean(rl))
        #else:
        #    metrics["traj/runlen_median"] = np.nan
        #    metrics["traj/runlen_p10"] = np.nan
        #    metrics["traj/runlen_p90"] = np.nan
        #    metrics["traj/runlen_mean"] = np.nan

        if val_obs_seqs is not None:
            metrics["val/loglik"] = float(total_val_loglik) # Sum of log-likelihood on the validation set
            metrics["val/loglik_per_obs"] = float(total_val_loglik) / max(int(val_num_obs), 1)
        else:
            metrics["val/loglik"] = np.nan

        # Posterior entropy summaries
        #e = np.asarray(ent_all_train, dtype=np.float64).ravel()
        #e = e[np.isfinite(e)]
        #metrics["traj/entropy_num_timesteps"] = int(e.size)
        #if e.size > 0:
        #    metrics["traj/entropy_mean"] = float(np.mean(e))
        #    metrics["traj/entropy_median"] = float(np.median(e))
        #    metrics["traj/entropy_p95"] = float(np.quantile(e, 0.95))
        #else:
        #    metrics["traj/entropy_mean"] = np.nan
        #    metrics["traj/entropy_median"] = np.nan
        #    metrics["traj/entropy_p95"] = np.nan

        #mt = np.asarray(runlen_median_per_traj).ravel()
        #mt = mt[np.isfinite(mt)]
        #metrics["traj/runlen_median_per_traj_mean"] = float(np.mean(mt)) if mt.size > 0 else np.nan
        #metrics["traj/runlen_median_per_traj_p10"] = float(np.quantile(mt, 0.10)) if mt.size > 0 else np.nan

        #et = np.asarray(ent_mean_per_traj, dtype=np.float64).ravel()
        #et = et[np.isfinite(et)]
        #metrics["traj/entropy_mean_per_traj_mean"] = float(np.mean(et)) if et.size > 0 else np.nan
        #metrics["traj/entropy_mean_per_traj_p10"]  = float(np.quantile(et, 0.10)) if et.size > 0 else np.nan

        # Bernoulli/Lane scalar summaries 
        bern_mean_per_state = None
        if getattr(trainer.emissions, "bin_dim", 0) > 0 and getattr(trainer.emissions, "bern_p", None) is not None:
            bern_p = trainer.emissions.bern_p
            if hasattr(bern_p, "detach"):
                bern_p = bern_p.detach().cpu().numpy()
            bern_p = np.asarray(bern_p, dtype=np.float64)
            bern_mean_overall = bern_p.mean()          # (N,B)
            bern_mean_per_state = bern_p.mean(axis=1)  # (N,)
        #    metrics["bern/mean_overall"] = float(bern_mean_overall)

        lane_p = getattr(trainer.emissions, "lane_p", None)
        if lane_p is not None:
            lp = np.asarray(lane_p, dtype=np.float64) # (N, K)
            lp = np.clip(lp, 1e-15, 1.0)
            lp = lp / lp.sum(axis=1, keepdims=True)
            lane_entropy = -np.sum(lp * np.log(lp), axis=1) # (N,)
            lp_max = lp.max(axis=1)
            lp_min = lp.min(axis=1)
        #    metrics["lane/entropy_mean"] = float(np.mean(lane_entropy))
        #    metrics["lane/entropy_std"]  = float(np.std(lane_entropy))   
        #    metrics["lane/p_max_mean"]   = float(np.mean(lp_max))     
        #    metrics["lane/p_min_mean"]   = float(np.mean(lp_min)) 

        lc_p = getattr(trainer.emissions, "lc_p", None)
        if lc_p is not None:
            lcp = np.asarray(lc_p, dtype=np.float64)  # (N, 3)
            lcp = np.clip(lcp, 1e-15, 1.0)
            lcp = lcp / lcp.sum(axis=1, keepdims=True)
            lc_entropy = -np.sum(lcp * np.log(lcp), axis=1)  # (N,)
        #    metrics["lc/entropy_mean"] = float(np.mean(lc_entropy))
        #    metrics["lc/entropy_std"]  = float(np.std(lc_entropy))
        #    metrics["lc/p_max_mean"]   = float(np.mean(lcp.max(axis=1)))
        #    metrics["lc/p_min_mean"]   = float(np.mean(lcp.min(axis=1)))

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
            metrics["pi_s/plot"] = wandb.Image(fig)
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

            #fig = plot_joint_index_grid(
            #    S=int(trainer.S),
            #    A=int(trainer.A),
            #    title="Joint state index z = s*A + a (Style × Action)",
            #)
            #metrics["debug/joint_index_grid"] = wandb.Image(fig)
            #WandbLogger._safe_close(fig)

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

            # Bernoulli mean per joint state
            if bern_mean_per_state is not None:
                fig = plot_bernoulli_means_per_state(bern_mean_per_state, trainer.num_states)
                metrics["bern/mean_per_state_plot"] = wandb.Image(fig)
                WandbLogger._safe_close(fig)

            # Lane categorical heatmap p(lane | z)
            if lane_p is not None:
                fig = plot_lane_heatmap(lane_p)
                metrics["lane/heatmap"] = wandb.Image(fig)
                WandbLogger._safe_close(fig)

            # Run-length + entropy distributions
            fig = plot_run_length_distribution(
                run_lengths_train,
                title="Run-length distribution (argmax gamma)",
                xlabel="segment length (timesteps)",
            )
            metrics["traj/runlen_dist"] = wandb.Image(fig)
            WandbLogger._safe_close(fig)

            #fig = plot_entropy_distribution(ent_all_train)
            #metrics["traj/entropy_dist"] = wandb.Image(fig)
            #WandbLogger._safe_close(fig)

            # Posterior-weighted key feature semantics (derived features)
            #if sem_means is not None and sem_stds is not None and sem_feat_names is not None and len(sem_feat_names) > 0:
            #    figs = plot_key_feature_per_feature(sem_means, sem_stds, sem_feat_names, title_prefix="Posterior-weighted (scaled)")
            #    for fname, fig in figs.items():
            #        key = fname.replace(" ", "_").replace("|", "").replace("=", "").replace("__", "_")
            #        metrics[f"semantics_scaled/{key}"] = wandb.Image(fig)
            #        WandbLogger._safe_close(fig)

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

                #figs = plot_semantics_by_style(
                #    sem_means_raw,
                #    sem_feat_names,
                #    S=int(trainer.S),
                #    A=int(trainer.A),
                #    title_prefix="Semantics (raw)",
                #)
                #for k, fig in figs.items():
                #    metrics[f"semantics_raw/by_style/{k}"] = wandb.Image(fig)
                #    WandbLogger._safe_close(fig)
                
                #figs = plot_key_feature_per_feature(sem_means_raw, sem_stds_raw, sem_feat_names, title_prefix="Posterior-weighted (raw)")
                #for fname, fig in figs.items():
                #    key = fname.replace(" ", "_").replace("|", "").replace("=", "").replace("__", "_")
                #    metrics[f"semantics_raw/{key}"] = wandb.Image(fig)
                #    WandbLogger._safe_close(fig)
            
            # LC categorical heatmap p(lc | z)
            if lc_p is not None:
                fig = plot_lc_heatmap(lc_p)
                metrics["lc/heatmap"] = wandb.Image(fig)
                WandbLogger._safe_close(fig)

                # Expected LC value per state: E[lc] = (-1)pL + 0*p0 + (+1)pR
                lcp = np.asarray(lc_p, dtype=np.float64)
                lcp = np.clip(lcp, 1e-15, 1.0)
                lcp = lcp / lcp.sum(axis=1, keepdims=True)
                lc_expect = (-1.0) * lcp[:, 0] + 0.0 * lcp[:, 1] + (+1.0) * lcp[:, 2]

                fig = plot_state_line(lc_expect, "Expected lane-change E[lc] per joint state", "E[lc] (-1 left, +1 right)")
                metrics["lc/expected_per_state_plot"] = wandb.Image(fig)
                WandbLogger._safe_close(fig)

                fig = plot_joint_sa_grid(lc_expect, S=int(trainer.S), A=int(trainer.A),
                                        title="Expected lane-change E[lc] on (Style × Action) grid")
                metrics["lc/expected_sa_grid"] = wandb.Image(fig)
                WandbLogger._safe_close(fig)

        except Exception as e:
            if int(getattr(trainer, "verbose", 0)) >= 0:
                print(f"[WandbLogger] Plotting failed at iter {it+1}: {e}")

        wandb_run.log(metrics)

    