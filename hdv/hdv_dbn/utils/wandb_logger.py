"""
W&B logger for the structured Style/Action DBN trainer (Pattern A).

Key design:
- Trainer calls: WandbLogger.log_iteration(**log_kwargs)
- Logger tolerates missing keys and extra keys.
- Uses .get(...) everywhere, never requires exact signature alignment.
"""
from ..config import BERNOULLI_FEATURES

import time
from typing import Any, Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .wandb_plots import (
    plot_pi_s0,
    plot_pi_a0_given_s0,
    plot_A_s,
    plot_A_s_diag,
    plot_A_a_per_style,
    plot_A_a_diag_per_style,
    plot_joint_grid,
    plot_joint_index_grid,
    plot_bar,
    plot_line,
    plot_hist,
    plot_heatmap,
    plot_semantics_tables_by_style,
    plot_semantics_tables_by_action,
)


class WandbLogger:
    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _safe_close(fig):
        try:
            plt.close(fig)
        except Exception:
            pass

    @staticmethod
    def _as_np(x, dtype=np.float64):
        if x is None:
            return None
        return np.asarray(x, dtype=dtype)

    @staticmethod
    def _entropy(p: np.ndarray, eps: float = 1e-15) -> float:
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        p = np.clip(p, 0.0, 1.0)
        s = float(p.sum())
        if s <= 0:
            return float("nan")
        p = p / s
        return float(-np.sum(p * np.log(p + eps)))

    @staticmethod
    def _joint_from_structured(pi_s0, pi_a0_given_s0, A_s, A_a) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagnostic-only conversion to joint z=(s,a) using z=s*A + a:

          pi_z[z]     = pi_s0[s] * pi_a0_given_s0[s,a]
          A_zz[zp,zc]  = A_s[s_prev, s_cur] * A_a[s_cur, a_prev, a_cur]
        """
        pi_s0 = np.asarray(pi_s0, dtype=np.float64).reshape(-1)
        pi_a0_given_s0 = np.asarray(pi_a0_given_s0, dtype=np.float64)
        A_s = np.asarray(A_s, dtype=np.float64)
        A_a = np.asarray(A_a, dtype=np.float64)

        S = int(pi_s0.size)
        A = int(pi_a0_given_s0.shape[1])

        pi_z = (pi_s0[:, None] * pi_a0_given_s0).reshape(S * A)
        A_joint = A_s[:, :, None, None] * A_a[None, :, :, :]
        A_zz = A_joint.reshape(S * A, S * A)
        return pi_z, A_zz

    # -------------------------------------------------------------------------
    # main entry
    # -------------------------------------------------------------------------
    @staticmethod
    def log_iteration(**kwargs):
        """
        Pattern A: accept any kwargs dict from trainer.

        Expected common keys (but all optional):
          wandb_run, trainer, it, iter_start,
          total_train_ll, train_ll_per_obs, total_val_ll, val_ll_per_obs,
          criterion_for_stop, criterion_source, improvement, bad_epochs,
          transitions_stable, pi_stable,
          train_total_seqs, train_used_seqs, skipped_empty, skipped_bad_logB, skipped_bad_ll,
          delta_pi_s0, delta_pi_a0_given_s0, delta_A_s, delta_A_a, delta_pi, delta_A,
          pi_s0, pi_a0_given_s0, A_s, A_a,
          state_weights_sa, state_frac_sa, total_responsibility_mass,
          run_lengths_train, ent_all_train, runlen_median_per_traj, ent_mean_per_traj,
          sem_feat_names, sem_means, sem_stds, sem_means_raw, sem_stds_raw
        """
        wandb_run = kwargs.get("wandb_run", None)
        if wandb_run is None:
            return

        # import wandb lazily so training can run without it
        import wandb

        trainer = kwargs.get("trainer", None)
        it = int(kwargs.get("it", -1))
        iter_start = kwargs.get("iter_start", None)
        if iter_start is None:
            iter_time = np.nan
        else:
            try:
                iter_time = float(time.perf_counter() - float(iter_start))
            except Exception:
                iter_time = np.nan

        pi_s0 = WandbLogger._as_np(kwargs.get("pi_s0", None))
        pi_a0_given_s0 = WandbLogger._as_np(kwargs.get("pi_a0_given_s0", None))
        A_s = WandbLogger._as_np(kwargs.get("A_s", None))
        A_a = WandbLogger._as_np(kwargs.get("A_a", None))

        # determine S,A if possible
        S = None
        A = None
        if trainer is not None:
            S = getattr(trainer, "S", None)
            A = getattr(trainer, "A", None)

        metrics: Dict[str, Any] = {
            "em/iter": it+1,
            "time/iter_seconds": iter_time,
            "train/total_train_ll": float(kwargs.get("total_train_ll", np.nan)),
            "train/train_ll_per_obs": float(kwargs.get("train_ll_per_obs", np.nan)),
            "val/total_val_ll": float(kwargs.get("total_val_ll", np.nan)),
            "val/val_ll_per_obs": float(kwargs.get("val_ll_per_obs", np.nan)),

            "early_stop/criterion": float(kwargs.get("criterion_for_stop", np.nan)),
            "early_stop/source": str(kwargs.get("criterion_source", "")),
            "early_stop/improvement": float(kwargs.get("improvement", np.nan)),
            "early_stop/bad_epochs": int(kwargs.get("bad_epochs", 0)),
            #"early_stop/transitions_stable": kwargs.get("transitions_stable", np.nan),
            #"early_stop/pi_stable": kwargs.get("pi_stable", np.nan),

            "e_step/train_total_seqs": int(kwargs.get("train_total_seqs", 0)),
            "e_step/train_used_seqs": int(kwargs.get("train_used_seqs", 0)),
            "e_step/skipped_empty": int(kwargs.get("skipped_empty", 0)),
            "e_step/skipped_bad_logB": int(kwargs.get("skipped_bad_logB", 0)),
            "e_step/skipped_bad_ll": int(kwargs.get("skipped_bad_ll", 0)),

            "delta/pi_s0": float(kwargs.get("delta_pi_s0", np.nan)),
            "delta/pi_a0_given_s0": float(kwargs.get("delta_pi_a0_given_s0", np.nan)),
            "delta/A_s": float(kwargs.get("delta_A_s", np.nan)),
            "delta/A_a": float(kwargs.get("delta_A_a", np.nan)),
            # aggregate
            "delta/emission_mean": float(kwargs.get("log_emission_delta_mean", np.nan)),
            # hierarchical emissions (keys exist only in hierarchical mode)
            "delta/emission/gauss_mean": float(kwargs.get("log_delta_gauss_mean", np.nan)),
            "delta/emission/gauss_var": float(kwargs.get("log_delta_gauss_var", np.nan)),
            "delta/emission/bern_p": float(kwargs.get("log_delta_bern_p", np.nan)),
            # PoE emissions (keys exist only in PoE mode)
            "delta/emission/style_gauss_mean": float(kwargs.get("log_delta_style_gauss_mean", np.nan)),
            "delta/emission/style_gauss_var": float(kwargs.get("log_delta_style_gauss_var", np.nan)),
            "delta/emission/action_gauss_mean": float(kwargs.get("log_delta_action_gauss_mean", np.nan)),
            "delta/emission/action_gauss_var": float(kwargs.get("log_delta_action_gauss_var", np.nan)),
            "delta/emission/style_bern_p": float(kwargs.get("log_delta_style_bern_p", np.nan)),
            "delta/emission/action_bern_p": float(kwargs.get("log_delta_action_bern_p", np.nan)),

            #"post/total_responsibility_mass": float(kwargs.get("total_responsibility_mass", np.nan)),
        }

        # Extra posterior summaries
        runlen_median_per_traj = kwargs.get("runlen_median_per_traj", None)
        ent_mean_per_traj = kwargs.get("ent_mean_per_traj", None)

        #try:
        #    if runlen_median_per_traj is not None:
        #        metrics["post/runlen_median_over_traj"] = float(np.nanmedian(np.asarray(runlen_median_per_traj, dtype=np.float64)))
        #    else:
        #        metrics["post/runlen_median_over_traj"] = np.nan
        #except Exception:
        #    metrics["post/runlen_median_over_traj"] = np.nan

        #try:
        #    if ent_mean_per_traj is not None:
        #        metrics["post/entropy_mean_over_traj"] = float(np.nanmean(np.asarray(ent_mean_per_traj, dtype=np.float64)))
        #    else:
        #        metrics["post/entropy_mean_over_traj"] = np.nan
        #except Exception:
        #    metrics["post/entropy_mean_over_traj"] = np.nan
    

        # ---------------------------------------------------------------------
        # Plots 
        # ---------------------------------------------------------------------
        try:                                               
            if pi_s0 is not None:
                fig = plot_pi_s0(pi_s0)
                metrics["plots/pi_s0"] = wandb.Image(fig)
                WandbLogger._safe_close(fig)

            if pi_a0_given_s0 is not None:
                fig = plot_pi_a0_given_s0(pi_a0_given_s0)
                metrics["plots/pi_a0_given_s0"] = wandb.Image(fig)
                WandbLogger._safe_close(fig)

            if A_s is not None:
                fig = plot_A_s(A_s)
                metrics["plots/A_s"] = wandb.Image(fig)
                WandbLogger._safe_close(fig)

                fig = plot_A_s_diag(A_s)
                metrics["plots/A_s_diag"] = wandb.Image(fig)
                WandbLogger._safe_close(fig)

            if A_a is not None:
                figs = plot_A_a_per_style(A_a)
                for k, fig in figs.items():
                    metrics[f"plots/A_a/{k}"] = wandb.Image(fig)
                    WandbLogger._safe_close(fig)

                figs = plot_A_a_diag_per_style(A_a)
                for k, fig in figs.items():
                    metrics[f"plots/A_a_diag/{k}"] = wandb.Image(fig)
                    WandbLogger._safe_close(fig)

            # --- semantics tables ---
            sem_feat_names = kwargs.get("sem_feat_names", None)

            sem_means = kwargs.get("sem_means", None)
            sem_stds = kwargs.get("sem_stds", None)

            sem_means_raw = kwargs.get("sem_means_raw", None)
            sem_stds_raw = kwargs.get("sem_stds_raw", None)

            if sem_feat_names is not None and S is not None and A is not None:
                S = int(S)
                A = int(A)

                #if sem_means is not None:
                #    try:
                 #       figs = plot_semantics_tables_by_style(
                #            sem_means, sem_feat_names, S=S, A=A,
                #            stds=sem_stds,
                #            title_prefix="Semantics (scaled)",
                #            max_cols=10,
                #            fmt="{:.2f}",
                #            wrap_header_at=18,
                 #       )
                #        for k, fig in figs.items():
                #            metrics[f"plots/semantics_scaled/by_style/{k}"] = wandb.Image(fig)
                 #           WandbLogger._safe_close(fig)
                 #   except Exception:
                 #       pass
#
#                    try:
#                        figs = plot_semantics_tables_by_action(
#                            sem_means, sem_feat_names, S=S, A=A,
 #                           stds=sem_stds,
 #                           title_prefix="Semantics (scaled)",
 #                           max_cols=10,
 #                           fmt="{:.2f}",
 #                           wrap_header_at=18,
 #                       )
 #                       for k, fig in figs.items():
  #                          metrics[f"plots/semantics_scaled/by_action/{k}"] = wandb.Image(fig)
  #                          WandbLogger._safe_close(fig)
  #                  except Exception:
  #                      pass

                if sem_means_raw is not None:
                    # sem_means_raw: (S,A,F)  -> (S*A,F)
                    M = np.asarray(sem_means_raw, dtype=np.float64)
                    SD = np.asarray(sem_stds_raw, dtype=np.float64) if sem_stds_raw is not None else None

                    if M.ndim == 3:
                        M2 = M.reshape(S * A, M.shape[2])
                        SD2 = SD.reshape(S * A, SD.shape[2]) if (SD is not None and SD.ndim == 3) else None
                    elif M.ndim == 2:
                        M2 = M
                        SD2 = SD if (SD is not None and SD.ndim == 2) else None
                    else:
                        raise ValueError(f"Unexpected sem_means_raw shape: {M.shape}")
                    # Hide ±std for Bernoulli features in semantics tables (show probabilities only)
                    if SD2 is not None and sem_feat_names is not None:
                        bern = set(BERNOULLI_FEATURES)
                        bern_idx = [j for j, name in enumerate(sem_feat_names) if name in bern]
                        if bern_idx:
                            SD2 = SD2.copy()
                            SD2[:, bern_idx] = np.nan
                    try:
                        figs = plot_semantics_tables_by_style(
                            M2, sem_feat_names, S=S, A=A,
                            stds=SD2,
                            title_prefix="Semantics (raw units)",
                            max_cols=10,
                            fmt="{:.2f}",
                            wrap_header_at=18,
                        )
                        for k, fig in figs.items():
                            metrics[f"plots/semantics_raw/by_style/{k}"] = wandb.Image(fig)
                            WandbLogger._safe_close(fig)
                    except Exception as e:
                        print(f"[WandbLogger] semantics plotting failed: {e}")
                        pass

                    #try:
                    #    figs = plot_semantics_tables_by_action(
                    #        M2, sem_feat_names, S=S, A=A,
                    #        stds=SD2,
                    #        title_prefix="Semantics (raw units)",
                    #        max_cols=10,
                    #        fmt="{:.2f}",
                    #        wrap_header_at=18,
                    #    )
                    #    for k, fig in figs.items():
                    #        metrics[f"plots/semantics_raw/by_action/{k}"] = wandb.Image(fig)
                    #        WandbLogger._safe_close(fig)
                    #except Exception:
                    #    print(f"[WandbLogger] semantics plotting failed: {e}")
                    #    pass

        except Exception as e:
            # Never crash training because of plotting/logging
            try:
                if int(getattr(trainer, "verbose", 1)) >= 1:
                    print(f"[WandbLogger] logging failed at iter={it}: {e}")
            except Exception:
                pass

        wandb_run.log(metrics)
