import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from pedal_visualizer import PedalVisualizer
from steering_visualizer import SteeringVisualizer
from frame_feature_engineer import OnlineFrameFeatureEngineer
from live_windowizer import LiveWindowizer
from matplotlib_backend import ensure_interactive_backend, is_non_interactive_backend
from hdv.hdv_dbn.prediction.model_interface import HDVDbnModel
from hdv.hdv_dbn.prediction.online_predictor import BatchedOnlinePredictor
from road_renderer import RoadSceneRenderer
from hdv.hdv_dbn.config import WINDOW_FEATURE_COLS, CONTINUOUS_FEATURES


class MultiVehicleSimulation:
    """
    Simultaneous multi-vehicle simulation with batched online prediction.

    SYNCED HUD MODE (2 figures, 1 animation clock):
      - Figure A: road scene subplots (one per vehicle)
      - Figure B: HUD subplots (one row per vehicle: pedal | steering | prediction)
    Both figures update inside the same FuncAnimation(update) callback,
    so pedal/steering/prediction stay synchronized with the animated frame index.
    """

    def __init__(
        self,
        *,
        model: HDVDbnModel,
        renderer: RoadSceneRenderer,
        recording_id,
        vehicle_ids,
        vehicle_tracks,
        tracks_meta_df,
        recording_meta,
        maneuver_labels,
        scaler_mean_by_vehicle,
        scaler_std_by_vehicle,
        warmup_steps=2,
    ):
        self.model = model
        self.renderer = renderer

        self.recording_id = int(recording_id)
        self.vehicle_ids = sorted(int(v) for v in vehicle_ids)
        self.vehicle_tracks = vehicle_tracks
        self.tracks_meta_df = tracks_meta_df
        self.recording_meta = recording_meta

        self.maneuver_labels = list(maneuver_labels)

        self.scaler_mean_by_vehicle = {
            int(k): np.asarray(v) for k, v in scaler_mean_by_vehicle.items()
        }
        self.scaler_std_by_vehicle = {
            int(k): np.asarray(v) for k, v in scaler_std_by_vehicle.items()
        }

        self.scale_idx = [
            i for i, name in enumerate(WINDOW_FEATURE_COLS)
            if name in CONTINUOUS_FEATURES
        ]

        self.slot_to_vid = list(self.vehicle_ids)
        self.vid_to_slot = {vid: i for i, vid in enumerate(self.slot_to_vid)}
        self.batch_size = len(self.slot_to_vid)

        self.predictor = BatchedOnlinePredictor(self.model, warmup_steps=warmup_steps)
        self.frame_engineer = OnlineFrameFeatureEngineer()
        self.windowizers = {vid: LiveWindowizer() for vid in self.vehicle_ids}

        # HUD helpers
        self._pedal_vis = PedalVisualizer()
        self._steer_vis = SteeringVisualizer()
        self._probs_by_vid = {vid: None for vid in self.vehicle_ids}
        self._pred_frame_by_vid = {vid: None for vid in self.vehicle_ids}

        self._ani = None

    def _vehicle_direction(self, vehicle_id):
        try:
            return int(self.renderer._meta_by_id.get(vehicle_id)[2])
        except Exception:
            return 2

    @staticmethod # helper for camera xlim calculation based on ego position and direction
    def _camera_xlim(x_pos, direction, x_offset, window_width):
        if direction == 2:
            return (x_pos - x_offset, x_pos - x_offset + window_width)
        return (x_pos + x_offset - window_width, x_pos + x_offset)

    def _reset_simulation(self):
        self.predictor.reset(self.batch_size)
        self.frame_engineer.reset()
        for w in self.windowizers.values():
            w.reset()

        for vid in self.vehicle_ids:
            self._probs_by_vid[vid] = None
            self._pred_frame_by_vid[vid] = None

    def _draw_prediction_axis(self, ax: plt.Axes, vid):
        ax.clear()
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("P")
        ax.tick_params(axis="x", labelrotation=35)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")

        probs = self._probs_by_vid.get(vid, None)
        pred_frame = self._pred_frame_by_vid.get(vid, None)

        if probs is None:
            ax.set_title(f"Vehicle {vid} - Prediction (warming up)")
            ax.text(
                0.5, 0.5,
                "No prediction yet",
                ha="center", va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            return

        labels = self.maneuver_labels
        bars = ax.bar(labels, probs)
        best = int(np.argmax(probs)) if len(probs) else 0
        if len(bars) > 0:
            bars[best].set_color("red")

        if pred_frame is None:
            ax.set_title(f"Vehicle {vid} - Prediction")
        else:
            ax.set_title(f"Vehicle {vid} - Prediction (updated @ frame {pred_frame})")

    def run(self):
        self._reset_simulation()

        backend_name = ensure_interactive_backend(
            plt.get_backend,
            plt.switch_backend,
            log_prefix="[multi_vehicle]",
        )

        if is_non_interactive_backend(backend_name):
            print(
                f"[multi_vehicle] Non-interactive matplotlib backend detected ({backend_name}). "
                f"Skipping visualization for recording {self.recording_id:02d}."
            )
            return

        if self.vehicle_tracks.empty:
            print(f"[multi_vehicle] No tracks for recording {self.recording_id}")
            return

        # frame -> df lookup for rendering + feature engineering
        frame_to_df = {
            int(frame): df
            for frame, df in self.vehicle_tracks.groupby("frame", sort=False)
        }

        # ego rows per vehicle for fast access
        ego_rows_by_vid = {}
        for vid in self.vehicle_ids:
            ego_rows = {
                int(row.frame): row
                for row in self.vehicle_tracks[self.vehicle_tracks["id"] == vid][
                    ["frame", "x", "y", "xAcceleration", "xVelocity"]
                ].itertuples(index=False)
            }
            ego_rows_by_vid[vid] = ego_rows

        frames_union = sorted(frame_to_df.keys())
        if not frames_union:
            print(f"[multi_vehicle] No frames in recording {self.recording_id}")
            return

        min_frame, max_frame = int(frames_union[0]), int(frames_union[-1])

        num_vid = max(1, len(self.vehicle_ids))
        window_width = 150.0
        x_offset = 10.0

        # -------------------------
        # Figure A: roads
        # -------------------------
        fig_road, axes_arr = plt.subplots(1, num_vid, figsize=(6 * num_vid, 5), squeeze=False)
        road_axes = list(axes_arr[0])

        # One renderer per vehicle to cache static road elements (lanes, curbs, medians) and only update dynamic vehicles each frame.
        road_renderers = [
            RoadSceneRenderer(
                self.recording_meta,
                self.tracks_meta_df,
                lane_color=self.renderer.lane_color,
                road_color=self.renderer.road_color,
                median_color=self.renderer.median_color,
            )
            for _ in self.vehicle_ids
        ]

        full_xmin = float(self.vehicle_tracks["x"].min()) - 50.0
        full_xmax = float(self.vehicle_tracks["x"].max()) + 50.0
        full_xlims = (full_xmin, full_xmax)

        for idx, vid in enumerate(self.vehicle_ids):
            ax = road_axes[idx]
            road_renderers[idx].render_road(ax, xlims=full_xlims)
            ax.set_title(f"Vehicle {vid}")

        # -------------------------
        # Figure B: HUD
        # -------------------------
        fig_hud, hud_axes_arr = plt.subplots(
            num_vid, 3, # pedal | steering | prediction
            figsize=(12, max(3, 2.6 * num_vid)),
            squeeze=False, # always 2D array for consistent indexing, even if num_vid=1
        )
        fig_hud.suptitle("HUD: Pedal | Steering | Prediction", fontsize=12)
        fig_hud.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])

        # performance counters
        fps_count = 0
        fps_window_start = time.perf_counter()
        last_frame_num = None

        zero_obs = np.zeros((self.batch_size, len(WINDOW_FEATURE_COLS)), dtype=np.float64)

        def update(frame_num):
            nonlocal fps_count, fps_window_start, last_frame_num

            # loop reset (repeat=True)
            if last_frame_num is not None and frame_num < last_frame_num:
                self._reset_simulation()
            last_frame_num = frame_num

            frame_df = frame_to_df.get(frame_num)
            if frame_df is None:
                return

            active_vids = [
                vid for vid in self.vehicle_ids
                if frame_num in ego_rows_by_vid.get(vid, {})
            ]

            # Build cached per-frame context once, then extract one ego cheaply per vehicle.
            try:
                frame_ctx = self.frame_engineer.prepare_frame(
                    raw_frame_df=frame_df,
                    tracks_meta_df=self.tracks_meta_df,
                    recording_meta=self.recording_meta,
                )
            except Exception:
                frame_ctx = None

            # -------------------------
            # A) Road rendering
            # -------------------------
            for idx, vid in enumerate(self.vehicle_ids):
                ax = road_axes[idx]
                row = ego_rows_by_vid.get(vid, {}).get(frame_num)

                if row is None:
                    # No ego row for this frame -> keep road visible, hide any stale vehicle rectangles.
                    empty_view = frame_df.iloc[0:0]
                    road_renderers[idx].render_vehicles(ax, empty_view, test_vehicle_id=vid)
                    ax.set_xlim(full_xlims)
                    ax.set_title(f"Vehicle {vid} (inactive) | Frame {frame_num}")
                    continue

                direction = self._vehicle_direction(vid)
                xlim = self._camera_xlim(float(row.x), direction, x_offset, window_width)

                # Filter to camera window to reduce the number of rectangles updated on this axis.
                pad = 30.0
                view = frame_df[(frame_df["x"] >= xlim[0] - pad) & (frame_df["x"] <= xlim[1] + pad)]

                road_renderers[idx].render_vehicles(ax, view, test_vehicle_id=vid)
                ax.set_xlim(xlim)
                ax.set_title(f"Vehicle {vid} | Frame {frame_num}")

            fig_road.suptitle(
                f"Recording {self.recording_id:02d} | Frame {frame_num} | Active {len(active_vids)}",
                fontsize=12,
            )

            # -------------------------
            # B) Prediction pipeline (batched)
            # -------------------------
            obs_batch = zero_obs.copy()
            update_mask = np.zeros(self.batch_size, dtype=bool)

            if frame_ctx is not None:
                for vid in active_vids:
                    try:
                        frame_vec = self.frame_engineer.compute_ego(
                            frame_ctx,
                            vid,
                            recording_id=self.recording_id,
                        )
                    except Exception:
                        continue

                    windows = self.windowizers[vid].add_frame(frame_vec)
                    if not windows:
                        continue

                    win = windows[-1].copy()
                    if self.scale_idx:
                        m = self.scaler_mean_by_vehicle[vid]
                        s = self.scaler_std_by_vehicle[vid]
                        win[self.scale_idx] = (
                            (win[self.scale_idx] - m[self.scale_idx]) /
                            (s[self.scale_idx] + 1e-12)
                        )

                    slot = self.vid_to_slot[vid]
                    obs_batch[slot] = win
                    update_mask[slot] = True

            if np.any(update_mask):
                obs_tensor = torch.tensor(
                    obs_batch,
                    dtype=self.predictor.dtype,
                    device=self.predictor.device,
                )
                active_tensor = torch.as_tensor(update_mask, device=self.predictor.device)

                self.predictor.update(obs_tensor, active_mask=active_tensor)
                out = self.predictor.predict_next(active_mask=active_tensor, strict_ready=False)

                ready_slots = torch.nonzero(
                    out.ready_mask & out.active_mask,
                    as_tuple=False,
                ).flatten().tolist()

                for slot in ready_slots:
                    vid = self.slot_to_vid[int(slot)]
                    probs = torch.softmax(out.pred_logprob[slot].flatten(), dim=0).detach().cpu().numpy()
                    self._probs_by_vid[int(vid)] = probs
                    self._pred_frame_by_vid[int(vid)] = int(frame_num)

            # -------------------------
            # C) HUD updates
            # -------------------------
            for idx, vid in enumerate(self.vehicle_ids):
                ax_pedal = hud_axes_arr[idx, 0]
                ax_steer = hud_axes_arr[idx, 1]
                ax_pred = hud_axes_arr[idx, 2]

                row = ego_rows_by_vid.get(vid, {}).get(frame_num)
                if row is None:
                    ax_pedal.clear()
                    ax_pedal.set_axis_off()
                    ax_pedal.set_title(f"V{vid} Pedal (inactive)")

                    ax_steer.clear()
                    ax_steer.set_axis_off()
                    ax_steer.set_title(f"V{vid} Steering (inactive)")

                    ax_pred.clear()
                    ax_pred.set_axis_off()
                    ax_pred.set_title(f"V{vid} Prediction (inactive)")
                    continue

                # Re-enable axes in case the vehicle was inactive in a previous frame.
                ax_pedal.set_axis_on()
                ax_steer.set_axis_on()
                ax_pred.set_axis_on()

                direction = self._vehicle_direction(vid)

                raw_ax = float(row.xAcceleration) if hasattr(row, "xAcceleration") else 0.0
                raw_vx = float(row.xVelocity) if hasattr(row, "xVelocity") else 0.0

                if direction == 1:
                    ax_val = -raw_ax
                    vx_val = -raw_vx
                else:
                    ax_val = raw_ax
                    vx_val = raw_vx

                prev_frame = frame_num - 1
                prev_row = ego_rows_by_vid[vid].get(prev_frame, row)
                prev_y = float(prev_row.y)
                curr_y = float(row.y)

                self._pedal_vis.draw(ax_pedal, ax_val, vx_val)
                ax_pedal.set_title(f"Vehicle {vid} - Pedal")

                self._steer_vis.draw(ax_steer, direction, prev_y, curr_y)
                ax_steer.set_title(f"Vehicle {vid} - Steering")

                self._draw_prediction_axis(ax_pred, int(vid))

            # Force HUD figure refresh on the same animation tick.
            fig_hud.canvas.draw_idle()

            fps_count += 1
            now = time.perf_counter()
            elapsed = now - fps_window_start
            if elapsed >= 1.0:
                print(f"[multi_render] {fps_count / elapsed:.2f} FPS")
                fps_count = 0
                fps_window_start = now

        self._ani = animation.FuncAnimation(
            fig_road,
            update,
            frames=range(min_frame, max_frame + 1),
            interval=40,
            repeat=True,
        )
        plt.show()