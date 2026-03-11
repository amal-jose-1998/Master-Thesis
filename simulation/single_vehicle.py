import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from frame_feature_engineer import OnlineFrameFeatureEngineer
from live_windowizer import LiveWindowizer
from matplotlib_backend import ensure_interactive_backend, is_non_interactive_backend
from pedal_visualizer import PedalVisualizer
from steering_visualizer import SteeringVisualizer

from hdv.hdv_dbn.prediction.online_predictor import OnlinePredictor
from road_renderer import RoadSceneRenderer
from hdv.hdv_dbn.config import WINDOW_FEATURE_COLS, CONTINUOUS_FEATURES


class SingleVehicleSimulation:
    """
    Simulates a single vehicle's trajectory and actions.

    SYNCED HUD MODE:
      - Figure A: road scene
      - Figure B: HUD = pedal | steering | prediction
    Both figures update inside the same FuncAnimation(update) callback,
    so the displayed frame, pedal/steering state, and prediction remain synchronized.
    """

    def __init__(
        self,
        *,
        predictor: OnlinePredictor,
        renderer: RoadSceneRenderer,
        vehicle_id,
        recording_id,
        vehicle_tracks,
        tracks_meta_df,
        recording_meta,
        maneuver_labels,
        scaler_mean,
        scaler_std,
        vehicle_class=None,
    ):
        self.predictor = predictor
        self.renderer = renderer

        self.vehicle_id = int(vehicle_id)
        self.recording_id = int(recording_id)
        self.vehicle_class = vehicle_class

        # Raw tracks used for rendering and extracting ego state
        self.vehicle_tracks = vehicle_tracks

        # Metadata used for online feature engineering
        self.tracks_meta_df = tracks_meta_df
        self.recording_meta = recording_meta

        self.maneuver_labels = list(maneuver_labels)

        # Online feature engineering + live windowization
        self.frame_engineer = OnlineFrameFeatureEngineer()
        self.live_windowizer = LiveWindowizer()

        # Scaling config (must match training)
        self.scale_idx = [
            i for i, name in enumerate(WINDOW_FEATURE_COLS)
            if name in CONTINUOUS_FEATURES
        ]
        self.scaler_mean = np.asarray(scaler_mean)
        self.scaler_std = np.asarray(scaler_std)

        # HUD helpers
        self._pedal_vis = PedalVisualizer()
        self._steer_vis = SteeringVisualizer()
        self._latest_probs = None
        self._latest_pred_frame = None

        self._ani = None

    def _reset_simulation(self):
        self.predictor.reset()
        self.live_windowizer.reset()
        self.frame_engineer.reset()
        self._latest_probs = None
        self._latest_pred_frame = None

    def _vehicle_direction(self):
        try:
            return int(self.renderer._meta_by_id.get(self.vehicle_id)[2])
        except Exception:
            try:
                return int(
                    self.renderer.tracks_meta_df[
                        self.renderer.tracks_meta_df["id"] == self.vehicle_id
                    ]["drivingDirection"].values[0]
                )
            except Exception:
                return 2

    def _draw_prediction_axis(self, ax):
        ax.clear()
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("P")
        ax.tick_params(axis="x", labelrotation=35)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")

        if self._latest_probs is None:
            ax.set_title(f"Vehicle {self.vehicle_id} - Prediction (warming up)")
            ax.text(
                0.5, 0.5,
                "No prediction yet",
                ha="center", va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            return

        bars = ax.bar(self.maneuver_labels, self._latest_probs)
        best = int(np.argmax(self._latest_probs)) if len(self._latest_probs) else 0
        if len(bars) > 0:
            bars[best].set_color("red")

        if self._latest_pred_frame is None:
            ax.set_title(f"Vehicle {self.vehicle_id} - Prediction")
        else:
            ax.set_title(
                f"Vehicle {self.vehicle_id} - Prediction (updated @ frame {self._latest_pred_frame})"
            )

    @staticmethod
    def _camera_xlim(x_pos, direction, x_offset, window_width):
        if direction == 2:
            return (x_pos - x_offset, x_pos - x_offset + window_width)
        return (x_pos + x_offset - window_width, x_pos + x_offset)

    def run(self):
        """
        Runs the single-vehicle simulation with synchronized road + HUD updates.
        """
        self._reset_simulation()

        backend_name = ensure_interactive_backend(
            plt.get_backend,
            plt.switch_backend,
            log_prefix="[single_vehicle]",
        )

        if is_non_interactive_backend(backend_name):
            print(
                f"[single_vehicle] Non-interactive matplotlib backend detected ({backend_name}). "
                f"Skipping visualization for recording {self.recording_id:02d}, vehicle {self.vehicle_id}."
            )
            return

        raw_ego = (
            self.vehicle_tracks[self.vehicle_tracks["id"] == self.vehicle_id]
            .sort_values("frame")
        )
        frames = raw_ego["frame"].to_numpy(dtype=int)
        if frames.size == 0:
            print(f"[single_vehicle] No frames for test vehicle {self.vehicle_id}")
            return

        min_frame, max_frame = int(frames[0]), int(frames[-1])

        # Precompute lookups once
        frame_to_df = {
            int(frame): df
            for frame, df in self.vehicle_tracks.groupby("frame", sort=False)
        }
        ego_rows = {
            int(row.frame): row
            for row in raw_ego[["frame", "x", "y", "xAcceleration", "xVelocity"]].itertuples(index=False)
        }

        # -------------------------
        # Figure A: road
        # -------------------------
        fig_road, ax_road = plt.subplots(figsize=(10, 6))

        xmin = float(self.vehicle_tracks["x"].min()) - 50.0
        xmax = float(self.vehicle_tracks["x"].max()) + 50.0
        full_xlims = (xmin, xmax)

        self.renderer.render_road(ax_road, xlims=full_xlims)
        ylims = ax_road.get_ylim()
        ax_road.set_ylim(ylims)

        # -------------------------
        # Figure B: HUD
        # -------------------------
        fig_hud, hud_axes = plt.subplots(1, 3, figsize=(12, 3.4), squeeze=False)
        ax_pedal = hud_axes[0, 0]
        ax_steer = hud_axes[0, 1]
        ax_pred = hud_axes[0, 2]
        fig_hud.suptitle(f"Vehicle {self.vehicle_id}: Pedal | Steering | Prediction", fontsize=12)
        fig_hud.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

        last_frame_num = None
        fps_count = 0
        fps_window_start = time.perf_counter()

        direction = self._vehicle_direction()
        window_width = 150.0
        x_offset = 10.0

        def update(frame_num):
            nonlocal last_frame_num, fps_count, fps_window_start, direction

            if last_frame_num is not None:
                if frame_num < last_frame_num:
                    self._reset_simulation()
                elif frame_num == last_frame_num:
                    return
            last_frame_num = frame_num

            ego_row = ego_rows.get(frame_num)
            if ego_row is None:
                return

            frame_df = frame_to_df.get(frame_num)
            if frame_df is None:
                return

            # -------------------------
            # A) Road rendering
            # -------------------------
            x0 = float(ego_row.x)
            y0 = float(ego_row.y)
            xlim = self._camera_xlim(x0, direction, x_offset, window_width)

            # Update only vehicles near the camera window
            pad = 30.0
            view = frame_df[(frame_df["x"] >= xlim[0] - pad) & (frame_df["x"] <= xlim[1] + pad)]

            self.renderer.render_vehicles(ax_road, view, self.vehicle_id)
            ax_road.set_xlim(xlim)
            ax_road.set_title(f"Frame {frame_num} - Test Vehicle {self.vehicle_id}")

            # -------------------------
            # B) Online feature engineering + prediction
            # -------------------------
            try:
                frame_ctx = self.frame_engineer.prepare_frame(
                    raw_frame_df=frame_df,
                    tracks_meta_df=self.tracks_meta_df,
                    recording_meta=self.recording_meta,
                )

                frame_vec = self.frame_engineer.compute_ego(
                    frame_ctx,
                    self.vehicle_id,
                    recording_id=self.recording_id,
                )

                windows = self.live_windowizer.add_frame(frame_vec)
                if windows:
                    win = windows[-1].copy()

                    if self.scale_idx:
                        win[self.scale_idx] = (
                            (win[self.scale_idx] - self.scaler_mean[self.scale_idx]) /
                            (self.scaler_std[self.scale_idx] + 1e-12)
                        )

                    win_tensor = torch.tensor(
                        win,
                        dtype=self.predictor.dtype,
                        device=self.predictor.device,
                    )

                    self.predictor.update(win_tensor)

                    if self.predictor.is_ready:
                        pred = self.predictor.predict_next()
                        probs = torch.softmax(pred.pred_logprob.flatten(), dim=0).detach().cpu().numpy()
                        self._latest_probs = probs
                        self._latest_pred_frame = int(frame_num)

            except Exception as e:
                print(f"[single_vehicle] Prediction update failed: {e}")

            # -------------------------
            # C) HUD updates
            # -------------------------
            raw_ax = float(ego_row.xAcceleration) if hasattr(ego_row, "xAcceleration") else 0.0
            raw_vx = float(ego_row.xVelocity) if hasattr(ego_row, "xVelocity") else 0.0

            if direction == 1:
                ax_val = -raw_ax
                vx_val = -raw_vx
            else:
                ax_val = raw_ax
                vx_val = raw_vx

            prev_frame = frame_num - 1
            prev_row = ego_rows.get(prev_frame, ego_row)
            prev_y = float(prev_row.y)
            curr_y = float(y0)

            self._pedal_vis.draw(ax_pedal, ax_val, vx_val)
            ax_pedal.set_title(f"Vehicle {self.vehicle_id} - Pedal")

            self._steer_vis.draw(ax_steer, direction, prev_y, curr_y)
            ax_steer.set_title(f"Vehicle {self.vehicle_id} - Steering")

            self._draw_prediction_axis(ax_pred)

            # Refresh the HUD on the same animation tick
            fig_hud.canvas.draw_idle()

            fps_count += 1
            now = time.perf_counter()
            elapsed = now - fps_window_start
            if elapsed >= 1.0:
                print(f"[single_render] {fps_count / elapsed:.2f} FPS")
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