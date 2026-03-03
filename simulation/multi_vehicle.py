import time
import multiprocessing as mp

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

from frame_feature_engineer import OnlineFrameFeatureEngineer
from live_windowizer import LiveWindowizer
from hdv.hdv_dbn.prediction.model_interface import HDVDbnModel
from hdv.hdv_dbn.prediction.online_predictor import BatchedOnlinePredictor
from road_renderer import RoadSceneRenderer
from hdv.hdv_dbn.config import WINDOW_FEATURE_COLS, CONTINUOUS_FEATURES


class MultiVehicleSimulation:
    """
    Simultaneous multi-vehicle simulation with batched online prediction.

    - One batched predictor handles all tracked vehicles in a recording.
    - Each vehicle keeps its own windowizer stream.
    - Frame feature engineering is shared and keyed by (recording_id, vehicle_id).
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
        pedal_queue: mp.Queue,
        prediction_queue: mp.Queue,
        maneuver_labels,
        scaler_mean_by_vehicle,
        scaler_std_by_vehicle,
        vehicle_class_by_vehicle,
        warmup_steps=2,
    ):
        self.model = model
        self.renderer = renderer

        self.recording_id = int(recording_id)
        self.vehicle_ids = sorted(int(v) for v in vehicle_ids)
        self.vehicle_tracks = vehicle_tracks
        self.tracks_meta_df = tracks_meta_df
        self.recording_meta = recording_meta

        self.pedal_queue = pedal_queue
        self.prediction_queue = prediction_queue
        self.maneuver_labels = list(maneuver_labels)

        self.scaler_mean_by_vehicle = {int(k): np.asarray(v) for k, v in scaler_mean_by_vehicle.items()}
        self.scaler_std_by_vehicle = {int(k): np.asarray(v) for k, v in scaler_std_by_vehicle.items()}

        self.scale_idx = [i for i, name in enumerate(WINDOW_FEATURE_COLS) if name in CONTINUOUS_FEATURES]

        self.slot_to_vid = list(self.vehicle_ids)
        self.vid_to_slot = {vid: i for i, vid in enumerate(self.slot_to_vid)}
        self.batch_size = len(self.slot_to_vid)

        self.predictor = BatchedOnlinePredictor(self.model, warmup_steps=warmup_steps)
        self.frame_engineer = OnlineFrameFeatureEngineer()
        self.windowizers = {vid: LiveWindowizer() for vid in self.vehicle_ids}

    def _vehicle_direction(self, vehicle_id: int) -> int:
        try:
            return int(self.renderer._meta_by_id.get(vehicle_id)[2])
        except Exception:
            return 2

    @staticmethod
    def _camera_xlim(x_pos: float, direction: int, x_offset: float, window_width: float):
        if direction == 2:
            return (x_pos - x_offset, x_pos - x_offset + window_width)
        return (x_pos + x_offset - window_width, x_pos + x_offset)

    def _reset_simulation(self):
        self.predictor.reset(self.batch_size)
        self.frame_engineer.reset()
        for w in self.windowizers.values():
            w.reset()
        try:
            self.prediction_queue.put_nowait(
                {
                    "reset": True,
                    "maneuver_labels": self.maneuver_labels,
                    "vehicle_ids": list(self.vehicle_ids),
                }
            )
        except Exception:
            pass
        try:
            self.pedal_queue.put_nowait(
                {
                    "reset": True,
                    "vehicle_ids": list(self.vehicle_ids),
                }
            )
        except Exception:
            pass

    def run(self):
        self._reset_simulation()

        if self.vehicle_tracks.empty:
            print(f"[multi_vehicle] No tracks for recording {self.recording_id}")
            return

        frame_to_df = {int(frame): df for frame, df in self.vehicle_tracks.groupby("frame", sort=False)}

        ego_rows_by_vid = {}
        for vid in self.vehicle_ids:
            ego_rows = {
                int(row.frame): row
                for row in self.vehicle_tracks[self.vehicle_tracks["id"] == vid][["frame", "x", "y", "xAcceleration", "xVelocity"]].itertuples(index=False)
            }
            ego_rows_by_vid[vid] = ego_rows

        frames_union = sorted(frame_to_df.keys())
        if not frames_union:
            print(f"[multi_vehicle] No frames in recording {self.recording_id}")
            return

        min_frame, max_frame = int(frames_union[0]), int(frames_union[-1])

        num_vid = max(1, len(self.vehicle_ids))
        fig, axes_arr = plt.subplots(1, num_vid, figsize=(6 * num_vid, 5), squeeze=False)
        axes = list(axes_arr[0])

        window_width = 150
        x_offset = 10

        fps_count = 0
        fps_window_start = time.perf_counter()
        last_frame_num = None

        zero_obs = np.zeros((self.batch_size, len(WINDOW_FEATURE_COLS)), dtype=np.float64)

        def update(frame_num):
            nonlocal fps_count, fps_window_start, last_frame_num

            if last_frame_num is not None and frame_num < last_frame_num:
                self._reset_simulation()
            last_frame_num = frame_num

            frame_df = frame_to_df.get(frame_num)
            if frame_df is None:
                return

            active_vids = [vid for vid in self.vehicle_ids if frame_num in ego_rows_by_vid.get(vid, {})]
            if not active_vids:
                try:
                    self.prediction_queue.put_nowait(
                        {
                            "active_vehicle_ids": [],
                            "vehicle_ids": list(self.vehicle_ids),
                        }
                    )
                except Exception:
                    pass
                return

            try:
                self.prediction_queue.put_nowait(
                    {
                        "active_vehicle_ids": [int(v) for v in active_vids],
                        "vehicle_ids": list(self.vehicle_ids),
                    }
                )
            except Exception:
                pass

            for idx, vid in enumerate(self.vehicle_ids):
                ax = axes[idx]
                ax.clear()

                row = ego_rows_by_vid.get(vid, {}).get(frame_num)
                if row is None:
                    ax.set_title(f"Vehicle {vid} (inactive)")
                    ax.set_axis_off()
                    continue

                direction = self._vehicle_direction(vid)
                xlim = self._camera_xlim(float(row.x), direction, x_offset, window_width)

                self.renderer.render_road(ax, xlims=xlim)
                self.renderer._render_vehicles(ax, frame_df, test_vehicle_id=vid)
                ax.set_xlim(xlim)
                ax.set_title(f"Vehicle {vid} | Frame {frame_num}")

            obs_batch = zero_obs.copy()
            update_mask = np.zeros(self.batch_size, dtype=bool)

            for vid in active_vids:
                try:
                    frame_vec = self.frame_engineer.add_frame(
                        raw_frame_df=frame_df,
                        ego_vehicle_id=vid,
                        tracks_meta_df=self.tracks_meta_df,
                        recording_meta=self.recording_meta,
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
                    win[self.scale_idx] = (win[self.scale_idx] - m[self.scale_idx]) / (s[self.scale_idx] + 1e-12)

                slot = self.vid_to_slot[vid]
                obs_batch[slot] = win
                update_mask[slot] = True

            if np.any(update_mask):
                obs_tensor = torch.tensor(obs_batch, dtype=self.predictor.dtype, device=self.predictor.device)
                active_tensor = torch.as_tensor(update_mask, device=self.predictor.device)
                self.predictor.update(obs_tensor, active_mask=active_tensor)

                out = self.predictor.predict_next(active_mask=active_tensor, strict_ready=False)
                ready_slots = torch.nonzero(out.ready_mask & out.active_mask, as_tuple=False).flatten().tolist()

                for slot in ready_slots:
                    vid = self.slot_to_vid[int(slot)]
                    probs = torch.softmax(out.pred_logprob[slot].flatten(), dim=0).detach().cpu().numpy()
                    msg = {
                        "probs": probs,
                        "maneuver_labels": self.maneuver_labels,
                        "vehicle_id": int(vid),
                        "frame": int(frame_num),
                    }
                    try:
                        self.prediction_queue.put_nowait(msg)
                    except Exception:
                        pass

            for vid in active_vids:
                row = ego_rows_by_vid.get(vid, {}).get(frame_num)
                if row is None:
                    continue
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

                try:
                    self.pedal_queue.put_nowait(
                        {
                            "vehicle_id": int(vid),
                            "ax_val": ax_val,
                            "vx_val": vx_val,
                            "direction": int(direction),
                            "prev_y": prev_y,
                            "curr_y": curr_y,
                        }
                    )
                except Exception:
                    pass

            fig.suptitle(
                f"Recording {self.recording_id:02d} | Frame {frame_num} | Active {len(active_vids)}",
                fontsize=12,
            )

            fps_count += 1
            now = time.perf_counter()
            elapsed = now - fps_window_start
            if elapsed >= 1.0:
                print(f"[multi_render] {fps_count / elapsed:.2f} FPS")
                fps_count = 0
                fps_window_start = now

        self._ani = animation.FuncAnimation(
            fig,
            update,
            frames=range(min_frame, max_frame + 1),
            interval=40,
            repeat=True,
        )
        plt.show()
