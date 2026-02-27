import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

from frame_feature_engineer import OnlineFrameFeatureEngineer
from live_windowizer import LiveWindowizer
from hdv.hdv_dbn.prediction.online_predictor import OnlinePredictor
from road_renderer import RoadSceneRenderer
from hdv.hdv_dbn.config import FRAME_FEATURE_COLS, WINDOW_FEATURE_COLS, CONTINUOUS_FEATURES


class SingleVehicleSimulation:
	"""
	Simulates a single vehicle's trajectory and actions, rendering the scene and sending prediction updates to the visualizer.
	"""
	def __init__(self, *, predictor: OnlinePredictor, renderer: RoadSceneRenderer, 
			  vehicle_id, recording_id, vehicle_tracks, tracks_meta_df, recording_meta, pedal_queue: mp.Queue, prediction_queue: mp.Queue, 
			  maneuver_labels, scaler_mean, scaler_std, vehicle_class):
		
		self.predictor = predictor
		self.renderer = renderer
		
		self.vehicle_id = vehicle_id # test vehicle ID to simulate
		self.recording_id = recording_id
		self.vehicle_class = vehicle_class

		# Raw tracks used for rendering + pedal/steering (MUST stay raw)
		self.vehicle_tracks = vehicle_tracks # this contains the vehicles present in the same frames as the test vehicle

		# Metadata used for feature engineering (MUST stay unscaled)
		self.tracks_meta_df = tracks_meta_df 
		self.recording_meta = recording_meta 

		self.pedal_queue = pedal_queue
		self.prediction_queue = prediction_queue
		
		self.maneuver_labels = list(maneuver_labels)

		# Online frame feature engineering + windowizer
		self.frame_engineer = OnlineFrameFeatureEngineer()
		self.live_windowizer = LiveWindowizer()

		# Scaling config (match training)
		self.scale_idx = [i for i, name in enumerate(WINDOW_FEATURE_COLS) if name in CONTINUOUS_FEATURES]
		self.scaler_mean = np.asarray(scaler_mean)
		self.scaler_std = np.asarray(scaler_std)

	def _reset_simulation(self):
		self.predictor.reset()
		self.live_windowizer.reset()
		self.frame_engineer.reset()
		try:
			self.prediction_queue.put_nowait({"reset": True, "maneuver_labels": self.maneuver_labels})
		except Exception:
			pass
	
	def run(self):
		"""
		Runs the simulation for the single vehicle, animating the scene and sending prediction updates.
		"""
		self._reset_simulation() # Reset the predictor, windowizer, and feature engineer to clear any state before starting the animation loop

		raw_ego = self.vehicle_tracks[self.vehicle_tracks["id"] == self.vehicle_id].sort_values("frame") # Get the raw track data for the test vehicle, sorted by frame number to ensure correct temporal order for feature engineering and rendering
		frames = raw_ego["frame"].to_numpy(dtype=int) 
		if frames.size == 0:
			print(f"No frames for test vehicle {self.vehicle_id}")
			return

		min_frame, max_frame = int(frames[0]), int(frames[-1]) # minimum and maximum frame numbers for the test vehicle to set the animation range

		# Precompute frame lookups once to avoid repeated DataFrame scans inside update()
		frame_to_df = {int(frame): df for frame, df in self.vehicle_tracks.groupby("frame", sort=False)} # dict mapping frame number to DataFrame of all vehicles in that frame
		ego_rows = {
			int(row.frame): row
			for row in raw_ego[["frame", "x", "y", "xAcceleration", "xVelocity"]].itertuples(index=False)
		} # for quick access to the test vehicle's row in each frame, used for rendering and pedal/steering state extraction
		ego_y_by_frame = {frame: row.y for frame, row in ego_rows.items()} # for quick access to the test vehicle's y position by frame

		# Setup for animation
		fig, ax = plt.subplots(figsize=(10, 6))

		# Draw static road ONCE over a wide span so we only move the camera (xlim) per frame
		xmin = float(self.vehicle_tracks["x"].min()) - 50.0
		xmax = float(self.vehicle_tracks["x"].max()) + 50.0
		self.renderer.render_road(ax, xlims=(xmin, xmax))
		ylims = ax.get_ylim()
		ax.set_ylim(ylims)

		last_frame_num = None
		
		fps_count = 0
		fps_window_start = time.perf_counter()
		
		try:
			direction = int(self.renderer._meta_by_id.get(self.vehicle_id)[2]) # Get driving direction for the test vehicle from metadata cache (default to 1 if not found)
		except Exception:
			direction = self.renderer.tracks_meta_df[self.renderer.tracks_meta_df['id'] == self.vehicle_id]['drivingDirection'].values[0]
			print(f"Direction metadata not found for vehicle {self.vehicle_id}, cannot determine camera orientation. Check metadata cache.")

		window_width = 150 # width of the camera view in meters
		x_offset = 10 # offset from the test vehicle's position to place camera 

		def update(frame_num):
			nonlocal last_frame_num, fps_count, fps_window_start, direction
			if last_frame_num is not None and frame_num < last_frame_num: # reset if we loop back to the start
				self._reset_simulation()
			last_frame_num = frame_num # Update the last seen frame number for loop detection
			
			ego_row = ego_rows.get(frame_num) # Get current frame row for the test vehicle
			if ego_row is None:
				return
			
			# Rendering (raw tracks)
			x0 = float(ego_row.x)
			y0 = float(ego_row.y)
			
			# Set window so test vehicle is always at x=10, direction-dependent
			if direction == 2: 
				xlim = (x0 - x_offset, x0 - x_offset + window_width)
			else:
				xlim = (x0 + x_offset - window_width, x0 + x_offset)

			frame_df = frame_to_df.get(frame_num) # all vehicles in the current frame
			if frame_df is None:
				return
			self.renderer.render_vehicles(ax, frame_df, self.vehicle_id) # Render all vehicles in the current frame, highlighting the test vehicle

			# ONLINE feature engineering + windowization + prediction
			try:
				frame_vec = self.frame_engineer.add_frame(
					raw_frame_df=frame_df,
					ego_vehicle_id=self.vehicle_id,
					tracks_meta_df=self.tracks_meta_df,
					recording_meta=self.recording_meta,
					recording_id=self.recording_id,
				)  # Engineer features from full current frame and extract ego vector

				windows = self.live_windowizer.add_frame(frame_vec) # Add the engineered feature vector to the live windowizer and get any new windows that are ready for prediction
				if windows:
					win = windows[-1]  # Get the most recent window 
					win_scaled = win.copy()
					if self.scale_idx:
						win_scaled[self.scale_idx] = (
							(win_scaled[self.scale_idx] - self.scaler_mean[self.scale_idx])
							/ (self.scaler_std[self.scale_idx] + 1e-12)
						)

					win_tensor = torch.tensor(
						win_scaled, dtype=self.predictor.dtype, device=self.predictor.device
					)
					self.predictor.update(win_tensor)

					if self.predictor.is_ready:
						pred = self.predictor.predict_next()
						probs = torch.softmax(pred.pred_logprob.flatten(), dim=0).detach().cpu().numpy()
						msg = {"probs": probs, "maneuver_labels": self.maneuver_labels}
						try:
							self.prediction_queue.put_nowait(msg)
						except Exception:
							pass
			except Exception as e:
					print(f"[single_vehicle] Prediction update failed: {e}")

			# Pedal/steering state for visualizer
			raw_ax = float(ego_row.xAcceleration) if hasattr(ego_row, "xAcceleration") else 0.0
			raw_vx = float(ego_row.xVelocity) if hasattr(ego_row, "xVelocity") else 0.0
			if direction == 1:
				ax_val = -raw_ax
				vx_val = -raw_vx
			else:
				ax_val = raw_ax
				vx_val = raw_vx
			prev_frame = frame_num - 1
			prev_y = ego_y_by_frame.get(prev_frame, y0)
	
			# Send pedal/steering state to visualizer
			try:
				self.pedal_queue.put_nowait((ax_val, vx_val, direction, prev_y, y0))
			except Exception:
				pass

			ax.set_xlim(xlim)
			ax.set_title(f'Frame {frame_num} - Test Vehicle {self.vehicle_id}')

			fps_count += 1
			now = time.perf_counter()
			elapsed = now - fps_window_start
			if elapsed >= 1.0:
				print(f"[render] {fps_count / elapsed:.2f} FPS")
				fps_count = 0
				fps_window_start = now

		ani = animation.FuncAnimation(fig, update, frames=range(min_frame, max_frame+1), interval=40, repeat=True)
		plt.show()
