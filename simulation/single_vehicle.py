import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from live_windowizer import LiveWindowizer, OnlineFrameFeatureEngineer
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
		
		self.vehicle_id = vehicle_id
		self.recording_id = recording_id
		self.vehicle_class = vehicle_class

		# Raw tracks used for rendering + pedal/steering (MUST stay raw)
		self.vehicle_tracks: pd.DataFrame = vehicle_tracks

		self.tracks_meta_df = tracks_meta_df
		self.recording_meta = recording_meta

		self.pedal_queue = pedal_queue
		self.prediction_queue = prediction_queue
		self.maneuver_labels = list(maneuver_labels)

		# Online frame feature engineering + windowizer
		self.obs_names = list(FRAME_FEATURE_COLS)

		self.frame_engineer = OnlineFrameFeatureEngineer(
			tracks_meta_df=self.tracks_meta_df,
			recording_meta=self.recording_meta,
			recording_id=self.recording_id,
			buffer_len=200,
			strict=False,
		)

		self.live_windowizer = LiveWindowizer(obs_names=self.obs_names)

		# Scaling config (match training)
		feature_cols = list(WINDOW_FEATURE_COLS)
		cont_set = set(CONTINUOUS_FEATURES)
		self.scale_idx = [i for i, name in enumerate(feature_cols) if name in cont_set]

		self.scaler_mean = np.asarray(scaler_mean)
		self.scaler_std = np.asarray(scaler_std)

	
	def run(self):
		"""
		Runs the simulation for the single vehicle, animating the scene and sending prediction updates.
		"""
		raw_ego = self.vehicle_tracks[self.vehicle_tracks["id"] == self.vehicle_id].sort_values("frame")
		frames = raw_ego["frame"].to_numpy(dtype=int)
		if frames.size == 0:
			print(f"No frames for test vehicle {self.vehicle_id}")
			return

		min_frame, max_frame = int(frames[0]), int(frames[-1]) # Get the minimum and maximum frame numbers for the test vehicle to set the animation range

		upper = self.renderer.recording_meta["lane_markings_upper"]
		lower = self.renderer.recording_meta["lane_markings_lower"]
		road_top = upper[0]
		road_bottom = lower[-1]
		y_center = (road_top + road_bottom) / 2

		window_width = 150
		window_height = 50
		x_offset = 10

		def _reset_prediction_state():
			self.predictor.reset()
			self.live_windowizer.reset()
			self.frame_engineer.reset()
			try:
				self.prediction_queue.put_nowait({"reset": True, "maneuver_labels": self.maneuver_labels})
			except Exception:
				pass
		
		_reset_prediction_state() # Reset all online state before starting animation

		# Setup for animation
		fig, ax = plt.subplots(figsize=(10, 6))

		# Draw static road ONCE over a wide span so we only move the camera (xlim) per frame
		xmin = float(self.vehicle_tracks["x"].min()) - 50.0
		xmax = float(self.vehicle_tracks["x"].max()) + 50.0
		self.renderer.render_road(ax, xlim=(xmin, xmax), ylim=(y_center - window_height/2, y_center + window_height/2))

		last_frame_num = None
		
		def _safe_get_direction():
			"""Safely get the driving direction for the test vehicle"""
			try:
				d = self.renderer.tracks_meta_df[self.renderer.tracks_meta_df["id"] == self.vehicle_id]["drivingDirection"].values
				if len(d) > 0:
					return int(d[0])
			except Exception:
				print(f"Could not get driving direction for vehicle {self.vehicle_id}")
		

		def update(frame_num):
			nonlocal last_frame_num
			if last_frame_num is not None and frame_num < last_frame_num:
				_reset_prediction_state()
			last_frame_num = frame_num
			
			tv_row: pd.DataFrame = self.vehicle_tracks[ # Get the row for the test vehicle at the current frame number
				(self.vehicle_tracks["id"] == self.vehicle_id) & (self.vehicle_tracks["frame"] == frame_num)
			]
			if tv_row.empty:
				return
			
			# Rendering (raw tracks)
			x0 = float(tv_row["x"].values[0])
			y0 = float(tv_row["y"].values[0])
			direction = _safe_get_direction()
			
			# Set window so test vehicle is always at x=10, direction-dependent
			if direction == 2: 
				xlim = (x0 - x_offset, x0 - x_offset + window_width)
			else:
				xlim = (x0 + x_offset - window_width, x0 + x_offset)
				
			ylim = (y_center - window_height/2, y_center + window_height/2) # Set ylim so that lower y is at the top (image coordinates)

			frame_df = self.vehicle_tracks[self.vehicle_tracks["frame"] == frame_num] 
			self.renderer.render_vehicles(ax, frame_df, self.vehicle_id) # Render all vehicles in the current frame, highlighting the test vehicle

			# ONLINE feature engineering + windowization + prediction
			try:
				raw_row_dict = tv_row.iloc[0].to_dict() # Convert the row for the test vehicle at the current frame to a dictionary for feature engineering
				frame_vec = self.frame_engineer.push_row(raw_row_dict)  # Engineer features for the current frame and get the feature vector
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
			raw_ax = tv_row['xAcceleration'].values[0] if 'xAcceleration' in tv_row else 0
			raw_vx = tv_row['xVelocity'].values[0] if 'xVelocity' in tv_row else 0
			if direction == 1:
				ax_val = -raw_ax
				vx_val = -raw_vx
			else:
				ax_val = raw_ax
				vx_val = raw_vx
			prev_frame = frame_num - 1
			prev_tv_row: pd.DataFrame = self.vehicle_tracks[(self.vehicle_tracks['id'] == self.vehicle_id) & (self.vehicle_tracks['frame'] == prev_frame)] 
			if not prev_tv_row.empty:
				prev_y = prev_tv_row['y'].values[0]
			else:
				prev_y = y0
	
			# Send pedal/steering state to visualizer
			try:
				self.pedal_queue.put_nowait((ax_val, vx_val, direction, prev_y, y0))
			except Exception:
				pass

			ax.set_xlim(xlim)
			ax.set_ylim(ylim)
			ax.invert_yaxis() # Invert y-axis so that lower y is at the top (image coordinates)
			ax.set_title(f'Frame {frame_num} - Test Vehicle {self.vehicle_id}')

		ani = animation.FuncAnimation(fig, update, frames=range(min_frame, max_frame+1), interval=40, repeat=True)
		plt.show()
