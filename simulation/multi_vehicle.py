import torch
from live_windowizer import LiveWindowizer

class MultiVehicleSimulation:
    def __init__(self, trainer, predictor, prediction_queue, renderer, select_meaningful_vehicles_in_test, load_tracks, WINDOW_CONFIG, WINDOW_FEATURE_COLS, CONTINUOUS_FEATURES):
        self.trainer = trainer
        self.predictor = predictor
        self.prediction_queue = prediction_queue
        self.renderer = renderer
        self.select_meaningful_vehicles_in_test = select_meaningful_vehicles_in_test
        self.load_tracks = load_tracks
        self.WINDOW_CONFIG = WINDOW_CONFIG
        self.WINDOW_FEATURE_COLS = WINDOW_FEATURE_COLS
        self.CONTINUOUS_FEATURES = CONTINUOUS_FEATURES

    def run(self, rec, tracks_meta_path, tracks_csv_path, recording_meta, tracks_meta_df, test_vehicle_ids, LANE_CHANGE, ACCEL_BRAKE, FOLLOWING, ACCEL_THRESHOLD, MIN_FRAMES):
        # If all selection flags are False, render all test vehicles one by one
        if not (LANE_CHANGE or ACCEL_BRAKE or FOLLOWING):
            vehicle_ids = test_vehicle_ids[rec]
        else:
            selected = self.select_meaningful_vehicles_in_test(
                tracks_meta_path, test_vehicle_ids,
                lane_change=LANE_CHANGE,
                accel_brake=ACCEL_BRAKE,
                following=FOLLOWING,
                accel_threshold=ACCEL_THRESHOLD,
                min_frames=MIN_FRAMES
            )
            if selected.empty:
                print(f"No meaningful vehicles found in test set for recording {rec:02d}.")
                return
            flags = []
            if LANE_CHANGE:
                flags.append('LANE_CHANGE')
            if ACCEL_BRAKE:
                flags.append('ACCEL_BRAKE')
            if FOLLOWING:
                flags.append('FOLLOWING')
            flags_str = ', '.join(flags) if flags else 'None'
            print(f"Test vehicles for recording {rec:02d} with flags [{flags_str}]: {len(selected)}")
            vehicle_ids = selected['id']

        from live_windowizer import rename_and_check_features
        for vehicle_id in vehicle_ids:
            print(f'Animating vehicle {vehicle_id} in recording {rec:02d}...')
            vehicle_tracks = self.load_tracks(tracks_csv_path, vehicle_id, tracks_meta_df)
            vehicle_tracks = rename_and_check_features(
                vehicle_tracks,
                tracks_meta_df=tracks_meta_df,
                recording_meta=recording_meta,
                recording_id=rec,
            )

            # --- Setup windowizer and predictor for each vehicle ---
            W = self.WINDOW_CONFIG.W
            stride = self.WINDOW_CONFIG.stride
            obs_names = list(vehicle_tracks.columns)
            windowizer = LiveWindowizer(W=W, stride=stride, obs_names=obs_names)

            feature_cols = list(self.WINDOW_FEATURE_COLS)
            cont_set = set(self.CONTINUOUS_FEATURES)
            scale_idx = [i for i, name in enumerate(feature_cols) if name in cont_set]
            scaler_mean = self.trainer.scaler_mean if not isinstance(self.trainer.scaler_mean, dict) else list(self.trainer.scaler_mean.values())[0]
            scaler_std = self.trainer.scaler_std if not isinstance(self.trainer.scaler_std, dict) else list(self.trainer.scaler_std.values())[0]

            self.predictor.reset()

            # --- Animate and predict (live, one window per frame) ---
            for _, frame in vehicle_tracks.iterrows():
                windows = windowizer.add_frame(frame.values)
                if not windows:
                    continue
                win = windows[-1]
                win_scaled = win.copy()
                win_scaled[scale_idx] = (win_scaled[scale_idx] - scaler_mean[scale_idx]) / scaler_std[scale_idx]
                win_tensor = torch.tensor(win_scaled, dtype=self.predictor.dtype, device=self.predictor.device)
                self.predictor.update(win_tensor)
                if self.predictor.is_ready:
                    try:
                        pred = self.predictor.predict_next()
                        probs = torch.exp(pred.pred_logprob).cpu().numpy().flatten()
                        self.prediction_queue.put({'probs': probs})
                    except Exception as e:
                        print(f"Prediction error: {e}")
            self.renderer.animate_scene(vehicle_tracks, test_vehicle_id=vehicle_id)
