import os
import sys
import multiprocessing as mp
import pandas as pd
from pathlib import Path

# Ensure workspace root is in sys.path for hdv imports
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from hdv.hdv_dbn.config import DBN_STATES
from hdv.hdv_dbn.trainer import HDVTrainer
from hdv.hdv_dbn.datasets.highd_loader import load_highd_folder
from hdv.hdv_dbn.prediction.semantic_label_utils import load_semantic_labels_from_yaml
from hdv.hdv_dbn.prediction.model_interface import HDVDbnModel
from hdv.hdv_dbn.prediction.online_predictor import OnlinePredictor
from road_renderer import RoadSceneRenderer
from vehicle_utils import get_test_vehicle_ids, select_meaningful_vehicles_in_test, get_vehicle_class, pick_classwise_scaler
from combined_visualizer import visualizer_process
from prediction_visualizer import prediction_visualizer_process


EXP_DIR = os.path.join(os.path.dirname(__file__), '..', 'hdv', 'models', 'main-model-sticky_S2_A4_hierarchical')
DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'hdv', 'data', 'highd')
CHECKPOINT_NAME = 'final.npz'

# User parameters for single vehicle simulation
SIMULATE_SINGLE_VEHICLE = True  # Set True to simulate a specific vehicle
SINGLE_REC_ID = 33                # Recording ID (int)
SINGLE_VEHICLE_ID = 1008            # Vehicle ID (int)


# User options for meaningful vehicle selection from test set
LANE_CHANGE = True      # Set True to include lane-changing vehicles
ACCEL_BRAKE = False      # Set True to include acceleration/braking vehicles
FOLLOWING = False       # Set True to include following vehicles
ACCEL_THRESHOLD = 5.0   # Threshold for acceleration/braking
MIN_FRAMES = 150        # Minimum frames for a vehicle


def load_tracks(tracks_csv_path, vehicle_id=None, tracks_meta_df=None):
    """
    Loads tracks CSV and optionally filters for a specific vehicle and frame range.
    
    parameters:
    - tracks_csv_path: Path to the tracks CSV file. 
    - vehicle_id: (Optional) ID of the vehicle to filter for. If None, loads all vehicles.
    - tracks_meta_df: (Optional) DataFrame containing metadata for all vehicles. Required if vehicle_id is provided to determine the frame range for that vehicle.

    returns:
    - DataFrame containing the tracks data, filtered by vehicle_id and frame range if specified.
    """
    df = pd.read_csv(tracks_csv_path)
    if vehicle_id is not None and tracks_meta_df is not None:
        meta = tracks_meta_df[tracks_meta_df['id'] == vehicle_id].iloc[0]
        initial_frame = meta['initialFrame']
        final_frame = meta['finalFrame']
        # Keep all vehicles, but only frames in the test vehicle's range
        df = df[(df['frame'] >= initial_frame) & (df['frame'] <= final_frame)]
    return df

def load_recording_meta(recording_meta_path):
    """
    Loads recording metadata from the specified CSV file and extracts lane markings and other relevant info.

    parameters:
    - recording_meta_path: Path to the recording metadata CSV file.

    returns:
    - Dictionary containing lane markings for the recording.
    """
    df = pd.read_csv(recording_meta_path)
    # Extract lane markings and other relevant info
    lane_markings_upper = [float(x) for x in df['upperLaneMarkings'].iloc[0].split(';')]
    lane_markings_lower = [float(x) for x in df['lowerLaneMarkings'].iloc[0].split(';')]
    return {
        'lane_markings_upper': lane_markings_upper,
        'lane_markings_lower': lane_markings_lower,
    }

def load_tracks_meta(tracks_meta_path):
    """
    Reads the tracks metadata CSV file and returns it as a DataFrame.

    parameters:
    - tracks_meta_path: Path to the tracks metadata CSV file.

    returns:
    - DataFrame containing the tracks metadata.
    """
    return pd.read_csv(tracks_meta_path)


def main():
    # Load maneuver/action labels from semantic_map.yaml
    S = len(DBN_STATES.driving_style)
    A = len(DBN_STATES.action)
    SEMANTIC_MAP_PATH = os.path.join(EXP_DIR, 'semantic_map.yaml')
    SPLIT_JSON_PATH = os.path.join(EXP_DIR, 'split.json')
    maneuver_labels = load_semantic_labels_from_yaml(SEMANTIC_MAP_PATH, S, A) # semantic labels for each (style, action) pair, used for annotation in the prediction visualizer
    test_vehicle_ids = get_test_vehicle_ids(SPLIT_JSON_PATH) # {recording_id: set of vehicle_ids} 

    # Start the combined visualizer for pedal and steering in a separate process
    pedal_steering_queue = mp.Queue() # Create a multiprocessing queue for communication between the main process and the visualizer process for pedal/steering updates
    vis_process = mp.Process(target=visualizer_process, args=(pedal_steering_queue,)) # Create the visualizer process, passing the queue as an argument
    vis_process.start() # Start the visualizer process

    # Start the prediction visualizer in a separate process
    prediction_queue = mp.Queue()
    prediction_vis_process = mp.Process(target=prediction_visualizer_process, args=(prediction_queue, maneuver_labels))
    prediction_vis_process.start()

    ckpt_path = os.path.join(EXP_DIR, CHECKPOINT_NAME) # Path to the model checkpoint file
    trainer = HDVTrainer.load(ckpt_path) # Load the HDVTrainer instance from the checkpoint, which includes the trained model and scalers for feature normalization
    model = HDVDbnModel(trainer) # Create an instance of the HDVDbnModel using the loaded trainer, which will be used for making online predictions during the simulation
    predictor = OnlinePredictor(model, warmup_steps=2) # Create an instance of the OnlinePredictor, which wraps the HDVDbnModel and provides an interface for making predictions in an online manner during the simulation

    try:
        if SIMULATE_SINGLE_VEHICLE: # If SIMULATE_SINGLE_VEHICLE is True, simulate only that vehicle instead of looping through all test vehicles in all recordings
            from single_vehicle import SingleVehicleSimulation

            tracks_meta_path = os.path.join(DATA_ROOT, f"{SINGLE_REC_ID:02d}_tracksMeta.csv")
            tracks_csv_path = os.path.join(DATA_ROOT, f"{SINGLE_REC_ID:02d}_tracks.csv")
            recording_meta_path = os.path.join(DATA_ROOT, f"{SINGLE_REC_ID:02d}_recordingMeta.csv")
            tracks_meta_df = load_tracks_meta(tracks_meta_path)
            recording_meta = load_recording_meta(recording_meta_path)
            vehicle_tracks = load_tracks(tracks_csv_path, SINGLE_VEHICLE_ID, tracks_meta_df) # Load the tracks for the single vehicle, filtering for the vehicle ID and frame range based on the tracks metadata
            renderer = RoadSceneRenderer(recording_meta, tracks_meta_df, visualizer_queue=pedal_steering_queue)
            vehicle_class = get_vehicle_class(tracks_meta_df, SINGLE_VEHICLE_ID)
            scaler_mean_vec = pick_classwise_scaler(trainer.scaler_mean, vehicle_class)
            scaler_std_vec  = pick_classwise_scaler(trainer.scaler_std,  vehicle_class)
            
            print(f"[main] vehicle_id={SINGLE_VEHICLE_ID} class='{vehicle_class}' "
                  f"scaler_mean_type={'dict' if isinstance(trainer.scaler_mean, dict) else 'array'} "
                  f"scaler_std_type={'dict' if isinstance(trainer.scaler_std, dict) else 'array'}")
            
            simulator = SingleVehicleSimulation(
                predictor=predictor,
                renderer=renderer,
                vehicle_id=SINGLE_VEHICLE_ID,
                recording_id=SINGLE_REC_ID,
                vehicle_tracks=vehicle_tracks,
                tracks_meta_df=tracks_meta_df,
                recording_meta=recording_meta,
                pedal_queue=pedal_steering_queue,
                prediction_queue=prediction_queue,
                maneuver_labels=maneuver_labels,
                scaler_mean=scaler_mean_vec,
                scaler_std=scaler_std_vec,
                vehicle_class=vehicle_class,
            )
            simulator.run()
            return

    finally:
        pedal_steering_queue.put('quit')
        vis_process.join()
        prediction_queue.put('quit')
        prediction_vis_process.join()

if __name__ == '__main__':
    main()
