import os
import pandas as pd
from road_renderer import RoadSceneRenderer
from vehicle_utils import get_test_vehicle_ids, select_meaningful_vehicles_in_test
import multiprocessing as mp
from combined_visualizer import visualizer_process


# Path to split.json
SPLIT_JSON_PATH = os.path.join(
    os.path.dirname(__file__),
    '..', 'hdv', 'models', 'main-model-sticky_S2_A4_hierarchical', 'split.json'
)

# User parameters for single vehicle simulation
SIMULATE_SINGLE_VEHICLE = False  # Set True to simulate a specific vehicle
SINGLE_REC_ID = 33                # Recording ID (int)
SINGLE_VEHICLE_ID = 1008            # Vehicle ID (int)


# User options for meaningful vehicle selection from test set
LANE_CHANGE = False      # Set True to include lane-changing vehicles
ACCEL_BRAKE = True      # Set True to include acceleration/braking vehicles
FOLLOWING = False       # Set True to include following vehicles
ACCEL_THRESHOLD = 5.0   # Threshold for acceleration/braking
MIN_FRAMES = 200        # Minimum frames for a vehicle



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
    test_vehicle_ids = get_test_vehicle_ids(SPLIT_JSON_PATH) # {recording_id: set of vehicle_ids}
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'hdv', 'data', 'highd') 

    # Start the combined visualizer in a separate process
    queue = mp.Queue() # Create a multiprocessing queue for communication between the main process and the visualizer process
    vis_process = mp.Process(target=visualizer_process, args=(queue,)) # Create the visualizer process, passing the queue as an argument
    vis_process.start() # Start the visualizer process

    try:
        # If SIMULATE_SINGLE_VEHICLE is True, simulate only that vehicle
        if SIMULATE_SINGLE_VEHICLE:
            from vehicle_utils import simulate_single_vehicle
            simulate_single_vehicle(
                SINGLE_REC_ID,
                SINGLE_VEHICLE_ID,
                data_dir,
                # Pass a lambda that creates a RoadSceneRenderer with the visualizer queue to the simulate_single_vehicle function, so it can send pedal/steering state updates to the visualizer
                lambda *args, **kwargs: RoadSceneRenderer(*args, **kwargs, visualizer_queue=queue), # Pass the visualizer queue to the renderer
                load_tracks_meta,
                load_recording_meta,
                load_tracks
            )
            return

        for rec in sorted(test_vehicle_ids.keys()): # Only process recordings that have test vehicles
            tracks_meta_path = os.path.join(data_dir, f"{rec:02d}_tracksMeta.csv")
            tracks_csv_path = os.path.join(data_dir, f"{rec:02d}_tracks.csv")
            recording_meta_path = os.path.join(data_dir, f"{rec:02d}_recordingMeta.csv")
            tracks_meta_df = load_tracks_meta(tracks_meta_path)
            recording_meta = load_recording_meta(recording_meta_path)
            renderer = RoadSceneRenderer(recording_meta, tracks_meta_df, visualizer_queue=queue) # Pass the visualizer queue to the renderer so it can send pedal/steering state updates
            # If all selection flags are False, render all test vehicles one by one
            if not (LANE_CHANGE or ACCEL_BRAKE or FOLLOWING):
                for vehicle_id in test_vehicle_ids[rec]:
                    print(f'Animating test vehicle {vehicle_id} in recording {rec:02d}...')
                    vehicle_tracks = load_tracks(tracks_csv_path, vehicle_id, tracks_meta_df) # Load tracks for the specific vehicle, using the tracks_meta_df to filter for the correct frame range
                    renderer.animate_scene(vehicle_tracks, test_vehicle_id=vehicle_id) # Animate the scene for the specific vehicle
                continue
            selected = select_meaningful_vehicles_in_test( # Select meaningful vehicles from the test set based on the specified criteria and flags
                tracks_meta_path, test_vehicle_ids,
                lane_change=LANE_CHANGE,
                accel_brake=ACCEL_BRAKE,
                following=FOLLOWING,
                accel_threshold=ACCEL_THRESHOLD,
                min_frames=MIN_FRAMES
            )
            if selected.empty:
                print(f"No meaningful vehicles found in test set for recording {rec:02d}.")
                continue
            flags = []
            if LANE_CHANGE:
                flags.append('LANE_CHANGE')
            if ACCEL_BRAKE:
                flags.append('ACCEL_BRAKE')
            if FOLLOWING:
                flags.append('FOLLOWING')
            flags_str = ', '.join(flags) if flags else 'None'
            print(f"Test vehicles for recording {rec:02d} with flags [{flags_str}]: {len(selected)}")
            for vehicle_id in selected['id']:
                print(f'Animating vehicle {vehicle_id}...')
                vehicle_tracks = load_tracks(tracks_csv_path, vehicle_id, tracks_meta_df)
                renderer.animate_scene(vehicle_tracks, test_vehicle_id=vehicle_id)
    finally:
        # Tell the visualizer process to quit and wait for it to finish
        queue.put('quit')
        vis_process.join() 

if __name__ == '__main__':
    main()
