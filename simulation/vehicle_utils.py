import json
import pandas as pd
import os

from road_renderer import RoadSceneRenderer

def get_test_vehicle_ids(split_json_path):
    """
    Reads the split JSON file and returns a dictionary mapping recording IDs to sets of test vehicle IDs.

    parameters: 
    - split_json_path: Path to the JSON file containing the train/test split information.

    returns:
    - Dictionary where keys are recording IDs (integers) and values are sets of vehicle IDs (integers) that are in the test set for that recording.
    """
    with open(split_json_path, 'r') as f:
        split = json.load(f)
    test_keys = split['keys']['test']
    test_ids = {}
    for key in test_keys:
        rec, vid = key.split(':')
        rec = int(float(rec))
        vid = int(float(vid))
        if rec not in test_ids:
            test_ids[rec] = set()
        test_ids[rec].add(vid)
    return test_ids # {recording_id: set of vehicle_ids}

def select_meaningful_vehicles_in_test(tracks_meta_path, test_vehicle_ids, lane_change=True, accel_brake=True, following=False, accel_threshold=2.0, min_frames=150):
    """
    Selects meaningful vehicles from the test set based on specified criteria and flags.

    parameters:
    - tracks_meta_path: Path to the tracks metadata CSV file for a specific recording.  
    - test_vehicle_ids: Dictionary mapping recording IDs to sets of test vehicle IDs, as returned by get_test_vehicle_ids().
    - lane_change: If True, include vehicles that performed lane changes.
    - accel_brake: If True, include vehicles that showed significant acceleration or braking (change in speed above accel_threshold).
    - following: If True, include vehicles that were mostly following (not changing lanes and not showing significant accel/brake).
    - accel_threshold: Threshold for change in speed to consider a vehicle as accelerating or braking.
    - min_frames: Minimum number of frames a vehicle must have to be considered.

    returns:
    - DataFrame containing the selected vehicles from the test set based on the specified criteria.
    """
    df = pd.read_csv(tracks_meta_path)
    rec_id = int(os.path.basename(tracks_meta_path).split('_')[0])
    if rec_id not in test_vehicle_ids:
        return pd.DataFrame()
    df = df[df['id'].isin(test_vehicle_ids[rec_id])]
    selected = pd.DataFrame()
    if lane_change:
        lane_changers = df[(df['numLaneChanges'] > 0) & (df['numFrames'] >= min_frames)]
        selected = pd.concat([selected, lane_changers])
    if accel_brake:
        accel_vehicles = df[(df['numLaneChanges'] == 0) &
                            (abs(df['maxXVelocity'] - df['minXVelocity']) > accel_threshold) &
                            (df['numFrames'] >= min_frames)]
        selected = pd.concat([selected, accel_vehicles])
    if following:
        followers = df[(df['numLaneChanges'] == 0) &
                       (abs(df['maxXVelocity'] - df['minXVelocity']) <= accel_threshold) &
                       (df['numFrames'] >= min_frames)]
        selected = pd.concat([selected, followers])
    selected = selected.drop_duplicates(subset=['id'])
    return selected


def simulate_single_vehicle(rec, vehicle_id, data_dir, renderer_class, load_tracks_meta, load_recording_meta, load_tracks):
    """
    Simulate a particular vehicle from a particular recording. 
    This function loads the necessary data for the specified vehicle and recording, initializes the renderer, 
    and runs the animation for that vehicle.

    parameters:
    - rec: Recording ID (integer) of the vehicle to simulate.
    - vehicle_id: Vehicle ID (integer) of the vehicle to simulate.
    - data_dir: Directory where the data files are located.
    - renderer_class: The class of the renderer to use for visualization (e.g., RoadSceneRenderer).
    - load_tracks_meta: Function to load the tracks metadata DataFrame from a given path.
    - load_recording_meta: Function to load the recording metadata from a given path.
    - load_tracks: Function to load the tracks DataFrame for a specific vehicle, given the tracks CSV path, vehicle ID, and tracks metadata DataFrame.
    """
    tracks_meta_path = os.path.join(data_dir, f"{rec:02d}_tracksMeta.csv")
    tracks_csv_path = os.path.join(data_dir, f"{rec:02d}_tracks.csv")
    recording_meta_path = os.path.join(data_dir, f"{rec:02d}_recordingMeta.csv")
    tracks_meta_df = load_tracks_meta(tracks_meta_path)
    recording_meta = load_recording_meta(recording_meta_path)
    print(f'Animating test vehicle {vehicle_id} in recording {rec:02d}...')
    vehicle_tracks = load_tracks(tracks_csv_path, vehicle_id, tracks_meta_df)
    renderer: RoadSceneRenderer = renderer_class(recording_meta, tracks_meta_df)
    renderer.animate_scene(vehicle_tracks, test_vehicle_id=vehicle_id)


def get_vehicle_class(tracks_meta_df, vehicle_id):
    """
    Retrieves the vehicle class for a given vehicle ID from the tracks metadata DataFrame. 
    If the vehicle ID is not found or the class information is missing, it returns None.
    
    parameters:
    - tracks_meta_df: DataFrame containing the metadata for all vehicles, including their IDs and classes.
    - vehicle_id: The ID of the vehicle for which to retrieve the class.
    returns:
    - The vehicle class as a lowercase string if found, otherwise None.
    """
    row = tracks_meta_df[tracks_meta_df["id"] == int(vehicle_id)]
    if row.empty:
        raise ValueError(f"Vehicle ID {vehicle_id} not found in tracks_meta_df")
    if "class" in row.columns:
        return str(row["class"].iloc[0]).strip().lower() # Return the vehicle class as a lowercase string, or None if it's NaN
    return None

def pick_classwise_scaler(scalar_dict, vehicle_class):
    """
    Picks the appropriate scaler (mean or std) for a vehicle based on its class.

    parameters:
    - scalar_dict: Either a single scaler value (if not class-specific) or a dictionary mapping vehicle classes to scalers.
    - vehicle_class: The class of the vehicle for which to pick the scaler.

    returns:
    - The scaler value corresponding to the vehicle's class
    """
    vc = vehicle_class.strip().lower()
    if vc in scalar_dict:
        return scalar_dict[vc]

    raise ValueError(f"Vehicle class '{vehicle_class}' not found in scaler dict keys: {list(scalar_dict.keys())}")