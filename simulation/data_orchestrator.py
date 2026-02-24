import os
import json
import pandas as pd
from road_renderer import RoadSceneRenderer

# Path to split.json
SPLIT_JSON_PATH = os.path.join(
    os.path.dirname(__file__),
    '..', 'hdv', 'models', 'main-model-sticky_S2_A4_hierarchical', 'split.json'
)

# Select a recording
RECORDING_ID = '04'  # Change as needed
TRACKS_CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    '..', 'hdv', 'data', 'highd', f'{RECORDING_ID}_tracks.csv'
)

# Load split.json and extract test vehicles for a recording from the 'test' group
def load_test_vehicles(split_json_path, recording_id, group='test'):
    with open(split_json_path, 'r') as f:
        split_data = json.load(f)
    test_keys = split_data['keys'][group]
    # Extract vehicle IDs for the selected recording
    vehicles = []
    for key in test_keys:
        rec_id, veh_id = key.split(':')
        if float(rec_id) == float(recording_id):
            vehicles.append(int(float(veh_id)))  # Convert to int
    return vehicles

# Load tracks CSV
def load_tracks(tracks_csv_path, vehicle_id=None, tracks_meta_df=None):
    df = pd.read_csv(tracks_csv_path)
    if vehicle_id is not None and tracks_meta_df is not None:
        meta = tracks_meta_df[tracks_meta_df['id'] == vehicle_id].iloc[0]
        initial_frame = meta['initialFrame']
        final_frame = meta['finalFrame']
        # Keep all vehicles, but only frames in the test vehicle's range
        df = df[(df['frame'] >= initial_frame) & (df['frame'] <= final_frame)]
    return df

# Load recording and tracks metadata
def load_recording_meta(recording_meta_path):
    df = pd.read_csv(recording_meta_path)
    # Extract lane markings and other relevant info
    lane_markings_upper = [float(x) for x in df['upperLaneMarkings'].iloc[0].split(';')]
    lane_markings_lower = [float(x) for x in df['lowerLaneMarkings'].iloc[0].split(';')]
    return {
        'lane_markings_upper': lane_markings_upper,
        'lane_markings_lower': lane_markings_lower,
    }

def load_tracks_meta(tracks_meta_path):
    return pd.read_csv(tracks_meta_path)


# Main 
def main():
    test_vehicles = load_test_vehicles(SPLIT_JSON_PATH, RECORDING_ID)
    tracks_csv_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'hdv', 'data', 'highd', f'{RECORDING_ID}_tracks.csv'
    )
    recording_meta_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'hdv', 'data', 'highd', f'{RECORDING_ID}_recordingMeta.csv'
    )
    tracks_meta_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'hdv', 'data', 'highd', f'{RECORDING_ID}_tracksMeta.csv'
    )
    tracks_meta_df = load_tracks_meta(tracks_meta_path)
    recording_meta = load_recording_meta(recording_meta_path)
    
    print(f'Test vehicles for recording {RECORDING_ID}: ', len(test_vehicles))
    print(f'Lane markings from recording meta:', recording_meta)
    
    renderer = RoadSceneRenderer(recording_meta, tracks_meta_df)
    for vehicle_id in test_vehicles:
        print(f'Animating vehicle {vehicle_id}...')
        vehicle_tracks = load_tracks(tracks_csv_path, vehicle_id, tracks_meta_df)
        renderer.animate_scene(vehicle_tracks, test_vehicle_id=vehicle_id)


if __name__ == '__main__':
    main()
