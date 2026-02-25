import json
import pandas as pd
import os

def get_test_vehicle_ids(split_json_path):
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

def select_meaningful_vehicles_in_test(tracks_meta_path, test_vehicle_ids, lane_change=True, accel_brake=True, following=False, accel_threshold=2.0, min_frames=100):
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
    """
    tracks_meta_path = os.path.join(data_dir, f"{rec:02d}_tracksMeta.csv")
    tracks_csv_path = os.path.join(data_dir, f"{rec:02d}_tracks.csv")
    recording_meta_path = os.path.join(data_dir, f"{rec:02d}_recordingMeta.csv")
    tracks_meta_df = load_tracks_meta(tracks_meta_path)
    recording_meta = load_recording_meta(recording_meta_path)
    print(f'Animating test vehicle {vehicle_id} in recording {rec:02d}...')
    vehicle_tracks = load_tracks(tracks_csv_path, vehicle_id, tracks_meta_df)
    renderer = renderer_class(recording_meta, tracks_meta_df)
    renderer.animate_scene(vehicle_tracks, test_vehicle_id=vehicle_id)
