import json
import pandas as pd
import os

def get_test_vehicle_ids(split_json_path):
    with open(split_json_path, 'r') as f:
        split = json.load(f)
    test_keys = split['keys']['test']
    # Parse keys like '13.0:1219.0' into (recording_id, vehicle_id)
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
    # Only keep vehicles in test set for this recording
    if rec_id not in test_vehicle_ids:
        return pd.DataFrame()  # No test vehicles in this recording
    df = df[df['id'].isin(test_vehicle_ids[rec_id])] # Keep only test vehicles in this recording
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

if __name__ == "__main__":
    split_json_path = r"C:\Users\amalj\OneDrive\Desktop\Master's Thesis\Implementation\hdv\models\main-model-sticky_S2_A4_hierarchical\split.json"
    test_vehicle_ids = get_test_vehicle_ids(split_json_path)
    data_dir = r"C:\Users\amalj\OneDrive\Desktop\Master's Thesis\Implementation\hdv\data\highd"
    all_selected = []
    for rec in test_vehicle_ids:
        tracks_meta_path = os.path.join(data_dir, f"{rec:02d}_tracksMeta.csv")
        selected = select_meaningful_vehicles_in_test(tracks_meta_path, test_vehicle_ids)
        if not selected.empty:
            selected['recording'] = rec
            all_selected.append(selected)
    if all_selected:
        result = pd.concat(all_selected)
        print(result[['recording', 'id', 'numLaneChanges', 'minXVelocity', 'maxXVelocity', 'numFrames']])
    else:
        print("No meaningful vehicles found in test set.")
