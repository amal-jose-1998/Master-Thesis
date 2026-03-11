from pathlib import Path
import pandas as pd

# Map highD column names to the generic internal names.
HIGHD_COL_MAP = {
    "id": "vehicle_id",
    "frame": "frame",
    "x": "x", 
    "y": "y",
    "width": "width",     
    "height": "height",
    "xVelocity": "vx",
    "yVelocity": "vy",
    "xAcceleration": "ax",
    "yAcceleration": "ay",
    "laneId": "lane_id",

    "frontSightDistance": "front_sight_dist",
    "backSightDistance": "back_sight_dist",
    "dhw": "dhw",
    "thw": "thw",
    "ttc": "ttc",

    "precedingXVelocity": "preceding_vx",
    "precedingId": "preceding_id",
    "followingId": "following_id",

    "leftPrecedingId": "left_preceding_id",
    "leftAlongsideId": "left_alongside_id",
    "leftFollowingId": "left_following_id",

    "rightPrecedingId": "right_preceding_id",
    "rightAlongsideId": "right_alongside_id",
    "rightFollowingId": "right_following_id",
}

def _read_recording_bundle(root, rec_id):
    """
    Read the three CSVs for one recording ID:
        {rec_id}_tracks.csv
        {rec_id}_tracksMeta.csv
        {rec_id}_recordingMeta.csv
    
    Parameters
    root : Path
        Directory containing the highD CSV files.
    rec_id : int
        Recording number (e.g. 1 for `01_tracks.csv`).
    
    Returns
    df_tracks : pd.DataFrame
        DataFrame from `{rec_id}_tracks.csv`.
    df_tracksmeta : pd.DataFrame
        DataFrame from `{rec_id}_tracksMeta.csv`.   
    df_recmeta : pd.DataFrame
        DataFrame from `{rec_id}_recordingMeta.csv`.
    """
    tracks_path = root / f"{rec_id:02d}_tracks.csv"
    tracksmeta_path = root / f"{rec_id:02d}_tracksMeta.csv"
    recmeta_path = root / f"{rec_id:02d}_recordingMeta.csv"

    if not tracks_path.exists():
        raise FileNotFoundError(tracks_path)
    if not tracksmeta_path.exists():
        raise FileNotFoundError(tracksmeta_path)
    if not recmeta_path.exists():
        raise FileNotFoundError(recmeta_path)

    df_tracks = pd.read_csv(tracks_path)
    df_tracksmeta = pd.read_csv(tracksmeta_path)
    df_recmeta = pd.read_csv(recmeta_path)

    return df_tracks, df_tracksmeta, df_recmeta
