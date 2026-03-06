"""
main - entry point for running single-vehicle or multi-vehicle simulations with the trained HDV-DBN model on the HighD dataset. 
Configurable options allow for selecting specific vehicles, applying meaningful selection criteria, and choosing between simultaneous 
multi-vehicle simulation or sequential single-vehicle simulations.
"""
import sys
from pathlib import Path
import pandas as pd

# Ensure workspace root is in sys.path for hdv imports
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from vehicle_utils import (
    get_test_vehicle_ids,
    select_meaningful_vehicles_in_test,
    get_vehicle_class,
    pick_classwise_scaler,
)
from hdv.hdv_dbn.config import DBN_STATES
from hdv.hdv_dbn.trainer import HDVTrainer
from hdv.hdv_dbn.prediction.semantic_label_utils import load_semantic_labels_from_yaml
from hdv.hdv_dbn.prediction.model_interface import HDVDbnModel
from hdv.hdv_dbn.prediction.online_predictor import OnlinePredictor
from road_renderer import RoadSceneRenderer
from single_vehicle import SingleVehicleSimulation
from multi_vehicle import MultiVehicleSimulation

# Paths
EXP_DIR = WORKSPACE_ROOT / "hdv" / "models" / "main-model-sticky_S2_A4_hierarchical"
DATA_ROOT = WORKSPACE_ROOT / "hdv" / "data" / "highd"
CHECKPOINT_NAME = "final.npz"

# User parameters for single vehicle simulation
SIMULATE_SINGLE_VEHICLE = False
SINGLE_REC_ID = 33
SINGLE_VEHICLE_ID = 1008

# Multi-vehicle options
SIMULATE_MULTI_VEHICLES_SIMULTANEOUS = True # if False, will simulate vehicles sequentially (one at a time) even within the same recording
MAX_SIM_VEHICLES = 2  # set None to disable cap

# Meaningful selection options
LANE_CHANGE = False
ACCEL_BRAKE = False
FOLLOWING = False
ACCEL_THRESHOLD = 5.0
MIN_FRAMES = 150


def build_recording_paths(rec_id):
    """
    Builds paths to tracks meta, tracks CSV, and recording meta files for a recording.
    """
    tracks_meta_path = DATA_ROOT / f"{rec_id:02d}_tracksMeta.csv"
    tracks_csv_path = DATA_ROOT / f"{rec_id:02d}_tracks.csv"
    recording_meta_path = DATA_ROOT / f"{rec_id:02d}_recordingMeta.csv"
    return tracks_meta_path, tracks_csv_path, recording_meta_path


def load_tracks_meta(tracks_meta_path):
    """Loads tracks metadata CSV into a DataFrame."""
    return pd.read_csv(tracks_meta_path)


def load_recording_meta(recording_meta_path):
    """
    Loads recording metadata and extracts lane markings.
    """
    df = pd.read_csv(recording_meta_path)
    up_raw = str(df["upperLaneMarkings"].iloc[0]) if "upperLaneMarkings" in df.columns else ""
    lo_raw = str(df["lowerLaneMarkings"].iloc[0]) if "lowerLaneMarkings" in df.columns else ""

    def _parse_marks(s: str):
        parts = [p for p in s.split(";") if p.strip() != ""]
        out = []
        for p in parts:
            try:
                out.append(float(p))
            except Exception:
                continue
        return out

    return {
        "upperLaneMarkings": _parse_marks(up_raw),
        "lowerLaneMarkings": _parse_marks(lo_raw),
    }


def load_tracks(tracks_csv_path, *, frame_min=None, frame_max=None):
    """
    Loads tracks CSV and optionally filters by frame range (inclusive).
    """
    df = pd.read_csv(tracks_csv_path)
    if frame_min is not None:
        df = df[df["frame"] >= int(frame_min)]
    if frame_max is not None:
        df = df[df["frame"] <= int(frame_max)]
    return df


def select_vehicle_ids_for_recording(rec_id, tracks_meta_path, test_vehicle_ids):
    """
    Selects vehicle IDs to simulate for a recording based on configured flags.
    """
    if rec_id not in test_vehicle_ids:
        return []

    # default: all test vehicles
    if not (LANE_CHANGE or ACCEL_BRAKE or FOLLOWING):
        return sorted(int(v) for v in test_vehicle_ids[rec_id])

    selected = select_meaningful_vehicles_in_test(
        str(tracks_meta_path),
        test_vehicle_ids,
        lane_change=LANE_CHANGE,
        accel_brake=ACCEL_BRAKE,
        following=FOLLOWING,
        accel_threshold=ACCEL_THRESHOLD,
        min_frames=MIN_FRAMES,
    )
    if selected.empty:
        return []

    return sorted(selected["id"].astype(int).tolist())


def _classwise_scaler_for_vehicle(trainer: HDVTrainer, tracks_meta_df, vehicle_id):
    vehicle_class = get_vehicle_class(tracks_meta_df, vehicle_id)
    mean_vec = pick_classwise_scaler(trainer.scaler_mean, vehicle_class)
    std_vec = pick_classwise_scaler(trainer.scaler_std, vehicle_class)
    return vehicle_class, mean_vec, std_vec


def main():
    # Load maneuver/action labels
    S = len(DBN_STATES.driving_style)
    A = len(DBN_STATES.action)
    semantic_map_path = EXP_DIR / "semantic_map.yaml"
    split_json_path = EXP_DIR / "split.json"
    maneuver_labels = load_semantic_labels_from_yaml(str(semantic_map_path), S, A)

    # Test split vehicle ids
    test_vehicle_ids = get_test_vehicle_ids(str(split_json_path))

    # Load trainer + model
    ckpt_path = EXP_DIR / CHECKPOINT_NAME
    trainer = HDVTrainer.load(str(ckpt_path))
    model = HDVDbnModel(trainer)

    # -------------------------
    # Single specified vehicle
    # -------------------------
    if SIMULATE_SINGLE_VEHICLE:
        tracks_meta_path, tracks_csv_path, recording_meta_path = build_recording_paths(SINGLE_REC_ID)
        if not (tracks_meta_path.exists() and tracks_csv_path.exists() and recording_meta_path.exists()):
            print(f"[main] Missing files for recording {SINGLE_REC_ID:02d}.")
            return

        tracks_meta_df = load_tracks_meta(tracks_meta_path)
        recording_meta = load_recording_meta(recording_meta_path)

        # frame range for this vehicle (so neighbors exist but only within ego lifespan)
        meta_row = tracks_meta_df[tracks_meta_df["id"] == SINGLE_VEHICLE_ID].iloc[0]
        frame_min = int(meta_row["initialFrame"])
        frame_max = int(meta_row["finalFrame"])
        tracks_df = load_tracks(tracks_csv_path, frame_min=frame_min, frame_max=frame_max)

        renderer = RoadSceneRenderer(recording_meta, tracks_meta_df)

        vehicle_class, scaler_mean_vec, scaler_std_vec = _classwise_scaler_for_vehicle(trainer, tracks_meta_df, SINGLE_VEHICLE_ID)
        print(
            f"[main] single vehicle_id={SINGLE_VEHICLE_ID} rec={SINGLE_REC_ID:02d} "
            f"class='{vehicle_class}'"
        )

        simulator = SingleVehicleSimulation(
            predictor=OnlinePredictor(model, warmup_steps=5),
            renderer=renderer,
            vehicle_id=SINGLE_VEHICLE_ID,
            recording_id=SINGLE_REC_ID,
            vehicle_tracks=tracks_df,
            tracks_meta_df=tracks_meta_df,
            recording_meta=recording_meta,
            maneuver_labels=maneuver_labels,
            scaler_mean=scaler_mean_vec,
            scaler_std=scaler_std_vec,
            vehicle_class=vehicle_class,
        )
        simulator.run()
        return

    # -------------------------
    # Loop recordings in test split
    # -------------------------
    for rec in sorted(test_vehicle_ids.keys()):
        tracks_meta_path, tracks_csv_path, recording_meta_path = build_recording_paths(rec)
        if not (tracks_meta_path.exists() and tracks_csv_path.exists() and recording_meta_path.exists()):
            print(f"[main] Skipping recording {rec:02d}: missing one or more data files.")
            continue

        tracks_meta_df = load_tracks_meta(tracks_meta_path)
        recording_meta = load_recording_meta(recording_meta_path)

        vehicle_ids = select_vehicle_ids_for_recording(rec, tracks_meta_path, test_vehicle_ids)
        if not vehicle_ids:
            print(f"[main] No matching test vehicles selected for recording {rec:02d}.")
            continue

        if MAX_SIM_VEHICLES is not None and len(vehicle_ids) > int(MAX_SIM_VEHICLES):
            original_count = len(vehicle_ids)
            vehicle_ids = vehicle_ids[: int(MAX_SIM_VEHICLES)]
            print(
                f"[main] Recording {rec:02d}: limiting vehicles "
                f"from {original_count} to {len(vehicle_ids)} (MAX_SIM_VEHICLES={MAX_SIM_VEHICLES})"
            )

        print(f"[main] Recording {rec:02d}: simulating {len(vehicle_ids)} vehicle(s)")

        # -------------------------
        # Simultaneous batched simulation
        # -------------------------
        if SIMULATE_MULTI_VEHICLES_SIMULTANEOUS:
            scaler_mean_by_vehicle = {}
            scaler_std_by_vehicle = {}
            valid_vehicle_ids = []

            for vehicle_id in vehicle_ids:
                try:
                    _, mean_vec, std_vec = _classwise_scaler_for_vehicle(trainer, tracks_meta_df, vehicle_id)
                    scaler_mean_by_vehicle[int(vehicle_id)] = mean_vec
                    scaler_std_by_vehicle[int(vehicle_id)] = std_vec
                    valid_vehicle_ids.append(int(vehicle_id))
                except Exception as e:
                    print(f"[main] Skipping vehicle {vehicle_id} in recording {rec:02d}: {e}")

            if not valid_vehicle_ids:
                print(f"[main] Recording {rec:02d}: no valid vehicles after scaler/class checks.")
                continue

            # Restrict tracks to union lifespan of selected vehicles for efficiency
            meta_sel = tracks_meta_df[tracks_meta_df["id"].isin(valid_vehicle_ids)]
            frame_min = int(meta_sel["initialFrame"].min())
            frame_max = int(meta_sel["finalFrame"].max())
            tracks_df = load_tracks(tracks_csv_path, frame_min=frame_min, frame_max=frame_max)

            print(
                f"[main] Recording {rec:02d}: simultaneous multi-vehicle mode with "
                f"{len(valid_vehicle_ids)} vehicle(s) (frames {frame_min}..{frame_max})"
            )

            renderer = RoadSceneRenderer(recording_meta, tracks_meta_df)
            simulator = MultiVehicleSimulation(
                model=model,
                renderer=renderer,
                recording_id=rec,
                vehicle_ids=valid_vehicle_ids,
                vehicle_tracks=tracks_df,
                tracks_meta_df=tracks_meta_df,
                recording_meta=recording_meta,
                maneuver_labels=maneuver_labels,
                scaler_mean_by_vehicle=scaler_mean_by_vehicle,
                scaler_std_by_vehicle=scaler_std_by_vehicle,
                warmup_steps=5,
            )
            simulator.run()
            continue

        # -------------------------
        # Sequential single-vehicle simulations
        # -------------------------
        for vehicle_id in vehicle_ids:
            try:
                vehicle_class, scaler_mean_vec, scaler_std_vec = _classwise_scaler_for_vehicle(
                    trainer, tracks_meta_df, vehicle_id
                )
            except Exception as e:
                print(f"[main] Skipping vehicle {vehicle_id} in recording {rec:02d}: {e}")
                continue

            meta_row = tracks_meta_df[tracks_meta_df["id"] == vehicle_id].iloc[0]
            frame_min = int(meta_row["initialFrame"])
            frame_max = int(meta_row["finalFrame"])
            tracks_df = load_tracks(tracks_csv_path, frame_min=frame_min, frame_max=frame_max)

            if tracks_df.empty:
                print(f"[main] Skipping vehicle {vehicle_id} in recording {rec:02d}: no tracks found.")
                continue

            print(f"[main] Simulating vehicle {vehicle_id} in recording {rec:02d} (class='{vehicle_class}')")

            renderer = RoadSceneRenderer(recording_meta, tracks_meta_df)
            simulator = SingleVehicleSimulation(
                predictor=OnlinePredictor(model, warmup_steps=5),
                renderer=renderer,
                vehicle_id=vehicle_id,
                recording_id=rec,
                vehicle_tracks=tracks_df,
                tracks_meta_df=tracks_meta_df,
                recording_meta=recording_meta,
                maneuver_labels=maneuver_labels,
                scaler_mean=scaler_mean_vec,
                scaler_std=scaler_std_vec,
                vehicle_class=vehicle_class,
            )
            simulator.run()


if __name__ == "__main__":
    main()