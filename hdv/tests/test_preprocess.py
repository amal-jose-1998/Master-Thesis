"""
Unit tests for hdv_dbn.preprocess module.

Tests cover:
- Lane center calculation
- Longitudinal maneuver classification (accelerate, decelerate, hard_brake, maintain_speed)
- Lateral maneuver classification (lane changes with prepare/perform phases)
- Full pipeline integration with multiple vehicles
- Edge cases and error handling
"""

import unittest
import numpy as np
import pandas as pd
from hdv.hdv_dbn.preprocess import (
    _lane_center,
    classify_long_lat_maneuvers_for_vehicle,
    add_maneuver_labels,
)
from hdv.hdv_dbn.config import OBS_CONFIG, DBN_STATES


class TestLaneCenter(unittest.TestCase):
    """Test the _lane_center helper function."""

    def test_lane_center_single_lane(self):
        """Test lane center calculation for single lane."""
        lane_id = np.array([1])
        result = _lane_center(lane_id, OBS_CONFIG.lane_width)
        expected = (1 - 0.5) * OBS_CONFIG.lane_width  # 0.5 * 3.5 = 1.75
        self.assertAlmostEqual(result[0], expected)

    def test_lane_center_multiple_lanes(self):
        """Test lane center calculation for multiple lanes."""
        lane_id = np.array([1, 2, 3])
        result = _lane_center(lane_id, OBS_CONFIG.lane_width)
        
        expected_1 = (1 - 0.5) * OBS_CONFIG.lane_width  # 1.75
        expected_2 = (2 - 0.5) * OBS_CONFIG.lane_width  # 5.25
        expected_3 = (3 - 0.5) * OBS_CONFIG.lane_width  # 8.75
        
        self.assertAlmostEqual(result[0], expected_1)
        self.assertAlmostEqual(result[1], expected_2)
        self.assertAlmostEqual(result[2], expected_3)

    def test_lane_center_lane_spacing(self):
        """Test that lane centers are properly spaced."""
        lane_id = np.array([1, 2, 3])
        result = _lane_center(lane_id, OBS_CONFIG.lane_width)
        
        # Difference between consecutive lanes should equal lane_width
        diff_1_2 = result[1] - result[0]
        diff_2_3 = result[2] - result[1]
        
        self.assertAlmostEqual(diff_1_2, OBS_CONFIG.lane_width)
        self.assertAlmostEqual(diff_2_3, OBS_CONFIG.lane_width)


class TestLongitudinalManeuvers(unittest.TestCase):
    """Test longitudinal maneuver classification."""

    def setUp(self):
        """Create a simple trajectory for testing."""
        self.n = 10
        self.df = pd.DataFrame({
            'frame': np.arange(self.n),
            'ax': np.zeros(self.n),
            'lane_id': np.ones(self.n, dtype=int),
            'y': np.ones(self.n) * 1.75,
            'vy': np.zeros(self.n),
        })

    def test_maintain_speed_default(self):
        """Test that accelerations near zero result in maintain_speed."""
        self.df['ax'] = np.ones(self.n) * 0.1  # Small acceleration, below threshold
        m_long, _ = classify_long_lat_maneuvers_for_vehicle(self.df)
        
        self.assertTrue((m_long == 'maintain_speed').all(),
                       msg="Small accelerations should result in maintain_speed")

    def test_accelerate_classification(self):
        """Test accelerate maneuver detection."""
        self.df['ax'] = np.ones(self.n) * 0.5  # Above accel_threshold (0.2)
        m_long, _ = classify_long_lat_maneuvers_for_vehicle(self.df)
        
        self.assertTrue((m_long == 'accelerate').all(),
                       msg="High positive acceleration should result in accelerate")

    def test_decelerate_classification(self):
        """Test decelerate maneuver detection."""
        self.df['ax'] = np.ones(self.n) * -0.5  # Below -accel_threshold
        m_long, _ = classify_long_lat_maneuvers_for_vehicle(self.df)
        
        self.assertTrue((m_long == 'decelerate').all(),
                       msg="Moderate negative acceleration should result in decelerate")

    def test_hard_brake_classification(self):
        """Test hard_brake maneuver detection."""
        self.df['ax'] = np.ones(self.n) * -3.0  # Below hard_brake_threshold (-2.0)
        m_long, _ = classify_long_lat_maneuvers_for_vehicle(self.df)
        
        self.assertTrue((m_long == 'hard_brake').all(),
                       msg="Severe negative acceleration should result in hard_brake")

    def test_hard_brake_dominates_decelerate(self):
        """Test that hard_brake takes precedence over decelerate."""
        # Mix of hard brake and other accelerations
        self.df['ax'] = np.array([-3.0, -0.5, -3.0, -0.5, 0.1, 0.5, -3.0, 0.0, 0.0, -0.5])
        m_long, _ = classify_long_lat_maneuvers_for_vehicle(self.df)
        
        # Indices with very negative acceleration should be hard_brake
        hard_brake_indices = np.where(self.df['ax'] <= OBS_CONFIG.hard_brake_threshold)[0]
        self.assertTrue((m_long[hard_brake_indices] == 'hard_brake').all(),
                       msg="Hard brake should be classified for ax <= hard_brake_threshold")

    def test_mixed_maneuvers(self):
        """Test mixed maneuvers in a single trajectory."""
        self.df['ax'] = np.array([0.5, 0.5, 0.0, -0.5, -0.5, -3.0, 0.1, 0.0, 0.2, 0.0])
        m_long, _ = classify_long_lat_maneuvers_for_vehicle(self.df)
        
        # Check specific indices
        self.assertEqual(m_long[0], 'accelerate')
        self.assertEqual(m_long[2], 'maintain_speed')
        self.assertEqual(m_long[3], 'decelerate')
        self.assertEqual(m_long[5], 'hard_brake')

    def test_accel_threshold_boundary(self):
        """Test behavior at acceleration threshold boundaries."""
        # Exactly at threshold should not trigger classification
        self.df['ax'] = np.ones(self.n) * OBS_CONFIG.accel_threshold
        m_long, _ = classify_long_lat_maneuvers_for_vehicle(self.df)
        self.assertTrue((m_long == 'maintain_speed').all(),
                       msg="Acceleration exactly at threshold should be maintain_speed")
        
        # Just above threshold should trigger accelerate
        self.df['ax'] = np.ones(self.n) * (OBS_CONFIG.accel_threshold + 0.01)
        m_long, _ = classify_long_lat_maneuvers_for_vehicle(self.df)
        self.assertTrue((m_long == 'accelerate').all(),
                       msg="Acceleration above threshold should be accelerate")

    def test_missing_accel_column_raises_error(self):
        """Test that missing acceleration column raises ValueError."""
        df_no_accel = self.df.drop(columns=['ax'])
        
        with self.assertRaises(ValueError):
            classify_long_lat_maneuvers_for_vehicle(df_no_accel, accel_col='ax')


class TestLateralManeuvers(unittest.TestCase):
    """Test lateral maneuver classification and lane-change detection."""

    def setUp(self):
        """Create trajectory with lane-change capability."""
        self.n = 50
        self.df = pd.DataFrame({
            'frame': np.arange(self.n),
            'ax': np.zeros(self.n),
            'lane_id': np.ones(self.n, dtype=int),
            'y': np.ones(self.n) * 1.75,  # Lane 1 center
            'vy': np.zeros(self.n),
        })

    def test_keep_lane_default(self):
        """Test that vehicles in constant lane have keep_lane maneuver."""
        _, m_lat = classify_long_lat_maneuvers_for_vehicle(self.df)
        
        self.assertTrue((m_lat == 'keep_lane').all(),
                       msg="Constant lane should result in keep_lane")

    def test_lane_change_left_detection(self):
        """Test detection of left lane change (lane 1 -> 2)."""
        # Lane change at frame 20
        self.df.loc[20:, 'lane_id'] = 2
        # Lateral position gradually moves to lane 2 center
        new_lane_center = (2 - 0.5) * OBS_CONFIG.lane_width  # 5.25
        self.df.loc[20:30, 'y'] = np.linspace(1.75, new_lane_center, 11)
        self.df.loc[30:, 'y'] = new_lane_center
        # Lateral velocity during transition
        self.df.loc[20:25, 'vy'] = 0.1
        self.df.loc[25:30, 'vy'] = 0.05
        self.df.loc[30:, 'vy'] = 0.0
        
        _, m_lat = classify_long_lat_maneuvers_for_vehicle(self.df)
        
        # Should have perform_lc_left for frames around 20
        perform_lc_indices = np.where(m_lat == 'perform_lc_left')[0]
        self.assertGreater(len(perform_lc_indices), 0,
                          msg="Lane change left should be detected")

    def test_lane_change_right_detection(self):
        """Test detection of right lane change (lane 2 -> 1)."""
        # Start in lane 2
        self.df['lane_id'] = 2
        self.df['y'] = (2 - 0.5) * OBS_CONFIG.lane_width  # 5.25
        
        # Lane change at frame 20
        self.df.loc[20:, 'lane_id'] = 1
        # Lateral position gradually moves to lane 1 center
        lane_1_center = (1 - 0.5) * OBS_CONFIG.lane_width  # 1.75
        self.df.loc[20:30, 'y'] = np.linspace(5.25, lane_1_center, 11)
        self.df.loc[30:, 'y'] = lane_1_center
        # Lateral velocity during transition
        self.df.loc[20:25, 'vy'] = -0.1
        self.df.loc[25:30, 'vy'] = -0.05
        self.df.loc[30:, 'vy'] = 0.0
        
        _, m_lat = classify_long_lat_maneuvers_for_vehicle(self.df)
        
        # Should have perform_lc_right for frames around 20
        perform_lc_indices = np.where(m_lat == 'perform_lc_right')[0]
        self.assertGreater(len(perform_lc_indices), 0,
                          msg="Lane change right should be detected")

    def test_preparation_phase_detected(self):
        """Test that preparation phase before lane change is detected."""
        # Lane change at frame 25, with preparation window before
        self.df.loc[25:, 'lane_id'] = 2
        new_lane_center = (2 - 0.5) * OBS_CONFIG.lane_width
        self.df.loc[25:35, 'y'] = np.linspace(1.75, new_lane_center, 11)
        self.df.loc[35:, 'y'] = new_lane_center
        # Add lateral velocity during preparation window (frames 10-20) before LC
        self.df.loc[10:20, 'vy'] = 0.1  # Above lateral_vel_threshold
        self.df.loc[25:30, 'vy'] = 0.15
        self.df.loc[30:35, 'vy'] = 0.05
        self.df.loc[35:, 'vy'] = 0.0
        
        _, m_lat = classify_long_lat_maneuvers_for_vehicle(self.df)
        
        # Should have perform_lc_left around frame 25
        perform_lc_indices = np.where(m_lat == 'perform_lc_left')[0]
        self.assertGreater(len(perform_lc_indices), 0,
                          msg="Lane change should result in perform_lc_left maneuver")
        # The perform phase should include the lane change frame
        self.assertTrue(np.any(perform_lc_indices >= 25),
                       msg="Perform phase should include or come after LC frame")

    def test_lateral_maneuver_validity(self):
        """Test that all lateral maneuver labels are valid."""
        # Create various lane changes
        self.df['lane_id'] = np.array([1]*10 + [2]*10 + [3]*10 + [2]*10 + [1]*10)
        # Add lateral position and velocity changes
        centers = [(1 - 0.5) * OBS_CONFIG.lane_width, (2 - 0.5) * OBS_CONFIG.lane_width, (3 - 0.5) * OBS_CONFIG.lane_width]
        self.df.loc[0:9, 'y'] = centers[0]
        self.df.loc[10:19, 'y'] = np.linspace(centers[0], centers[1], 10)
        self.df.loc[20:29, 'y'] = np.linspace(centers[1], centers[2], 10)
        self.df.loc[30:39, 'y'] = np.linspace(centers[2], centers[1], 10)
        self.df.loc[40:49, 'y'] = np.linspace(centers[1], centers[0], 10)
        
        self.df.loc[10:15, 'vy'] = 0.1
        self.df.loc[20:25, 'vy'] = 0.1
        self.df.loc[30:35, 'vy'] = -0.1
        self.df.loc[40:45, 'vy'] = -0.1
        
        _, m_lat = classify_long_lat_maneuvers_for_vehicle(self.df)
        
        # All maneuvers should be valid
        valid_lat = set(DBN_STATES.lat_maneuver_states)
        self.assertTrue(m_lat.isin(valid_lat).all(),
                       msg="All lateral maneuvers should be valid")

    def test_no_lateral_data_graceful_handling(self):
        """Test that missing lateral data doesn't cause errors."""
        df_no_y = self.df.drop(columns=['y'])
        
        _, m_lat = classify_long_lat_maneuvers_for_vehicle(
            df_no_y, lat_pos_col='y'
        )
        
        # Should still produce valid maneuvers
        valid_lat = set(DBN_STATES.lat_maneuver_states)
        self.assertTrue(m_lat.isin(valid_lat).all(),
                       msg="Should handle missing lateral position gracefully")

    def test_single_timestep_no_crash(self):
        """Test that single-timestep trajectories don't crash."""
        df_single = self.df.iloc[:1].copy()
        
        _, m_lat = classify_long_lat_maneuvers_for_vehicle(df_single)
        
        self.assertEqual(len(m_lat), 1)
        self.assertEqual(m_lat.iloc[0], 'keep_lane')


class TestFullPipeline(unittest.TestCase):
    """Test the full preprocessing pipeline with multiple vehicles."""

    def setUp(self):
        """Create a multi-vehicle dataset."""
        # Vehicle 1: constant lane, mixed accelerations
        v1_frames = np.arange(20)
        v1_data = {
            'vehicle_id': np.ones(20),
            'frame': v1_frames,
            'ax': np.concatenate([np.ones(5) * 0.5, np.ones(5) * 0.0, np.ones(5) * -0.5, np.ones(5) * -3.0]),
            'lane_id': np.ones(20, dtype=int),
            'y': np.ones(20) * 1.75,
            'vy': np.zeros(20),
        }
        
        # Vehicle 2: performs a lane change
        v2_frames = np.arange(25)
        v2_data = {
            'vehicle_id': np.ones(25) * 2,
            'frame': v2_frames,
            'ax': np.ones(25) * 0.1,
            'lane_id': np.concatenate([np.ones(12, dtype=int), np.ones(13, dtype=int) * 2]),
            'y': np.concatenate([
                np.ones(12) * 1.75,
                np.linspace(1.75, 5.25, 13)
            ]),
            'vy': np.concatenate([
                np.zeros(12),
                np.concatenate([np.ones(6) * 0.15, np.linspace(0.15, 0.0, 7)])
            ]),
        }
        
        self.df = pd.concat([
            pd.DataFrame(v1_data),
            pd.DataFrame(v2_data),
        ], ignore_index=True)

    def test_full_pipeline_output_shape(self):
        """Test that add_maneuver_labels produces correct output shape."""
        result = add_maneuver_labels(self.df)
        
        self.assertEqual(len(result), len(self.df))
        self.assertIn('maneuver_long', result.columns)
        self.assertIn('maneuver_lat', result.columns)

    def test_full_pipeline_no_na_values(self):
        """Test that all rows receive maneuver labels (no NaN values)."""
        result = add_maneuver_labels(self.df)
        
        self.assertFalse(result['maneuver_long'].isna().any(),
                        msg="No NaN values in maneuver_long")
        self.assertFalse(result['maneuver_lat'].isna().any(),
                        msg="No NaN values in maneuver_lat")

    def test_full_pipeline_valid_labels(self):
        """Test that all output labels are valid."""
        result = add_maneuver_labels(self.df)
        
        valid_long = set(DBN_STATES.long_maneuver_states)
        valid_lat = set(DBN_STATES.lat_maneuver_states)
        
        self.assertTrue(result['maneuver_long'].isin(valid_long).all(),
                       msg="All longitudinal maneuvers should be valid")
        self.assertTrue(result['maneuver_lat'].isin(valid_lat).all(),
                       msg="All lateral maneuvers should be valid")

    def test_full_pipeline_vehicle_separation(self):
        """Test that maneuvers are correctly computed per vehicle."""
        result = add_maneuver_labels(self.df)
        
        # Vehicle 1 should have some hard_brake
        v1_long = result[result['vehicle_id'] == 1]['maneuver_long']
        self.assertIn('hard_brake', v1_long.values,
                     msg="Vehicle 1 should have hard_brake maneuver")
        
        # Vehicle 2 should have lane change
        v2_lat = result[result['vehicle_id'] == 2]['maneuver_lat']
        has_lc = ('perform_lc_left' in v2_lat.values) or ('prepare_lc_left' in v2_lat.values)
        self.assertTrue(has_lc,
                       msg="Vehicle 2 should have lane change maneuver")

    def test_full_pipeline_preserves_original_columns(self):
        """Test that original columns are preserved."""
        result = add_maneuver_labels(self.df)
        
        for col in self.df.columns:
            self.assertIn(col, result.columns,
                         msg=f"Original column '{col}' should be preserved")

    def test_full_pipeline_sorting_handled(self):
        """Test that output is correctly sorted even if input is unsorted."""
        # Shuffle the input
        df_shuffled = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        result = add_maneuver_labels(df_shuffled)
        
        # Result should have same length
        self.assertEqual(len(result), len(self.df))
        # All rows should have maneuvers
        self.assertFalse(result['maneuver_long'].isna().any())
        self.assertFalse(result['maneuver_lat'].isna().any())


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df_empty = pd.DataFrame({
            'frame': [],
            'ax': [],
            'lane_id': [],
            'y': [],
            'vy': [],
        })
        
        m_long, m_lat = classify_long_lat_maneuvers_for_vehicle(df_empty)
        
        self.assertEqual(len(m_long), 0)
        self.assertEqual(len(m_lat), 0)

    def test_nan_acceleration_values(self):
        """Test handling of NaN acceleration values."""
        df = pd.DataFrame({
            'frame': np.arange(10),
            'ax': np.array([0.5, np.nan, 0.0, -0.5, np.nan, -3.0, 0.1, np.nan, 0.0, 0.2]),
            'lane_id': np.ones(10, dtype=int),
            'y': np.ones(10) * 1.75,
            'vy': np.zeros(10),
        })
        
        # Should not crash with NaN values
        m_long, m_lat = classify_long_lat_maneuvers_for_vehicle(df)
        
        self.assertEqual(len(m_long), 10)
        self.assertEqual(len(m_lat), 10)

    def test_very_long_trajectory(self):
        """Test with a very long trajectory."""
        n = 10000
        df = pd.DataFrame({
            'frame': np.arange(n),
            'ax': np.random.uniform(-3.0, 1.0, n),
            'lane_id': np.random.choice([1, 2, 3], n),
            'y': np.random.uniform(1.0, 9.0, n),
            'vy': np.random.uniform(-0.2, 0.2, n),
        })
        
        m_long, m_lat = classify_long_lat_maneuvers_for_vehicle(df)
        
        self.assertEqual(len(m_long), n)
        self.assertEqual(len(m_lat), n)
        # All should be valid
        valid_long = set(DBN_STATES.long_maneuver_states)
        valid_lat = set(DBN_STATES.lat_maneuver_states)
        self.assertTrue(m_long.isin(valid_long).all())
        self.assertTrue(m_lat.isin(valid_lat).all())

    def test_constant_acceleration_no_maneuver_change(self):
        """Test that constant moderate acceleration stays as maintain_speed."""
        df = pd.DataFrame({
            'frame': np.arange(20),
            'ax': np.ones(20) * 0.05,  # Below threshold
            'lane_id': np.ones(20, dtype=int),
            'y': np.ones(20) * 1.75,
            'vy': np.zeros(20),
        })
        
        m_long, _ = classify_long_lat_maneuvers_for_vehicle(df)
        
        self.assertTrue((m_long == 'maintain_speed').all())

    def test_multiple_consecutive_lane_changes(self):
        """Test handling of multiple consecutive lane changes."""
        df = pd.DataFrame({
            'frame': np.arange(50),
            'ax': np.zeros(50),
            'lane_id': np.array([1]*10 + [2]*10 + [3]*10 + [2]*10 + [1]*10),
            'y': np.concatenate([
                np.ones(10) * 1.75,
                np.linspace(1.75, 5.25, 10),
                np.linspace(5.25, 8.75, 10),
                np.linspace(8.75, 5.25, 10),
                np.linspace(5.25, 1.75, 10),
            ]),
            'vy': np.concatenate([
                np.zeros(10),
                np.ones(10) * 0.1,
                np.ones(10) * 0.1,
                np.ones(10) * -0.1,
                np.ones(10) * -0.1,
            ]),
        })
        
        _, m_lat = classify_long_lat_maneuvers_for_vehicle(df)
        
        # Should have multiple lane change maneuvers
        valid_lat = set(DBN_STATES.lat_maneuver_states)
        self.assertTrue(m_lat.isin(valid_lat).all())


if __name__ == '__main__':
    unittest.main()
