import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from pedal_visualizer import PedalVisualizer
from steering_visualizer import SteeringVisualizer

class RoadSceneRenderer:
    def __init__(self, recording_meta, tracks_meta_df, lane_color='white', road_color="#807D7D", median_color="#BDDA90", visualizer_queue=None):
        self.recording_meta = recording_meta
        self.tracks_meta_df = tracks_meta_df
        self.lane_color = lane_color
        self.road_color = road_color
        self.median_color = median_color
        self.pedal_visualizer = PedalVisualizer()
        self.steering_visualizer = SteeringVisualizer()
        self.visualizer_queue = visualizer_queue  # If set, send pedal/steering state here

    def render_road(self, ax, xlim=(0, 1000), ylim=None):
        offset = 0.5  # meters
        upper = self.recording_meta['lane_markings_upper']
        lower = self.recording_meta['lane_markings_lower']
        road_top = upper[0]
        road_bottom = lower[-1]
        median_top = upper[-1]
        median_bottom = lower[0]

        # Set y-limits so that lower y is at the top (image coordinates)
        ylims = (road_top - 2*offset, road_bottom + 2*offset) if ylim is None else ylim

        # Fill top margin with median color
        ax.fill_between(xlim, ylims[0], road_top-offset, color=self.median_color, zorder=0)
        # Fill bottom margin with median color
        ax.fill_between(xlim, road_bottom+offset, ylims[1], color=self.median_color, zorder=0)

        # Fill the main road area
        ax.fill_between(xlim, road_top-offset, road_bottom+offset, color=self.road_color, zorder=1)
        # Draw median
        ax.fill_between(xlim, median_top+offset, median_bottom-offset, color=self.median_color, zorder=2)

        # Draw lane markings (solid white on top)
        for i, y in enumerate(upper):
            if i == 0 or i == len(upper) - 1:
                ax.axhline(y, color=self.lane_color, linestyle='-', linewidth=3, zorder=4)
            else:
                ax.axhline(y, color=self.lane_color, linestyle='--', linewidth=2, zorder=3)
        for i, y in enumerate(lower):
            if i == len(lower) - 1 or i == 0:
                ax.axhline(y, color=self.lane_color, linestyle='-', linewidth=3, zorder=4)
            else:
                ax.axhline(y, color=self.lane_color, linestyle='--', linewidth=2, zorder=3)
        ax.set_xlim(xlim)
        ax.set_ylim(ylims)
        ax.invert_yaxis()  # Keep inverted to match highD convention
        ax.set_aspect('equal')
        ax.set_xlabel('x (meters)')
        ax.set_ylabel('y (meters)')
        ax.set_title('Road Scene')

    def render_vehicles(self, ax, tracks_df, test_vehicle_id=None):
        for vid, vehicle_data in tracks_df.groupby('id'):
            vehicle_meta = self.tracks_meta_df[self.tracks_meta_df['id'] == vid].iloc[0]
            width = vehicle_meta['width']
            height = vehicle_meta['height']
            direction = vehicle_meta['drivingDirection']
            x0 = vehicle_data['x'].iloc[0]
            y0 = vehicle_data['y'].iloc[0]
            if vid == test_vehicle_id:
                color = 'red'
            elif direction == 1:
                color = 'blue'
            else:
                color = 'green'
            rect = Rectangle((x0, y0), width, height, edgecolor=color, facecolor="none", lw=2, zorder=3)
            ax.add_patch(rect)

    def animate_scene(self, tracks_df, test_vehicle_id, window_width=150, window_height=50, x_offset=10):
        # Get all frames for the test vehicle
        test_vehicle_frames = tracks_df[tracks_df['id'] == test_vehicle_id]['frame'].values
        if len(test_vehicle_frames) == 0:
            print(f"No frames for test vehicle {test_vehicle_id}")
            return
        min_frame, max_frame = test_vehicle_frames[0], test_vehicle_frames[-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        upper = self.recording_meta['lane_markings_upper']
        lower = self.recording_meta['lane_markings_lower']
        road_top = upper[0]
        road_bottom = lower[-1]
        y_center = (road_top + road_bottom) / 2
        # Only create pedal/steering axes if not using external visualizer
        if self.visualizer_queue is None:
            pedal_ax = fig.add_axes([0.7, 0.05, 0.25, 0.18])  # [left, bottom, width, height]
            pedal_ax.axis('off')
        def update(frame_num):
            ax.clear()
            ax.set_aspect('auto')  # Allow zooming and minimize margins
            # Get test vehicle position at this frame
            tv_row = tracks_df[(tracks_df['id'] == test_vehicle_id) & (tracks_df['frame'] == frame_num)]
            if tv_row.empty:
                return
            x0 = tv_row['x'].values[0]
            y0 = tv_row['y'].values[0]
            direction = self.tracks_meta_df[self.tracks_meta_df['id'] == test_vehicle_id]['drivingDirection'].values[0]
            # Set window so test vehicle is always at x=10, direction-dependent
            if direction == 2:
                xlim = (x0 - x_offset, x0 - x_offset + window_width)
            else:
                xlim = (x0 + x_offset - window_width, x0 + x_offset)
            # Set ylim so that lower y is at the top (image coordinates)
            ylim = (y_center - window_height/2, y_center + window_height/2)
            self.render_road(ax, xlim, ylim)
            # Draw all vehicles at this frame 
            self.render_vehicles(ax, tracks_df[tracks_df['frame'] == frame_num], test_vehicle_id)
            # Compute pedal/steering state
            raw_ax = tv_row['xAcceleration'].values[0] if 'xAcceleration' in tv_row else 0
            raw_vx = tv_row['xVelocity'].values[0] if 'xVelocity' in tv_row else 0
            if direction == 1:
                ax_val = -raw_ax
                vx_val = -raw_vx
            else:
                ax_val = raw_ax
                vx_val = raw_vx
            prev_frame = frame_num - 1
            prev_tv_row = tracks_df[(tracks_df['id'] == test_vehicle_id) & (tracks_df['frame'] == prev_frame)]
            if not prev_tv_row.empty:
                prev_y = prev_tv_row['y'].values[0]
            else:
                prev_y = y0
            # If using external visualizer, send state to queue
            if self.visualizer_queue is not None:
                try:
                    self.visualizer_queue.put_nowait((ax_val, vx_val, direction, prev_y, y0))
                except Exception:
                    pass
            else:
                self.pedal_visualizer.draw(pedal_ax, ax_val, vx_val)
                steering_ax = fig.add_axes([0.7, 0.25, 0.25, 0.18])
                steering_ax.axis('off')
                self.steering_visualizer.draw(steering_ax, direction, prev_y, y0)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.set_xlabel('x (meters)')
            ax.set_ylabel('y (meters)')
            ax.set_title(f'Frame {frame_num} - Test Vehicle {test_vehicle_id}')
        ani = animation.FuncAnimation(fig, update, frames=range(min_frame, max_frame+1), interval=40, repeat=True)
        plt.show()
