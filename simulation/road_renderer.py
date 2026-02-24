import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

class RoadSceneRenderer:
    def __init__(self, recording_meta, tracks_meta_df, lane_color='white', road_color="#807D7D", median_color="#BDDA90"):
        self.recording_meta = recording_meta
        self.tracks_meta_df = tracks_meta_df
        self.lane_color = lane_color
        self.road_color = road_color
        self.median_color = median_color

    def render_road(self, ax, xlim=(0, 1000)):
        offset = 0.5  # meters
        upper = self.recording_meta['lane_markings_upper']
        lower = self.recording_meta['lane_markings_lower']
        road_top = upper[0]
        road_bottom = lower[-1]
        median_top = upper[-1]
        median_bottom = lower[0]

        # Fill the main road area
        ax.fill_between(xlim, road_top-offset, road_bottom+offset, color=self.road_color, zorder=0)
        # Draw median
        ax.fill_between(xlim, median_top+offset, median_bottom-offset, color=self.median_color, zorder=1)

        # Draw lane markings (solid white on top)
        for i, y in enumerate(upper):
            if i == 0 or i == len(upper) - 1:
                ax.axhline(y, color=self.lane_color, linestyle='-', linewidth=3, zorder=3)
            else:
                ax.axhline(y, color=self.lane_color, linestyle='--', linewidth=2, zorder=2)
        for i, y in enumerate(lower):
            if i == len(lower) - 1 or i == 0:
                ax.axhline(y, color=self.lane_color, linestyle='-', linewidth=3, zorder=3)
            else:
                ax.axhline(y, color=self.lane_color, linestyle='--', linewidth=2, zorder=2)
        # Set y-limits so that lower y is at the top (image coordinates)
        ylims = (road_top - 2*offset, road_bottom + 2*offset)
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
                z = 3
            elif direction == 1:
                color = 'blue'
                z = 2
            else:
                color = 'green'
                z = 2
            rect = Rectangle((x0, y0), width, height, edgecolor=color, facecolor='none', lw=2, zorder=z)
            ax.add_patch(rect)

    def render_scene(self, tracks_df, test_vehicle_id=None, xlim=None):
        # Automatically set xlim based on min/max x in tracks_df if not provided
        if xlim is None:
            min_x = tracks_df['x'].min()
            max_x = tracks_df['x'].max()
            xlim = (min_x - 10, max_x + 10)  # Add some padding
        fig, ax = plt.subplots()
        self.render_road(ax, xlim)
        self.render_vehicles(ax, tracks_df, test_vehicle_id)
        plt.show()
        input("Press Enter to continue...")

    def animate_scene(self, tracks_df, test_vehicle_id, window_width=100, window_height=50, x_offset=10):
        # Get all frames for the test vehicle
        test_vehicle_frames = tracks_df[tracks_df['id'] == test_vehicle_id]['frame'].values
        if len(test_vehicle_frames) == 0:
            print(f"No frames for test vehicle {test_vehicle_id}")
            return
        min_frame, max_frame = test_vehicle_frames[0], test_vehicle_frames[-1]
        fig, ax = plt.subplots()
        upper = self.recording_meta['lane_markings_upper']
        lower = self.recording_meta['lane_markings_lower']
        road_top = upper[0]
        road_bottom = lower[-1]
        y_center = (road_top + road_bottom) / 2
        def update(frame_num):
            ax.clear()
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
            self.render_road(ax, xlim)
            # Draw all vehicles at this frame
            for vid, vehicle_data in tracks_df[tracks_df['frame'] == frame_num].groupby('id'):
                vehicle_meta = self.tracks_meta_df[self.tracks_meta_df['id'] == vid].iloc[0]
                width = vehicle_meta['width']
                height = vehicle_meta['height']
                direction_v = vehicle_meta['drivingDirection']
                vx = vehicle_data['x'].values[0]
                vy = vehicle_data['y'].values[0]
                if vid == test_vehicle_id:
                    color = 'red'
                    z = 3
                elif direction_v == 1:
                    color = 'blue'
                    z = 2
                else:
                    color = 'green'
                    z = 2
                rect = Rectangle((vx, vy), width, height, edgecolor=color, facecolor='none', lw=2, zorder=z)
                ax.add_patch(rect)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.set_xlabel('x (meters)')
            ax.set_ylabel('y (meters)')
            ax.set_title(f'Frame {frame_num} - Test Vehicle {test_vehicle_id}')
        ani = animation.FuncAnimation(fig, update, frames=range(min_frame, max_frame+1), interval=40, repeat=False)
        plt.show()
