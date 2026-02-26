from queue import Queue

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from pedal_visualizer import PedalVisualizer
from steering_visualizer import SteeringVisualizer
import pandas as pd

class RoadSceneRenderer:
    """Renders the road scene with lane markings, vehicles, and visualizes pedal/steering state for a test vehicle."""
    def __init__(self, recording_meta, tracks_meta_df, lane_color='white', road_color="#807D7D", median_color="#BDDA90", visualizer_queue=None):
        self.recording_meta = recording_meta
        self.tracks_meta_df = tracks_meta_df
        self.lane_color = lane_color 
        self.road_color = road_color
        self.median_color = median_color
        self.pedal_visualizer = PedalVisualizer() # For internal pedal visualization if not using external visualizer
        self.steering_visualizer = SteeringVisualizer() # For internal steering visualization if not using external visualizer
        self.visualizer_queue: Queue = visualizer_queue  # Queue for sending pedal/steering state updates to external visualizer, if provided

        # caches for fast rendering 
        self._road_xspan = None  # (xmin, xmax) used when road was drawn 

        # meta cache: id -> (width, height, drivingDirection)
        self._meta_by_id = {}
        try:
            for row in self.tracks_meta_df.itertuples(index=False):
                self._meta_by_id[int(row.id)] = (float(row.width), float(row.height), int(row.drivingDirection))
        except Exception:
            print("Error caching tracks metadata for rendering. Vehicle dimensions and directions will be looked up on the fly, which may cause slowdowns during rendering.")

        # vehicle artists
        self._rect_by_id = {}       # id -> Rectangle
        self._visible_last = set()  # ids visible last frame

    def render_road(self, ax: plt.Axes, xlim=(0, 1000), ylim=None):
        """
        Renders the road with lane markings based on the recording metadata.

        parameters:
        - ax: Matplotlib axis to draw on
        - xlim: Tuple specifying the x-axis limits for rendering the road
        - ylim: Tuple specifying the y-axis limits for rendering the road (if None, it will be set based on lane markings)
        """
        offset = 0.5  # Offset for lane marking thickness and road margins
        upper = self.recording_meta['lane_markings_upper']
        lower = self.recording_meta['lane_markings_lower']
        road_top = upper[0] # y value of the top lane marking (smallest y since y increases downwards)
        road_bottom = lower[-1] # y value of the bottom lane marking (largest y)
        median_top = upper[-1] # y value of the bottom lane marking in the upper set (largest y in upper)
        median_bottom = lower[0] # y value of the top lane marking in the lower set (smallest y in lower)
        
        ylims = (road_top - 2*offset, road_bottom + 2*offset) if ylim is None else ylim # Set y limits to show some margin around the road if not provided

        ax.fill_between(xlim, ylims[0], road_top-offset, color=self.median_color, zorder=0) # Fill top margin with median color
        ax.fill_between(xlim, road_bottom+offset, ylims[1], color=self.median_color, zorder=0) # Fill bottom margin with median color
        ax.fill_between(xlim, road_top-offset, road_bottom+offset, color=self.road_color, zorder=1) # Fill the main road area with road color
        ax.fill_between(xlim, median_top+offset, median_bottom-offset, color=self.median_color, zorder=2) # Draw median strip between upper and lower lane markings

        # Draw lane markings, using solid lines for the outermost markings and dashed lines for inner markings
        for i, y in enumerate(upper):
            ls = '-' if (i == 0 or i == len(upper) - 1) else '--'
            ax.axhline(y, color=self.lane_color, linestyle=ls, linewidth=1, zorder=3)
        for i, y in enumerate(lower):
            ls = '-' if (i == 0 or i == len(lower) - 1) else '--'
            ax.axhline(y, color=self.lane_color, linestyle=ls, linewidth=1, zorder=3)

        ax.set_xlim(xlim)
        ax.set_ylim(ylims)
        ax.invert_yaxis()  # Keep inverted to match highD convention
        ax.set_aspect('equal') # Keep aspect ratio so lanes look correct and minimize margins
        ax.set_xlabel('x (meters)')
        ax.set_ylabel('y (meters)')
        ax.set_title('Road Scene')
        self._road_drawn = True
        self._road_xspan = xlim

    def _render_vehicles(self, ax: plt.Axes, tracks_df: pd.DataFrame, test_vehicle_id=None):
        """
        Renders vehicles as rectangles based on their position and dimensions from the tracks dataframe.
        
        parameters:
        - ax: Matplotlib axis to draw on
        - tracks_df: DataFrame containing the tracks data for the current frame, filtered to only include vehicles present in that frame
        - test_vehicle_id: ID of the test vehicle to highlight (optional)
        """
        seen = set()
        for vid, vehicle_data in tracks_df.groupby('id'):
            vehicle_meta = self.tracks_meta_df[self.tracks_meta_df['id'] == vid].iloc[0] # Get metadata for this vehicle to determine dimensions and direction
            width = vehicle_meta['width']
            height = vehicle_meta['height']
            direction = vehicle_meta['drivingDirection']
            x0 = vehicle_data['x'].iloc[0]
            y0 = vehicle_data['y'].iloc[0]
            if vid == test_vehicle_id:
                color = 'red' # Highlight test vehicle in red
            elif direction == 1:
                color = 'blue' 
            else:
                color = 'green'
            rect = Rectangle((x0, y0), width, height, edgecolor=color, facecolor="none", lw=2, zorder=3) 
            ax.add_patch(rect) # Draw the vehicle rectangle on the plot
    
    def render_vehicles(self, ax: plt.Axes, frame_df, test_vehicle_id=None):
        """
        Renders vehicles for a single frame, using caching to optimize performance. 
        Vehicles are colored based on their direction, with the test vehicle highlighted in red.
        """
        seen = set()

        for row in frame_df.itertuples(index=False):
            vid = int(row.id)
            x0 = float(row.x)
            y0 = float(row.y)

            meta = self._meta_by_id.get(vid, None)
            if meta is None:
                # fallback (should be rare)
                try:
                    m = self.tracks_meta_df[self.tracks_meta_df['id'] == vid].iloc[0]
                    width = float(m['width']); height = float(m['height']); direction = int(m['drivingDirection'])
                    self._meta_by_id[vid] = (width, height, direction)
                    meta = self._meta_by_id[vid]
                except Exception:
                    continue

            width, height, direction = meta

            if vid == test_vehicle_id:
                color = 'red'
            elif direction == 1:
                color = 'blue'
            else:
                color = 'green'

            rect = self._rect_by_id.get(vid)
            if rect is None:
                rect = Rectangle((x0, y0), width, height, edgecolor=color, facecolor="none", lw=2, zorder=4)
                ax.add_patch(rect)
                self._rect_by_id[vid] = rect
            else:
                rect.set_xy((x0, y0))
                # width/height are constant in highD, but safe to keep correct:
                rect.set_width(width)
                rect.set_height(height)
                rect.set_edgecolor(color)
                rect.set_visible(True)

            seen.add(vid)

        # Hide rectangles that were visible last frame but are not present now
        for vid in (self._visible_last - seen):
            r = self._rect_by_id.get(vid)
            if r is not None:
                r.set_visible(False)

        self._visible_last = seen

    def animate_scene(self, tracks_df: pd.DataFrame, test_vehicle_id, window_width=150, window_height=50, x_offset=10):
        """
        Animates the road scene for the specified test vehicle, showing its movement and pedal/steering state over time.
        
        parameters:
        - tracks_df: DataFrame containing the tracks data for the test vehicle, filtered to only include frames for that vehicle
        - test_vehicle_id: ID of the test vehicle to animate
        - window_width: Width of the x-axis window to show around the test vehicle
        - window_height: Height of the y-axis window to show around the center of the road
        - x_offset: Distance to keep the test vehicle from the left edge of the window (if driving right) or right edge (if driving left)
        """
        test_vehicle_frames = tracks_df[tracks_df['id'] == test_vehicle_id]['frame'].values # Get the frames for the test vehicle to determine animation range
        if len(test_vehicle_frames) == 0:
            print(f"No frames for test vehicle {test_vehicle_id}")
            return
        min_frame, max_frame = test_vehicle_frames[0], test_vehicle_frames[-1] # Get the minimum and maximum frame numbers for the test vehicle to set the animation range
        fig, ax = plt.subplots(figsize=(10, 6)) 
        upper = self.recording_meta['lane_markings_upper'] 
        lower = self.recording_meta['lane_markings_lower']
        road_top = upper[0]
        road_bottom = lower[-1]
        y_center = (road_top + road_bottom) / 2
        # Only create pedal/steering axes if not using external visualizer
        if self.visualizer_queue is None:
            pedal_ax: plt.Axes = fig.add_axes([0.7, 0.05, 0.25, 0.18])  # Position for pedal visualizer [left, bottom, width, height]
            pedal_ax.axis('off') 

        def update(frame_num):
            """
            Update function for animation, called for each frame number in the animation range. Renders the road, vehicles, 
            and updates pedal/steering visualization for the current frame.

            parameters:
            - frame_num: Current frame number to render
            """
            ax.clear() # Clear the main axis for redrawing the road and vehicles
            ax.set_aspect('auto')  # Set aspect to auto for the main plot to allow it to fill the space, while keeping the road rendering correct with fixed aspect ratio
            tv_row: pd.DataFrame = tracks_df[(tracks_df['id'] == test_vehicle_id) & (tracks_df['frame'] == frame_num)] # Get the row for the test vehicle at the current frame to determine its position and state
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

            ylim = (y_center - window_height/2, y_center + window_height/2) # Set ylim so that lower y is at the top (image coordinates)
            self.render_road(ax, xlim, ylim) # Render the road for the current window  
            self.render_vehicles(ax, tracks_df[tracks_df['frame'] == frame_num], test_vehicle_id) # Draw all vehicles at this frame, highlighting the test vehicle

            # Get pedal/steering state for the test vehicle at this frame to update visualizations
            raw_ax = tv_row['xAcceleration'].values[0] if 'xAcceleration' in tv_row else 0 
            raw_vx = tv_row['xVelocity'].values[0] if 'xVelocity' in tv_row else 0
            if direction == 1:
                ax_val = -raw_ax
                vx_val = -raw_vx
            else:
                ax_val = raw_ax
                vx_val = raw_vx
            prev_frame = frame_num - 1
            prev_tv_row: pd.DataFrame = tracks_df[(tracks_df['id'] == test_vehicle_id) & (tracks_df['frame'] == prev_frame)] 
            if not prev_tv_row.empty:
                prev_y = prev_tv_row['y'].values[0]
            else:
                prev_y = y0
            
            # If using external visualizer queue, send the pedal/steering state to it; otherwise, draw the visualizations on the main plot
            if self.visualizer_queue is not None:
                try:
                    self.visualizer_queue.put_nowait((ax_val, vx_val, direction, prev_y, y0)) # Send pedal/steering state to external visualizer without blocking, so it doesn't slow down the animation if the visualizer is not keeping up
                except Exception:
                    pass
            else: 
                self.pedal_visualizer.draw(pedal_ax, ax_val, vx_val)
                steering_ax: plt.Axes = fig.add_axes([0.7, 0.25, 0.25, 0.18])
                steering_ax.axis('off')
                self.steering_visualizer.draw(steering_ax, direction, prev_y, y0)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.set_xlabel('x (meters)')
            ax.set_ylabel('y (meters)')
            ax.set_title(f'Frame {frame_num} - Test Vehicle {test_vehicle_id}')
        ani = animation.FuncAnimation(fig, update, frames=range(min_frame, max_frame+1), interval=40, repeat=True) # Create the animation, calling the update function for each frame in the test vehicle's frame range, with a 40ms interval between frames (25 FPS)
        plt.show()
