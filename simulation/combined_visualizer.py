import multiprocessing as mp
import matplotlib.pyplot as plt
from pedal_visualizer import PedalVisualizer
from steering_visualizer import SteeringVisualizer
import numpy as np

class CombinedVisualizer:
    """
    CombinedVisualizer manages both the pedal and steering visualizations in a single figure. 
    It can receive updates from the RoadSceneRenderer via a multiprocessing queue and update the visualizations accordingly.
    """
    def __init__(self):
        self.pedal = PedalVisualizer() # Initialize the pedal visualizer
        self.steering = SteeringVisualizer() # Initialize the steering visualizer
        self.fig = None
        self.axes = None
        self.vehicle_ids = []
        self.state = {}
        self._setup([0])

    def _setup(self, vehicle_ids):
        ids = [int(v) for v in vehicle_ids] if vehicle_ids else [0]
        self.vehicle_ids = ids
        if self.fig is not None:
            plt.close(self.fig)

        n = len(self.vehicle_ids)
        self.fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n), squeeze=False)
        self.axes = axes
        plt.tight_layout()
        self.fig.suptitle('Pedal & Steering Visualizer (Per Vehicle)')

        self.state = {
            vid: {"ax_val": 0.0, "vx_val": 0.0, "direction": 2, "prev_y": 0.0, "curr_y": 0.0}
            for vid in self.vehicle_ids
        }
        self._draw_all()

    def _draw_all(self):
        for r, vid in enumerate(self.vehicle_ids):
            ax_pedal = self.axes[r, 0]
            ax_steer = self.axes[r, 1]
            ax_pedal.axis('off')
            ax_steer.axis('off')

            s = self.state.get(vid)
            if s is None:
                s = {"ax_val": 0.0, "vx_val": 0.0, "direction": 2, "prev_y": 0.0, "curr_y": 0.0}

            self.pedal.draw(ax_pedal, s["ax_val"], s["vx_val"])
            self.steering.draw(ax_steer, s["direction"], s["prev_y"], s["curr_y"])
            ax_pedal.set_title(f"Vehicle {vid} - Pedal")
            ax_steer.set_title(f"Vehicle {vid} - Steering")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, vehicle_id, ax_val, vx_val, direction, prev_y, curr_y):
        """
        Update the pedal and steering visualizations with the latest values.

        parameters:
        - ax_val: Acceleration value for the pedal visualizer.
        - vx_val: Velocity value for the pedal visualizer.
        - direction: Steering direction for the steering visualizer.
        - prev_y: Previous y position for the steering visualizer.
        - curr_y: Current y position for the steering visualizer.
        """
        vid = int(vehicle_id)
        if vid not in self.state:
            new_ids = list(self.vehicle_ids) + [vid]
            self._setup(new_ids)

        self.state[vid] = {
            "ax_val": float(ax_val),
            "vx_val": float(vx_val),
            "direction": int(direction),
            "prev_y": float(prev_y),
            "curr_y": float(curr_y),
        }
        self._draw_all()
        plt.pause(0.0001) # Pause briefly to allow the plot to update

def visualizer_process(queue: mp.Queue):
    """
    This function runs in a separate process and manages the CombinedVisualizer. 
    It listens for updates from the main process via a multiprocessing queue and updates the visualizations accordingly.

    parameters:
    - queue: A multiprocessing.Queue object for receiving updates from the main process.
    """
    vis = CombinedVisualizer() # Create an instance of the CombinedVisualizer
    plt.ion() # Enable interactive mode for the plot
    plt.show() # Show the plot
    while True:
        try:
            msg = queue.get(timeout=0.1) # Wait for a message from the main process with a timeout to allow for graceful shutdown
            if msg == 'quit': # If the message is 'quit', break the loop and close the visualizer
                break
            if isinstance(msg, dict) and msg.get("reset"):
                vis._setup(msg.get("vehicle_ids", [0]))
                continue

            if isinstance(msg, dict):
                vis.update(
                    msg.get("vehicle_id", 0),
                    msg.get("ax_val", 0.0),
                    msg.get("vx_val", 0.0),
                    msg.get("direction", 2),
                    msg.get("prev_y", 0.0),
                    msg.get("curr_y", 0.0),
                )
                continue

            # Backward compatibility: tuple payload from single-vehicle mode
            ax_val, vx_val, direction, prev_y, curr_y = msg # Unpack the message into the respective values
            vis.update(0, ax_val, vx_val, direction, prev_y, curr_y) # Update the visualizer with the latest values
        except Exception:
            plt.pause(0.01) # If no message is received, just pause briefly to allow the plot to update
    plt.close() # Close the plot when done
    plt.close('all') # Ensure all figures are closed
