import multiprocessing as mp
import matplotlib.pyplot as plt
from pedal_visualizer import PedalVisualizer
from steering_visualizer import SteeringVisualizer

class CombinedVisualizer:
    """
    CombinedVisualizer manages both the pedal and steering visualizations in a single figure. 
    It can receive updates from the RoadSceneRenderer via a multiprocessing queue and update the visualizations accordingly.
    """
    def __init__(self):
        self.pedal = PedalVisualizer() # Initialize the pedal visualizer
        self.steering = SteeringVisualizer() # Initialize the steering visualizer
        self.fig, (self.ax_pedal, self.ax_steering) = plt.subplots(2, 1, figsize=(4, 6)) # Create a figure with 2 subplots (one for pedal, one for steering)
        plt.tight_layout() 
        self.ax_pedal.axis('off')
        self.ax_steering.axis('off')
        self.fig.suptitle('Pedal & Steering Visualizer')

    def update(self, ax_val, vx_val, direction, prev_y, curr_y):
        """
        Update the pedal and steering visualizations with the latest values.

        parameters:
        - ax_val: Acceleration value for the pedal visualizer.
        - vx_val: Velocity value for the pedal visualizer.
        - direction: Steering direction for the steering visualizer.
        - prev_y: Previous y position for the steering visualizer.
        - curr_y: Current y position for the steering visualizer.
        """
        self.pedal.draw(self.ax_pedal, ax_val, vx_val) # Update the pedal visualization with the latest acceleration and velocity values
        self.steering.draw(self.ax_steering, direction, prev_y, curr_y) # Update the steering visualization with the latest direction and y position values
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
            ax_val, vx_val, direction, prev_y, curr_y = msg # Unpack the message into the respective values
            vis.update(ax_val, vx_val, direction, prev_y, curr_y) # Update the visualizer with the latest values
        except Exception:
            plt.pause(0.01) # If no message is received, just pause briefly to allow the plot to update
    plt.close() # Close the plot when done
    plt.close('all') # Ensure all figures are closed
