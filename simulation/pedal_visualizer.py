from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt

class PedalVisualizer:
    """
    Visualizes accelerator and brake pedal states based on acceleration and speed values. 
    The accelerator pedal is shown in green when accelerating forward, and the brake pedal 
    is shown in red when braking (negative acceleration). Speed and acceleration values are also displayed as text.
    """
    def __init__(self):
        pass

    def draw(self, ax: plt.Axes, ax_val, vx_val):
        """
        Draws the pedal visualization on the given axes based on the acceleration (ax_val) and speed (vx_val).

        parameters:
        - ax: The matplotlib Axes object to draw the visualization on.
        - ax_val: The acceleration value (positive for acceleration, negative for braking).
        - vx_val: The speed value to display on the visualization.
        """
        ax.clear()
        ax.axis('off')
        # Accelerator pedal logic: green if accelerating forward
        accel_color = 'green' if ax_val > 0 else 'gray'
        # Brake pedal logic: red if braking (negative acceleration)
        brake_color = 'red' if ax_val < 0 else 'gray'
        # Accelerator pedal
        accel_rect = Rectangle((0.1, 0.6), 0.8, 0.3, facecolor=accel_color, edgecolor='black', lw=2)
        ax.add_patch(accel_rect)
        ax.text(0.5, 0.75, 'Accelerate', ha='center', va='center', fontsize=10, color='white' if accel_color=='green' else 'black')
        # Brake pedal
        brake_rect = Rectangle((0.1, 0.1), 0.8, 0.3, facecolor=brake_color, edgecolor='black', lw=2)
        ax.add_patch(brake_rect)
        ax.text(0.5, 0.25, 'Brake', ha='center', va='center', fontsize=10, color='white' if brake_color=='red' else 'black')
        # Speedometer and Accelerometer display
        ax.text(0.5, 0.55, f'Speed: {vx_val:.2f} m/s', ha='center', va='center', fontsize=10, color='black')
        ax.text(0.5, 0.45, f'Accel: {ax_val:.2f} m/s²', ha='center', va='center', fontsize=10, color='black')
