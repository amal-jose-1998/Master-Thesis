from matplotlib.patches import Rectangle

class PedalVisualizer:
    """
    Visualizes accelerator and brake pedal states based on speed and acceleration.
    """
    def __init__(self):
        pass

    def draw(self, ax, ax_val, vx_val):
        ax.clear()
        ax.axis('off')
        # Accelerator pedal logic: green if accelerating forward
        accel_color = 'green' if ax_val > 0 else 'gray'
        # Brake pedal logic: red if braking (negative acceleration and speed)
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
