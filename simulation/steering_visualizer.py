from matplotlib.patches import FancyArrowPatch

class SteeringVisualizer:
    """
    Visualizes steering/turn indicator based on y movement and direction.
    """
    def __init__(self):
        pass

    def draw(self, ax, direction, prev_y, curr_y):
        ax.clear()
        ax.axis('off')
        # Determine turn direction
        if direction == 1:
            # y increases: left, y decreases: right
            if curr_y > prev_y:
                turn = 'left'
            elif curr_y < prev_y:
                turn = 'right'
            else:
                turn = 'straight'
        else:
            # direction == 2: y increases: right, y decreases: left
            if curr_y > prev_y:
                turn = 'right'
            elif curr_y < prev_y:
                turn = 'left'
            else:
                turn = 'straight'
        # Draw steering wheel or arrow
        if turn == 'left':
            color = 'blue'
            arrow = FancyArrowPatch((0.2, 0.5), (0.05, 0.8), mutation_scale=40, color=color, lw=3)
            ax.text(0.5, 0.2, 'Left', ha='center', va='center', fontsize=12, color=color)
        elif turn == 'right':
            color = 'orange'
            arrow = FancyArrowPatch((0.8, 0.5), (0.95, 0.8), mutation_scale=40, color=color, lw=3)
            ax.text(0.5, 0.2, 'Right', ha='center', va='center', fontsize=12, color=color)
        else:
            color = 'gray'
            arrow = FancyArrowPatch((0.5, 0.2), (0.5, 0.8), mutation_scale=40, color=color, lw=3)
            ax.text(0.5, 0.1, 'Straight', ha='center', va='center', fontsize=12, color=color)
        ax.add_patch(arrow)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
