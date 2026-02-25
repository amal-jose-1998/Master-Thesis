import multiprocessing as mp
import matplotlib.pyplot as plt
from pedal_visualizer import PedalVisualizer
from steering_visualizer import SteeringVisualizer
import time

class CombinedVisualizer:
    def __init__(self):
        self.pedal = PedalVisualizer()
        self.steering = SteeringVisualizer()
        self.fig, (self.ax_pedal, self.ax_steering) = plt.subplots(2, 1, figsize=(4, 6))
        plt.tight_layout()
        self.ax_pedal.axis('off')
        self.ax_steering.axis('off')
        self.fig.suptitle('Pedal & Steering Visualizer')

    def update(self, ax_val, vx_val, direction, prev_y, curr_y):
        self.pedal.draw(self.ax_pedal, ax_val, vx_val)
        self.steering.draw(self.ax_steering, direction, prev_y, curr_y)
        plt.pause(0.001)

def visualizer_process(queue):
    vis = CombinedVisualizer()
    plt.ion()
    while True:
        try:
            msg = queue.get(timeout=0.1)
            if msg == 'quit':
                break
            ax_val, vx_val, direction, prev_y, curr_y = msg
            vis.update(ax_val, vx_val, direction, prev_y, curr_y)
        except Exception:
            plt.pause(0.01)
    plt.close('all')

if __name__ == '__main__':
    # For manual test/demo
    q = mp.Queue()
    p = mp.Process(target=visualizer_process, args=(q,))
    p.start()
    for i in range(100):
        q.put((i % 10 - 5, 1 if i % 2 == 0 else 2, 0, i % 3 - 1))
        time.sleep(0.1)
    q.put('quit')
    p.join()
