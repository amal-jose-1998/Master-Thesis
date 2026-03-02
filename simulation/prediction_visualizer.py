import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp


def prediction_visualizer_process(queue: mp.Queue, maneuver_names):
    """
    This function runs in a separate process and manages the prediction visualization. 
    It listens for predicted probabilities from the main process via a multiprocessing queue and updates a bar chart
    to show the predicted maneuver probabilities.

    parameters:
    - queue: A multiprocessing.Queue object for receiving predicted probabilities from the main process.
    - maneuver_names: A list of maneuver names corresponding to the predicted probabilities.

    The function creates a bar chart with the maneuver names on the x-axis and their corresponding probabilities on the y-axis.
    It updates the bar colors to highlight the most likely predicted maneuver and displays the predicted maneuver as
    text above the bars. The visualization runs in an interactive mode and updates in real-time as new predictions are received.
    """
    maneuver_names = list(maneuver_names)

    vehicle_ids = [0]
    probs_by_vehicle = {0: np.zeros(len(maneuver_names), dtype=float)}

    fig = None
    axes = None
    status = None

    def _setup_figure(new_vehicle_ids):
        nonlocal fig, axes, status, vehicle_ids, probs_by_vehicle
        vehicle_ids = [int(v) for v in new_vehicle_ids] if new_vehicle_ids else [0]
        probs_by_vehicle = {
            vid: probs_by_vehicle.get(vid, np.zeros(len(maneuver_names), dtype=float)).copy()
            for vid in vehicle_ids
        }

        if fig is not None:
            plt.close(fig)

        n = len(vehicle_ids)
        fig, axes_arr = plt.subplots(n, 1, figsize=(10, max(4, 3 * n)), squeeze=False)
        axes = list(axes_arr[:, 0])
        fig.subplots_adjust(bottom=0.12, hspace=0.8)
        status = fig.suptitle('Waiting for predictions...', color='red', fontsize=12)
        

    def _setup_axes(ax, vid):
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title(f'Vehicle {vid} - Predicted Maneuver Probabilities')
        ax.tick_params(axis='x', labelrotation=35)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment('right')

    def _draw_all():
        best_lines = []
        for idx, vid in enumerate(vehicle_ids):
            ax = axes[idx]
            ax.clear()
            probs = probs_by_vehicle.get(vid, np.zeros(len(maneuver_names), dtype=float))
            bars_local = ax.bar(maneuver_names, probs, color='blue')
            _setup_axes(ax, vid)
            if len(probs) > 0:
                max_idx = int(np.argmax(probs))
                bars_local[max_idx].set_color('red')
                best_lines.append(f"V{vid}: {maneuver_names[max_idx]}")

        if best_lines:
            status.set_text(' | '.join(best_lines))
        else:
            status.set_text('Waiting for predictions...')

        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ion()
    _setup_figure(vehicle_ids)
    _draw_all()
    plt.show()

    while True:
        msg = queue.get()
        if msg == 'quit':
            break
        if isinstance(msg, dict) and msg.get('reset'):
            msg_labels = msg.get('maneuver_labels')
            if msg_labels is not None:
                msg_labels = list(msg_labels)
                if len(msg_labels) > 0:
                    maneuver_names = msg_labels

            msg_vehicle_ids = msg.get('vehicle_ids')
            if msg_vehicle_ids is not None:
                _setup_figure(msg_vehicle_ids)
            else:
                _setup_figure(vehicle_ids)

            for vid in vehicle_ids:
                probs_by_vehicle[vid] = np.zeros(len(maneuver_names), dtype=float)

            _draw_all()
            continue

        if isinstance(msg, dict) and 'probs' in msg:
            probs = np.asarray(msg['probs']).reshape(-1)

            msg_labels = msg.get('maneuver_labels')
            if msg_labels is not None:
                msg_labels = list(msg_labels)
                if len(msg_labels) == len(probs) and msg_labels != maneuver_names:
                    maneuver_names = msg_labels

            if len(probs) != len(maneuver_names):
                print(
                    f"[prediction_visualizer] Skipping update: probs length {len(probs)} "
                    f"!= maneuver_names length {len(maneuver_names)}"
                )
                continue

            vid = int(msg.get('vehicle_id', 0))
            if vid not in probs_by_vehicle:
                _setup_figure(vehicle_ids + [vid])
            probs_by_vehicle[vid] = probs
            _draw_all()
            
    plt.close(fig)

