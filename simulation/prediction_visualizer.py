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

    def _setup_axes():
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Predicted Maneuver Probabilities')
        ax.tick_params(axis='x', labelrotation=35)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment('right')

    def _draw(prob_values, highlight_best=True):
        ax.clear()
        bars_local = ax.bar(maneuver_names, prob_values, color='blue')
        _setup_axes()
        if highlight_best:
            max_idx = int(np.argmax(prob_values))
            bars_local[max_idx].set_color('red')
            status.set_text(f"Predicted: {maneuver_names[max_idx]}")
        else:
            status.set_text('Waiting for predictions...')
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.subplots_adjust(bottom=0.35)
    status = fig.suptitle('Waiting for predictions...', color='red', fontsize=14) 
    _draw(np.zeros(len(maneuver_names)), highlight_best=False)
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
            _draw(np.zeros(len(maneuver_names)), highlight_best=False)
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

            _draw(probs, highlight_best=True)
            
    plt.close(fig)

