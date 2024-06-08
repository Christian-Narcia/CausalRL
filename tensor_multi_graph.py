import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import numpy as np

def plot_tensorboard_log(directories_with_labels):
    plt.figure(figsize=(8, 8))
    # Define a set of highly distinct colors
    distinct_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Yellow-Green
        '#17becf'   # Teal
    ]
    
    for idx, (event_dir, label) in enumerate(directories_with_labels):
        # Verify that the directory exists
        if not os.path.exists(event_dir):
            print(f"Directory not found: {event_dir}")
            continue
        
        # Find the event file in the directory
        event_files = [f for f in os.listdir(event_dir) if 'events.out.tfevents' in f]
        if not event_files:
            print(f"No event file found in directory: {event_dir}")
            continue

        event_file = os.path.join(event_dir, event_files[0])

        # Load the TensorBoard event file
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()

        # Get all the available tags
        tags = event_acc.Tags()['scalars']
        
        # Assuming you want to plot the first tag, adjust if necessary
        tag = tags[0]
        events = event_acc.Scalars(tag)
        
        # Prepare data for plotting
        steps = [event.step for event in events]
        values = [event.value for event in events]

        # Create a pandas DataFrame
        data = pd.DataFrame({'steps': steps, 'values': values})

        # Calculate the exponential moving average
        smoothing_factor = 0.02  # You can adjust the smoothing factor as needed
        data['ema'] = data['values'].ewm(alpha=smoothing_factor).mean()

        # Get the color for this graph
        color = distinct_colors[idx % len(distinct_colors)]

        # Plot the original data
        # plt.plot(data['steps'], data['values'], alpha=0.3, color=color)# label=f'Original Data ({label})'

        # Plot the exponential moving average
        plt.plot(data['steps'], data['ema'], label=f'Smoothed Data ({label})', color=color)

    # Set plot title and labels
    plt.title('Original PPO Pusher VS Self-Connecting VS FC VS Random Edges')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Specify the list of directories and their corresponding labels
    directories_with_labels = [
        ("./runs/Pusher-v4__1_PPO_0__1717657993/", "Original PPO Pusher"),
        # ("./runs/Pusher-v4_GNN_Self-Connected_1__1__1717656676/", "Self-Connecting Graph Pusher"),
        ("./runs/Pusher-v4_GNN_FC_1__1__1717641772/", "FC Graph Pusher"),
        ("./runs/Pusher-v4_GNN_Random_1__1__1717662562/", "Random Graph Pusher"),
        # ("./runs/Pusher-v4__1_PPO_0__1717657993/", "Original PPO Pusher"),
        # ("./runs/Pusher-v4_GNN_Self-Connected_1__1__1717656676/", "Self-Connecting Graph Pusher"),
        ("./runs/Pusher-v4_GNN_sl_FC_1__1__1717697490/", "FC No sl Graph Pusher"),
        ("./runs/Pusher-v4_GNN_sl_Random_1__1__1717697611/", "Random No sl Graph Pusher")
    ]
    
    # Plot the TensorBoard log data
    plot_tensorboard_log(directories_with_labels)
