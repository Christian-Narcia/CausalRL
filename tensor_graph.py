import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from concurrent.futures import ProcessPoolExecutor
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import savgol_filter

def process_event_file(event_file, max_steps):
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    # Get all the available tags
    tags = event_acc.Tags()['scalars']
    if not tags:
        return None

    tag = tags[0]
    events = event_acc.Scalars(tag)

    # Prepare data for plotting
    steps = [event.step for event in events if event.step <= max_steps]
    values = [event.value for event in events if event.step <= max_steps]

    print(event_file)

    # Create a pandas DataFrame
    return pd.DataFrame({'steps': steps, 'values': values})

def plot_tensorboard_logs(runs_dir, max_steps=500000):
    plt.figure(figsize=(8, 8))
    all_data = []

    # Collect event files
    event_files = []
    for subdir, _, files in os.walk(runs_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_files.append(os.path.join(subdir, file))

    # Process event files in parallel
    with ProcessPoolExecutor(max_workers=15) as executor:
        results = executor.map(process_event_file, event_files, [max_steps] * len(event_files))
    
    for result in results:
        if result is not None:
            all_data.append(result)
            # plt.plot(result['steps'], result['values'], color='blue', alpha=0.1)

    if not all_data:
        raise ValueError("No valid data found in the event files.")

    # Combine all data into a single DataFrame
    combined_data = pd.concat(all_data).groupby('steps').agg(['mean', 'min', 'max'])
    combined_data.columns = combined_data.columns.droplevel(0)

    # Apply moving average to min and max values
    window_length = min(len(combined_data), 101)  # Choose an appropriate window length
    combined_data['min_smooth'] = savgol_filter(combined_data['min'], window_length, 3)
    combined_data['max_smooth'] = savgol_filter(combined_data['max'], window_length, 3)

    # Calculate the exponential moving average
    smoothing_factor = 0.03  # You can adjust the smoothing factor as needed
    combined_data['ema'] = combined_data['mean'].ewm(alpha=smoothing_factor).mean()

    # Plot the exponential moving average
    plt.plot(combined_data.index, combined_data['ema'], color='Purple', label='Smoothed Average(Original PPO Pusher)')

    # Fill the area between smoothed min and max
    plt.fill_between(combined_data.index, combined_data['min_smooth'], combined_data['max_smooth'], color='Purple', alpha=0.3)

    # Set plot title and labels
    plt.suptitle('PPO Pusher (100 Runs)', fontsize=14)
    plt.title('lr: 0.0003, mini batch: 32, num steps: 2048', fontsize=12)
    plt.xlabel('Steps',fontsize=10)
    plt.ylabel('Reward',fontsize=10)
    plt.legend()
    plt.grid(True)
    
    # Customize x-axis ticks
    plt.xticks(ticks=range(0, max_steps + 1, max_steps // 20), rotation=90, fontsize=9)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
    plt.xlim([0, max_steps])

    plt.yticks(fontsize=9)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Specify the directory containing your TensorBoard event files
    runs_dir = "./runsPusherPPO/"
    
    # Ensure the directory exists
    if not os.path.exists(runs_dir):
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    
    # Plot the TensorBoard log data from all event files in the directory
    plot_tensorboard_logs(runs_dir)