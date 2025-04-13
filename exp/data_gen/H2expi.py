import numpy as np
import json
import os
import random
from tqdm import tqdm

######################################################
### Hawkes process
######################################################
def generate_hawkes1(time_interval=[100, 200]):
    """
    Generate a Hawkes process (type 1) sequence over a specific time interval.
    
    Args:
        time_interval (list): [start_time, end_time] for simulation
    
    Returns:
        list: List of event timestamps
    """
    mu = 0.2
    alpha = [0.4, 0.0]
    beta = [1, 0]
    
    return simulate_hawkes(mu, alpha, beta, time_interval[1])

def generate_hawkes2(time_interval=[100, 200]):
    """
    Generate a Hawkes process (type 2) sequence over a specific time interval.
    
    Args:
        time_interval (list): [start_time, end_time] for simulation
    
    Returns:
        list: List of event timestamps
    """
    mu = 0.2
    alpha = [0.4, 0]
    beta = [20, 0]
    
    return simulate_hawkes(mu, alpha, beta, time_interval[1])

def simulate_hawkes(mu, alpha, beta, t_end):
    """
    Simulate a Hawkes process up to time t_end.
    
    Args:
        mu (float): Baseline intensity
        alpha (list): Jump in intensity parameters
        beta (list): Exponential decay rate parameters
        t_end (float): End time for simulation
        
    Returns:
        numpy.ndarray: Array of event timestamps
    """
    T = []
    
    x = 0
    l_trg1 = 0
    l_trg2 = 0
    
    while x < t_end:
        l = mu + l_trg1 + l_trg2
        step = np.random.exponential()/l
        x = x + step
        
        if x >= t_end:
            break
            
        l_trg1 *= np.exp(-beta[0]*step)
        l_trg2 *= np.exp(-beta[1]*step)
        l_next = mu + l_trg1 + l_trg2
        
        if np.random.rand() < l_next/l:  # accept
            T.append(x)
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
    
    return np.array(T)

def format_multiple_simulations_to_hf(simulations, mark, start_time=0):
    """
    Formats multiple Hawkes process simulations to the Hugging Face dataset format.
    
    Args:
        simulations (list): List of arrays, each containing event timestamps for one simulation
        mark (int): Event type marker (0 for Hawkes1, 1 for Hawkes2)
        start_time (float): Only include timestamps greater than this value
        
    Returns:
        list: A list of dictionaries, where each dictionary represents a sequence
    """
    formatted_data = []
    
    for seq_idx, timestamps in enumerate(tqdm(simulations, desc=f"Formatting simulations (mark {mark})")):
        timestamps = np.array(timestamps)
        
        # Filter timestamps greater than start_time
        valid_timestamps = timestamps[timestamps > start_time]
        
        if len(valid_timestamps) == 0:
            continue
            
        time_since_start = valid_timestamps - valid_timestamps[0]
        time_since_last_event = np.zeros_like(valid_timestamps)
        time_since_last_event[1:] = np.diff(valid_timestamps)
        type_events = [mark] * len(valid_timestamps)
        
        temp_dict = {
            'dim_process': 1,
            'seq_len': len(valid_timestamps),
            'seq_idx': seq_idx,
            'time_since_start': time_since_start.tolist(),
            'time_since_last_event': time_since_last_event.tolist(),
            'type_event': type_events
        }
        formatted_data.append(temp_dict)
    
    return formatted_data

def merge_simulations(simulations1, simulations2, start_time=0):
    """
    Merge two sets of simulations with proper timestamps ordering.
    
    Args:
        simulations1 (list): List of arrays with timestamps for first process (mark 0)
        simulations2 (list): List of arrays with timestamps for second process (mark 1)
        start_time (float): Only include timestamps greater than this value
        
    Returns:
        list: List of dictionaries with merged sequences
    """
    merged_data = []
    
    for i in tqdm(range(min(len(simulations1), len(simulations2))), desc="Merging simulations"):
        # Get timestamps from both processes
        timestamps1 = simulations1[i]
        timestamps2 = simulations2[i]
        
        # Filter timestamps greater than start_time
        valid_timestamps1 = timestamps1[timestamps1 > start_time]
        valid_timestamps2 = timestamps2[timestamps2 > start_time]
        
        if len(valid_timestamps1) == 0 or len(valid_timestamps2) == 0:
            continue
        
        time_diff1 = np.diff(valid_timestamps1, prepend=valid_timestamps1[0])
        time_diff2 = np.diff(valid_timestamps2, prepend=valid_timestamps2[0])
        
        # Merge timestamps and create corresponding event types
        merged_timestamps = np.concatenate([valid_timestamps1, valid_timestamps2])
        event_types = np.concatenate([np.zeros(len(valid_timestamps1)), np.ones(len(valid_timestamps2))])
        merged_time_diff = np.concatenate([time_diff1, time_diff2])
        
        # Sort by timestamps
        sort_idx = np.argsort(merged_timestamps)
        sorted_timestamps = merged_timestamps[sort_idx]
        sorted_event_types = event_types[sort_idx].astype(int).tolist()
        
        # Calculate time since start and time since last event
        time_since_start = sorted_timestamps - sorted_timestamps[0]
        time_since_last_event = merged_time_diff[sort_idx]
        
        # Create dictionary for the merged sequence
        temp_dict = {
            'dim_process': 2,  # Now we have two processes
            'seq_len': len(sorted_timestamps),
            'seq_idx': i,
            'time_since_start': time_since_start.tolist(),
            'time_since_last_event': time_since_last_event.tolist(),
            'type_event': sorted_event_types
        }
        merged_data.append(temp_dict)
    
    return merged_data

def split_data(data, train_ratio=0.6, test_ratio=0.2, dev_ratio=0.2):
    assert abs(train_ratio + test_ratio + dev_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    n = len(data)
    train_size = int(n * train_ratio)
    test_size = int(n * test_ratio)
    
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    dev_data = data[train_size + test_size:]
    
    return train_data, test_data, dev_data

def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def generate_and_save_data(num_simulations=15000, output_dir='data', time_interval=[100, 200]):
    """
    Generate multiple shorter simulations for different point processes and save the data.
    
    Args:
        num_simulations (int): Number of simulations to generate
        output_dir (str): Directory to save the output files
        time_interval (list): [min_time, max_time] for each simulation
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate simulations
    print(f"Generating {num_simulations} Hawkes1 simulations...")
    hawkes1_simulations = [generate_hawkes1(time_interval) for _ in tqdm(range(num_simulations), desc="Generating Hawkes1")]
    
    print(f"Generating {num_simulations} Hawkes2 simulations...")
    hawkes2_simulations = [generate_hawkes2(time_interval) for _ in tqdm(range(num_simulations), desc="Generating Hawkes2")]
    
    # Merge simulations
    print("Merging simulations...")
    merged_data = merge_simulations(hawkes1_simulations, hawkes2_simulations, start_time=time_interval[0])
    
    print("Splitting data into train/test/dev sets...")
    train_data_merged, test_data_merged, dev_data_merged = split_data(merged_data)
    
    # Save merged data
    print("Saving data...")
    save_json(train_data_merged, os.path.join(output_dir, 'train.json'))
    save_json(test_data_merged, os.path.join(output_dir, 'test.json'))
    save_json(dev_data_merged, os.path.join(output_dir, 'dev.json'))
    
    # Generate and save metadata
    print("Generating metadata...")
    
    # Merged data metadata
    merged_metadata = {
        'parameters': {
                'mu': [0.2, 0.2],
                'alpha': [
                    [0.4, 0.0],
                    [0, 0.4]
                ],
                'beta': [
                    [1.0, 0.0],
                    [0.0, 20.0]
                ]
            },
            'simulation_interval': time_interval,
            'num_simulations': num_simulations,
        'split_info': {
            'train_size': len(train_data_merged),
            'test_size': len(test_data_merged),
            'dev_size': len(dev_data_merged),
            'train_ratio': 0.6,
            'test_ratio': 0.2,
            'dev_ratio': 0.2
        },
        'total_events': sum(item['seq_len'] for item in merged_data)
    }
    
    # Save metadata
    save_json(merged_metadata, os.path.join(output_dir, 'metadata.json'))
    
    print(f"Data generation complete. Files saved to {output_dir}")
    
if __name__ == "__main__":
    generate_and_save_data(num_simulations=15000, output_dir='data/hawkes2', time_interval=[100, 200])