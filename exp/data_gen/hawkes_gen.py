import numpy as np
import json
import os
import random
from tqdm import tqdm

######################################################
### Correlated Multi-dimensional Hawkes process
######################################################
def generate_hawkes(time_interval=[100, 200]):
    """
    Generate a 2D correlated Hawkes process sequence.
    
    Args:
        time_interval (list): [start_time, end_time] for simulation
    
    Returns:
        tuple: Arrays of timestamps for each dimension
    """
    mu = np.array([0.2])  # Baseline intensities
    
    alpha = np.array([[0.8]])
    
    # Decay rates can be different for each pair of processes
    beta = np.array([[1]])
    
    
    return simulate_hawkes_multivariate(mu, alpha, beta, time_interval[1])

def generate_hawkes_3d_correlated(time_interval=[100, 200]):
    """
    Generate a 3D correlated Hawkes process sequence with randomly sampled parameters.
    
    Args:
        time_interval (list): [start_time, end_time] for simulation
    
    Returns:
        tuple: Arrays of timestamps for each dimension
    """
    # Randomly sample parameters from uniform distributions
    mu = np.random.uniform(0.1, 0.5, size=3)  # Baseline intensities in [0.1, 0.5]
    
    # Alpha matrix (excitation) in [0.5, 0.8]
    alpha = np.random.uniform(0.5, 0.8, size=(3, 3))
    
    # Decay rates in [0.4, 1.2]
    beta = np.random.uniform(0.4, 1.2, size=(3, 3))
    
    
    return simulate_hawkes_multivariate(mu, alpha, beta, time_interval[1])

def simulate_hawkes_multivariate(mu, alpha, beta, t_end):
    """
    Simulate a multivariate Hawkes process up to time t_end.
    
    Args:
        mu (numpy.ndarray): Baseline intensities for each dimension
        alpha (numpy.ndarray): Jump in intensity matrix (i,j: effect of j on i)
        beta (numpy.ndarray): Exponential decay rate matrix
        t_end (float): End time for simulation
        
    Returns:
        tuple: Arrays of event timestamps for each dimension
    """
    dim = len(mu)
    events = [[] for _ in range(dim)]
    
    # Current time
    t = 0
    
    # Initial intensity contribution matrix [to_process][from_process]
    lambda_trg = np.zeros((dim, dim))
    
    while t < t_end:
        # Total intensity for each dimension
        lambda_total = np.array([mu[i] + np.sum(lambda_trg[i]) for i in range(dim)])
        lambda_sum = np.sum(lambda_total)
        
        # Sample waiting time until next event
        dt = np.random.exponential(scale=1/lambda_sum) if lambda_sum > 0 else float('inf')
        t = t + dt
        
        if t >= t_end:
            break
            
        # Update intensity contributions based on exponential decay
        for i in range(dim):
            for j in range(dim):
                lambda_trg[i, j] *= np.exp(-beta[i, j] * dt)
        
        lambda_next = np.array([mu[i] + np.sum(lambda_trg[i]) for i in range(dim)])
        lambda_next = np.sum(lambda_next)
        
        if np.random.rand() < lambda_next / lambda_sum:  # Accept the event
        
            # Randomly select which dimension the event belongs to
            event_dim = np.random.choice(dim, p=lambda_total/lambda_sum)
            
            # Add the event to the corresponding process
            events[event_dim].append(t)
            
            # Update intensity contributions
            for i in range(dim):
                lambda_trg[i, event_dim] += alpha[i, event_dim]
    
    # Convert to numpy arrays
    return tuple(np.array(events_dim) for events_dim in events)

def format_multivariate_simulations(simulations, dim_process, start_time=0):
    """
    Format multivariate Hawkes simulations to the Hugging Face dataset format.
    
    Args:
        simulations (list): List of tuples, each containing arrays of timestamps for each dimension
        dim_process (int): Number of dimensions in the process
        start_time (float): Only include timestamps greater than this value
        
    Returns:
        list: A list of dictionaries, each representing a sequence
    """
    formatted_data = []
    
    for seq_idx, sim in enumerate(tqdm(simulations, desc=f"Formatting {dim_process}D simulations")):
        # Merge timestamps from all dimensions with their type
        all_timestamps = []
        all_types = []
        all_time_diff = []
        
        for dim, timestamps in enumerate(sim):
            # Filter timestamps greater than start_time
            valid_timestamps = timestamps[timestamps > start_time]
            
            if len(valid_timestamps) > 0:
                all_timestamps.extend(valid_timestamps)
                all_types.extend([dim] * len(valid_timestamps))
                all_time_diff.extend(np.diff(np.array(valid_timestamps), prepend=valid_timestamps[0]))
        
        if len(all_timestamps) == 0:
            continue
        
        # Convert to numpy arrays and sort by time
        all_timestamps = np.array(all_timestamps)
        all_types = np.array(all_types)
        all_time_diff = np.array(all_time_diff)
        sort_idx = np.argsort(all_timestamps)
        sorted_timestamps = all_timestamps[sort_idx]
        sorted_types = all_types[sort_idx].tolist()
        
        # Calculate time since start and time differences
        time_since_start = sorted_timestamps - sorted_timestamps[0]
        time_since_last_event = all_time_diff[sort_idx]
            
        temp_dict = {
            'dim_process': dim_process,
            'seq_len': len(sorted_timestamps),
            'seq_idx': seq_idx,
            'time_since_start': time_since_start.tolist(),
            'time_since_last_event': time_since_last_event.tolist(),
            'type_event': sorted_types
        }
        formatted_data.append(temp_dict)
    
    return formatted_data

def split_data(data, train_ratio=0.6, test_ratio=0.2, dev_ratio=0.2):
    """
    Split data into train, test, and dev sets.
    
    Args:
        data (list): List of formatted sequences
        train_ratio, test_ratio, dev_ratio (float): Split ratios
        
    Returns:
        tuple: Train, test, dev data lists
    """
    assert abs(train_ratio + test_ratio + dev_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    n = len(data)
    train_size = int(n * train_ratio)
    test_size = int(n * test_ratio)
    
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    dev_data = data[train_size + test_size:]
    
    return train_data, test_data, dev_data

def save_json(data, filepath):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath (str): Path to save the JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def generate_and_save_correlated_data(num_simulations = 15000, output_dir='data', time_interval=[100, 200]):
    """
    Generate and save correlated Hawkes process simulations.
    
    Args:
        num_simulations (int): Number of simulations to generate
        output_dir (str): Directory to save output files
        time_interval (list): [min_time, max_time] for simulations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate 2D correlated Hawkes processes 
    print(f"Generating {num_simulations} 2D correlated Hawkes simulations...")
    hawkes_2d = [generate_hawkes(time_interval) 
                 for _ in tqdm(range(num_simulations), desc="Generating 2D Hawkes")]
    
    # # Generate 3D correlated Hawkes processes
    # print(f"Generating {num_simulations} 3D correlated Hawkes simulations...")
    # hawkes_3d = [generate_hawkes_3d_correlated(time_interval) 
    #              for _ in tqdm(range(num_simulations), desc="Generating 3D Hawkes")]
    
    # Format data with start_time filtering
    print("Formatting simulation data...")
    data_2d = format_multivariate_simulations(hawkes_2d, 1, start_time=time_interval[0])
    #data_3d = format_multivariate_simulations(hawkes_3d, 3, start_time=time_interval[0])
    
    # Split data
    print("Splitting data into train/test/dev sets...")
    train_data_2d, test_data_2d, dev_data_2d = split_data(data_2d)
    #train_data_3d, test_data_3d, dev_data_3d = split_data(data_3d)
    
    # Create directories
    os.makedirs(os.path.join(output_dir, '2d'), exist_ok=True)
    #os.makedirs(os.path.join(output_dir, '3d'), exist_ok=True)
    
    # Save data
    print("Saving data...")
    
    # 2D data
    save_json(train_data_2d, os.path.join(output_dir, 'train.json'))
    save_json(test_data_2d, os.path.join(output_dir, 'test.json'))
    save_json(dev_data_2d, os.path.join(output_dir, 'dev.json'))
    
    # # 3D data
    # save_json(train_data_3d, os.path.join(output_dir, 'HDLHP/train.json'))
    # save_json(test_data_3d, os.path.join(output_dir, 'HDLHP/test.json'))
    # save_json(dev_data_3d, os.path.join(output_dir, 'HDLHP/dev.json'))
    
    # Generate and save metadata
    print("Generating metadata...")
    metadata = {
        'parameters': {
                'mu': 0.2,  # Valeurs réelles de mu
                'alpha': 0.8,  # Valeurs réelles de alpha
                'beta': 1  # Valeurs réelles de beta
            },
            'simulation_interval': time_interval,
            'num_simulations': num_simulations,
        'split_info': {
            'train_size': len(train_data_2d),
            'test_size': len(test_data_2d),
            'dev_size': len(dev_data_2d),
            'train_ratio': 0.6,
            'test_ratio': 0.2,
            'dev_ratio': 0.2
        },
        'total_events': sum(item['seq_len'] for item in data_2d)
    }
    
    save_json(metadata, os.path.join(output_dir, 'metadata.json'))
    
    print(f"All data saved to {output_dir}")

if __name__ == "__main__":
    generate_and_save_correlated_data(num_simulations = 15000, output_dir='data/hawkes1', time_interval=[100, 200])
