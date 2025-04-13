import numpy as np
import json
import os
import random

######################################################
### self-correcting process
######################################################
def generate_self_correcting(time_interval=[100, 200]):
    """
    Generate a self-correcting process sequence over a specific time interval.
    
    Args:
        time_interval (list): [start_time, end_time] for simulation
    
    Returns:
        list: List of event timestamps
    """
    
    def self_correcting_process(mu, alpha, t_end):
        t = 0
        x = 0
        T = []
        
        while t < t_end:
            e = np.random.exponential()
            tau = np.log(e*mu/np.exp(x) + 1)/mu  # e = (np.exp(mu*tau) - 1)*np.exp(x)/mu
            t = t + tau
            
            if t >= t_end:
                break
                
            T.append(t)
            x = x + mu*tau
            x = x - alpha
        
        return np.array(T)
    
    mu = 1.0
    alpha = 1.0
    
    return self_correcting_process(mu, alpha, time_interval[1])


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
    alpha = [0.8, 0.0]
    beta = [1.0, 20.0]
    t_end = random.uniform(time_interval[0], time_interval[1])
    
    return simulate_hawkes(mu, alpha, beta, t_end)

def generate_hawkes2(time_interval=[100, 200]):
    """
    Generate a Hawkes process (type 2) sequence over a specific time interval.
    
    Args:
        time_interval (list): [start_time, end_time] for simulation
    
    Returns:
        list: List of event timestamps
    """
    mu = 0.2
    alpha = [0.4, 0.4]
    beta = [1.0, 20.0]
    
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

def format_multiple_simulations_to_hf(all_simulations, start_time = 0):
    """
    Formats multiple Hawkes process simulations to the Hugging Face dataset format.
    
    Args:
        all_simulations (list): List of arrays, each containing event timestamps for one simulation
        
    Returns:
        list: A list of dictionaries, where each dictionary represents a sequence
    """
    formatted_data = []
    
    for seq_idx, timestamps in enumerate(all_simulations):
        # Filtrer uniquement les timestamps supérieurs à start_time
        valid_timestamps = timestamps[timestamps > start_time]
        
        if len(valid_timestamps) == 0:
            continue
            
        time_since_start = valid_timestamps - valid_timestamps[0]
        time_since_last_event = np.zeros_like(valid_timestamps)
        time_since_last_event[1:] = np.diff(valid_timestamps)
        type_events = [0] * len(valid_timestamps)
        
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
    # Generate data for all types of processes
    all_simulations1 = []  # Hawkes1
    all_simulations2 = []  # Hawkes2
    all_simulations_sc = []  # Self-correcting
    
    for i in range(num_simulations):
        # Generate one sequence for each process type
        T1 = generate_hawkes1(time_interval)
        T_sc = generate_self_correcting(time_interval)
        
        all_simulations1.append(T1)
        all_simulations_sc.append(T_sc)
        
        if i % 1000 == 0:
            print(f"Completed {i}/{num_simulations} simulations")

    # Create directories for all types
    os.makedirs(os.path.join(output_dir, 'hawkes1'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'self_correcting'), exist_ok=True)

    # Process and save Hawkes1 data
    train_data1, test_data1, dev_data1 = split_data(all_simulations1)
    train_data1 = format_multiple_simulations_to_hf(train_data1, start_time=time_interval[0])
    test_data1 = format_multiple_simulations_to_hf(test_data1, start_time=time_interval[0])
    dev_data1 = format_multiple_simulations_to_hf(dev_data1, start_time=time_interval[0])

    # Process and save Self-correcting data
    train_data_sc, test_data_sc, dev_data_sc = split_data(all_simulations_sc)
    train_data_sc = format_multiple_simulations_to_hf(train_data_sc, start_time=time_interval[0])
    test_data_sc = format_multiple_simulations_to_hf(test_data_sc, start_time=time_interval[0])
    dev_data_sc = format_multiple_simulations_to_hf(dev_data_sc, start_time=time_interval[0])

    # Save Hawkes1 data
    save_json(train_data1, os.path.join(output_dir, 'hawkes1/train.json'))
    save_json(test_data1, os.path.join(output_dir, 'hawkes1/test.json'))
    save_json(dev_data1, os.path.join(output_dir, 'hawkes1/dev.json'))
    
    # Save Self-correcting data
    save_json(train_data_sc, os.path.join(output_dir, 'self_correcting/train.json'))
    save_json(test_data_sc, os.path.join(output_dir, 'self_correcting/test.json'))
    save_json(dev_data_sc, os.path.join(output_dir, 'self_correcting/dev.json'))

    # Save metadata for Hawkes1
    metadata1 = {
        'parameters': {
            'mu': 0.2,
            'alpha': [0.8, 0.0],
            'beta': [1.0, 20.0],
            'simulation_interval': time_interval,
            'num_simulations': num_simulations
        },
        'split_info': {
            'train_size': len(train_data1),
            'test_size': len(test_data1),
            'dev_size': len(dev_data1),
            'train_ratio': 0.6,
            'test_ratio': 0.2,
            'dev_ratio': 0.2
        },
        'total_events': sum(len(sim) for sim in all_simulations1)
    }


    # Save metadata for Self-correcting
    metadata_sc = {
        'parameters': {
            'mu': 1.0,
            'alpha': 1.0,
            'simulation_interval': time_interval,
            'num_simulations': num_simulations
        },
        'split_info': {
            'train_size': len(train_data_sc),
            'test_size': len(test_data_sc),
            'dev_size': len(dev_data_sc),
            'train_ratio': 0.6,
            'test_ratio': 0.2,
            'dev_ratio': 0.2
        },
        'total_events': sum(len(sim) for sim in all_simulations_sc)
    }

    save_json(metadata1, os.path.join(output_dir, 'hawkes1/Hawkes_process_metadata.json'))
    save_json(metadata_sc, os.path.join(output_dir, 'self_correcting/Self_correcting_process_metadata.json'))

if __name__ == "__main__":
    generate_and_save_data(time_interval=[100, 200])