import numpy as np
from easy_tpp.utils.misc import save_json
from tqdm import tqdm

def generate_synthetic_data(n_nodes=3, end_time=1000, baseline=0.1, adjacency=0.5, decay=1.0):
    """
    Generates synthetic data using a multivariate Hawkes process with exponential kernels.

    Args:
        n_nodes (int): Number of nodes (or dimensions) in the Hawkes process.
        end_time (float): The time until which the process is simulated.
        baseline (float): Baseline intensity for each node.
        adjacency (float): Adjacency matrix value for the influence between nodes.
        decay (float): Decay parameter for the exponential kernel.

    Returns:
        list: A list of lists, where each sublist contains dictionaries representing events for a node.
    """
    baseline_vector = np.full(n_nodes, baseline)
    adjacency_matrix = np.full((n_nodes, n_nodes), adjacency)
    events = [[] for _ in range(n_nodes)]
    current_time = 0

    while current_time < end_time:
        # Calculate the intensity for each node
        intensities = baseline_vector.copy()
        for i in range(n_nodes):
            for j in range(n_nodes):
                if events[j]:
                    last_event_time = events[j][-1]['time_since_start']
                    intensities[i] += adjacency_matrix[i, j] * np.exp(-decay * (current_time - last_event_time))

        # Determine the next event time
        total_intensity = np.sum(intensities)
        if total_intensity == 0:
            break
        time_to_next_event = np.random.exponential(1 / total_intensity)
        current_time += time_to_next_event

        if current_time >= end_time:
            break

        # Determine which node the event occurs in
        probabilities = intensities / total_intensity
        node = np.random.choice(n_nodes, p=probabilities)

        # Record the event as a dictionary
        if events[node]:
            last_event_time = events[node][-1]['time_since_start']
        else:
            last_event_time = 0

        event = {
            'time_since_start': current_time,
            'time_since_last_event': current_time - last_event_time,
            'type_event': node
        }
        events[node].append(event)

    return events

def format_multivariate_simulations(simulations: list[dict], dim_process) -> list[dict]:
    """
    Formats the raw simulation results into a list of dictionaries, one per sequence.

    Each dictionary follows a structure similar to Hugging Face datasets,
    containing event times, time deltas, event types, sequence length, etc.

    Args:
        simulations (List[Dict]): A list where each dict contains tensors
                                    ('time_seq', 'time_delta_seq', 'event_seq')
                                    for a single simulated sequence.
        dim_process (Optional[int]): The number of event types (dimensionality) in the process.

    Returns:
        List[Dict]: A list of dictionaries, each representing a formatted sequence.
    """
    formatted_data = []
    
    for seq_idx, sim in enumerate(tqdm(simulations, desc="Formatting sequences")):
        times = sim['time_seq']
        events = sim['event_seq']
        time_deltas = sim['time_delta_seq']
        
        times = times - times[0]  

        times_list = times.cpu().tolist()
        events_list = events.cpu().long().tolist()
        time_deltas_list = time_deltas.cpu().tolist()

        seq_dict = {
            'dim_process': dim_process if dim_process is not None else -1,
            'seq_len': len(times_list),
            'seq_idx': seq_idx,
            'time_since_start': times_list,
            'time_since_last_event': time_deltas_list,
            'type_event': events_list
        }
        formatted_data.append(seq_dict)
    
    return formatted_data

def format_tick_data_to_hf(events, dim_process, max_seq_len):
    """
    Formats the synthetic data from a multivariate Hawkes process to the Hugging Face dataset format.

    Args:
        events (list): A list of lists, where each sublist contains dictionaries representing events for a node.
        dim_process (int): Number of nodes (or dimensions) in the Hawkes process.
        max_seq_len (int): Maximum sequence length.

    Returns:
        list: A list of dictionaries, where each dictionary represents a sequence.
    """
    # Flatten all events into a single list
    all_events = [event for node_events in events for event in node_events]
    
    # Sort events by time_since_start
    all_events.sort(key=lambda x: x['time_since_start'])

    # Split into multiple sequences based on max_seq_len
    formatted_data = []
    for seq_idx in range(0, len(all_events), max_seq_len):
        seq_events = all_events[seq_idx:seq_idx + max_seq_len]
        
        # Adjust time_since_start to have zero start timestamps
        start_time = seq_events[0]['time_since_start']
        time_since_start = [event['time_since_start'] - start_time for event in seq_events]
        time_since_last_event = [event['time_since_last_event'] for event in seq_events]
        type_event = [event['type_event'] for event in seq_events]

        temp_dict = {
            'dim_process': dim_process,
            'seq_idx': seq_idx // max_seq_len,
            'seq_len': len(seq_events),
            'time_since_start': time_since_start,
            'time_since_last_event': time_since_last_event,
            'type_event': type_event
        }
        formatted_data.append(temp_dict)

    return formatted_data

def format_gen_data_to_hf(sample_data, dim_process, max_seq_len=None):
    """
    Formats generated data from BaseGenerator.sample() to the Hugging Face dataset format.

    Args:
        sample_data (dict): A dictionary containing:
            - 'type_seqs': List of tensors representing event types
            - 'time_delta_seqs': List of tensors representing time differences between events
            - 'simul_mask': List of tensors representing simulation masks
        dim_process (int): Number of nodes (or dimensions) in the process.
        max_seq_len (int, optional): Not used. Kept for backward compatibility.

    Returns:
        list: A list of dictionaries, where each dictionary represents a sequence in HF format.
    """
    formatted_data = []
    seq_idx = 0
    
    type_seqs = sample_data['type_seqs']
    time_delta_seqs = sample_data['time_delta_seqs']
    time_seqs = sample_data['time_seqs']
    simul_masks = sample_data.get('simul_mask', None)
    
    for batch_idx, (type_seq_batch, time_delta_seq_batch, time_seq_batch) in enumerate(zip(type_seqs, time_delta_seqs, time_seqs)):
        # Get batch dimension
        batch_size = type_seq_batch.shape[0]
        
        # Process each sequence in the batch separately
        for i in range(batch_size):
            # Extract individual sequences
            type_seq = type_seq_batch[i]
            time_delta_seq = time_delta_seq_batch[i]
            time_since_start = time_seq_batch[i]
            
            # Apply mask if provided to filter out only interesting events
            if simul_masks is not None:
                simul_mask = simul_masks[batch_idx][i]
                time_since_start = time_since_start[simul_mask]
                time_delta_seq = time_delta_seq[simul_mask]
                type_seq = type_seq[simul_mask]
            
            # Convert to numpy and then to list
            type_seq = type_seq.detach().cpu().numpy().tolist()
            time_delta_seq = time_delta_seq.detach().cpu().numpy().tolist()
            time_since_start = time_since_start.detach().cpu().numpy().tolist()
            
            # Create a sequence with a unique seq_idx
            temp_dict = {
                'dim_process': dim_process,
                'seq_idx': seq_idx,
                'seq_len': len(type_seq),
                'time_since_start': time_since_start,
                'time_since_last_event': time_delta_seq,
                'type_event': type_seq
            }
            formatted_data.append(temp_dict)
            seq_idx += 1
    
    return formatted_data

def generate_and_save_json(n_nodes, end_time, baseline, adjacency, decay, max_seq_len, target_file):
    """
    Generates synthetic data, formats it, and saves it to a file in Hugging Face format.

    Args:
        n_nodes (int): Number of nodes (or dimensions) in the Hawkes process.
        end_time (float): The time until which the process is simulated.
        baseline (float): Baseline intensity for each node.
        adjacency (float): Adjacency matrix value for the influence between nodes.
        decay (float): Decay parameter for the exponential kernel.
        max_seq_len (int): Maximum sequence length.
        target_file (str): Path to the file where the formatted data will be saved.

    Raises:
        IOError: If the file cannot be opened or written to.
    """
    events = generate_synthetic_data(n_nodes, end_time, baseline, adjacency, decay)
    formatted_data = format_tick_data_to_hf(events, dim_process=n_nodes, max_seq_len=max_seq_len)
    save_json(formatted_data, target_file)