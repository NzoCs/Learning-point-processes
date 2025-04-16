from easy_tpp.config_factory import SimulatorConfig
import json
import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

class Simulator:
    
    def __init__(self, simulator_config: SimulatorConfig) -> None:
        
        self.save_dir = simulator_config.save_dir
        self.start_time = simulator_config.start_time
        self.end_time = simulator_config.end_time
        self.history_data = simulator_config.history_data
        self.model = simulator_config.pretrained_model
        self.splits = simulator_config.splits
        self.seed = simulator_config.seed
        
        # Add num_simulations attribute which was referenced but not initialized in original code
        self.num_simulations = getattr(simulator_config, 'num_simulations', 100)
        
        if self.seed is not None:
            import random
            import torch
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            

    def run(self) -> None:
        """
        Run the simulation process and save the results in HuggingFace format.
        Requires history_data to be provided.
        """
        # Create the output directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        model = self.model
        history_data = self.history_data
        start_time = self.start_time
        end_time = self.end_time
        
        # Validate that history_data is provided
        if history_data is None:
            raise ValueError("history_data must be provided for simulation. No simulations can be performed without historical data.")
        
        print(f"Generating {self.num_simulations} simulations...")
        simulations = []
        
        data_loader = history_data.get_dataloader(split='test')
        
        for batch in tqdm(data_loader, desc="Simulating from batches"):
            batch_values = batch.values()
            
            # Run the simulation using the model with the entire batch
            time_seq, time_delta_seq, event_seq, simul_mask = model.simulate(
                start_time=start_time,
                end_time=end_time,
                batch=batch_values,
                batch_size=None  # Model will use the actual batch size
            )
            
            # Store the direct simulation results for each sample in the batch
            batch_size = time_seq.size(0)
            for i in range(batch_size):
                mask_i = simul_mask[i]
                simulations.append({
                    'time_seq': time_seq[i][mask_i],
                    'time_delta_seq': time_delta_seq[i][mask_i],
                    'event_seq': event_seq[i][mask_i],
                    'mask': mask_i
                })
            
            if len(simulations) >= self.num_simulations:
                # Truncate to exactly num_simulations if we have more
                simulations = simulations[:self.num_simulations]
                break
        
        print(f"Successfully generated {len(simulations)} simulations")
        
        # Format the simulations for HuggingFace dataset
        formatted_data = self.format_multivariate_simulations(
            simulations, model.num_event_types
        )
        
        # Save the formatted data
        self.save_data(formatted_data)
    
    def format_multivariate_simulations(self, simulations, dim_process):
        """
        Format multivariate simulations to the Hugging Face dataset format.
        
        Args:
            simulations (list): List of dictionaries containing simulation results:
                               - time_seq: Timestamps of events
                               - time_delta_seq: Time differences between events
                               - event_seq: Event types
                               - mask: Mask indicating valid events
            dim_process (int): Number of dimensions in the process
            
        Returns:
            list: A list of dictionaries, each representing a sequence
        """
        formatted_data = []
        
        for seq_idx, sim in enumerate(simulations):
            # Get the simulated sequences - these are already filtered by the mask
            times = sim['time_seq']
            events = sim['event_seq']
            time_deltas = sim['time_delta_seq']
            
            # Convert to numpy arrays for easier processing
            times_np = times.cpu().numpy()
            events_np = events.cpu().numpy()
            time_deltas_np = time_deltas.cpu().numpy()
            
            # Skip if no events are present
            if len(times_np) == 0:
                continue
            
            # Calculate time since start (relative to the first event)
            first_time = times_np[0]
            time_since_start = times_np - first_time
            
            temp_dict = {
                'dim_process': dim_process,
                'seq_len': len(times_np),
                'seq_idx': seq_idx,
                'time_since_start': time_since_start.tolist(),
                'time_since_last_event': time_deltas_np.tolist(),
                'type_event': events_np.astype(int).tolist()
            }
            formatted_data.append(temp_dict)
        
        return formatted_data
    
    def save_data(self, formatted_data):
        """
        Save all formatted data to a single file.
        
        Args:
            formatted_data (list): List of formatted sequences
        """
        # Save all data to a single file
        self.save_json(formatted_data, os.path.join(self.save_dir, 'simulation_data.json'))
        
        # Save metadata
        self.save_metadata(formatted_data)
    
    def save_json(self, data, filepath):
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            filepath (str): Path to save the JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_metadata(self, formatted_data):
        """
        Save metadata about the simulations.
        
        Args:
            formatted_data (list): Formatted data
        """
        metadata = {
            'simulation_info': {
                'num_simulations': len(formatted_data),
                'dimension': self.model.num_event_types if hasattr(self.model, 'num_event_types') else None,
                'time_interval': [self.start_time, self.end_time],
                'model_name': self.model.__class__.__name__ if self.model else None
            },
            'total_events': sum(item['seq_len'] for item in formatted_data)
        }
        
        # Add model-specific metadata if available
        if hasattr(self.model, 'get_model_metadata'):
            model_metadata = self.model.get_model_metadata()
            metadata['model_parameters'] = model_metadata
        
        self.save_json(metadata, os.path.join(self.save_dir, 'metadata.json'))
        print(f"All simulated data has been saved in {self.save_dir}")
