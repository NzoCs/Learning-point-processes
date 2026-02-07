from typing import Callable, Dict, List, Tuple

import numpy as np
from tqdm import tqdm


class SimulationManager:
    """
    Simulation manager that orchestrates generation and formatting of data.
    Accepts a simulation function and handles bulk processing and formatting.
    """

    def __init__(
        self,
        simulation_func: Callable[[], Tuple[np.ndarray, np.ndarray]],
        dim_process: int,
        start_time: float,
        end_time: float,
    ):
        """
        Initialize the simulation manager.

        Args:
            simulation_func: Function that simulates a process and returns (times, marks)
            dim_process: Dimension of the process
            start_time: Simulation start time
            end_time: Simulation end time
        """
        self.simulation_func = simulation_func
        self.dim_process = dim_process
        self.start_time = start_time
        self.end_time = end_time

    def bulk_simulate(self, num_simulations: int) -> List[Dict]:
        """
        Generate multiple simulations and format them.

        Args:
            num_simulations: Number of simulations to generate

        Returns:
            A list of formatted simulations
        """
        simulations = []

        for _ in tqdm(
            range(num_simulations), desc=f"Simulating {num_simulations} processes"
        ):
            times, marks = self.simulation_func()
            simulations.append((times, marks))

        # Format simulations for dataset
        formatted_data = self.format_simulations(simulations)

        return formatted_data

    def format_simulations(
        self, simulations: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Dict]:
        """
        Format simulations into a Hugging Face-style dataset format.

        Args:
            simulations: List of tuples (times, marks) produced by the simulation function

        Returns:
            A list of dictionaries, each representing a sequence
        """
        formatted_data = []

        for seq_idx, (times, marks) in enumerate(simulations):
            # Filter timestamps greater than start_time
            mask = times > self.start_time
            valid_times = times[mask]
            valid_marks = marks[mask]

            if len(valid_times) == 0:
                continue

            # Sort by time (should already be sorted, but done for safety)
            sort_idx = np.argsort(valid_times)
            sorted_times = valid_times[sort_idx]
            sorted_marks = valid_marks[sort_idx]

            # Compute time since start and time differences
            time_since_start = sorted_times - sorted_times[0]
            time_since_last_event = np.diff(sorted_times, prepend=sorted_times[0])

            temp_dict = {
                "dim_process": self.dim_process,
                "seq_len": len(sorted_times),
                "seq_idx": seq_idx,
                "time_since_start": time_since_start.tolist(),
                "time_since_last_event": time_since_last_event.tolist(),
                "type_event": sorted_marks.tolist(),
            }
            formatted_data.append(temp_dict)

        return formatted_data
