"""
Data Generator Runner

Runner for synthetic TPP data generation.
"""

from typing import Dict, List, Optional

import numpy as np

from new_ltpp.data.generation import (
    HawkesSimulator,
    IOSimulator,
    SelfCorrecting,
    SimulationManager,
)

from .cli_base import CLIRunnerBase


class DataGenerator(CLIRunnerBase):
    """
    Runner for synthetic data generation.
    """

    def __init__(self, debug: bool = False):
        super().__init__("DataGenerator", debug=debug)

    def _build_simulator(
        self,
        generation_method: str,
        dim_process: int,
        start_time: float,
        end_time: float,
        mu: Optional[List[float]],
        alpha: Optional[List[List[float]]],
        beta: Optional[List[List[float]]],
        seed: Optional[int] = None,
    ):
        """Instantiate the right simulator with default parameters."""
        if generation_method.lower() == "hawkes":
            if mu is None:
                mu = [0.2] * dim_process
            if alpha is None:
                alpha = (
                    [[0.4, 0], [0, 0.8]]
                    if dim_process == 2
                    else [
                        [0.3 if i == j else 0.1 for j in range(dim_process)]
                        for i in range(dim_process)
                    ]
                )
            if beta is None:
                beta = (
                    [[1, 0], [0, 20]]
                    if dim_process == 2
                    else [
                        [2.0 if i == j else 1.0 for j in range(dim_process)]
                        for i in range(dim_process)
                    ]
                )
            return HawkesSimulator(
                mu=np.array(mu),
                alpha=np.array(alpha),
                beta=np.array(beta),
                dim_process=dim_process,
                start_time=start_time,
                end_time=end_time,
                seed=seed,
            )

        elif generation_method.lower() == "self_correcting":
            if mu is None:
                mu = [0.2] * dim_process
            if alpha is None:
                alpha = [
                    [0.3 if i == j else 0.1 for j in range(dim_process)]
                    for i in range(dim_process)
                ]
            return SelfCorrecting(
                mu=np.array(mu),
                alpha=np.array(alpha),
                dim_process=dim_process,
                start_time=start_time,
                end_time=end_time,
                seed=seed,
            )

        else:
            raise ValueError(
                f"Generation method not supported: {generation_method}. "
                "Available methods: hawkes, self_correcting"
            )

    def generate_data(
        self,
        output_dir: Optional[str] = None,
        num_simulations: int = 50,
        generation_method: str = "hawkes",
        splits: Optional[Dict[str, float]] = None,
        start_time: float = 0,
        end_time: float = 100,
        dim_process: int = 2,
        mu: Optional[List[float]] = None,
        alpha: Optional[List[List[float]]] = None,
        beta: Optional[List[List[float]]] = None,
        save_local: bool = True,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        private: bool = False,
        token: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> bool:
        """
        Generate synthetic TPP data, save locally, and optionally push to Hugging Face Hub.

        Args:
            output_dir: Output directory (auto-generated if None)
            num_simulations: Number of simulations to generate
            generation_method: Generation method (hawkes, self_correcting)
            splits: Data splits ratios (must sum to 1.0)
            start_time: Simulation start time
            end_time: Simulation end time
            dim_process: Number of event types/dimensions
            mu: Baseline intensity parameters
            alpha: Excitation matrix (Hawkes only)
            beta: Decay matrix (Hawkes only)
            push_to_hub: Whether to push to Hugging Face Hub
            repo_id: HF repo ID (required if push_to_hub=True, format: "user/dataset-name")
            private: Make the HF dataset private
            token: Hugging Face authentication token
            seed: Random seed for reproducibility

        Returns:
            True if completed successfully
        """
        if not self.check_dependencies(["new_ltpp.data.generation"]):
            return False

        if push_to_hub and not repo_id:
            self.print_error("--repo-id is required when push_to_hub=True")
            return False

        try:
            if splits is None:
                splits = {"train": 0.6, "test": 0.2, "dev": 0.2}

            if abs(sum(splits.values()) - 1.0) > 1e-10:
                self.print_error("Split ratios must sum to 1.0")
                return False

            if output_dir is None:
                output_dir = str(self.get_output_path())

            self.print_info(
                f"Generating {num_simulations} simulations — method: {generation_method}"
            )
            self.print_info(f"Output directory: {output_dir}")

            # Build simulator
            try:
                simulator = self._build_simulator(
                    generation_method=generation_method,
                    dim_process=dim_process,
                    start_time=start_time,
                    end_time=end_time,
                    mu=mu,
                    alpha=alpha,
                    beta=beta,
                    seed=seed,
                )
            except ValueError as e:
                self.print_error(str(e))
                return False

            # Generate simulations
            sim_manager = SimulationManager(
                simulation_func=simulator.simulate,
                dim_process=dim_process,
                start_time=start_time,
                end_time=end_time,
                simulator=simulator,
            )

            if self.console:
                with self.console.status("[bold green]Simulation in progress..."):
                    formatted_data = sim_manager.bulk_simulate(num_simulations)
            else:
                formatted_data = sim_manager.bulk_simulate(num_simulations)

            metadata = simulator.get_metadata(num_simulations)
            io_handler = IOSimulator()

            # Save locally if requested
            if save_local:
                io_handler.save_to_json(
                    formatted_data=formatted_data,
                    output_dir=output_dir,
                    splits=splits,
                    metadata=metadata,
                )
                self.print_success(f"Dataset saved locally in: {output_dir}")
            else:
                self.print_info("Skipping local saving as requested.")

            # Optionally push to Hugging Face Hub
            if push_to_hub and repo_id:
                self.print_info(f"Pushing to Hugging Face Hub: {repo_id}")
                io_handler.push_to_hub(
                    formatted_data=formatted_data,
                    repo_id=repo_id,
                    splits=splits,
                    metadata=metadata,
                    private=private,
                    token=token,
                )
                self.print_success(
                    f"Dataset pushed to https://huggingface.co/datasets/{repo_id}"
                )

            # Display summary table
            if self.console:
                from rich.table import Table

                table = Table(title="Generation Summary")
                table.add_column("Parameter", style="cyan")
                table.add_column("Value", style="magenta")
                for key, value in {
                    "method": generation_method,
                    "num_simulations": num_simulations,
                    "dim_process": dim_process,
                    "time_range": f"{start_time} → {end_time}",
                    "splits": str(splits),
                    "output_dir": output_dir,
                    "pushed_to_hub": str(push_to_hub and bool(repo_id)),
                }.items():
                    table.add_row(key, str(value))
                self.console.print(table)

            return True

        except Exception as e:
            self.print_error_with_traceback(f"Error during generation: {e}", e)
            if self.debug:
                self.logger.exception("Error details:")
            return False
