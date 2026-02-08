"""
Data Generator Runner

Runner for synthetic TPP data generation.
"""

from pathlib import Path
from typing import Dict, List, Optional

from new_ltpp.data.generation import HawkesSimulator, SelfCorrecting

from .cli_base import CLIRunnerBase


class DataGenerator(CLIRunnerBase):
    """
    Runner for synthetic data generation.
    Uses SynGenConfigBuilder for configuration.
    """

    def __init__(self, debug: bool = False):
        super().__init__("DataGenerator", debug=debug)

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
        **kwargs,
    ) -> bool:
        """
        Generate synthetic TPP data.

        Args:
            output_dir: Output directory
            num_simulations: Number of simulations to generate
            generation_method: Generation method (hawkes, self_correcting)
            splits: Data splits (train/test/dev)
            start_time: Start time
            end_time: End time
            dim_process: Process dimension
            mu: Mu parameters for Hawkes
            alpha: Alpha parameters for Hawkes
            beta: Beta parameters for Hawkes
            **kwargs: Additional parameters

        Returns:
            True if generation completed successfully
        """
        # Check dependencies
        required_modules = ["new_ltpp.data.generation"]
        if not self.check_dependencies(required_modules):
            return False

        try:
            self.print_info(
                f"Generating {num_simulations} simulations - Method: {generation_method}"
            )

            # Create default output directory if needed
            if output_dir is None:
                output_dir = str(self.get_output_path())
                self.print_info(f"Output directory: {output_dir}")

            # Default values for splits
            if splits is None:
                splits = {"train": 0.6, "test": 0.2, "dev": 0.2}

            # Generation by method
            if generation_method.lower() == "hawkes":
                # Default parameters for Hawkes
                if mu is None:
                    mu = [0.2] * dim_process
                if alpha is None:
                    if dim_process == 2:
                        alpha = [[0.4, 0], [0, 0.8]]
                    else:
                        alpha = [
                            [0.3 if i == j else 0.1 for j in range(dim_process)]
                            for i in range(dim_process)
                        ]
                if beta is None:
                    if dim_process == 2:
                        beta = [[1, 0], [0, 20]]
                    else:
                        beta = [
                            [2.0 if i == j else 1.0 for j in range(dim_process)]
                            for i in range(dim_process)
                        ]

                generator = HawkesSimulator(
                    mu=mu,
                    alpha=alpha,
                    beta=beta,
                    dim_process=dim_process,
                    start_time=start_time,
                    end_time=end_time,
                )

            elif generation_method.lower() == "self_correcting":
                # Default parameters for SelfCorrecting
                if mu is None:
                    mu = [0.2] * dim_process
                if alpha is None:
                    alpha = [
                        [0.3 if i == j else 0.1 for j in range(dim_process)]
                        for i in range(dim_process)
                    ]

                generator = SelfCorrecting(
                    mu=mu,
                    alpha=alpha,
                    dim_process=dim_process,
                    start_time=start_time,
                    end_time=end_time,
                )

            else:
                self.print_error(
                    f"Generation method not supported: {generation_method}"
                )
                self.print_info("Available methods: hawkes, self_correcting")
                return False

            # Generate and save data
            self.print_info("Generation in progress...")

            if self.console:
                with self.console.status(
                    "[bold green]Generation in progress..."
                ) as status:
                    generator.generate_and_save(
                        output_dir=output_dir,
                        num_simulations=num_simulations,
                        splits=splits,
                    )
                    status.update("[bold green]Save completed")
            else:
                generator.generate_and_save(
                    output_dir=output_dir,
                    num_simulations=num_simulations,
                    splits=splits,
                )

            # Statistics of generated data
            stats = {
                "generation_method": generation_method,
                "num_simulations": num_simulations,
                "dim_process": dim_process,
                "time_range": f"{start_time} - {end_time}",
                "splits": splits,
                "output_directory": output_dir,
            }

            # Afficher les statistiques
            if self.console:
                from rich.table import Table

                table = Table(title="Generated synthetic data")
                table.add_column("Statistic", style="cyan")
                table.add_column("Value", style="magenta")

                for key, value in stats.items():
                    if isinstance(value, (dict, list)):
                        table.add_row(key, str(value))
                    elif isinstance(value, float):
                        table.add_row(key, f"{value:.2f}")
                    else:
                        table.add_row(key, str(value))

                self.console.print(table)

            # Save metadata
            metadata = {
                "generation_config": {
                    "generation_method": generation_method,
                    "num_simulations": num_simulations,
                    "dim_process": dim_process,
                    "start_time": start_time,
                    "end_time": end_time,
                    "mu": mu,
                    "alpha": alpha,
                    "beta": beta,
                },
                "statistics": stats,
            }

            import json

            metadata_path = Path(output_dir) / "generation_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            self.print_success(f"Data generated successfully in: {output_dir}")
            self.print_success(f"Metadata: {metadata_path}")

            return True

        except Exception as e:
            self.print_error_with_traceback(f"Error during generation: {e}", e)
            if self.debug:
                self.logger.exception("Error details:")
            return False
