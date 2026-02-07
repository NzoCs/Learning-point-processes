"""
Data Inspector Runner

Runner for inspection and visualization of TPP data.
"""

from pathlib import Path
from typing import Optional

from new_ltpp.configs import DataConfigBuilder
from new_ltpp.data.preprocess import Visualizer

from .cli_base import CLIRunnerBase


class DataInspector(CLIRunnerBase):
    """
    Runner for inspecting and visualizing data.
    Uses `DataConfigBuilder` and the Visualizer API.
    """

    def __init__(self, debug: bool = False):
        super().__init__("DataInspector", debug=debug)

    def inspect_data(
        self,
        data_dir: str,
        num_event_types: int,
        data_format: str = "json",
        output_dir: Optional[str] = None,
        save_graphs: bool = True,
        show_graphs: bool = False,
        max_sequences: Optional[int] = None,
    ) -> bool:
        """
        Inspect and visualize TPP data.

        Args:
            data_dir: Directory containing the data
            data_format: Data format (json, csv, etc.)
            output_dir: Output directory for generated plots
            save_graphs: Whether to save plots
            show_graphs: Whether to display plots interactively
            max_sequences: Maximum number of sequences to analyze

        Returns:
            True if inspection completed successfully
        """
        # Check dependencies
        required_modules = ["new_ltpp.configs", "new_ltpp.data.preprocess"]
        if not self.check_dependencies(required_modules):
            return False

        try:
            self.print_info(f"Inspecting data: {data_dir}")

            # Configure data via builder
            builder = DataConfigBuilder()
            (builder
                .set_src_dir(
                    data_dir
                )  # Uses set_src_dir which sets train/valid/test
                .set_dataset_id("test")
                .set_data_format(data_format)
                # Default data loading specs
                .set_data_loading_specs(
                    batch_size=32,
                    num_workers=4,
                    shuffle=False,
                )
                # Default tokenizer specs (may be updated after reading data)
                .set_tokenizer_specs()
                .set_num_event_types(num_event_types)
            )

            data_config = builder.build()

            # Create the data module (as in the example)
            from new_ltpp.data.preprocess import TPPDataModule

            datamodule = TPPDataModule(data_config)

            datamodule.setup(stage="test")

            # Use train if available, otherwise use test
            split_to_use = "train"
            visualizer = Visualizer(
                data_module=datamodule,
                split=split_to_use,
                max_events=max_sequences if max_sequences else 10000,
            )

            self.print_info("Generating visualizations with Visualizer...")

            # Generate individual visualizations as well
            visualizer.plot_inter_event_times(
                show=show_graphs, save=save_graphs
            )
            visualizer.plot_event_types(
                show=show_graphs, save=save_graphs
            )
            visualizer.plot_sequence_lengths(
                show=show_graphs, save=save_graphs
            )

            results = {
                "show_all_distributions": True,
                "delta_times_distribution": True,
                "event_type_distribution": True,
                "sequence_length_distribution": True,
            }
            self.print_success("✓ All visualizations generated")


            # Save metadata
            if save_graphs and output_dir:
                metadata = {
                    "data_dir": data_dir,
                    "data_format": data_format,
                    "max_sequences": max_sequences,
                    "num_event_types": num_event_types,
                    "visualizations_generated": list(results.keys()),
                    "timestamp": str(Path().absolute()),
                }

                import json

                metadata_path = Path(output_dir) / "inspection_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                self.print_success(f"Metadata saved: {metadata_path}")

            # Summary of results
            if self.console and results:
                from rich.table import Table

                table = Table(title="Inspection Results")
                table.add_column("Visualization", style="cyan")
                table.add_column("Status", style="green")

                for viz_name, result in results.items():
                    status = "✓ Generated" if result else "✗ Failed"
                    table.add_row(viz_name, status)

                self.console.print(table)

            self.print_success("Data inspection completed")
            return True

        except Exception as e:
            self.print_error_with_traceback(f"Error during inspection: {e}", e)
            self.logger.exception("Error details:")
            return False

    def _generate_summary_report(
        self, save_dir, seq_lengths, all_event_types, all_time_deltas, event_counts
    ):
        """Generate a summary report."""
        import numpy as np

        try:
            summary = {
                "total_sequences": int(len(seq_lengths)),
                "total_events": int(len(all_event_types)),
                "unique_event_types": int(len(set(all_event_types))),
                "avg_sequence_length": float(np.mean(seq_lengths)),
                "median_sequence_length": float(np.median(seq_lengths)),
                "min_sequence_length": int(np.min(seq_lengths)),
                "max_sequence_length": int(np.max(seq_lengths)),
                "event_type_distribution": {
                    str(k): int(v) for k, v in event_counts.items()
                },
            }

            if all_time_deltas:
                positive_deltas = [d for d in all_time_deltas if d > 0]
                if positive_deltas:
                    summary.update(
                        {
                            "avg_time_interval": float(np.mean(positive_deltas)),
                            "median_time_interval": float(np.median(positive_deltas)),
                            "min_time_interval": float(np.min(positive_deltas)),
                            "max_time_interval": float(np.max(positive_deltas)),
                        }
                    )

            # Save the report
            import json

            report_path = Path(save_dir) / "summary_report.json"
            with open(report_path, "w") as f:
                json.dump(summary, f, indent=2)

            self.print_success(f"Summary report: {report_path}")

            # Display the summary in the console
            if self.console:
                from rich.table import Table

                table = Table(title="Analysis Summary")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Total sequences", str(summary["total_sequences"]))
                table.add_row("Total events", str(summary["total_events"]))
                table.add_row("Unique event types", str(summary["unique_event_types"]))
                table.add_row("Average length", f"{summary['avg_sequence_length']:.2f}")
                table.add_row("Median length", f"{summary['median_sequence_length']:.2f}")

                self.console.print(table)

        except Exception as e:
            self.print_error_with_traceback(f"Error generating report: {e}", e)
