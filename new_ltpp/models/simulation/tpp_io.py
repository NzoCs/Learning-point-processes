from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from new_ltpp.utils import logger
from new_ltpp.shared_types import SimulationResult


class IncrementalParquetWriter:
    """Saves batches of data to a single Parquet file by appending Row Groups.

    This is efficient for large simulations as it avoids keeping everything in memory
    and doesn't create thousands of small files.
    """

    def __init__(
        self, output_dir: Union[str, Path], filename: str = "simulations.parquet"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.output_dir / filename
        self._writer: Optional[pq.ParquetWriter] = None
        self._schema: Optional[pa.Schema] = None
        self._total_rows = 0

    def write_batch(self, data_list: List[Dict[str, Any]]) -> None:
        """Append a batch of dictionary records to the Parquet file."""
        if not data_list:
            return

        # Convert to DataFrame then to Arrow Table
        df = pd.DataFrame(data_list)
        table = pa.Table.from_pandas(df)

        if self._writer is None:
            # First batch: initialize schema and writer
            self._schema = table.schema
            self._writer = pq.ParquetWriter(str(self.filepath), self._schema)
            logger.info(
                f"IncrementalParquetWriter: Created single file at {self.filepath}"
            )

        # Write the row group
        self._writer.write_table(table)
        self._total_rows += len(data_list)
        logger.debug(
            f"IncrementalParquetWriter: Appended {len(data_list)} rows to {self.filepath.name}"
        )

    def close(self) -> None:
        """Properly close the Parquet writer."""
        if self._writer:
            self._writer.close()
            self._writer = None
            logger.info(
                f"IncrementalParquetWriter: Closed file {self.filepath.name} (Total rows: {self._total_rows})"
            )

    def finalize_and_get_filepath(self) -> Path:
        """Close the writer and return the path to the final file."""
        self.close()
        return self.filepath

    def finalize_and_get_all_data(self) -> List[Dict]:
        """Read the file and return all data (use with caution for large datasets)."""
        self.close()
        if not self.filepath.exists():
            return []
        table = pq.read_table(str(self.filepath))
        return table.to_pylist()


class SimulationIOManager:
    """Orchestrates simulation data formatting, local storage, and Hub integration.

    This class is designed to be used by the Simulator to offload all IO concerns.
    """

    def __init__(
        self,
        num_event_types: int,
        prefix: str = "sim_batch",
    ):
        self.num_event_types = num_event_types
        self._accumulated_data: List[Dict] = []
        self._total_sequences = 0
        self.prefix = prefix
        self._writer: Optional[IncrementalParquetWriter] = None

    def setup_io(
        self, output_dir: Union[str, Path], filename: Optional[str] = None
    ) -> None:
        """(Re)configure the IO directory and initialize the incremental writer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        fname = filename or f"{self.prefix}.parquet"
        self._writer = IncrementalParquetWriter(self.output_dir, filename=fname)
        logger.info(
            f"SimulationIOManager: configured for single-file saving at {self._writer.filepath}"
        )

    def update(self, result: SimulationResult) -> None:
        """Alias for append_batch to match statistics collector API."""
        self.append_batch(result)

    def append_batch(self, result: SimulationResult) -> None:
        """Format and accumulate/save a simulation batch."""
        formatted = self.format_simulation_result(
            result, self.num_event_types, self._total_sequences
        )

        if self._writer:
            self._writer.write_batch(formatted)
        else:
            self._accumulated_data.extend(formatted)

        self._total_sequences += len(formatted)

    @staticmethod
    def format_simulation_result(
        result: SimulationResult, num_event_types: int, offset: int = 0
    ) -> List[Dict]:
        """Converts raw SimulationResult (Batch) tensors into a flattened list of events."""
        flattened_events = []

        # Ensure data is on CPU
        times = result.time_seqs.cpu()
        types = result.type_seqs.cpu()
        masks = result.valid_event_mask.cpu()

        batch_size = times.shape[0]

        for i in range(batch_size):
            mask_i = masks[i]
            seq_times = times[i][mask_i]
            seq_types = types[i][mask_i]

            if len(seq_times) == 0:
                continue

            t0 = seq_times[0].item()
            
            # Compute relative times
            time_since_start = (seq_times - t0).tolist()
            time_since_last_event = torch.cat(
                [torch.tensor([0.0]), seq_times[1:] - seq_times[:-1]]
            ).tolist()
            event_types = seq_types.tolist()

            # Create one row per event
            for j in range(len(event_types)):
                flattened_events.append(
                    {
                        "seq_idx": offset + i,
                        "event_idx": j,
                        "dim_process": num_event_types,
                        "time_since_start": time_since_start[j],
                        "time_since_last_event": time_since_last_event[j],
                        "type_event": event_types[j],
                    }
                )

        return flattened_events

    def finalize(self) -> None:
        """Properly close the Parquet writer."""
        if self._writer:
            self._writer.close()
            logger.info("SimulationIOManager: IO finalized.")

    def _get_all_data(self) -> List[Dict]:
        """Merge local parquet files or return in-memory list."""
        if self._writer:
            return self._writer.finalize_and_get_all_data()
        return self._accumulated_data

    def clear(self) -> None:
        """Reset state and close writer."""
        self._accumulated_data = []
        self._total_sequences = 0
        if self._writer:
            self._writer.close()
            self._writer = None
