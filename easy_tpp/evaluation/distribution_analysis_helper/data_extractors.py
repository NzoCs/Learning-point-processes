"""
Data Extractors for Temporal Point Process Analysis
"""

from easy_tpp.utils import logger
from easy_tpp.data.preprocess.dataset import TPPDataset
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


class TPPDatasetExtractor:
    """Extracts data directly from TPPDataset for improved performance (SRP).

    This extractor works directly with TPPDataset objects, avoiding the overhead
    of DataLoader iteration and batch processing. This results in significantly
    faster data extraction compared to the legacy LabelDataExtractor.
    """

    def __init__(self, dataset: TPPDataset, dataset_size: int = 10**5):
        """Initialize the extractor.

        Args:
            dataset: TPPDataset object containing the data
            dataset_size: Maximum number of events to extract
        """
        self.dataset = dataset
        self.dataset_size = dataset_size
        self._time_deltas = []
        self._event_types = []
        self._sequence_lengths = []

    def extract_time_deltas(self) -> np.ndarray:
        """Extract time delta sequences.

        Returns:
            np.ndarray: Array of time deltas
        """
        if not self._time_deltas:
            self._extract_all_data()
        return np.array(self._time_deltas[: self.dataset_size])

    def extract_event_types(self) -> np.ndarray:
        """Extract event type sequences.

        Returns:
            np.ndarray: Array of event types
        """
        if not self._event_types:
            self._extract_all_data()
        return np.array(self._event_types[: self.dataset_size])

    def extract_sequence_lengths(self) -> List[int]:
        """Extract sequence length information.

        Returns:
            List[int]: List of sequence lengths
        """
        if not self._sequence_lengths:
            self._extract_all_data()
        return self._sequence_lengths

    def _extract_all_data(self) -> None:
        """Extract all data from the TPPDataset directly.

        This method processes the dataset sequentially without batching overhead,
        resulting in improved performance compared to DataLoader-based extraction.
        """
        logger.info(
            f"Extracting ground truth data from TPPDataset with {len(self.dataset)} sequences..."
        )

        dataset_len = len(self.dataset)
        total_events = 0
        processed_sequences = 0

        for idx in range(dataset_len):
            if total_events >= self.dataset_size:
                logger.info(f"Reached dataset size limit ({self.dataset_size} events)")
                break

            try:
                sample = self.dataset[idx]
                time_delta_seq = sample["time_delta_seqs"]
                type_seq = sample["type_seqs"]

                # Convert to numpy if they are tensors
                if torch.is_tensor(time_delta_seq):
                    time_delta_seq = time_delta_seq.cpu().numpy()
                if torch.is_tensor(type_seq):
                    type_seq = type_seq.cpu().numpy()

                # Filter out padding tokens (assuming -1 is padding for type sequences)
                if isinstance(type_seq, np.ndarray):
                    valid_mask = type_seq != -1
                    time_deltas = (
                        time_delta_seq[valid_mask]
                        if isinstance(time_delta_seq, np.ndarray)
                        else time_delta_seq
                    )
                    event_types = type_seq[valid_mask]
                else:
                    time_deltas = time_delta_seq
                    event_types = type_seq

                seq_length = len(time_deltas)
                if seq_length > 0:  # Only process non-empty sequences
                    self._sequence_lengths.append(seq_length)
                    self._time_deltas.extend(time_deltas)
                    self._event_types.extend(event_types)
                    total_events += seq_length
                    processed_sequences += 1

            except Exception as e:
                logger.warning(f"Error processing sequence {idx}: {str(e)}")
                continue

        logger.info(
            f"Successfully processed {processed_sequences}/{dataset_len} sequences, "
            f"extracted {total_events} events"
        )


class LabelDataExtractor:
    """Extracts data from PyTorch DataLoader (SRP)."""

    def __init__(self, data_loader, dataset_size: int = 10**5):
        self.data_loader = data_loader
        self.dataset_size = dataset_size
        self._time_deltas = []
        self._event_types = []
        self._sequence_lengths = []

    def extract_time_deltas(self) -> np.ndarray:
        if not self._time_deltas:
            self._extract_all_data()
        return np.array(self._time_deltas[: self.dataset_size])

    def extract_event_types(self) -> np.ndarray:
        if not self._event_types:
            self._extract_all_data()
        return np.array(self._event_types[: self.dataset_size])

    def extract_sequence_lengths(self) -> List[int]:
        if not self._sequence_lengths:
            self._extract_all_data()
        return self._sequence_lengths

    def _extract_all_data(self) -> None:
        """Extract all data from the data loader."""
        logger.info("Extracting ground truth data from dataloader...")

        for batch_idx, batch in enumerate(self.data_loader):
            if self._process_batch(batch):
                break

    def _process_batch(self, batch) -> bool:
        """Process a single batch. Returns True if dataset size limit reached."""
        try:
            time_delta_seqs, type_seqs, attention_mask = self._extract_batch_tensors(
                batch
            )

            if time_delta_seqs is None or type_seqs is None:
                return False

            batch_size = len(time_delta_seqs)
            for i in range(batch_size):
                if self._process_sequence(
                    time_delta_seqs[i],
                    type_seqs[i],
                    attention_mask[i] if attention_mask is not None else None,
                ):
                    return True
            return False

        except Exception as e:
            logger.warning(f"Error processing batch: {str(e)}")
            return False

    def _extract_batch_tensors(
        self, batch
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract relevant tensors from batch with flexible format handling.

        Args:
            batch: Batch data in various possible formats

        Returns:
            Tuple of (time_delta_seqs, type_seqs, attention_mask)
        """
        try:
            if isinstance(batch, dict):
                return (
                    batch.get("time_delta_seqs"),
                    batch.get("type_seqs"),
                    batch.get("attention_mask"),
                )
            elif hasattr(batch, "values"):
                batch_values = list(batch.values())
                if len(batch_values) >= 3:
                    return batch_values[1], batch_values[2], batch_values[3]
            elif isinstance(batch, (list, tuple)) and len(batch) >= 3:
                return batch[1], batch[2], batch[3] if len(batch) > 3 else None

            return None, None, None

        except Exception as e:
            logger.warning(f"Failed to extract tensors from batch: {str(e)}")
            return None, None, None

    def _process_sequence(
        self,
        time_deltas: torch.Tensor,
        event_types: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> bool:
        """
        Process a single sequence and add to appropriate data containers.

        Args:
            time_deltas: Time delta sequence tensor
            event_types: Event type sequence tensor
            attention_mask: Optional attention mask tensor

        Returns:
            True if dataset size limit reached, False otherwise
        """
        try:
            if attention_mask is not None:
                valid_mask = attention_mask.bool()
            else:
                valid_mask = torch.ones_like(time_deltas, dtype=torch.bool)

            valid_time_deltas = time_deltas[valid_mask].cpu().numpy()
            valid_event_types = event_types[valid_mask].cpu().numpy()

            self._sequence_lengths.append(len(valid_time_deltas))
            self._time_deltas.extend(valid_time_deltas)
            self._event_types.extend(valid_event_types)

            return len(self._time_deltas) >= self.dataset_size

        except Exception as e:
            logger.warning(f"Error processing sequence: {str(e)}")
            return False


class SimulationDataExtractor:
    """Extracts data from simulation sequences (SRP)."""

    def __init__(self, simulation: List[Dict], dataset_size: int = 10**5):
        self.simulation = simulation
        self.dataset_size = dataset_size
        self._time_deltas = []
        self._event_types = []
        self._sequence_lengths = []

    def extract_time_deltas(self) -> np.ndarray:
        if not self._time_deltas:
            self._extract_all_data()
        return np.array(self._time_deltas[: self.dataset_size])

    def extract_event_types(self) -> np.ndarray:
        if not self._event_types:
            self._extract_all_data()
        return np.array(self._event_types[: self.dataset_size])

    def extract_sequence_lengths(self) -> List[int]:
        if not self._sequence_lengths:
            self._extract_all_data()
        return self._sequence_lengths

    def _extract_all_data(self) -> None:
        """Extract all data from simulation sequences."""
        logger.info("Processing simulation data...")

        for seq_idx, seq in enumerate(self.simulation):
            if self._process_sequence(seq):
                break

    def _process_sequence(self, seq: Dict) -> bool:
        """Process a single simulation sequence."""
        try:
            if "time_delta_seq" not in seq or "event_seq" not in seq:
                return False

            time_deltas = seq["time_delta_seq"].cpu().numpy()
            event_types = seq["event_seq"].cpu().numpy()

            self._sequence_lengths.append(len(time_deltas))
            self._time_deltas.extend(time_deltas)
            self._event_types.extend(event_types)

            return len(self._time_deltas) >= self.dataset_size

        except Exception as e:
            logger.warning(f"Error processing simulation sequence: {str(e)}")
            return False
