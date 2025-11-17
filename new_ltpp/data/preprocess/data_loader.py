import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Iterator, Literal, Optional
import numpy as np

from new_ltpp.configs.data_config import DataConfig
from new_ltpp.data.preprocess.data_collator import TPPDataCollator
from new_ltpp.data.preprocess.dataset import TPPDataset
from new_ltpp.data.preprocess.event_tokenizer import EventTokenizer
from new_ltpp.shared_types import Batch, DataStats
from new_ltpp.utils import load_pickle, logger, py_assert


class TypedDataLoader:
    """Typed DataLoader for Temporal Point Process event sequences.

    This class wraps a PyTorch DataLoader to provide typed access to TPP event
    sequences, ensuring that batches are returned in the expected format.

    Args:
        data_loader (DataLoader): The underlying PyTorch DataLoader.
    """

    def __init__(self, data_loader: DataLoader[Batch]):
        self.data_loader = data_loader

    def __iter__(self) -> Iterator[Batch]:
        """Return an iterator over the DataLoader."""
        for batch in self.data_loader:
            yield batch

    def __len__(self) -> int:
        """Return the number of batches in the DataLoader."""
        return len(self.data_loader)


# PyTorch Lightning DataModule for TPP
class TPPDataModule(pl.LightningDataModule):
    
    def __init__(self, data_config: DataConfig):
        """Initialize the PyTorch Lightning DataModule.

        Args:
            data_config (DataConfig): Configuration for the dataset
        """
        super().__init__()
        self.data_config = data_config
        self.num_event_types = data_config.tokenizer_specs.num_event_types
        self.batch_size = data_config.data_loading_specs.batch_size
        self.tokenizer = EventTokenizer(data_config.tokenizer_specs)
        self.tokenizer_specs = data_config.tokenizer_specs

        data_loading_specs = data_config.data_loading_specs
        self.num_workers = data_loading_specs.num_workers

        # Initialize data containers
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.predict_data = None
        
    def estimate_dtime_range(self, split: str = "train", quantile: float = 0.995, nb_samples: int = 10**5):
        """
        Estimate the minimum and maximum time deltas (dtime) from the dataset.
        This is useful for understanding the range of time intervals in the data.

        Args:
            split (str): Dataset split to use ("train", "val", "test").
            quantile (float): Quantile to use for robustness (default=0.995).
        Returns:
            Tuple[float, float]: Estimated (dtime_min, dtime_max).
        """
        # Ensure data is loaded using setup()
        stage = "fit" if split == "train" else split
        data_attr = f"{split}_data"
        
        if getattr(self, data_attr, None) is None:
            self.setup(stage)

        # Sélectionner le bon dataset

        data = getattr(self, f"{split}_data")
        time_deltas = []
        dt_count = 0
        # Collect up to nb_samples time deltas, cap at nb_samples to avoid excessive memory use
        for seq in data["time_delta_seqs"]:
            for dt in seq[:nb_samples]:
                time_deltas.append(dt)
                dt_count += 1
            if dt_count >= nb_samples:
                break

        if len(time_deltas) == 0:
            raise ValueError("No time deltas found for estimating dtime range.")

        # Calcul robuste : quantiles pour éviter les outliers extrêmes
        dtime_min_estimate = float(np.quantile(time_deltas, 1 - quantile))
        dtime_max_estimate = float(np.quantile(time_deltas, quantile))

        logger.info(
            f"Estimated dtime range from {split} data: "
            f"({dtime_min_estimate:.4f}, {dtime_max_estimate:.4f}) (quantile={quantile})"
        )

        return dtime_min_estimate, dtime_max_estimate
    
    def estimate_end_time_max(self, split: str = "train", quantile: float = 0.995, nb_samples: int = 10**5):
        """
        Estimate the maximum end time (last event time) from the dataset.
        This is useful for understanding the temporal extent of sequences.

        Args:
            split (str): Dataset split to use ("train", "val", "test").
            quantile (float): Quantile to use for robustness (default=0.995).
            nb_samples (int): Maximum number of sequences to sample (default=100000).
        Returns:
            float: Estimated maximum end time.
        """
        # Ensure data is loaded using setup()
        stage = "fit" if split == "train" else split
        data_attr = f"{split}_data"
        
        if getattr(self, data_attr, None) is None:
            self.setup(stage)

        # Sélectionner le bon dataset
        data = getattr(self, f"{split}_data")
        end_times = []
        count = 0
        # Collect up to nb_samples end times (last event in each sequence)
        for seq in data["time_seqs"]:
            if len(seq) > 0:
                end_times.append(seq[-1])  # Last event time in sequence
                count += 1
            if count >= nb_samples:
                break

        if len(end_times) == 0:
            raise ValueError("No end times found for estimating maximum end time.")

        # Calcul robuste : quantile pour éviter les outliers extrêmes
        end_time_max_estimate = float(np.quantile(end_times, quantile))
        self.end_time_max_estimate = end_time_max_estimate

        logger.info(
            f"Estimated maximum end time from {split} data: "
            f"{end_time_max_estimate:.4f} (quantile={quantile})"
        )

        return end_time_max_estimate
    
    def get_data_stats(self) -> DataStats:
        """Get dataset statistics for model configuration.

        Returns:
            DataStats: Dataset statistics including number of event types,
                       maximum sequence length, and maximum time delta.
        """
        # Estimate dtime_max from training data
        dtime_min, dtime_max = self.estimate_dtime_range(split="train", quantile=0.995)
        end_time_max = self.estimate_end_time_max(split="train", quantile=0.995)

        # Get number of event types and max sequence length from tokenizer specs
        num_event_types = self.tokenizer_specs.num_event_types

        data_stats = DataStats(
            num_event_types=num_event_types,
            end_time_max=end_time_max,
            dtime_max=dtime_max,
        )

        return data_stats


    def build_input(
        self, source_dir: str, data_format: Literal["json", "pkl", "hf"] = "json", split: str = "train"
    ) -> dict:
        """Helper function to load and process dataset based on file format.

        Args:
            source_dir (str): Path to dataset directory.
            data_format (str): json or pkl
            split (str): Dataset split, e.g., 'train', 'dev', 'test'.

        Returns:
            dict: Dictionary containing sequences of event times, types, and intervals.
        """
        if data_format == "pkl":
            return self._build_input_from_pkl(source_dir, split)
        elif data_format == "json":
            try:
                return self._build_input_from_json(source_dir, split)
            except ValueError as e:
                logger.error(f"Error loading data from {source_dir}: {e}")
                raise e
        elif data_format == "hf":
            return self._build_input_from_hf(source_dir, split)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
    def _build_input_from_hf(self, source_dir: str, split: str) -> dict:
        """Load and process data from a Hugging Face dataset.

        Args:
            source_dir (str): Hugging Face dataset name.
            split (str): Dataset split, e.g., 'train', 'dev', 'test'.
        Returns:
            dict: Dictionary with processed event sequences.
        """
        from datasets import load_dataset

        split_mapped = "validation" if split == "dev" else split
        data = load_dataset(source_dir, split=split_mapped)

        py_assert(
            data["dim_process"][0] == self.num_event_types,
            ValueError,
            "Inconsistent dim_process in different splits.",
        )

        return {
            "time_seqs": data["time_since_start"],
            "type_seqs": data["type_event"],
            "time_delta_seqs": data["time_since_last_event"],
        }

    def _build_input_from_pkl(self, source_dir: str, split: str) -> dict:
        """Load and process data from a pickle file.

        Args:
            source_dir (str): Path to the pickle file.
            split (str): Dataset split, e.g., 'train', 'dev', 'test'.

        Returns:
            dict: Dictionary with processed event sequences.
        """
        data = load_pickle(source_dir)
        py_assert(
            data["dim_process"] == self.num_event_types,
            ValueError,
            "Inconsistent dim_process in different splits.",
        )

        source_data = data[split]
        return {
            "time_seqs": [[x["time_since_start"] for x in seq] for seq in source_data],
            "type_seqs": [[x["type_event"] for x in seq] for seq in source_data],
            "time_delta_seqs": [
                [x["time_since_last_event"] for x in seq] for seq in source_data
            ],
        }

    def _build_input_from_json(self, source_dir: str, split: str) -> dict:
        """Load and process data from a JSON file.

        Args:
            source_dir (str): Path to the JSON file or Hugging Face dataset name.
            split (str): Dataset split, e.g., 'train', 'dev', 'test'.

        Returns:
            dict: Dictionary with processed event sequences.
        """
        from datasets import load_dataset

        split_mapped = "validation" if split == "dev" else split
        data = load_dataset("json", data_files={split: source_dir}, split=split_mapped)

        py_assert(
            data["dim_process"][0] == self.num_event_types,
            ValueError,
            "Inconsistent dim_process in different splits.",
        )

        return {
            "time_seqs": data["time_since_start"],
            "type_seqs": data["type_event"],
            "time_delta_seqs": data["time_since_last_event"],
        }

    def prepare_data(self):
        """Prepare data if needed (download data, etc.)."""
        # This method is called only once and on only one GPU
        pass

    def setup(self, stage: str) -> None:
        """Set up datasets for each stage.

        Args:
            stage (str): 'fit', 'validate', 'test', or 'predict'
        """

        # Validate stage input
        valid_stages = ["fit", "train", "test", "predict", "simulation"]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}.")
        logger.info(f"Setting up data for stage: {stage}")

        # Set up datasets for training and validation
        if stage == "fit" or stage == "train":
            # Only load train data if not already loaded
            if self.train_data is None:
                train_data_dir = self.data_config.get_data_dir("train")
                data_format: Literal["pkl", "json", "hf"] = self.data_config.data_format
                self.train_data = self.build_input(
                    train_data_dir, data_format, "train"
                )
                self.train_dataset = TPPDataset(self.train_data)
                logger.info(
                    f"Train dataset created with {len(self.train_dataset)} sequences"
                )
            else:
                logger.info("Train data already loaded, skipping setup")

            # Only load validation data if not already loaded
            if self.val_data is None:
                val_data_dir = self.data_config.get_data_dir("dev")
                data_format: Literal["pkl", "json", "hf"] = self.data_config.data_format
                self.val_data = self.build_input(
                    val_data_dir, data_format, "dev"
                )
                self.val_dataset = TPPDataset(self.val_data)
                logger.info(
                    f"Validation dataset created with {len(self.val_dataset)} sequences"
                )
            else:
                logger.info("Validation data already loaded, skipping setup")

        # Set up dataset for testing
        if stage == "test" or stage == "predict" or stage == "simulation":
            # Only load test data if not already loaded
            if self.test_data is None:
                test_data_dir = self.data_config.get_data_dir("test")
                data_format: Literal["pkl", "json", "hf"] = self.data_config.data_format
                self.test_data = self.build_input(
                    test_data_dir, data_format, "test"
                )
                self.test_dataset = TPPDataset(self.test_data)
                logger.info(
                    f"Test dataset created with {len(self.test_dataset)} sequences"
                )
            else:
                logger.info("Test data already loaded, skipping setup")

    def train_dataloader(self):
        """Return the training data loader.

        Returns:
            DataLoader: PyTorch DataLoader for training data
        """
        collate_fn = TPPDataCollator(
            tokenizer=self.tokenizer,
        )

        train_data_loader = TypedDataLoader(DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True,
        ))

        return train_data_loader

    def val_dataloader(self):
        """Return the validation data loader.

        Returns:
            DataLoader: PyTorch DataLoader for validation data
        """
        collate_fn = TPPDataCollator(
            tokenizer=self.tokenizer,
        )

        val_data_loader = TypedDataLoader(DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True,
        ))

        return val_data_loader

    def test_dataloader(self):
        """Return the test data loader.

        Returns:
            DataLoader: PyTorch DataLoader for test data
        """
        collate_fn = TPPDataCollator(
            tokenizer=self.tokenizer,
        )

        test_data_loader = TypedDataLoader(DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True,
        ))

        return test_data_loader

    # Convenience methods that match the TPPDataLoader API
    def train_loader(self, **kwargs):
        """Alias for train_dataloader with optional kwargs.

        Returns:
            DataLoader: PyTorch DataLoader for training data
        """
        if kwargs:
            orig_batch_size = self.batch_size
            orig_shuffle = self.shuffle
            self.batch_size = kwargs.get("batch_size", self.batch_size)
            self.shuffle = kwargs.get("shuffle", self.shuffle)
            loader = self.train_dataloader()
            self.batch_size = orig_batch_size
            self.shuffle = orig_shuffle
            return loader
        return self.train_dataloader()

    def valid_loader(self, **kwargs):
        """Alias for val_dataloader with optional kwargs.

        Returns:
            DataLoader: PyTorch DataLoader for validation data
        """
        if kwargs:
            orig_batch_size = self.batch_size
            self.batch_size = kwargs.get("batch_size", self.batch_size)
            loader = self.val_dataloader()
            self.batch_size = orig_batch_size
            return loader
        return self.val_dataloader()

    def test_loader(self, **kwargs):
        """Alias for test_dataloader with optional kwargs.

        Returns:
            DataLoader: PyTorch DataLoader for test data
        """
        if kwargs:
            orig_batch_size = self.batch_size
            self.batch_size = kwargs.get("batch_size", self.batch_size)
            loader = self.test_dataloader()
            self.batch_size = orig_batch_size
            return loader
        return self.test_dataloader()

    def get_dataloader(self, split: str, **kwargs):
        """Get dataloader for a specific split.

        Args:
            split (str): One of 'train', 'dev'/'val', 'test', or 'predict'
            **kwargs: Optional arguments to override default settings

        Returns:
            DataLoader: PyTorch DataLoader for the specified split

        Raises:
            ValueError: If split is not recognized
        """
        if split == "train":
            return self.train_loader(**kwargs)
        elif split in ["val", "dev"]:
            return self.valid_loader(**kwargs)
        elif split == "test":
            return self.test_loader(**kwargs)
        elif split == "predict":
            return self.predict_dataloader()
        else:
            raise ValueError(
                f"Unrecognized split: {split}. Must be one of 'train', 'dev'/'val', 'test', or 'predict'"
            )
