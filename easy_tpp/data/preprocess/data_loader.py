import pytorch_lightning as pl
from torch.utils.data import DataLoader

from easy_tpp.configs.data_config import DataConfig
from easy_tpp.data.preprocess.data_collator import TPPDataCollator
from easy_tpp.data.preprocess.dataset import TPPDataset
from easy_tpp.data.preprocess.event_tokenizer import EventTokenizer
from easy_tpp.utils import load_pickle, logger, py_assert


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
        self.shuffle = data_config.data_loading_specs.shuffle
        self.tokenizer = EventTokenizer(data_config.tokenizer_specs)

        data_loading_specs = data_config.data_loading_specs
        self.padding = data_loading_specs.padding
        self.truncation = data_loading_specs.truncation
        self.tensor_type = data_loading_specs.tensor_type
        self.max_length = data_loading_specs.max_length
        self.num_workers = data_loading_specs.num_workers

        # Initialize data containers
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.predict_data = None

    def build_input(
        self, source_dir: str, data_format: str = "json", split: str = "train"
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
        else:
            try:
                data_format == "json"
                return self._build_input_from_json(source_dir, split)
            except ValueError as e:
                logger.error(f"Error loading data from {source_dir}: {e}")
                raise e

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
        if source_dir.endswith(".json"):
            data = load_dataset(
                "json", data_files={split_mapped: source_dir}, split=split_mapped
            )
        elif source_dir.startswith("easytpp"):
            data = load_dataset(source_dir, split=split_mapped)
        elif source_dir.startswith("NzoCs"):
            data = load_dataset(source_dir, split=split_mapped)
        else:
            raise ValueError("Unsupported source directory format for JSON.")

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
            train_data_dir = self.data_config.get_data_dir("train")
            self.train_data = self.build_input(
                train_data_dir, self.data_config.data_format, "train"
            )
            self.train_dataset = TPPDataset(self.train_data)
            logger.info(
                f"Train dataset created with {len(self.train_dataset)} sequences"
            )

            val_data_dir = self.data_config.get_data_dir("dev")
            self.val_data = self.build_input(
                val_data_dir, self.data_config.data_format, "dev"
            )
            self.val_dataset = TPPDataset(self.val_data)
            logger.info(
                f"Validation dataset created with {len(self.val_dataset)} sequences"
            )

        # Set up dataset for testing
        if stage == "test" or stage == "predict" or stage == "simulation":
            test_data_dir = self.data_config.get_data_dir("test")
            self.test_data = self.build_input(
                test_data_dir, self.data_config.data_format, "test"
            )
            self.test_dataset = TPPDataset(self.test_data)

    def train_dataloader(self):
        """Return the training data loader.

        Returns:
            DataLoader: PyTorch DataLoader for training data
        """
        collate_fn = TPPDataCollator(
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            return_tensors=self.tensor_type,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """Return the validation data loader.

        Returns:
            DataLoader: PyTorch DataLoader for validation data
        """

        collatefn = TPPDataCollator(
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            return_tensors=self.tensor_type,
        )

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collatefn,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """Return the test data loader.

        Returns:
            DataLoader: PyTorch DataLoader for test data
        """
        collatefn = TPPDataCollator(
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            return_tensors=self.tensor_type,
        )

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collatefn,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        """Return the prediction data loader.

        Returns:
            DataLoader: PyTorch DataLoader for prediction data
        """
        collatefn = TPPDataCollator(
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_length=self.max_length,
            truncation=self.truncation,
            return_tensors=self.tensor_type,
        )

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collatefn,
            num_workers=self.num_workers,
        )

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
