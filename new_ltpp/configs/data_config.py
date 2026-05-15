from typing import List, Literal, Optional, Union
from pathlib import Path
from pydantic import model_validator

from new_ltpp.configs.base_config import Config, ConfigValidationError
from new_ltpp.utils.const import PaddingStrategy, TruncationStrategy
from new_ltpp.configs.config_utils import load_yaml, extract


class TokenizerConfig(Config):
    """Configuration for event tokenizer with strategy-based processing."""

    num_event_types: int
    num_event_types_pad: int = 0  # Will be computed as num_event_types + 1
    pad_token_id: int = 0  # Will be computed as num_event_types
    padding_strategy: Literal["longest", "do_not_pad"] = "longest"
    truncation_strategy: Literal["longest_first", "do_not_truncate"] = "do_not_truncate"
    padding_side: Literal["left", "right"] = "left"
    truncation_side: Literal["left", "right"] = "left"
    model_input_names: Optional[List[str]] = None

    @model_validator(mode="before")
    @classmethod
    def compute_derived(cls, values: dict) -> dict:
        values["num_event_types_pad"] = values["num_event_types"] + 1
        values["pad_token_id"] = values["num_event_types"]
        return values

    @model_validator(mode="after")
    def validate_strategy(self) -> "TokenizerConfig":
        has_truncation = (
            self.truncation_strategy is not None
            and self.truncation_strategy != "do_not_truncate"
        )
        has_padding = (
            self.padding_strategy is not None and self.padding_strategy != "do_not_pad"
        )

        if has_truncation and has_padding:
            raise ConfigValidationError(
                "Cannot specify both an active padding strategy and an active truncation strategy. "
                "Use either padding OR truncation, not both."
            )

        return self

    @property
    def strategy(self) -> Union[PaddingStrategy, TruncationStrategy]:
        has_truncation = (
            self.truncation_strategy is not None
            and self.truncation_strategy != "do_not_truncate"
        )
        has_padding = (
            self.padding_strategy is not None and self.padding_strategy != "do_not_pad"
        )
        if has_truncation:
            return TruncationStrategy(self.truncation_strategy)
        elif has_padding:
            return PaddingStrategy(self.padding_strategy)
        return PaddingStrategy.LONGEST

    @property
    def is_padding_strategy(self) -> bool:
        return isinstance(self.strategy, PaddingStrategy)

    @property
    def is_truncation_strategy(self) -> bool:
        return isinstance(self.strategy, TruncationStrategy)


class DataLoadingSpecsConfig(Config):
    """Configuration for data loading specifications."""

    batch_size: int
    num_workers: int = 1
    shuffle: bool = False
    padding: bool = True
    truncation: bool = False
    max_len: Optional[int] = None


class DataConfig(Config):
    """Configuration for dataset and data processing."""

    num_event_types: int
    data_format: Literal["json", "pkl", "hf"]
    data_loading_specs: DataLoadingSpecsConfig
    tokenizer_specs: TokenizerConfig = TokenizerConfig(
        num_event_types=0
    )  # Will be overridden by validator

    # Internal variables, to be populated through src_dir or explicitly
    dataset_id: str = ""
    train_dir: str = ""
    valid_dir: str = ""
    test_dir: str = ""

    @model_validator(mode="before")
    @classmethod
    def compute_derived(cls, values: dict) -> dict:
        # Compute tokenizer_specs if not provided
        if "tokenizer_specs" not in values or values["tokenizer_specs"] is None:
            values["tokenizer_specs"] = TokenizerConfig(
                num_event_types=values["num_event_types"]
            )
        return values

    @model_validator(mode="before")
    @classmethod
    def resolve_dirs_and_tokenizer(cls, data: dict) -> dict:
        # Resolve src_dir → train/valid/test
        src_dir = data.pop("src_dir", None)
        if src_dir is not None:
            data.setdefault("train_dir", src_dir)
            data.setdefault("valid_dir", src_dir)
            data.setdefault("test_dir", src_dir)
        else:
            data.setdefault("train_dir", "")
            data.setdefault("valid_dir", "")
            data.setdefault("test_dir", "")

        # Default tokenizer_specs
        if data.get("tokenizer_specs") is None:
            data["tokenizer_specs"] = TokenizerConfig(
                num_event_types=data["num_event_types"]
            )

        return data

    @classmethod
    def from_yaml_components(
        cls,
        yaml_path: Union[str, Path],
        dataset_id: str,
        data_config_path: str,
        data_loading_config_path: str,
        tokenizer_config_path: Optional[str] = None,
    ) -> "DataConfig":
        data = load_yaml(yaml_path)

        data_info = extract(data, data_config_path)
        loading_info = extract(data, data_loading_config_path)

        src_dir = data_info.pop("src_dir", "")
        train_dir = data_info.pop("train_dir", src_dir)
        valid_dir = data_info.pop("valid_dir", src_dir)
        test_dir = data_info.pop("test_dir", src_dir)

        tokenizer_kwargs = (
            extract(data, tokenizer_config_path) if tokenizer_config_path else {}
        )
        tokenizer_specs = TokenizerConfig(
            num_event_types=data_info["num_event_types"], **tokenizer_kwargs
        )

        return cls(
            dataset_id=dataset_id,
            data_loading_specs=DataLoadingSpecsConfig(**loading_info),
            train_dir=train_dir,
            valid_dir=valid_dir,
            test_dir=test_dir,
            tokenizer_specs=tokenizer_specs,
            **data_info,
        )

    def get_data_dir(self, split: str) -> str:
        if split in ["train", "dev", "valid", "test"]:
            split = split.lower()
            if split == "train":
                return self.train_dir
            elif split in ["dev", "valid"]:
                return self.valid_dir
            elif split == "test":
                return self.test_dir
        raise ValueError(
            f"Unknown split: {split}. Valid splits are: train, valid, test."
        )
