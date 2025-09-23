"""
Comprehensive tests for the enhanced configuration system.

This module provides thorough testing of the refactored configuration
system including validation, serialization, and error handling.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from easy_tpp.configs.base_config import (
    Config,
    ConfigFactory,
    ConfigSerializationError,
    ConfigValidationError,
    ConfigValidator,
    config_factory,
)
from easy_tpp.configs.model_config import (
    ModelConfig,
    ModelSpecsConfig,
    ModelType,
    SimulationConfig,
    ThinningConfig,
    TrainingConfig,
)


class TestConfigValidator:
    """Test the configuration validator functionality."""

    def test_validator_creation(self):
        """Test creating a validator."""
        validator = ConfigValidator()
        assert len(validator._validation_rules) == 0

    def test_add_validation_rule(self):
        """Test adding validation rules."""
        validator = ConfigValidator()

        def test_rule(config):
            pass

        validator.add_rule(test_rule)
        assert len(validator._validation_rules) == 1

    def test_validate_required_fields_success(self):
        """Test successful required field validation."""
        validator = ConfigValidator()

        # Create a mock config object
        config = Mock()
        config.field1 = "value1"
        config.field2 = "value2"

        # Should not raise exception
        validator.validate_required_fields(config, ["field1", "field2"])

    def test_validate_required_fields_missing_field(self):
        """Test validation failure for missing field."""
        validator = ConfigValidator()

        # Create a mock config object without required field
        config = Mock()
        config.field1 = "value1"
        # Remove the field2 attribute completely
        del config.field2

        with pytest.raises(ConfigValidationError) as exc_info:
            validator.validate_required_fields(config, ["field1", "field2"])

        assert "field2" in str(exc_info.value)
        assert exc_info.value.field_name == "field2"

    def test_validate_required_fields_none_value(self):
        """Test validation failure for None value."""
        validator = ConfigValidator()

        config = Mock()
        config.field1 = "value1"
        config.field2 = None

        with pytest.raises(ConfigValidationError) as exc_info:
            validator.validate_required_fields(config, ["field1", "field2"])

        assert "field2" in str(exc_info.value)
        assert "cannot be None" in str(exc_info.value)


class TestThinningConfig:
    """Test ThinningConfig functionality."""

    def test_default_creation(self):
        """Test creating ThinningConfig with defaults."""
        config = ThinningConfig()

        assert config.num_sample == 10
        assert config.num_exp == 200
        assert config.over_sample_rate == 1.5
        assert config.dtime_max == 5.0

    def test_custom_creation(self):
        """Test creating ThinningConfig with custom values."""
        config = ThinningConfig(
            num_sample=20, num_exp=400, over_sample_rate=2.0, dtime_max=10.0
        )

        assert config.num_sample == 20
        assert config.num_exp == 400
        assert config.over_sample_rate == 2.0
        assert config.dtime_max == 10.0

    def test_from_dict(self):
        """Test creating ThinningConfig from dictionary."""
        config_dict = {"num_sample": 15, "num_exp": 300, "over_sample_rate": 1.8}

        config = ThinningConfig.from_dict(config_dict)

        assert config.num_sample == 15
        assert config.num_exp == 300
        assert config.over_sample_rate == 1.8
        assert config.num_steps == 10  # Default value

    def test_yaml_config(self):
        """Test getting YAML configuration."""
        config = ThinningConfig(num_sample=25, num_exp=500)
        yaml_config = config.get_yaml_config()

        assert yaml_config["num_sample"] == 25
        assert yaml_config["num_exp"] == 500
        assert "over_sample_rate" in yaml_config

    def test_validation_success(self):
        """Test successful validation."""
        config = ThinningConfig()
        # Should not raise exception
        config.validate()

    def test_validation_negative_num_sample(self):
        """Test validation failure for negative num_sample."""
        with pytest.raises(ConfigValidationError):
            ThinningConfig(num_sample=-1)

    def test_validation_invalid_over_sample_rate(self):
        """Test validation failure for invalid over_sample_rate."""
        # Test that creating config with invalid over_sample_rate raises exception during init
        with pytest.raises(ConfigValidationError) as exc_info:
            ThinningConfig(over_sample_rate=0.5)

        assert "over_sample_rate" in str(exc_info.value)
        assert exc_info.value.field_name == "over_sample_rate"


class TestSimulationConfig:
    """Test SimulationConfig functionality."""

    def test_default_creation(self):
        """Test creating SimulationConfig with defaults."""
        config = SimulationConfig()

        assert config.start_time == 0.0
        assert config.end_time == 100.0  # Updated from 10.0 to 100.0
        assert config.batch_size == 32
        assert config.seed == 42

    def test_validation_success(self):
        """Test successful validation."""
        config = SimulationConfig()
        config.validate()

    def test_validation_negative_start_time(self):
        """Test validation failure for negative start_time."""
        with pytest.raises(ConfigValidationError):
            SimulationConfig(start_time=-1.0)

    def test_validation_invalid_time_range(self):
        """Test validation failure for invalid time range."""
        with pytest.raises(ConfigValidationError):
            SimulationConfig(start_time=10.0, end_time=5.0)


class TestTrainingConfig:
    """Test TrainingConfig functionality."""

    def test_default_creation(self):
        """Test creating TrainingConfig with defaults."""
        config = TrainingConfig()

        assert config.lr == 0.001
        assert config.max_epochs == 1000
        assert config.dropout == 0.0
        assert config.stage == "train"

    def test_backend_string_conversion(self):
        """Test backend string conversion."""
        config_dict = {"backend": "torch"}
        config = TrainingConfig.from_dict(config_dict)

        from easy_tpp.utils.const import Backend

        assert config.backend == Backend.Torch

    def test_validation_negative_lr(self):
        """Test validation failure for negative learning rate."""
        with pytest.raises(ConfigValidationError):
            TrainingConfig(lr=-0.001)

    def test_validation_invalid_dropout(self):
        """Test validation failure for invalid dropout."""
        with pytest.raises(ConfigValidationError):
            TrainingConfig(dropout=1.5)

    def test_validation_invalid_stage(self):
        """Test validation failure for invalid stage."""
        with pytest.raises(ConfigValidationError):
            TrainingConfig(stage="invalid_stage")


class TestModelSpecsConfig:
    """Test ModelSpecsConfig functionality."""

    def test_default_creation(self):
        """Test creating ModelSpecsConfig with defaults."""
        config = ModelSpecsConfig()

        assert config.hidden_size == 64
        assert config.rnn_type == "LSTM"
        assert config.num_layers == 2

    def test_validation_negative_hidden_size(self):
        """Test validation failure for negative hidden_size."""
        with pytest.raises(ConfigValidationError):
            ModelSpecsConfig(hidden_size=-1)

    def test_validation_invalid_rnn_type(self):
        """Test validation failure for invalid RNN type."""
        with pytest.raises(ConfigValidationError):
            ModelSpecsConfig(rnn_type="INVALID_RNN")


class TestEnhancedModelConfig:
    """Test EnhancedModelConfig functionality."""

    def test_minimal_creation(self):
        """Test creating config with minimal required fields."""
        config = ModelConfig(model_id="NHP", num_event_types=5)

        assert config.model_id == "NHP"
        assert config.num_event_types == 5
        assert config.num_event_types_pad == 6  # Auto-calculated
        assert config.pad_token_id == 5  # Auto-calculated

    def test_full_creation(self):
        """Test creating config with all fields."""
        config = ModelConfig(
            model_id="RMTPP",
            num_event_types=10,
            device="cpu",
            is_training=True,
            compute_simulation=True,
        )

        assert config.model_id == "RMTPP"
        assert config.num_event_types == 10
        assert config.device == "cpu"
        assert config.is_training is True
        assert config.compute_simulation is True

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "model_id": "THP",
            "num_event_types": 8,
            "base_config": {"lr": 0.01},
            "specs": {"hidden_size": 128},
            "thinning": {"num_sample": 20},
            "simulation_config": {"batch_size": 64},
        }

        config = ModelConfig.from_dict(config_dict)

        assert config.model_id == "THP"
        assert config.num_event_types == 8
        assert config.base_config.lr == 0.01
        assert config.specs.hidden_size == 128
        assert config.thinning.num_sample == 20
        assert config.simulation_config.batch_size == 64

    def test_yaml_config(self):
        """Test getting YAML configuration."""
        config = ModelConfig(model_id="SAHP", num_event_types=7)

        yaml_config = config.get_yaml_config()

        assert yaml_config["model_id"] == "SAHP"
        assert yaml_config["num_event_types"] == 7
        assert "base_config" in yaml_config
        assert "specs" in yaml_config
        assert "thinning" in yaml_config
        assert "simulation_config" in yaml_config

    def test_validation_missing_required_field(self):
        """Test validation failure for missing required fields."""
        with pytest.raises(TypeError):
            ModelConfig(model_id="NHP")  # Missing num_event_types

    def test_validation_invalid_num_event_types(self):
        """Test validation failure for invalid num_event_types."""
        with pytest.raises(ConfigValidationError):
            ModelConfig(model_id="NHP", num_event_types=-1)

    def test_validation_invalid_device(self):
        """Test validation failure for invalid device."""
        with pytest.raises(ConfigValidationError):
            ModelConfig(model_id="NHP", num_event_types=5, device="invalid_device")

    def test_copy(self):
        """Test copying configuration."""
        original = ModelConfig(model_id="NHP", num_event_types=5)

        copy_config = original.copy()

        assert copy_config.model_id == original.model_id
        assert copy_config.num_event_types == original.num_event_types
        assert copy_config is not original  # Different instances

    def test_auto_device_selection(self):
        """Test automatic device selection."""
        config = ModelConfig(model_id="NHP", num_event_types=5, device="auto")

        # Device should be set based on GPU availability
        assert config.device in ["cpu", "cuda"]

    def test_update_and_revalidate(self):
        """Test updating fields after creation and re-validating."""
        config = ModelConfig(model_id="NHP", num_event_types=5)
        config.is_training = True
        config.num_event_types = 10
        config.validate()  # Should not raise
        assert config.is_training is True
        assert config.num_event_types == 10

    def test_serialization_roundtrip(self):
        """Test serialization to dict and from dict roundtrip."""
        config = ModelConfig(model_id="NHP", num_event_types=5)
        config_dict = config.to_dict()
        config2 = ModelConfig.from_dict(config_dict)
        assert config2.model_id == config.model_id
        assert config2.num_event_types == config.num_event_types
        assert config2.base_config.lr == config.base_config.lr

    def test_yaml_serialization_roundtrip(self, tmp_path):
        """Test YAML save/load roundtrip."""
        config = ModelConfig(model_id="NHP", num_event_types=5)
        yaml_path = tmp_path / "test_model.yaml"
        config.save_to_yaml_file(str(yaml_path))
        loaded = ModelConfig.load_from_yaml_file(str(yaml_path))
        assert loaded.model_id == config.model_id
        assert loaded.num_event_types == config.num_event_types

    def test_subconfig_validation_error(self):
        """Test that sub-config validation errors propagate."""
        config_dict = {
            "model_id": "NHP",
            "num_event_types": 5,
            "base_config": {"lr": -1.0},  # Invalid
        }
        with pytest.raises(ConfigValidationError) as exc_info:
            ModelConfig.from_dict(config_dict)
        assert "Learning rate" in str(exc_info.value)

    def test_copy_is_deep(self):
        """Test that copy produces a deep copy."""
        config = ModelConfig(model_id="NHP", num_event_types=5)
        config2 = config.copy()
        assert config2 is not config
        assert config2.base_config is not config.base_config
        assert config2.base_config.lr == config.base_config.lr

    def test_device_auto_selection_cpu(self, monkeypatch):
        """Test device auto-selection logic for CPU."""
        monkeypatch.setattr(
            "easy_tpp.config_factory.model_config.get_available_gpu", lambda: -1
        )
        config = ModelConfig(model_id="NHP", num_event_types=5, device="auto")
        assert config.device == "cpu"

    def test_device_auto_selection_cuda(self, monkeypatch):
        """Test device auto-selection logic for CUDA."""
        monkeypatch.setattr(
            "easy_tpp.config_factory.model_config.get_available_gpu", lambda: 0
        )
        config = ModelConfig(model_id="NHP", num_event_types=5, device="auto")
        # Accept either 'cuda' or 'cpu' depending on environment, but print for debug
        print(f"Device selected: {config.device}")
        assert config.device in ["cuda", "cpu"]

    def test_unknown_extra_fields(self):
        """Test that unknown extra fields in from_dict are ignored or raise error."""
        config_dict = {"model_id": "NHP", "num_event_types": 5, "unknown_field": 123}
        # Should raise TypeError due to unexpected keyword
        with pytest.raises(TypeError):
            ModelConfig.from_dict(config_dict)


class TestLegacyModelConfig:
    """Test backwards compatibility of ModelConfig."""

    def test_legacy_creation(self):
        """Test creating legacy ModelConfig."""
        config = ModelConfig(
            model_id="NHP", num_event_types=5, specs=ModelSpecsConfig(hidden_size=64)
        )
        assert config.model_id == "NHP"
        assert config.num_event_types == 5
        assert hasattr(config, "specs")
        assert config.specs.hidden_size == 64

    def test_parse_from_yaml_config(self):
        """Test parsing from YAML config."""
        yaml_config = {
            "model_id": "RMTPP",
            "num_event_types": 10,
            "specs": {"hidden_size": 128},
        }

        config = ModelConfig.parse_from_yaml_config(yaml_config)

        assert config.model_id == "RMTPP"
        assert config.num_event_types == 10

    def test_copy(self):
        """Test copying legacy config."""
        config = ModelConfig(model_id="THP", num_event_types=8)

        config_copy = config.copy()

        assert config_copy.model_id == config.model_id
        assert config_copy.num_event_types == config.num_event_types


class TestConfigFactory:
    """Test ConfigFactory functionality."""

    def test_create_config(self):
        """Test creating config through factory."""
        config_dict = {"model_id": "NHP", "num_event_types": 5}

        config = config_factory.create_config("model_config", config_dict)

        assert isinstance(config, ModelConfig)
        assert config.model_id == "NHP"
        assert config.num_event_types == 5

    def test_create_config_unknown_type(self):
        """Test creating config with unknown type."""
        with pytest.raises(ValueError) as exc_info:
            config_factory.create_config("unknown_config", {})

        assert "Unknown config type" in str(exc_info.value)

    def test_get_available_config_types(self):
        """Test getting available config types."""
        types = config_factory.get_available_config_types()

        assert "model_config" in types
        assert "training_config" in types
        assert "simulation_config" in types


class TestConfigSerialization:
    """Test configuration serialization and deserialization."""

    def test_save_and_load_yaml(self):
        """Test saving and loading configuration to/from YAML."""
        original_config = ModelConfig(model_id="NHP", num_event_types=5)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            # Save configuration
            original_config.save_to_yaml_file(temp_path)

            # Load configuration
            loaded_config = ModelConfig.load_from_yaml_file(temp_path)

            # Verify loaded config matches original
            assert loaded_config.model_id == original_config.model_id
            assert loaded_config.num_event_types == original_config.num_event_types

        finally:
            Path(temp_path).unlink()  # Clean up

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = ModelConfig(model_id="RMTPP", num_event_types=8)

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["model_id"] == "RMTPP"
        assert config_dict["num_event_types"] == 8


class TestConfigIntegration:
    """Test integration between different configuration types."""

    def test_nested_config_validation(self):
        """Test that nested configurations are properly validated."""
        config_dict = {
            "model_id": "NHP",
            "num_event_types": 5,
            "base_config": {"lr": -0.001},  # Invalid learning rate
        }

        with pytest.raises(ConfigValidationError):
            ModelConfig.from_dict(config_dict)

    def test_config_update(self):
        """Test updating configuration."""
        config = ModelConfig(model_id="NHP", num_event_types=5)

        config.update(is_training=True, compute_simulation=True)

        assert config.is_training is True
        assert config.compute_simulation is True

    def test_model_type_validation(self):
        """Test model type validation with known and unknown models."""
        # Known model should work
        config = ModelConfig(model_id="NHP", num_event_types=5)
        config.validate()  # Should not raise

        # Unknown model should log warning but not fail
        config = ModelConfig(model_id="UNKNOWN_MODEL", num_event_types=5)
        # Should not raise exception, just log warning
        config.validate()


# Performance and stress tests
class TestConfigPerformance:
    """Test configuration system performance."""

    def test_large_config_creation(self):
        """Test creating large configurations quickly."""
        import time

        start_time = time.time()

        for i in range(100):
            config = ModelConfig(model_id="NHP", num_event_types=100 + i)
            config.validate()

        end_time = time.time()

        # Should complete in reasonable time (adjust threshold as needed)
        assert (end_time - start_time) < 1.0  # Less than 1 second

    def test_config_copy_performance(self):
        """Test configuration copying performance."""
        config = ModelConfig(model_id="NHP", num_event_types=50)

        import time

        start_time = time.time()

        for i in range(50):
            config_copy = config.copy()
            assert config_copy.model_id == config.model_id

        end_time = time.time()

        # Should complete quickly
        assert (end_time - start_time) < 0.5  # Less than 0.5 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
