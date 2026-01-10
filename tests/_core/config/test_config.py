import os
import tempfile
from typing import Any, Dict, Generator

import pytest
import yaml
from axion._core.config.config import Config, ConfigurationError


class TestConfig:
    """Tests for the Config class."""

    @pytest.fixture
    def sample_config_dict(self) -> Dict[str, Any]:
        """Fixture providing a sample configuration dictionary."""
        return {
            'app': {'name': 'test-app', 'version': '1.0.0', 'debug': True},
            'database': {
                'host': 'localhost',
                'port': 5432,
                'credentials': {'username': 'test_user', 'password': 'test_password'},
            },
            'logging': {'level': 'INFO', 'file': '/var/log/app.log'},
            'boolean_value': False,
            'int_value': 42,
            'float_value': 3.14,
            'string_value': 'hello',
            'list_value': [1, 2, 3],
            'null_value': None,
        }

    @pytest.fixture
    def config_file_path(self, sample_config_dict) -> Generator:
        """Fixture creating a temporary YAML config file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp_file:
            temp_file_path = temp_file.name
            # Write the config to the file
            yaml.dump(sample_config_dict, temp_file)

        # Return the path to the temp file
        yield temp_file_path

        # Clean up the temp file after the test
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

    @pytest.fixture
    def invalid_yaml_file_path(self) -> Generator:
        """Fixture creating a temporary YAML file with invalid content."""
        # Create a temporary file with invalid YAML
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(
                b'app: {name: test-app, version: 1.0.0, debug: true,'
            )  # Missing closing brace

        # Return the path to the temp file
        yield temp_file_path

        # Clean up the temp file after the test
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

    def test_init_with_dict(self, sample_config_dict):
        """Test initializing Config with a dictionary."""
        config = Config(sample_config_dict)
        assert config.config == sample_config_dict

    def test_init_with_unsupported_type(self):
        """Test initializing Config with an unsupported type raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            Config(123)  # Integer is not a supported type

        assert 'Unsupported config source type' in str(excinfo.value)

    def test_load_config_file_not_found(self):
        """Test loading a non-existent config file raises ConfigurationError."""
        non_existent_file = '/path/to/non/existent/file.yaml'

        with pytest.raises(ConfigurationError) as excinfo:
            Config(non_existent_file)

        assert 'Configuration file not found' in str(excinfo.value)

    def test_load_config_invalid_yaml(self, invalid_yaml_file_path):
        """Test loading an invalid YAML file raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as excinfo:
            Config(invalid_yaml_file_path)

        assert 'Invalid YAML format' in str(excinfo.value)

    def test_get_top_level_key(self, sample_config_dict):
        """Test getting a top-level configuration value."""
        config = Config(sample_config_dict)
        assert config.get('int_value') == 42
        assert config.get('string_value') == 'hello'
        assert config.get('boolean_value') is False
        assert config.get('float_value') == 3.14
        assert config.get('list_value') == [1, 2, 3]
        assert config.get('null_value') is None

    def test_get_nested_key(self, sample_config_dict):
        """Test getting a nested configuration value using dot notation."""
        config = Config(sample_config_dict)
        assert config.get('app.name') == 'test-app'
        assert config.get('app.version') == '1.0.0'
        assert config.get('app.debug') is True
        assert config.get('database.port') == 5432
        assert config.get('database.credentials.username') == 'test_user'

    def test_get_with_default(self, sample_config_dict):
        """Test getting a non-existent key returns the default value."""
        config = Config(sample_config_dict)
        assert config.get('non_existent_key', 'default_value') == 'default_value'
        assert config.get('app.non_existent', 123) == 123
        assert config.get('database.credentials.non_existent', False) is False

    def test_get_non_existent_without_default(self, sample_config_dict):
        """Test getting a non-existent key without a default returns None."""
        config = Config(sample_config_dict)
        assert config.get('non_existent_key') is None

    def test_get_invalid_nested_path(self, sample_config_dict):
        """Test getting an invalid nested path returns the default value."""
        config = Config(sample_config_dict)
        # Trying to access a nested path on a non-dict value
        assert config.get('int_value.nested', 'default') == 'default'
        assert config.get('list_value.0', 'default') == 'default'

    def test_config_property_returns_copy(self, sample_config_dict):
        """Test config property returns a copy, not the original dictionary."""
        config = Config(sample_config_dict)
        config_copy = config.config

        # Verify it's a copy by modifying it and checking the original is unchanged
        config_copy['new_key'] = 'new_value'
        assert 'new_key' not in config.config

    def test_config_empty_dict(self):
        """Test Config with an empty dictionary."""
        config = Config({})
        assert config.config == {}
        assert config.get('any_key') is None
        assert config.get('any.nested.key', 'default') == 'default'

    def test_merge_two_configs(self, sample_config_dict):
        """Test merging two Config objects."""
        # Create two configs with overlapping and unique keys
        config1_dict = {
            'app': {'name': 'main-app', 'version': '2.0.0'},
            'feature_flag': {'enabled': True},
        }
        config2_dict = {'app': {'debug': True}, 'logging': {'level': 'DEBUG'}}

        config1 = Config(config1_dict)
        config2 = Config(config2_dict)
        config1.merge(config2)

        expected = {
            'app': {
                'name': 'main-app',  # from config1
                'version': '2.0.0',  # from config1
                'debug': True,  # from config2
            },
            'feature_flag': {'enabled': True},  # from config1
            'logging': {'level': 'DEBUG'},  # from config2
        }

        assert config1.config == expected
        assert 'debug' in config1.get('app', {})
        assert 'debug' in config2.get('app', {})
