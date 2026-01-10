from pathlib import PosixPath
from typing import Any, Dict, Union

import yaml
from yaml.parser import ParserError


class Config:
    """
    Configuration loaded from a YAML file or provided as a dictionary.
    """

    def __init__(self, config_source: Union[str, Dict[str, Any]]) -> None:
        """
        Initialize the Config class.

        Args:
            config_source: The configuration source, either a path to a YAML file or a dictionary
        """
        if isinstance(config_source, dict):
            self._config = config_source
        elif isinstance(config_source, str):
            self._config = self._load_config(config_source)
        elif isinstance(config_source, PosixPath):
            self._config = self._load_config(str(config_source))
        else:
            raise ValueError(
                f'Unsupported config source type: {type(config_source)}. '
                "Only 'str', 'dict' or PosixPath inputs are supported."
            )

    @staticmethod
    def _load_config(config_file: str) -> Dict[str, Any]:
        """
        Load the YAML configuration file.

        Args:
            config_file: Path to the YAML configuration file

        Returns:
            Dictionary containing the loaded configuration
        """
        try:
            with open(config_file, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise ConfigurationError(f'Configuration file not found: {config_file}')
        except ParserError as e:
            raise ConfigurationError(f'Invalid YAML format: {str(e)}')

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value by its key.

        Args:
            key: The configuration key to retrieve, supports dotted notation for nested dictionaries
            default: The default value to return if the key is not found

        Returns:
            The configuration value if found, otherwise the default value
        """
        keys = key.split('.')
        config = self._config

        for k in keys:
            if isinstance(config, dict) and k in config:
                config = config[k]
            else:
                return default

        return config

    @property
    def config(self) -> Dict[str, Any]:
        """
        Get the entire configuration dictionary.

        Returns:
            A copy of the entire configuration dictionary
        """
        return self._config.copy()

    def merge(self, other: 'Config') -> None:
        """
        Merge another Config object into this one.
        This updates values recursively.
        """

        def recursive_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in b.items():
                if key in a and isinstance(a[key], dict) and isinstance(value, dict):
                    a[key] = recursive_merge(a[key], value)
                else:
                    a[key] = value
            return a

        if not isinstance(other, Config):
            raise ValueError('Argument to merge must be a Config instance.')

        self._config = recursive_merge(self._config, other._config)


class ConfigurationError(Exception):
    """Raised when there's an error in the configuration file."""

    pass
