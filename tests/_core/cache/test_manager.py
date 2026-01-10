import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from axion._core.cache.manager import CacheManager
from axion._core.cache.schema import CacheConfig


class MockDiskCache:
    def __init__(self, directory):
        self.directory = directory
        self._data = {}
        self.closed = False

    def get(self, key, default=None):
        if self.closed:
            raise RuntimeError('Cache is closed')
        return self._data.get(key, default)

    def set(self, key, value):
        if self.closed:
            raise RuntimeError('Cache is closed')
        self._data[key] = value

    def close(self):
        self.closed = True

    def __contains__(self, key):
        return key in self._data


class TestCacheManagerInitialization:
    """Test CacheManager initialization with different configurations."""

    def test_init_memory_cache(self):
        """Test initialization with memory cache."""
        config = CacheConfig(cache_type='memory')
        manager = CacheManager(config)

        assert manager.config is config
        assert isinstance(manager.cache, dict)
        assert manager.cache == {}

    def test_init_cache_disabled(self):
        """Test initialization with cache disabled."""
        config = CacheConfig(use_cache=False)
        manager = CacheManager(config)

        assert manager.config is config
        assert manager.cache is None

    @patch('diskcache.Cache')
    def test_init_disk_cache_success(self, mock_diskcache):
        """Test successful disk cache initialization."""
        mock_cache_instance = MockDiskCache('/tmp/test')
        mock_diskcache.return_value = mock_cache_instance

        config = CacheConfig(cache_type='disk', cache_dir='/tmp/test')
        manager = CacheManager(config)

        assert manager.cache is mock_cache_instance
        mock_diskcache.assert_called_once_with('/tmp/test')

    def test_init_disk_cache_no_directory(self):
        """Test disk cache initialization without directory raises error."""
        config = CacheConfig(cache_type='disk', cache_dir=None)

        with pytest.raises(
            ValueError, match='cache_dir must be provided for disk cache'
        ):
            CacheManager(config)

    def test_init_disk_cache_missing_dependency(self):
        """Test disk cache initialization with missing diskcache dependency."""
        config = CacheConfig(cache_type='disk', cache_dir='/tmp/test')

        with patch(
            'builtins.__import__',
            side_effect=ImportError("No module named 'diskcache'"),
        ):
            with pytest.raises(
                ImportError, match='diskcache is required for disk caching'
            ):
                CacheManager(config)

    def test_init_unsupported_cache_type(self):
        """Test initialization with unsupported cache type."""
        config = CacheConfig(cache_type='redis')  # Unsupported type

        with pytest.raises(ValueError, match='Unsupported cache type: redis'):
            CacheManager(config)


class TestCacheManagerMemoryOperations:
    """Test CacheManager operations with memory cache."""

    def setup_method(self):
        """Set up memory cache manager for each test."""
        self.config = CacheConfig(cache_type='memory')
        self.manager = CacheManager(self.config)

    def test_memory_get_existing_key(self):
        """Test getting existing key from memory cache."""
        # Manually add data to cache
        self.manager.cache['test_key'] = 'test_value'

        result = self.manager.get('test_key')
        assert result == 'test_value'

    def test_memory_get_non_existing_key(self):
        """Test getting non-existing key from memory cache."""
        result = self.manager.get('non_existing_key')
        assert result is None

    def test_memory_set_value(self):
        """Test setting value in memory cache."""
        self.manager.set('new_key', 'new_value')

        assert self.manager.cache['new_key'] == 'new_value'
        assert self.manager.get('new_key') == 'new_value'

    def test_memory_set_overwrite_value(self):
        """Test overwriting existing value in memory cache."""
        self.manager.set('key', 'original_value')
        self.manager.set('key', 'updated_value')

        assert self.manager.get('key') == 'updated_value'

    def test_memory_set_various_types(self):
        """Test setting various data types in memory cache."""
        test_data = {
            'string': 'test_string',
            'number': 42,
            'list': [1, 2, 3],
            'dict': {'nested': 'value'},
            'none': None,
            'bool': True,
        }

        for key, value in test_data.items():
            self.manager.set(key, value)
            assert self.manager.get(key) == value

    def test_memory_close_no_effect(self):
        """Test that close() has no effect on memory cache."""
        self.manager.set('key', 'value')
        self.manager.close()

        # Should still work after close
        assert self.manager.get('key') == 'value'


class TestCacheManagerDiskOperations:
    """Test CacheManager operations with disk cache."""

    def setup_method(self):
        """Set up disk cache manager for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = CacheConfig(cache_type='disk', cache_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('diskcache.Cache')
    def test_disk_get_existing_key(self, mock_diskcache):
        """Test getting existing key from disk cache."""
        mock_cache = MockDiskCache(self.temp_dir)
        mock_cache.set('test_key', 'test_value')
        mock_diskcache.return_value = mock_cache

        manager = CacheManager(self.config)
        result = manager.get('test_key')

        assert result == 'test_value'

    @patch('diskcache.Cache')
    def test_disk_get_non_existing_key(self, mock_diskcache):
        """Test getting non-existing key from disk cache."""
        mock_cache = MockDiskCache(self.temp_dir)
        mock_diskcache.return_value = mock_cache

        manager = CacheManager(self.config)
        result = manager.get('non_existing_key')

        assert result is None

    @patch('diskcache.Cache')
    def test_disk_set_value(self, mock_diskcache):
        """Test setting value in disk cache."""
        mock_cache = MockDiskCache(self.temp_dir)
        mock_diskcache.return_value = mock_cache

        manager = CacheManager(self.config)
        manager.set('new_key', 'new_value')

        assert manager.get('new_key') == 'new_value'

    @patch('diskcache.Cache')
    def test_disk_close(self, mock_diskcache):
        """Test closing disk cache."""
        mock_cache = MockDiskCache(self.temp_dir)
        mock_diskcache.return_value = mock_cache

        manager = CacheManager(self.config)
        manager.close()

        assert mock_cache.closed is True


class TestCacheManagerConfigurationFlags:
    """Test CacheManager behavior with different configuration flags."""

    def test_use_cache_false_get_returns_none(self):
        """Test that get returns None when use_cache is False."""
        config = CacheConfig(use_cache=False, cache_type='memory')
        manager = CacheManager(config)

        # Even if we manually add to cache, get should return None
        if manager.cache:
            manager.cache['key'] = 'value'

        result = manager.get('key')
        assert result is None

    def test_write_cache_false_set_does_nothing(self):
        """Test that set does nothing when write_cache is False."""
        config = CacheConfig(write_cache=False, cache_type='memory')
        manager = CacheManager(config)

        manager.set('key', 'value')

        # Cache should be empty since write_cache is False
        assert manager.cache == {}
        assert manager.get('key') is None

    def test_use_cache_true_write_cache_false(self):
        """Test behavior when use_cache=True but write_cache=False."""
        config = CacheConfig(use_cache=True, write_cache=False, cache_type='memory')
        manager = CacheManager(config)

        # Manually populate cache to test reading
        manager.cache['existing_key'] = 'existing_value'

        # Should be able to read existing data
        assert manager.get('existing_key') == 'existing_value'

        # But setting new data should do nothing
        manager.set('new_key', 'new_value')
        assert manager.get('new_key') is None

    def test_use_cache_false_write_cache_true(self):
        """Test behavior when use_cache=False but write_cache=True."""
        config = CacheConfig(use_cache=False, write_cache=True, cache_type='memory')
        manager = CacheManager(config)

        # Should not initialize cache when use_cache is False
        assert manager.cache is None

        # Both get and set should do nothing
        manager.set('key', 'value')
        result = manager.get('key')
        assert result is None

    def test_cache_task_flag(self):
        """Test the cache_task flag (if used by implementation)."""
        # Test with cache_task=True (default)
        config1 = CacheConfig(cache_task=True)
        manager1 = CacheManager(config1)
        assert manager1.config.cache_task is True

        # Test with cache_task=False
        config2 = CacheConfig(cache_task=False)
        manager2 = CacheManager(config2)
        assert manager2.config.cache_task is False

    def test_default_cache_dir(self):
        """Test default cache_dir value."""
        config = CacheConfig()
        assert config.cache_dir == '.cache'

        # Test with custom cache_dir
        config_custom = CacheConfig(cache_dir='/custom/path')
        assert config_custom.cache_dir == '/custom/path'


class TestCacheManagerEdgeCases:
    """Test edge cases and error conditions."""

    def test_get_with_none_cache(self):
        """Test get operation when cache is None."""
        config = CacheConfig(use_cache=False)
        manager = CacheManager(config)

        result = manager.get('any_key')
        assert result is None

    def test_set_with_none_cache(self):
        """Test set operation when cache is None."""
        config = CacheConfig(use_cache=False)
        manager = CacheManager(config)

        # Should not raise any exception
        manager.set('key', 'value')

    def test_close_with_none_cache(self):
        """Test close operation when cache is None."""
        config = CacheConfig(use_cache=False)
        manager = CacheManager(config)

        # Should not raise any exception
        manager.close()

    def test_empty_string_key(self):
        """Test operations with empty string key."""
        config = CacheConfig(cache_type='memory')
        manager = CacheManager(config)

        manager.set('', 'empty_key_value')
        assert manager.get('') == 'empty_key_value'

    def test_special_characters_in_key(self):
        """Test operations with special characters in key."""
        config = CacheConfig(cache_type='memory')
        manager = CacheManager(config)

        special_key = 'key_with_!@#$%^&*()_+-={}[]|\\:;"\'<>,.?/~`'
        manager.set(special_key, 'special_value')
        assert manager.get(special_key) == 'special_value'

    def test_unicode_key_and_value(self):
        """Test operations with unicode key and value."""
        config = CacheConfig(cache_type='memory')
        manager = CacheManager(config)

        unicode_key = '🔑_ключ_キー'
        unicode_value = '🎯_значение_値'

        manager.set(unicode_key, unicode_value)
        assert manager.get(unicode_key) == unicode_value

    def test_large_value(self):
        """Test caching large values."""
        config = CacheConfig(cache_type='memory')
        manager = CacheManager(config)

        large_value = 'x' * 1_000_000  # 1MB string
        manager.set('large_key', large_value)
        assert manager.get('large_key') == large_value

    @patch('diskcache.Cache')
    def test_disk_cache_exception_handling(self, mock_diskcache):
        """Test exception handling in disk cache operations."""
        mock_cache = Mock()
        mock_cache.get.side_effect = Exception('Disk error')
        mock_cache.set.side_effect = Exception('Disk write error')
        mock_diskcache.return_value = mock_cache

        config = CacheConfig(cache_type='disk', cache_dir='/tmp/test')
        manager = CacheManager(config)

        # These should raise exceptions (not handled by CacheManager)
        with pytest.raises(Exception, match='Disk error'):
            manager.get('key')

        with pytest.raises(Exception, match='Disk write error'):
            manager.set('key', 'value')

    def test_cache_config_defaults(self):
        """Test CacheConfig default values."""
        config = CacheConfig()

        assert config.use_cache is True
        assert config.write_cache is True
        assert config.cache_type == 'memory'
        assert config.cache_dir == '.cache'
        assert config.cache_task is True

    def test_cache_config_custom_values(self):
        """Test CacheConfig with custom values."""
        config = CacheConfig(
            use_cache=False,
            write_cache=False,
            cache_type='disk',
            cache_dir='/custom/cache',
            cache_task=False,
        )

        assert config.use_cache is False
        assert config.write_cache is False
        assert config.cache_type == 'disk'
        assert config.cache_dir == '/custom/cache'
        assert config.cache_task is False


class TestCacheManagerIntegration:
    """Integration tests for CacheManager."""

    def test_memory_cache_workflow(self):
        """Test complete workflow with memory cache."""
        config = CacheConfig(cache_type='memory')
        manager = CacheManager(config)

        # Initially empty
        assert manager.get('workflow_key') is None

        # Set value
        manager.set('workflow_key', 'workflow_value')

        # Retrieve value
        assert manager.get('workflow_key') == 'workflow_value'

        # Update value
        manager.set('workflow_key', 'updated_value')
        assert manager.get('workflow_key') == 'updated_value'

        # Close (should have no effect on memory cache)
        manager.close()
        assert manager.get('workflow_key') == 'updated_value'

    @patch('diskcache.Cache')
    def test_disk_cache_workflow(self, mock_diskcache):
        """Test complete workflow with disk cache."""
        temp_dir = tempfile.mkdtemp()

        try:
            mock_cache = MockDiskCache(temp_dir)
            mock_diskcache.return_value = mock_cache

            config = CacheConfig(cache_type='disk', cache_dir=temp_dir)
            manager = CacheManager(config)

            # Initially empty
            assert manager.get('workflow_key') is None

            # Set value
            manager.set('workflow_key', 'workflow_value')

            # Retrieve value
            assert manager.get('workflow_key') == 'workflow_value'

            # Close
            manager.close()
            assert mock_cache.closed is True

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cache_type_switching(self):
        """Test behavior when switching between cache types."""
        # Start with memory cache
        memory_config = CacheConfig(cache_type='memory')
        memory_manager = CacheManager(memory_config)

        memory_manager.set('shared_key', 'memory_value')
        assert memory_manager.get('shared_key') == 'memory_value'

        # Create new manager with different cache type
        disabled_config = CacheConfig(use_cache=False)
        disabled_manager = CacheManager(disabled_config)

        # Should not have access to memory cache data
        assert disabled_manager.get('shared_key') is None


class TestCacheManagerPerformance:
    """Performance-related tests for CacheManager."""

    def test_memory_cache_performance(self):
        """Test performance with many operations on memory cache."""
        config = CacheConfig(cache_type='memory')
        manager = CacheManager(config)

        # Set many items
        for i in range(1000):
            manager.set(f'key_{i}', f'value_{i}')

        # Verify all items
        for i in range(1000):
            assert manager.get(f'key_{i}') == f'value_{i}'

        # Test non-existent keys
        for i in range(1000, 1100):
            assert manager.get(f'key_{i}') is None

    def test_memory_cache_large_objects(self):
        """Test caching large objects in memory."""
        config = CacheConfig(cache_type='memory')
        manager = CacheManager(config)

        # Create large objects
        large_dict = {f'key_{i}': f'value_{i}' * 1000 for i in range(100)}
        large_list = [f'item_{i}' * 100 for i in range(1000)]

        manager.set('large_dict', large_dict)
        manager.set('large_list', large_list)

        assert manager.get('large_dict') == large_dict
        assert manager.get('large_list') == large_list


# FIXTURES


@pytest.fixture
def memory_cache_manager():
    """Fixture providing a memory cache manager."""
    config = CacheConfig(cache_type='memory')
    return CacheManager(config)


@pytest.fixture
def disabled_cache_manager():
    """Fixture providing a disabled cache manager."""
    config = CacheConfig(use_cache=False)
    return CacheManager(config)


@pytest.fixture
def temp_directory():
    """Fixture providing a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_disk_cache_manager(temp_directory):
    """Fixture providing a mocked disk cache manager."""
    with patch('diskcache.Cache') as mock_diskcache:
        mock_cache = MockDiskCache(temp_directory)
        mock_diskcache.return_value = mock_cache

        config = CacheConfig(cache_type='disk', cache_dir=temp_directory)
        manager = CacheManager(config)
        yield manager, mock_cache


class TestWithFixtures:
    """Tests using pytest fixtures."""

    def test_memory_fixture_basic_operations(self, memory_cache_manager):
        """Test basic operations using memory cache fixture."""
        manager = memory_cache_manager

        manager.set('fixture_key', 'fixture_value')
        assert manager.get('fixture_key') == 'fixture_value'

    def test_disabled_fixture_no_operations(self, disabled_cache_manager):
        """Test that disabled cache fixture doesn't cache."""
        manager = disabled_cache_manager

        manager.set('key', 'value')
        assert manager.get('key') is None

    def test_disk_fixture_operations(self, mock_disk_cache_manager):
        """Test operations using disk cache fixture."""
        manager, mock_cache = mock_disk_cache_manager

        manager.set('disk_key', 'disk_value')
        assert manager.get('disk_key') == 'disk_value'

        manager.close()
        assert mock_cache.closed is True

    def test_temp_directory_fixture(self, temp_directory):
        """Test that temp directory fixture works."""
        assert isinstance(temp_directory, str)
        assert Path(temp_directory).exists()
        assert Path(temp_directory).is_dir()


class TestCacheManagerConcurrency:
    """Test concurrent operations (simplified for single-threaded testing)."""

    def test_concurrent_access_simulation(self):
        """Simulate concurrent access patterns."""
        config = CacheConfig(cache_type='memory')
        manager = CacheManager(config)

        # Simulate multiple "threads" setting and getting
        keys = [f'concurrent_key_{i}' for i in range(10)]
        values = [f'concurrent_value_{i}' for i in range(10)]

        # Interleaved set/get operations
        for i in range(10):
            manager.set(keys[i], values[i])
            if i > 0:
                assert manager.get(keys[i - 1]) == values[i - 1]

        # Verify all values
        for key, value in zip(keys, values):
            assert manager.get(key) == value


class TestCacheConfigValidation:
    """Test CacheConfig dataclass validation and edge cases."""

    def test_cache_config_dataclass_behavior(self):
        """Test that CacheConfig behaves as expected dataclass."""
        config1 = CacheConfig()
        config2 = CacheConfig()

        # Two instances with same values should be equal
        assert config1 == config2

        # Modification should make them different
        config2.use_cache = False
        assert config1 != config2

    def test_cache_config_field_modification(self):
        """Test modifying CacheConfig fields after creation."""
        config = CacheConfig()

        # Test field modifications
        config.use_cache = False
        config.write_cache = False
        config.cache_type = 'disk'
        config.cache_dir = '/new/path'
        config.cache_task = False

        assert config.use_cache is False
        assert config.write_cache is False
        assert config.cache_type == 'disk'
        assert config.cache_dir == '/new/path'
        assert config.cache_task is False

    def test_cache_config_with_manager_integration(self):
        """Test CacheConfig integration with CacheManager."""
        # Test that manager respects all config flags
        config = CacheConfig(
            use_cache=True,
            write_cache=True,
            cache_type='memory',
            cache_dir=None,
            cache_task=True,
        )

        manager = CacheManager(config)

        # Verify config is stored correctly
        assert manager.config.use_cache is True
        assert manager.config.write_cache is True
        assert manager.config.cache_type == 'memory'
        assert manager.config.cache_task is True

        # Verify functionality works as expected
        manager.set('test', 'value')
        assert manager.get('test') == 'value'


class TestCacheManagerWithRealDiskCache:
    """Tests with real filesystem operations (optional, slower)."""

    @pytest.mark.slow
    def test_real_disk_cache_operations(self):
        """Test with actual disk cache if diskcache is available."""
        pytest.importorskip('diskcache', reason='diskcache not available')

        temp_dir = tempfile.mkdtemp()
        try:
            config = CacheConfig(cache_type='disk', cache_dir=temp_dir)
            manager = CacheManager(config)

            # Test basic operations
            manager.set('real_key', 'real_value')
            assert manager.get('real_key') == 'real_value'

            # Test persistence by creating new manager with same directory
            manager.close()

            manager2 = CacheManager(config)
            assert manager2.get('real_key') == 'real_value'

            manager2.close()

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.slow
    def test_disk_cache_directory_creation(self):
        """Test that disk cache creates directories if they don't exist."""
        pytest.importorskip('diskcache', reason='diskcache not available')

        temp_base = tempfile.mkdtemp()
        cache_dir = Path(temp_base) / 'nested' / 'cache' / 'dir'

        try:
            config = CacheConfig(cache_type='disk', cache_dir=str(cache_dir))
            manager = CacheManager(config)

            # Directory should be created
            assert cache_dir.exists()

            # Should be able to cache
            manager.set('nested_key', 'nested_value')
            assert manager.get('nested_key') == 'nested_value'

            manager.close()

        finally:
            shutil.rmtree(temp_base, ignore_errors=True)


class TestCacheManagerErrorRecovery:
    """Test error recovery and resilience."""

    def test_memory_cache_after_error(self):
        """Test that memory cache continues working after errors."""
        config = CacheConfig(cache_type='memory')
        manager = CacheManager(config)

        # Set some data
        manager.set('good_key', 'good_value')

        # Simulate error by corrupting cache reference temporarily
        original_cache = manager.cache
        manager.cache = None

        # Operations should not crash
        assert manager.get('any_key') is None
        manager.set('any_key', 'any_value')  # Should not crash

        # Restore cache
        manager.cache = original_cache

        # Original data should still be there
        assert manager.get('good_key') == 'good_value'

    @patch('diskcache.Cache')
    def test_disk_cache_initialization_recovery(self, mock_diskcache):
        """Test recovery from disk cache initialization errors."""
        # First call fails, second succeeds
        mock_diskcache.side_effect = [
            Exception('Initialization failed'),
            MockDiskCache('/tmp/test'),
        ]

        config = CacheConfig(cache_type='disk', cache_dir='/tmp/test')

        # First attempt should fail
        with pytest.raises(Exception, match='Initialization failed'):
            CacheManager(config)

        # Second attempt should succeed
        manager = CacheManager(config)
        assert manager.cache is not None


class TestCacheManagerThreadSafety:
    """Test thread safety considerations (single-threaded simulation)."""

    def test_memory_cache_concurrent_modifications(self):
        """Simulate concurrent modifications to memory cache."""
        config = CacheConfig(cache_type='memory')
        manager = CacheManager(config)

        # Simulate rapid modifications
        for i in range(100):
            manager.set(f'key_{i % 10}', f'value_{i}')

            # Verify some previous values still exist
            if i > 10:
                prev_key = f'key_{(i - 5) % 10}'
                prev_value = manager.get(prev_key)
                assert prev_value is not None
                assert prev_value.startswith('value_')
