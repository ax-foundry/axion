from unittest.mock import MagicMock, patch

from axion._core.tracing.context import reset_tracer_context, set_current_tracer


class TestInitTracer:
    """Tests for init_tracer factory function."""

    def test_init_tracer_forwards_kwargs_to_create(self):
        """kwargs passed to init_tracer are forwarded to TracerClass.create()."""
        from axion._core.tracing.factory import init_tracer

        mock_tracer = MagicMock()
        mock_class = MagicMock(return_value=mock_tracer)
        mock_class.create = MagicMock(return_value=mock_tracer)

        with patch('axion._core.tracing.factory.get_tracer', return_value=mock_class):
            with patch(
                'axion._core.tracing.factory.get_current_tracer',
                side_effect=LookupError,
            ):
                with patch(
                    'axion._core.tracing.utils.get_default_global_tracer',
                    return_value=None,
                ):
                    init_tracer('llm', tags=['test'], environment='staging')

        mock_class.create.assert_called_once()
        call_kwargs = mock_class.create.call_args[1]
        assert call_kwargs['tags'] == ['test']
        assert call_kwargs['environment'] == 'staging'
        assert call_kwargs['metadata_type'] == 'llm'

    def test_init_tracer_force_new_skips_context_tracer(self):
        """force_new=True bypasses context tracer and creates a fresh one."""
        from axion._core.tracing.factory import init_tracer
        from axion._core.tracing.noop.tracer import NoOpTracer

        context_tracer = NoOpTracer.create(metadata_type='llm')
        token = set_current_tracer(context_tracer)
        try:
            result = init_tracer('llm', force_new=True)
            # Should be a new instance, not the context tracer
            assert result is not context_tracer
        finally:
            reset_tracer_context(token)

    def test_init_tracer_force_new_false_reuses_context_tracer(self):
        """Without force_new, init_tracer returns the context tracer."""
        from axion._core.tracing.factory import init_tracer
        from axion._core.tracing.noop.tracer import NoOpTracer

        context_tracer = NoOpTracer.create(metadata_type='llm')
        token = set_current_tracer(context_tracer)
        try:
            result = init_tracer('llm')
            assert result is context_tracer
        finally:
            reset_tracer_context(token)

    def test_init_tracer_explicit_tracer_ignores_force_new(self):
        """An explicit tracer is always returned regardless of force_new."""
        from axion._core.tracing.factory import init_tracer
        from axion._core.tracing.noop.tracer import NoOpTracer

        explicit = NoOpTracer.create(metadata_type='llm')
        result = init_tracer('llm', tracer=explicit, force_new=True)
        assert result is explicit

    def test_init_tracer_force_new_skips_global_tracer(self):
        """force_new=True bypasses global tracer fallback and creates a fresh one."""
        from axion._core.tracing.factory import init_tracer
        from axion._core.tracing.noop.tracer import NoOpTracer
        from axion._core.tracing.utils import set_default_global_tracer

        global_tracer = NoOpTracer.create(metadata_type='llm')
        set_default_global_tracer(global_tracer)
        try:
            result = init_tracer('llm', force_new=True)
            assert result is not global_tracer
        finally:
            set_default_global_tracer(None)

    def test_backward_compat_no_kwargs(self):
        """Calling init_tracer without new params preserves existing behavior."""
        from axion._core.tracing.factory import init_tracer
        from axion._core.tracing.noop.tracer import NoOpTracer

        # No context, no global → falls through to create()
        with patch(
            'axion._core.tracing.factory.get_current_tracer',
            side_effect=LookupError,
        ):
            with patch(
                'axion._core.tracing.utils.get_default_global_tracer',
                return_value=None,
            ):
                result = init_tracer('llm')

        assert result is not None
        assert isinstance(result, NoOpTracer)
