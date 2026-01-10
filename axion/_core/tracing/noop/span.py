from typing import Any, Dict

from axion._core.uuid import uuid7


class Span:
    """No-operation span that does nothing but maintains the same interface."""

    def __init__(
        self,
        operation_name: str,
        tracer,
        attributes: Dict[str, Any],
        auto_trace: bool = True,
    ):
        self.operation_name = operation_name
        self.tracer = tracer
        self.attributes = attributes
        self.auto_trace = auto_trace
        self.span_id = str(uuid7())
        self.trace_id = tracer._trace_id

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def add_trace(self, event_type: str, message: str, metadata: Dict[str, Any] = None):
        """No-op trace method."""
        pass

    def set_attribute(self, key: str, value: Any):
        """No-op attribute setting."""
        pass

    def set_attributes(self, attr_dict: Dict[str, Any]):
        """No-op attributes setting."""
        pass

    def set_output(self, key: str, value: Any):
        """No-op output setting."""
        pass

    def set_outputs(self, output_dict: Dict):
        """No-op outputs setting."""
        pass
