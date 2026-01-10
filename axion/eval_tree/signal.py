from typing import Any

from axion.metrics.schema import SignalDescriptor


class SignalNode:
    """
    An operational node that extracts and scores a value using a SignalDescriptor.

    This class acts as the worker that takes the blueprint (the descriptor) and
    applies it to a metric's structured result data to produce a scored signal.
    """

    def __init__(self, descriptor: SignalDescriptor):
        """Initializes the SignalNode with a specific descriptor."""
        self.descriptor = descriptor

    def extract(self, data: Any) -> Any:
        """
        Extracts the signal's value from the result data using the descriptor.

        Args:
            data: The structured Pydantic model returned by the parent metric.

        Returns:
            The raw value extracted by the descriptor's callable, or None if
            an error occurs.
        """
        try:
            return self.descriptor.extractor(data)
        except Exception:
            return None

    def to_score(self, value: Any) -> float:
        """
        Converts a raw extracted value into a standardized numerical score.

        This method applies a set of rules to transform common data types into a
        float score. It prioritizes a custom `score_mapping` if provided in the
        descriptor, otherwise falls back to extensive default logic.

        Args:
            value: The raw value returned by the extract method.

        Returns:
            A float score or 'nan' if the value cannot be converted.
        """
        if value is None:
            return float('nan')
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            # Prioritize the custom score mapping if it exists.
            if self.descriptor.score_mapping:
                return self.descriptor.score_mapping.get(value.lower(), float('nan'))

            # Fallback to the default string mapping.
            val_lower = value.lower()
            if val_lower in ['yes', 'relevant', 'faithful', 'correct']:
                return 1.0
            if val_lower in [
                'no',
                'irrelevant',
                'unfaithful',
                'incorrect',
                'contradictory',
            ]:
                return 0.0

        return float('nan')
