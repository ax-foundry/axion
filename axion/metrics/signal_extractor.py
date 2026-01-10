from collections import defaultdict
from typing import Any, Dict, Optional

from axion._core.logging import get_logger
from pydantic import BaseModel

logger = get_logger(__name__)


class SignalExtractor:
    """A helper class to abstract the logic of extracting signals from metric results."""

    @staticmethod
    def extract(metric: Any, structured_data: Any) -> Optional[Dict]:
        """
        Extracts and processes signals if the metric supports it.

        Args:
            metric: The metric instance that was run.
            structured_data: The output from the metric's execute method,
                             potentially containing signal data.

        Returns:
            A dictionary of grouped signals, or None if not applicable.
        """
        if not isinstance(structured_data, BaseModel) or not hasattr(
            metric, 'get_signals'
        ):
            return None

        try:
            from axion.eval_tree.signal import SignalNode

            signal_descriptors = metric.get_signals(structured_data)
            if not signal_descriptors:
                return None

            signal_nodes = [SignalNode(desc) for desc in signal_descriptors]
            grouped_signals = defaultdict(list)
            for node in signal_nodes:
                value = node.extract(structured_data)
                if value is not None:
                    group_name = node.descriptor.group or 'overall'
                    grouped_signals[group_name].append(
                        {
                            'name': node.descriptor.name,
                            'value': value,
                            'score': node.to_score(value),
                            'description': node.descriptor.description,
                            'headline_display': node.descriptor.headline_display,
                        }
                    )
            return dict(grouped_signals) if grouped_signals else None
        except Exception as e:
            logger.warning(f"Failed to extract signals for metric '{metric.name}': {e}")
            return None
