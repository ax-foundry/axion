import re
from typing import List, Optional, Tuple

from pydantic import Field

from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor


class LengthResult(RichBaseModel):
    char_count: int = Field(...)
    max_chars_allowed: Optional[int] = Field(None)
    sentence_count: int = Field(...)
    sentence_range: Optional[Tuple[Optional[int], Optional[int]]] = Field(None)
    passed: bool = Field(...)


@metric(
    name='Length Constraint',
    key='length_constraint',
    description='Verifies response does not exceed character limits or sentence count constraints.',
    required_fields=[],
    optional_fields=['actual_output', 'additional_output'],
    default_threshold=1.0,
    tags=['heuristic'],
)
class LengthConstraint(BaseMetric):
    def __init__(
        self,
        max_chars: Optional[int] = 2800,
        sentence_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
        **kwargs,
    ):
        """
        Args:
            max_chars: Maximum allowed characters.
            sentence_range: Tuple of (min_sentences, max_sentences).
                            Use None for open ends, e.g., (None, 5) is max 5.
            **kwargs: Supports BaseMetric field_mapping to remap actual_output
                (e.g., {'actual_output': 'additional_output.summary'}).
        """
        super().__init__(**kwargs)
        self.max_chars = max_chars
        self.sentence_range = sentence_range

    @trace(name='LengthConstraint', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        text = self.get_field(item, 'actual_output', default='') or ''
        text = str(text)

        # Check Characters
        char_count = len(text)
        char_passed = True
        if self.max_chars is not None:
            char_passed = char_count <= self.max_chars

        # Check Sentences (if range provided)
        # Splits on punctuation followed by space or end of string, ignores empty splits
        sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_count = len(sentences)

        sentence_passed = True
        range_desc = ''

        if self.sentence_range:
            min_s, max_s = self.sentence_range
            if min_s is not None and sentence_count < min_s:
                sentence_passed = False
            if max_s is not None and sentence_count > max_s:
                sentence_passed = False

            # Format range string for explanation (e.g., "3-5" or "<=5")
            min_str = str(min_s) if min_s is not None else '0'
            max_str = str(max_s) if max_s is not None else '∞'
            range_desc = f'(Range: {min_str}-{max_str})'

        passed = char_passed and sentence_passed
        score = 1.0 if passed else 0.0

        fail_reasons = []
        if not char_passed:
            fail_reasons.append(f'Exceeded chars ({char_count}/{self.max_chars})')
        if not sentence_passed:
            fail_reasons.append(f'Sentence count {sentence_count} outside {range_desc}')

        if passed:
            explanation = f'PASSED. Chars: {char_count}, Sentences: {sentence_count}.'
        else:
            explanation = 'FAILED. ' + ', '.join(fail_reasons) + '.'

        return MetricEvaluationResult(
            score=score,
            explanation=explanation,
            signals=LengthResult(
                char_count=char_count,
                max_chars_allowed=self.max_chars,
                sentence_count=sentence_count,
                sentence_range=self.sentence_range,
                passed=passed,
            ),
        )

    def get_signals(self, result: LengthResult) -> List[SignalDescriptor]:
        char_desc = f'Used {result.char_count} chars.'
        if result.max_chars_allowed is not None:
            char_desc = (
                f'Used {result.char_count} of {result.max_chars_allowed} characters.'
            )

        signals = [
            SignalDescriptor(
                name='char_count',
                description=char_desc,
                extractor=lambda r: r.char_count,
                headline_display=True,
            ),
            SignalDescriptor(
                name='status',
                description='Pass/Fail',
                extractor=lambda r: 'Pass' if r.passed else 'Fail',
            ),
        ]

        if result.sentence_range:
            signals.insert(
                1,
                SignalDescriptor(
                    name='sentence_count',
                    description=f'Count: {result.sentence_count}. Criteria: {result.sentence_range}',
                    extractor=lambda r: r.sentence_count,
                    headline_display=True,
                ),
            )

        return signals
