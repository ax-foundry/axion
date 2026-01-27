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
    DEFAULT_ABBREVIATIONS = [
        'mr',
        'mrs',
        'ms',
        'dr',
        'prof',
        'sr',
        'jr',
        'st',
        'mt',
        'vs',
        'etc',
        'e.g',
        'i.e',
        'u.s',
        'u.k',
        'p.s',
    ]

    def __init__(
        self,
        max_chars: Optional[int] = 2800,
        sentence_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
        abbreviations: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Args:
            max_chars: Maximum allowed characters.
            sentence_range: Tuple of (min_sentences, max_sentences).
                            Use None for open ends, e.g., (None, 5) is max 5.
            abbreviations: Abbreviations to avoid splitting on (no trailing dot).
            **kwargs: Supports BaseMetric field_mapping to remap actual_output
                (e.g., {'actual_output': 'additional_output.summary'}).
        """
        super().__init__(**kwargs)
        self.max_chars = max_chars
        self.sentence_range = sentence_range
        abbreviations = abbreviations or self.DEFAULT_ABBREVIATIONS
        self.abbreviations_set = {value.lower() for value in abbreviations}

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
        # Split on sentence-ending punctuation followed by whitespace and likely sentence
        # start, while skipping common abbreviations and initials.
        # Regex candidate split points, then validate with explicit checks to
        # avoid variable-length lookbehinds.
        sentence_pattern = r'([.!?])\s+(?=(?:(?:"|\'|`|\(|\[))?[A-Z])'
        sentences = []
        start = 0
        for match in re.finditer(sentence_pattern, text.strip()):
            end = match.end(1)
            candidate = text.strip()[start:end].strip()
            if candidate:
                last_token = candidate.split()[-1]
                token_base = last_token.rstrip('.').lower()
                is_initial = len(token_base) == 1 and token_base.isalpha()
                if not is_initial and token_base not in self.abbreviations_set:
                    sentences.append(candidate)
                    start = match.end()
        tail = text.strip()[start:].strip()
        if tail:
            sentences.append(tail)
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
