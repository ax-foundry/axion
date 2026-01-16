import math
from collections import Counter
from typing import List

from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import (
    BaseMetric,
    MetricEvaluationResult,
    metric,
)


def _get_ngrams(sequence: List[str], n: int) -> Counter:
    """Calculates n-grams for a sequence of tokens."""
    if len(sequence) < n:
        return Counter()
    return Counter(tuple(sequence[i : i + n]) for i in range(len(sequence) - n + 1))


def _calculate_brevity_penalty(candidate_len: int, reference_len: int) -> float:
    """
    Calculates the brevity penalty (BP).

    The BP penalizes generated translations that are too short compared to the
    closest reference length. It is 1.0 if the candidate length is greater than
    the reference length.
    """
    if candidate_len == 0:
        return 0.0
    if candidate_len > reference_len:
        return 1.0
    # If candidate is shorter, apply penalty
    return math.exp(1 - reference_len / candidate_len)


@metric(
    key='sentence_bleu',
    name='Sentence BLEU',
    description=(
        'Computes sentence-level BLEU score between a candidate and reference sentence. '
        'Returns a score between 0.0 and 1.0. Higher means more similar.'
    ),
    required_fields=['actual_output', 'expected_output'],
    default_threshold=0.5,
    score_range=(0, 1),
    tags=['heuristic'],
)
class SentenceBLEU(BaseMetric):
    """
    Calculates the BLEU score for a single candidate sentence against reference(s).
    This implementation is suitable for evaluating individual sentences but can be
    volatile for short sentences.
    """

    def __init__(
        self,
        n_grams: int = 4,
        case_sensitive: bool = False,
        smoothing: bool = True,
        **kwargs,
    ):
        """
        Initialize the SentenceBLEU metric.

        Args:
            n_grams (int): Maximum n-gram length to consider (e.g., 4 for BLEU-4).
            case_sensitive (bool): Whether the comparison is case-sensitive.
            smoothing (bool): Whether to apply smoothing for sentence-level BLEU.
        """
        self.n_grams = n_grams
        self.case_sensitive = case_sensitive
        self.smoothing = smoothing
        super().__init__(**kwargs)

    @trace(name='SentenceBLEU', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Computes the sentence-level BLEU score for a single DatasetItem."""
        candidate_str = item.actual_output or ''
        references_str = [item.expected_output or '']

        if not self.case_sensitive:
            candidate_str = candidate_str.lower()
            references_str = [ref.lower() for ref in references_str]

        candidate_tokens = candidate_str.split()
        references_tokens = [ref.split() for ref in references_str]

        if not candidate_tokens:
            return MetricEvaluationResult(score=0.0)

        # Calculate Modified Precision for each n-gram order
        precisions = []
        for n in range(1, self.n_grams + 1):
            hyp_ngrams = _get_ngrams(candidate_tokens, n)
            total_hyp_ngrams_count = sum(hyp_ngrams.values())

            if total_hyp_ngrams_count == 0:
                precisions.append(0.0)
                continue

            # Find max n-gram counts in any single reference
            max_ref_ngrams = Counter()
            for ref_tokens in references_tokens:
                ref_ngrams = _get_ngrams(ref_tokens, n)
                for ngram in ref_ngrams:
                    max_ref_ngrams[ngram] = max(
                        max_ref_ngrams[ngram], ref_ngrams[ngram]
                    )

            # Clip hypothesis n-grams against the max reference counts
            clipped_count = sum(
                min(count, max_ref_ngrams.get(ng, 0))
                for ng, count in hyp_ngrams.items()
            )

            if self.smoothing:
                # Add-one smoothing for sentence-level BLEU
                precision = (clipped_count + 1) / (total_hyp_ngrams_count + 1)
            else:
                precision = clipped_count / total_hyp_ngrams_count

            precisions.append(precision)

        # Handle zero precisions (if no smoothing)
        if not self.smoothing and any(p == 0 for p in precisions):
            return MetricEvaluationResult(score=0.0)

        # Combine with Geometric Mean
        log_precisions = [math.log(p) for p in precisions if p > 0]

        if len(log_precisions) != len(precisions):
            return MetricEvaluationResult(score=0.0)

        geometric_mean = math.exp(sum(log_precisions) / self.n_grams)

        # Calculate Brevity Penalty
        candidate_len = len(candidate_tokens)
        ref_lens = [len(ref) for ref in references_tokens]
        closest_ref_len = min(
            ref_lens, key=lambda ref_len: (abs(ref_len - candidate_len), ref_len)
        )
        bp = _calculate_brevity_penalty(candidate_len, closest_ref_len)

        bleu_score = bp * geometric_mean
        return MetricEvaluationResult(score=round(bleu_score, 4))
