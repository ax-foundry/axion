from typing import Dict, List, Literal, Optional

from axion._core.schema import RichBaseModel, RichEnum
from pydantic import Field


class RAGEvaluationError(Exception):
    """Base exception for RAG evaluation failures"""

    pass


class StatementExtractionError(RAGEvaluationError):
    """Failed to extract statements from text"""

    pass


class JudgmentError(RAGEvaluationError):
    """Failed to get judgment from LLM"""

    pass


class ConfigurationError(RAGEvaluationError):
    """Invalid configuration provided"""

    pass


class FaithfulnessVerdict(str, RichEnum):
    FULLY_SUPPORTED = 'Fully Supported'
    PARTIALLY_SUPPORTED = 'Partially Supported'
    CONTRADICTORY = 'Contradictory'
    NO_EVIDENCE = 'No Evidence'


class EvaluationMode(str, RichEnum):
    GRANULAR = 'granular'
    HOLISTIC = 'holistic'


class RelevancyVerdictModel(RichBaseModel):
    verdict: Literal['yes', 'no', 'idk'] = Field(description='Relevancy verdict')
    reason: Optional[str] = Field(default=None, description='Reason for no verdict')


class JudgedClaim(RichBaseModel):
    """Represents a single claim from the actual_output, with all associated judgments."""

    claim_text: str
    relevancy_verdict: Optional[Literal['yes', 'no', 'idk']] = Field(
        default=None,
        description="Verdict on the claim's relevance to the query, if computed.",
    )
    faithfulness_verdict: Optional[FaithfulnessVerdict] = Field(
        default=None,
        description="Verdict on the claim's support from the context, if computed.",
    )
    reason: Optional[str] = Field(
        default=None,
        description='Combined reasoning for judgments, populated only when computed.',
    )

    @property
    def is_relevant(self) -> Optional[bool]:
        """
        Helper property to get a boolean representation of relevancy.
        Returns None if relevancy was not assessed.
        """
        if self.relevancy_verdict is None:
            return None
        return self.relevancy_verdict != 'no'


class JudgedContextChunk(RichBaseModel):
    """Represents a chunk of the retrieved_content, judged for relevance and utility."""

    chunk_text: str
    is_relevant_to_query: bool
    is_useful_for_expected_output: bool
    is_utilized_in_answer: Optional[bool] = Field(
        default=None,
        description='Whether this chunk was actually used in generating the answer, if computed.',
    )


class JudgedGroundTruthStatement(RichBaseModel):
    """Represents a statement from the expected_output, judged for contextual support."""

    statement_text: str
    is_supported_by_context: bool


class ContextSufficiencyVerdict(RichBaseModel):
    """Represents the verdict on whether context is sufficient to answer the query."""

    is_sufficient: bool = Field(
        description='Whether the context contains sufficient information to answer the query'
    )
    reasoning: str = Field(
        description='Explanation of why the context is or is not sufficient'
    )


class AnalysisMetadata(RichBaseModel):
    """Contains metadata about the RAG analysis run."""

    mode: EvaluationMode
    execution_time: float = 0.0
    cache_hits: int = 0
    llm_calls: Dict[str, int] = Field(
        default_factory=lambda: {
            'batch_attempts': 0,
            'batch_success': 0,
            'granular_attempts': 0,
            'granular_success': 0,
            'failed_judgments': 0,
        }
    )

    # Cost tracking
    cost_estimate: float = 0.0
    cost_by_judge: Dict[str, float] = Field(
        default_factory=dict, description='Cost breakdown by judge type name'
    )
    judge_usage: Dict[str, int] = Field(
        default_factory=dict, description='Number of times each judge was called'
    )

    def get_cost_summary(self) -> str:
        """Get a formatted summary of costs."""
        if not self.cost_by_judge:
            return 'No cost data available'

        lines = [f'Total Cost: ${self.total_cost:.4f}']
        lines.append('\nCost Breakdown:')
        for judge_name, cost in sorted(self.cost_by_judge.items(), key=lambda x: -x[1]):
            count = self.judge_usage.get(judge_name, 0)
            lines.append(f'  {judge_name}: ${cost:.4f} ({count} calls)')

        return '\n'.join(lines)


class RAGAnalyzerResult(RichBaseModel):
    """The single, comprehensive result object returned by the pre-computation step."""

    judged_claims: List[JudgedClaim] = Field(default_factory=list)
    judged_context_chunks: List[JudgedContextChunk] = Field(default_factory=list)
    judged_ground_truth_statements: List[JudgedGroundTruthStatement] = Field(
        default_factory=list
    )
    context_sufficiency_verdict: Optional[ContextSufficiencyVerdict] = None
    metadata: Optional[AnalysisMetadata] = None
    error: Optional[str] = None

    def has_error(self) -> bool:
        return self.error is not None
