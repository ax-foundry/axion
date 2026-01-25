"""
Shared types, enums, and constants used across the axion package.
"""

from typing import Dict, List, Set

from axion._core.schema import RichEnum

# =============================================================================
# Enums
# =============================================================================


class ComponentType(RichEnum):
    AGENT = 'agent'
    KNOWLEDGE = 'knowledge'
    DATA = 'data'
    CUSTOM = 'custom'


class EvaluationType(RichEnum):
    SINGLE_TURN = 'Single Turn'
    MULTI_TURN = 'Multi Turn'


class EvaluationStatus(RichEnum):
    PASS = 'pass'
    FAIL = 'fail'
    NEUTRAL = 'neutral'
    ERROR = 'error'


class TraceGranularity(RichEnum):
    """
    Controls trace granularity during batch evaluation.

    Can be specified as enum or string (case-insensitive):
        - TraceGranularity.SINGLE_TRACE or 'single_trace' or 'single'
        - TraceGranularity.SEPARATE or 'separate'

    Attributes:
        SINGLE_TRACE: All evaluations under one parent trace
        SEPARATE: Each metric execution gets its own independent trace (default)
    """

    SINGLE_TRACE = 'single'
    SEPARATE = 'separate'


# =============================================================================
# Field Name Constants
# =============================================================================


class FieldNames:
    ID = 'id'
    QUERY = 'query'
    EXPECTED_OUTPUT = 'expected_output'
    ACTUAL_OUTPUT = 'actual_output'
    RETRIEVED_CONTENT = 'retrieved_content'
    CONVERSATION = 'conversation'
    ACCEPTANCE_CRITERIA = 'acceptance_criteria'
    TOOLS_CALLED = 'tools_called'
    EXPECTED_TOOLS = 'expected_tools'
    METADATA = 'metadata'
    TRACE = 'trace'
    ADDITIONAL_INPUT = 'additional_input'
    ADDITIONAL_OUTPUT = 'additional_output'
    ACTUAL_RANKING = 'actual_ranking'
    EXPECTED_RANKING = 'expected_ranking'
    EXPECTED_REFERENCE = 'expected_reference'
    ACTUAL_REFERENCE = 'actual_reference'
    DOCUMENT_TEXT = 'document_text'

    # Feedback
    JUDGMENT = 'judgment'
    CRITIQUE = 'critique'

    # Metadata
    CONVERSATION_EXTRACTION_STRATEGY = 'conversation_extraction_strategy'
    USER_TAGS = 'user_tags'
    NAME = 'name'
    SCORE = 'score'
    EXPLANATION = 'explanation'
    THRESHOLD = 'threshold'
    ISSUES = 'issues'

    TOKEN_COUNT = 'token_count'
    LATENCY = 'latency'
    COST_ESTIMATE = 'cost_estimate'

    METRICS = 'metrics'
    STATUS = 'status'
    PERFORMANCE = 'performance'

    @classmethod
    def get_input_fields(cls) -> list[str]:
        return [
            cls.ID,
            cls.QUERY,
            cls.EXPECTED_OUTPUT,
            cls.ACCEPTANCE_CRITERIA,
            cls.EXPECTED_TOOLS,
            cls.ADDITIONAL_INPUT,
            cls.METADATA,
        ]

    @classmethod
    def config_fields(cls) -> list[str]:
        return [
            cls.CONVERSATION_EXTRACTION_STRATEGY,
        ]

    @classmethod
    def get_runtime_fields(cls) -> list[str]:
        return [
            cls.ACTUAL_OUTPUT,
            cls.RETRIEVED_CONTENT,
            cls.TOOLS_CALLED,
            cls.LATENCY,
            cls.TRACE,
            cls.ADDITIONAL_OUTPUT,
        ]

    @classmethod
    def get_evaluation_fields(cls) -> List[str]:
        return [
            cls.QUERY,
            cls.CONVERSATION,
            cls.ACTUAL_OUTPUT,
            cls.EXPECTED_OUTPUT,
            cls.RETRIEVED_CONTENT,
            cls.ACCEPTANCE_CRITERIA,
            cls.TOOLS_CALLED,
            cls.EXPECTED_TOOLS,
            cls.ADDITIONAL_INPUT,
            cls.ADDITIONAL_OUTPUT,
        ]

    @classmethod
    def get_required_evaluation_input_fields(cls) -> List[str]:
        """
        Returns a list of required DatasetItem fields
        that must be present for evaluation.
        """
        return [cls.ACTUAL_OUTPUT, cls.TOOLS_CALLED, cls.EXPECTED_REFERENCE]

    @classmethod
    def get_output_fields(cls) -> List[str]:
        return [cls.NAME, cls.SCORE, cls.EXPLANATION, cls.THRESHOLD, cls.ISSUES]

    @classmethod
    def get_performance_fields(cls) -> List[str]:
        return [cls.TOKEN_COUNT, cls.LATENCY, cls.COST_ESTIMATE]

    @classmethod
    def get_feedback_fields(cls) -> List[str]:
        return [cls.JUDGMENT, cls.CRITIQUE]

    @classmethod
    def get_result_fields(cls) -> List[str]:
        return [cls.METRICS, cls.STATUS, cls.PERFORMANCE]

    @classmethod
    def get_full_dataset_fields(cls):
        return cls.get_input_fields() + cls.get_runtime_fields() + cls.config_fields()

    @classmethod
    def get_display_fields(cls):
        return [
            cls.QUERY,
            cls.ACTUAL_OUTPUT,
            cls.RETRIEVED_CONTENT,
            cls.EXPECTED_OUTPUT,
            cls.EXPECTED_TOOLS,
            cls.LATENCY,
        ]

    @classmethod
    def get_all_fields(cls) -> Dict[str, List[str]]:
        return {
            'input': cls.get_input_fields(),
            'output': cls.get_output_fields(),
            'performance': cls.get_performance_fields(),
            'result': cls.get_result_fields(),
        }

    @classmethod
    def get_aliased_model_field_keys(cls) -> Set[str]:
        return {
            'single_turn_query',
            'single_turn_expected_output',
            'multi_turn_conversation',
        }

    @classmethod
    def get_computed_field_keys(cls) -> Set[str]:
        """
        Returns computed field names that are derived from other data
        and should be excluded during serialization/deserialization.
        These fields are calculated on-the-fly and cannot be set directly.
        """
        return {
            'conversation_stats',
            'agent_trajectory',
            'has_errors',
        }


__all__ = [
    'ComponentType',
    'EvaluationType',
    'EvaluationStatus',
    'TraceGranularity',
    'FieldNames',
]
