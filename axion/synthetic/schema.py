from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from axion._core.schema import RichBaseModel


class GenerationParams(RichBaseModel):
    """
    Configuration parameters for controlling the QA (Question–Answer) generation pipeline.

    These settings define how QA pairs are generated from source documents, including
    the number of pairs, question style and complexity, chunking strategies, and validation
    thresholds. The configuration supports both factual and synthetic QA creation,
    enabling flexible generation for training, evaluation, and benchmarking.

    Attributes:
        num_pairs (int):
            Total number of QA pairs to generate per document.
        question_types (List[str]):
            List of question types to generate. Common options include:
            - 'factual'      : Direct, fact-based questions
            - 'conceptual'   : Understanding-based questions
            - 'application'  : Scenario-based application questions
            - 'analysis'     : Critical thinking and analysis questions
            - 'synthetic'    : Artificially created questions for stress-testing
        difficulty (str):
            Target difficulty of generated questions. Options include:
            'easy', 'medium', and 'hard'.
        splitter_type (Literal['semantic', 'sentence']):
            Chunking strategy for breaking documents into sections:
            - 'semantic': Embedding-aware splits for context preservation.
            - 'sentence': Rule-based splits by sentence length.
        chunk_size (int):
            Maximum size (in characters or tokens) of each chunk when using
            `splitter_type='sentence'`.
        statements_per_chunk (int):
            Number of candidate statements generated per chunk before filtering
            and validation.
        answer_length (str):
            Desired length for generated answers. Options:
            'short', 'medium', or 'long'.
        dimensions (Optional[Dict[str, Any]]):
            A dictionary guiding synthetic data generation. Possible keys:
            - 'features' : Data attributes to reflect real-world structure.
            - 'persona'  : Profiles simulating different perspectives.
            - 'scenarios': Contextual situations to ensure realism.
        custom_guidelines (Optional[str]):
            Additional free-text instructions to condition the QA generation
            process beyond default behavior.
        example_question (Optional[str]):
            An example question to guide style, tone, and complexity.
        example_answer (Optional[str]):
            An example answer to align generated responses with the desired
            style and depth.
        max_reflection_iterations (int):
            Maximum self-reflection and retry loops for improving QA quality
            during validation.
        validation_threshold (float):
            Minimum confidence or faithfulness score (0.0–1.0) required to
            accept a QA pair.
        breakpoint_percentile_threshold (int):
            Percentile threshold for determining sentence breakpoints in
            semantic chunking. Higher values create fewer, larger chunks.
    """

    num_pairs: int = Field(
        default=1, description='Total number of QA pairs to generate per document.'
    )

    question_types: List[str] = Field(
        default_factory=lambda: ['factual', 'conceptual', 'application', 'analysis'],
        description=(
            'List of question types to include. '
            "Options may include: 'factual', 'conceptual', 'application', 'analysis', 'synthetic', etc."
        ),
    )

    difficulty: str = Field(
        default='medium',
        description="Target difficulty level for generated questions (e.g., 'easy', 'medium', 'hard').",
    )

    splitter_type: Literal['semantic', 'sentence'] = Field(
        default='semantic',
        description="Chunking strategy: 'semantic' uses embedding-aware splits, 'sentence' uses rule-based.",
    )

    chunk_size: int = Field(
        default=4000,
        description="Maximum number of characters or tokens per document chunk (used if splitter_type='sentence').",
    )

    statements_per_chunk: int = Field(
        default=5,
        description='Number of candidate statements generated per chunk before filtering and validation.',
    )

    answer_length: str = Field(
        default='medium',
        description="Desired length of the generated answer: options include 'short', 'medium', or 'long'.",
    )

    dimensions: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            'A dictionary of dimension types and their descriptions used to guide synthetic data generation. '
            "Valid keys include: 'features' (specific data attributes to reflect real-world structure), "
            "'persona' (AI-driven profiles capturing distinct perspectives), and "
            "'scenarios' (simulated events or contexts ensuring relevance and realism)."
        ),
    )

    custom_guidelines: Optional[str] = Field(
        default=None,
        description='Optional additional instructions or guidelines to condition question generation.',
    )

    example_question: Optional[str] = Field(
        default=None, description='An example question to guide the generation style.'
    )

    example_answer: Optional[str] = Field(
        default=None, description='An example answer to guide the generation style.'
    )

    max_reflection_iterations: int = Field(
        default=3,
        description='Maximum number of self-reflection and retry loops during QA validation.',
    )

    validation_threshold: float = Field(
        default=0.7,
        description=(
            'Minimum confidence or faithfulness score required to accept a QA pair. '
            'Range: 0.0 to 1.0.'
        ),
    )

    breakpoint_percentile_threshold: int = Field(
        default=95,
        description=(
            'Percentile threshold used to identify sentence breakpoints in semantic chunking. '
            'Higher values result in fewer, larger chunks.'
        ),
    )
