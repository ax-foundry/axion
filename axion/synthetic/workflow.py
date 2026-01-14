from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypedDict, Union

from axion._core.logging import get_logger
from axion._core.metadata.schema import ToolMetadata
from axion._core.schema import EmbeddingRunnable, LLMRunnable, RichBaseModel
from axion._core.tracing import init_tracer, trace
from axion._core.tracing.handlers import BaseTraceHandler
from axion.synthetic.answer import AnswerGenerator
from axion.synthetic.preprocessing import DocumentProcessor
from axion.synthetic.question import QuestionGenerator
from axion.synthetic.reflection import (
    ENHANCEMENT_FEEDBACK_TEMPLATE,
    QAValidator,
)
from axion.synthetic.statement_extractor import StatementExtractor
from llama_index.core import Document
from pydantic import Field

from pydantic_graph import BaseNode, End, Graph, GraphRunContext


logger = get_logger(__name__)


def find_text_indices(corpus: str, snippet: str) -> Optional[Tuple[int, int]]:
    """Finds the start and end indices of a snippet within a larger corpus."""
    start_index = corpus.find(snippet)
    if start_index != -1:
        end_index = start_index + len(snippet)
        return start_index, end_index
    return None


class Message(TypedDict):
    """A dictionary representing a message, replacing BaseMessage."""

    role: str
    content: str


class QAWorkflowConfiguration(RichBaseModel):
    """Single source of truth for all workflow configuration."""

    splitter_type: Literal['semantic', 'sentence'] = Field(
        default='sentence', description='The type of text splitter to use.'
    )
    chunk_size: int = Field(
        default=2048, description='Target chunk size for sentence splitter.'
    )
    breakpoint_percentile_threshold: int = Field(
        default=95, ge=80, le=100, description='Threshold for semantic splitting.'
    )
    statements_per_chunk: int = Field(default=5, ge=1, le=20)
    num_pairs: int = Field(default=10, ge=1, le=100)
    question_types: List[str] = Field(default_factory=lambda: ['factual', 'analytical'])
    difficulty: Literal['easy', 'medium', 'hard'] = Field(default='medium')
    answer_length: Literal['short', 'medium', 'long'] = Field(default='medium')
    validation_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    max_reflection_iterations: int = Field(default=3, ge=1, le=10)
    custom_guidelines: Optional[str] = Field(default=None)
    dimensions: Dict[str, Any] = Field(default_factory=dict)
    example_question: Optional[str] = Field(default=None)
    example_answer: Optional[str] = Field(default=None)
    input_source: Literal['documents', 'sessions'] = Field(default='documents')
    session_analysis_depth: Literal['surface', 'deep'] = Field(default='surface')


@dataclass
class QAWorkflowState:
    """Streamlined state schema focused on workflow data and control."""

    # Input data
    content: Optional[str] = None
    session_messages: List[Message] = field(default_factory=list)
    session_metadata: Dict[str, Any] = field(default_factory=dict)

    # Processing pipeline data
    processed_content: str = ''
    chunks: List[str] = field(default_factory=list)
    statements: List[Dict[str, Any]] = field(default_factory=list)
    questions: List[Dict[str, Any]] = field(default_factory=list)
    qa_pairs: List[Dict[str, Any]] = field(default_factory=list)
    validated_qa_pairs: List[Dict[str, Any]] = field(default_factory=list)
    input_source: Literal['documents', 'sessions'] = 'documents'
    validation_threshold: float = 0.8
    max_reflection_iterations: int = 3

    # Workflow control
    current_iteration: int = 1
    average_quality: float = 0.0
    enhancement_feedback: str = ''
    processing_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for backward compatibility."""
        return {
            'content': self.content,
            'session_messages': self.session_messages,
            'session_metadata': self.session_metadata,
            'processed_content': self.processed_content,
            'chunks': self.chunks,
            'statements': self.statements,
            'questions': self.questions,
            'qa_pairs': self.qa_pairs,
            'validated_qa_pairs': self.validated_qa_pairs,
            'input_source': self.input_source,
            'validation_threshold': self.validation_threshold,
            'max_reflection_iterations': self.max_reflection_iterations,
            'current_iteration': self.current_iteration,
            'average_quality': self.average_quality,
            'enhancement_feedback': self.enhancement_feedback,
            'processing_errors': self.processing_errors,
        }


@dataclass
class QAWorkflowResult:
    """The final result of running the QA workflow."""

    validated_qa_pairs: List[Dict[str, Any]]
    statements: List[Dict[str, Any]]
    average_quality: float
    iterations: int
    errors: List[str]
    content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for backward compatibility."""
        return {
            'validated_qa_pairs': self.validated_qa_pairs,
            'statements': self.statements,
            'average_quality': self.average_quality,
            'current_iteration': self.iterations,
            'processing_errors': self.errors,
            'content': self.content,
        }


@dataclass
class QAWorkflowDeps:
    """Dependencies for the QA workflow nodes."""

    config: QAWorkflowConfiguration
    llm: LLMRunnable
    embed_model: Optional[EmbeddingRunnable] = None
    extra_transformations: List[Callable] = field(default_factory=list)
    statement_extractor: Optional[StatementExtractor] = None
    question_generator: Optional[QuestionGenerator] = None
    answer_generator: Optional[AnswerGenerator] = None
    qa_validator: Optional[QAValidator] = None
    document_processor: Optional[DocumentProcessor] = None
    document_processor_cls: Optional[DocumentProcessor] = None
    tracer: Optional[BaseTraceHandler] = None

    def __post_init__(self):
        """Initialize components that depend on LLM."""
        if self.statement_extractor is None:
            self.statement_extractor = StatementExtractor(self.llm, tracer=self.tracer)
        if self.question_generator is None:
            self.question_generator = QuestionGenerator(self.llm, tracer=self.tracer)
        if self.answer_generator is None:
            self.answer_generator = AnswerGenerator(self.llm, tracer=self.tracer)
        if self.qa_validator is None:
            self.qa_validator = QAValidator(self.llm, tracer=self.tracer)


def handle_node_errors(func: Callable) -> Callable:
    """Decorator to handle exceptions in node run methods."""

    @wraps(func)
    async def wrapper(self, ctx: GraphRunContext) -> Any:
        try:
            return await func(self, ctx)
        except Exception as e:
            error_msg = f'Error in {self.__class__.__name__}: {str(e)}'
            logger.error(error_msg, exc_info=True)
            ctx.state.processing_errors.append(error_msg)
            return HandleError()

    return wrapper


# Forward declarations for type hints
class ProcessDocuments:
    pass


class ProcessSessions:
    pass


class ChunkContent:
    pass


class ExtractStatements:
    pass


class GenerateQuestions:
    pass


class GenerateAnswers:
    pass


class ValidateQAPairs:
    pass


class PrepareEnhancement:
    pass


class HandleError:
    pass


@dataclass
class InitializeWorkflow(BaseNode[QAWorkflowState, QAWorkflowDeps, QAWorkflowResult]):
    """Initialize workflow and route to appropriate processor based on input source."""

    @handle_node_errors
    async def run(
        self, ctx: GraphRunContext[QAWorkflowState, QAWorkflowDeps]
    ) -> ProcessDocuments | ProcessSessions | HandleError:
        cfg = ctx.deps.config
        state = ctx.state

        logger.info(
            f"Initializing workflow. Splitter: '{cfg.splitter_type}', "
            f'Iterations: {cfg.max_reflection_iterations}, Threshold: {cfg.validation_threshold}'
        )

        # Initialize document processor
        splitter_kwargs = {
            'breakpoint_percentile_threshold': cfg.breakpoint_percentile_threshold,
            'chunk_size': cfg.chunk_size,
        }
        if ctx.deps.document_processor_cls:
            ctx.deps.document_processor = ctx.deps.document_processor_cls
        else:
            ctx.deps.document_processor = DocumentProcessor(
                splitter_type=cfg.splitter_type,
                embed_model=ctx.deps.embed_model,
                tracer=ctx.deps.tracer,
                **splitter_kwargs,
            )

        # Validate inputs based on source
        if cfg.input_source == 'documents' and not state.content:
            raise ValueError(
                "Document content is required for 'documents' input source."
            )
        if cfg.input_source == 'sessions' and not state.session_messages:
            raise ValueError(
                "Session messages are required for 'sessions' input source."
            )

        # Store config values in state for later use
        state.input_source = cfg.input_source
        state.validation_threshold = cfg.validation_threshold
        state.max_reflection_iterations = cfg.max_reflection_iterations

        # Route based on input source
        if cfg.input_source == 'documents':
            return ProcessDocuments()
        return ProcessSessions()


@dataclass
class ProcessDocuments(BaseNode[QAWorkflowState, QAWorkflowDeps, QAWorkflowResult]):
    """Process document input with optional transformations."""

    @handle_node_errors
    async def run(
        self, ctx: GraphRunContext[QAWorkflowState, QAWorkflowDeps]
    ) -> ChunkContent | HandleError:
        logger.info('Processing document input.')
        processed_content = ctx.state.content
        for transformation in ctx.deps.extra_transformations:
            processed_content = transformation(processed_content)
        ctx.state.processed_content = processed_content
        return ChunkContent()


@dataclass
class ProcessSessions(BaseNode[QAWorkflowState, QAWorkflowDeps, QAWorkflowResult]):
    """Process session messages into unified content."""

    @handle_node_errors
    async def run(
        self, ctx: GraphRunContext[QAWorkflowState, QAWorkflowDeps]
    ) -> ChunkContent | HandleError:
        logger.info('Processing session input.')
        conversation_text = '\n'.join(
            [
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in ctx.state.session_messages
            ]
        )
        metadata_text = (
            f'Session Context: {ctx.state.session_metadata}\n\n'
            if ctx.state.session_metadata
            else ''
        )
        ctx.state.processed_content = metadata_text + conversation_text
        return ChunkContent()


@dataclass
class ChunkContent(BaseNode[QAWorkflowState, QAWorkflowDeps, QAWorkflowResult]):
    """Chunk processed content using configuration parameters."""

    @trace
    @handle_node_errors
    async def run(
        self, ctx: GraphRunContext[QAWorkflowState, QAWorkflowDeps]
    ) -> ExtractStatements | HandleError:
        chunks = await ctx.deps.document_processor.process(ctx.state.processed_content)
        logger.info(f'Content split into {len(chunks)} chunks.')
        ctx.state.chunks = chunks
        return ExtractStatements()


@dataclass
class ExtractStatements(BaseNode[QAWorkflowState, QAWorkflowDeps, QAWorkflowResult]):
    """Extract key statements from content chunks."""

    @trace
    @handle_node_errors
    async def run(
        self, ctx: GraphRunContext[QAWorkflowState, QAWorkflowDeps]
    ) -> GenerateQuestions | HandleError:
        cfg = ctx.deps.config
        state = ctx.state

        unique_statement_texts = (
            await ctx.deps.statement_extractor.extract_statements_from_chunks(
                chunks=state.chunks, statements_per_chunk=cfg.statements_per_chunk
            )
        )

        statements_with_indices = []
        original_document = state.content

        for stmt_text in unique_statement_texts:
            indices = find_text_indices(original_document, stmt_text)
            if indices:
                statements_with_indices.append(
                    {
                        'content': stmt_text,
                        'start_index': indices[0],
                        'end_index': indices[1],
                    }
                )

        logger.info(
            f'Extracted {len(statements_with_indices)} unique statements with indices.'
        )
        state.statements = statements_with_indices
        return GenerateQuestions()


@dataclass
class GenerateQuestions(BaseNode[QAWorkflowState, QAWorkflowDeps, QAWorkflowResult]):
    """Generate questions using configuration parameters and feedback."""

    @trace
    @handle_node_errors
    async def run(
        self, ctx: GraphRunContext[QAWorkflowState, QAWorkflowDeps]
    ) -> GenerateAnswers | HandleError:
        cfg = ctx.deps.config
        state = ctx.state

        guidelines = cfg.custom_guidelines or ''
        statement_texts = [statement['content'] for statement in state.statements]

        if cfg.dimensions:
            features = cfg.dimensions.get('features')
            persona = cfg.dimensions.get('persona')
            scenarios = cfg.dimensions.get('scenarios')

            guidelines += (
                '\n\nBefore generating data, please understand the context defined below. '
                'These dimensions help guide how synthetic questions or examples should be structured:\n\n'
                f'- **Features** describe the core attributes or data characteristics involved. For this case: {features}.\n'
                f'- **Persona** represents the profile of the user or subject involved, with motivations and behaviors. Here: {persona}.\n'
                f'- **Scenarios** define the situation or use case context being simulated. In this case: {scenarios}.\n\n'
                'Use these dimensions to generate relevant, realistic synthetic examples that align with this setup. '
                'Focus on natural language, consistency, and plausible outputs.'
            )

        if cfg.example_question:
            guidelines += f"\n\nPlease generate questions in a similar style to this example:\n'{cfg.example_question}'"

        if state.current_iteration > 1 and state.enhancement_feedback:
            logger.info(
                f'Applying enhancement feedback for iteration {state.current_iteration}'
            )
            guidelines += f'\n\n--- Improvement Guidelines (Iteration {state.current_iteration}) ---\n{state.enhancement_feedback}'

        questions = await ctx.deps.question_generator.balance_question_types(
            statements=statement_texts,
            num_questions=cfg.num_pairs,
            question_types=cfg.question_types,
            difficulty=cfg.difficulty,
            custom_guidelines=guidelines,
        )
        logger.info(
            f'Generated {len(questions)} questions (iteration {state.current_iteration}).'
        )
        state.questions = questions
        return GenerateAnswers()


@dataclass
class GenerateAnswers(BaseNode[QAWorkflowState, QAWorkflowDeps, QAWorkflowResult]):
    """Generate answers using configuration parameters."""

    @trace
    @handle_node_errors
    async def run(
        self, ctx: GraphRunContext[QAWorkflowState, QAWorkflowDeps]
    ) -> ValidateQAPairs | HandleError:
        cfg = ctx.deps.config
        state = ctx.state

        guidelines = cfg.custom_guidelines or ''
        statement_texts = [statement['content'] for statement in state.statements]

        if cfg.example_answer:
            guidelines += f"\n\nPlease generate answers in a similar style to this example:\n'{cfg.example_answer}'"

        qa_pairs = await ctx.deps.answer_generator.generate_answers_batch(
            questions=state.questions,
            statements=statement_texts,
            answer_length=cfg.answer_length,
            custom_guidelines=guidelines,
        )
        logger.info(f'Generated {len(qa_pairs)} QA pairs.')
        state.qa_pairs = qa_pairs
        return ValidateQAPairs()


@dataclass
class ValidateQAPairs(BaseNode[QAWorkflowState, QAWorkflowDeps, QAWorkflowResult]):
    """Validate QA pairs and determine whether to continue or end."""

    @trace
    @handle_node_errors
    async def run(
        self, ctx: GraphRunContext[QAWorkflowState, QAWorkflowDeps]
    ) -> PrepareEnhancement | End[QAWorkflowResult] | HandleError:
        state = ctx.state
        statement_texts = [statement['content'] for statement in state.statements]

        validated_qa_pairs = await ctx.deps.qa_validator.validate_qa_pairs_batch(
            qa_pairs=state.qa_pairs,
            statements=statement_texts,
        )
        quality_scores = [pair.get('quality_score', 0.0) for pair in validated_qa_pairs]
        average_quality = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        )

        state.validated_qa_pairs = validated_qa_pairs
        state.average_quality = average_quality

        logger.info(
            f'Iteration {state.current_iteration}: Avg quality {average_quality:.2f} '
            f'(threshold: {state.validation_threshold:.2f})'
        )

        # Determine whether to continue, end, or handle errors
        if state.processing_errors:
            return HandleError()

        quality_threshold_met = average_quality >= state.validation_threshold
        max_iterations_reached = state.current_iteration >= state.max_reflection_iterations

        if quality_threshold_met or max_iterations_reached:
            if max_iterations_reached and not quality_threshold_met:
                logger.warning(
                    f'Max iterations ({state.max_reflection_iterations}) reached, '
                    f'but quality ({average_quality:.2f}) is still below threshold '
                    f'({state.validation_threshold:.2f}).'
                )
            logger.info(
                'Quality threshold met or max iterations reached. Ending workflow.'
            )
            return End(
                QAWorkflowResult(
                    validated_qa_pairs=state.validated_qa_pairs,
                    statements=state.statements,
                    average_quality=state.average_quality,
                    iterations=state.current_iteration,
                    errors=state.processing_errors,
                    content=state.content,
                )
            )
        else:
            logger.info('Quality below threshold. Continuing to next iteration.')
            return PrepareEnhancement()


@dataclass
class PrepareEnhancement(BaseNode[QAWorkflowState, QAWorkflowDeps, QAWorkflowResult]):
    """Prepare feedback for the next iteration using the prompt template."""

    @trace
    @handle_node_errors
    async def run(
        self, ctx: GraphRunContext[QAWorkflowState, QAWorkflowDeps]
    ) -> GenerateQuestions | HandleError:
        state = ctx.state

        logger.info(
            f'Preparing enhancement feedback for iteration {state.current_iteration + 1}.'
        )

        low_quality_pairs = [
            p
            for p in state.validated_qa_pairs
            if p.get('quality_score', 0.0) < state.validation_threshold
        ]
        feedback_examples = '\n'.join(
            [
                f"- Question: '{p['question'][:60]}...' | Issue: {p.get('validation_feedback', 'N/A')}"
                for p in low_quality_pairs[:3]
            ]
        )

        enhancement_feedback = ENHANCEMENT_FEEDBACK_TEMPLATE.format(
            average_quality=state.average_quality,
            validation_threshold=state.validation_threshold,
            num_low_quality=len(low_quality_pairs),
            feedback_examples=feedback_examples or 'No specific examples to show.',
        )

        state.current_iteration += 1
        state.enhancement_feedback = enhancement_feedback
        return GenerateQuestions()


@dataclass
class HandleError(BaseNode[QAWorkflowState, QAWorkflowDeps, QAWorkflowResult]):
    """Terminal node to handle errors gracefully."""

    async def run(
        self, ctx: GraphRunContext[QAWorkflowState, QAWorkflowDeps]
    ) -> End[QAWorkflowResult]:
        state = ctx.state

        logger.error(
            f'Workflow ended due to {len(state.processing_errors)} processing error(s).'
        )
        for error in state.processing_errors:
            logger.error(f'- {error}')

        return End(
            QAWorkflowResult(
                validated_qa_pairs=state.validated_qa_pairs,
                statements=state.statements,
                average_quality=state.average_quality,
                iterations=state.current_iteration,
                errors=state.processing_errors,
                content=state.content,
            )
        )


# Build the graph with all node types
qa_workflow_graph = Graph(
    nodes=(
        InitializeWorkflow,
        ProcessDocuments,
        ProcessSessions,
        ChunkContent,
        ExtractStatements,
        GenerateQuestions,
        GenerateAnswers,
        ValidateQAPairs,
        PrepareEnhancement,
        HandleError,
    ),
    name='qa_workflow_graph',
)


class QAWorkflowGraph:
    """Builds and runs the pydantic_graph workflow, focusing on orchestration."""

    PAIRS_NAME = 'qa_pairs'
    QUESTION_NAME = 'question'
    ANSWER_NAME = 'answer'

    def __init__(
        self,
        llm: LLMRunnable,
        embed_model: EmbeddingRunnable = None,
        extra_transformations: Optional[List[Callable]] = None,
        document_processor_cls: Optional[DocumentProcessor] = None,
        tracer: Optional[BaseTraceHandler] = None,
    ):
        self.llm = llm
        self.embed_model = embed_model
        self.extra_transformations = extra_transformations or []
        self.document_processor_cls = document_processor_cls
        self.tracer = init_tracer('llm', self._get_tool_metadata(), tracer)
        self.graph = qa_workflow_graph

    def _get_tool_metadata(self):
        return ToolMetadata(
            name=self.__class__.__name__,
            description='QA Workflow Actions',
            owner='AXION',
            version='1.0.0',
        )

    def _create_deps(self, config: QAWorkflowConfiguration) -> QAWorkflowDeps:
        """Create dependencies for the workflow."""
        return QAWorkflowDeps(
            config=config,
            llm=self.llm,
            embed_model=self.embed_model,
            extra_transformations=self.extra_transformations,
            document_processor_cls=self.document_processor_cls,
            tracer=self.tracer,
        )

    @staticmethod
    def extract_document_text(content: Union[str, Document]) -> str:
        """Extract Document Text"""
        if isinstance(content, Document):
            return content.text
        return content

    @trace
    async def run_from_documents(
        self, content: Union[str, Document], **config_kwargs
    ) -> Dict[str, Any]:
        """Run workflow with document input."""
        content = self.extract_document_text(content)
        config_kwargs['input_source'] = 'documents'
        config = QAWorkflowConfiguration(**config_kwargs)

        state = QAWorkflowState(content=content)
        deps = self._create_deps(config)

        result = await self.graph.run(InitializeWorkflow(), state=state, deps=deps)

        # Return dict for backward compatibility
        final_dict = state.to_dict()
        if result.output:
            final_dict.update(result.output.to_dict())
        return final_dict

    async def run_from_sessions(
        self,
        session_messages: List[Message],
        session_metadata: Optional[Dict[str, Any]] = None,
        **config_kwargs,
    ) -> Dict[str, Any]:
        """Run workflow with session input."""
        config_kwargs['input_source'] = 'sessions'
        config = QAWorkflowConfiguration(**config_kwargs)

        state = QAWorkflowState(
            session_messages=session_messages, session_metadata=session_metadata or {}
        )
        deps = self._create_deps(config)

        result = await self.graph.run(InitializeWorkflow(), state=state, deps=deps)

        # Return dict for backward compatibility
        final_dict = state.to_dict()
        if result.output:
            final_dict.update(result.output.to_dict())
        return final_dict

    def visualize_graph(self):
        """Generate a visual representation of the workflow graph"""
        try:
            from IPython.display import display, Markdown

            mermaid_code = self.graph.mermaid_code(start_node=InitializeWorkflow)
            display(Markdown(f'```mermaid\n{mermaid_code}\n```'))
        except ImportError:
            logger.info('Install IPython to visualize the graph in notebooks')
            print(self.graph.mermaid_code(start_node=InitializeWorkflow))
        except Exception as e:
            logger.error(f'Graph visualization error: {str(e)}')
