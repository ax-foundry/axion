from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from axion._core.schema import RichBaseModel, RichEnum
from pydantic import BaseModel, Field


class Status(str, RichEnum):
    STARTED = 'started'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'


class Trace(RichBaseModel):
    """Enhanced trace event for execution monitoring with Logfire integration."""

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description='Timestamp of when the trace event occurred.',
    )
    event_type: str = Field(
        ...,
        description="Category or type of the event (e.g., 'INFO', 'START', 'ERROR', 'WARNING').",
    )
    message: str = Field(
        ..., description='Human-readable description of the trace event.'
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description='Optional structured metadata associated with the trace event.',
    )
    span_id: Optional[str] = Field(
        default=None,
        description='Logfire span ID if this trace is associated with a span.',
    )
    trace_id: Optional[str] = Field(
        default=None, description='Logfire trace ID for grouping related events.'
    )


class ToolMetadata(RichBaseModel):
    """Enhanced metadata describing the tool with Logfire context."""

    name: str = Field(..., description='Name of the tool or system component.')
    description: Optional[str] = Field(
        default=None,
        description='Brief description of what the tool does or its primary function.',
    )
    task: Optional[str] = Field(
        default=None, description='Optional name of the specific task being executed.'
    )
    owner: Optional[str] = Field(
        default=None,
        description='Name or identifier of the team, service, or individual that owns this tool.',
    )
    version: Optional[str] = Field(
        default='0.0.0', description='Version string of the tool.'
    )
    run_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description='Optional runtime metadata passed in or generated during execution.',
    )


class LLMCallMetadata(BaseModel):
    """Metadata for a single LLM call with enhanced Logfire integration."""

    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency: float
    timestamp: datetime = Field(default_factory=datetime.now)
    prompt_length: Optional[int] = Field(
        default=None, description='Length of prompt in characters'
    )
    response_length: Optional[int] = Field(
        default=None, description='Length of response in characters'
    )
    cost_estimate: Optional[float] = Field(
        default=None, description='Estimated cost of the call'
    )
    error: Optional[str] = Field(
        default=None, description='Error message if call failed'
    )
    span_id: Optional[str] = Field(
        default=None, description='Associated Logfire span ID'
    )
    trace_id: Optional[str] = Field(
        default=None, description='Associated Logfire trace ID'
    )


class BaseExecutionMetadata(RichBaseModel):
    """Enhanced base metadata with Logfire tracing support."""

    id: UUID = Field(
        default_factory=uuid4,
        description='Unique identifier for the execution instance.',
    )
    name: str = Field(..., description='Descriptive name of the execution task or job.')
    session_id: UUID = Field(
        default_factory=uuid4,
        description='Unique identifier for the broader session or parent context this execution belongs to.',
    )
    start_time: Optional[str] = Field(
        default=None, description='Execution start time in ISO 8601 UTC string format.'
    )
    end_time: Optional[str] = Field(
        default=None, description='Execution end time in ISO 8601 UTC string format.'
    )
    latency: Optional[float] = Field(
        default=None, description='Duration of the execution in seconds.'
    )
    status: Status = Field(
        default=Status.STARTED, description='Current status of the execution.'
    )
    error: Optional[str] = Field(
        default=None, description='Error message if the execution failed.'
    )
    traces: List[Trace] = Field(
        default_factory=list, description='List of trace events for this execution.'
    )
    input_data: Dict[str, Any] = Field(
        default_factory=dict, description='Dictionary of input parameters or payload.'
    )
    output_data: Dict[str, Any] = Field(
        default_factory=dict, description='Dictionary of output results or response.'
    )
    tool_metadata: ToolMetadata = Field(
        ..., description='Additional metadata describing the tool.'
    )
    # Logfire fields
    logfire_span_id: Optional[str] = Field(
        default=None, description='Main Logfire span ID for this execution.'
    )
    logfire_trace_id: Optional[str] = Field(
        default=None, description='Logfire trace ID for grouping related executions.'
    )

    def add_trace(
        self,
        event_type: str,
        message: str,
        metadata: Dict[str, Any] = None,
        span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ):
        """Add a trace event with optional Logfire context."""
        self.traces.append(
            Trace(
                event_type=event_type,
                message=message,
                metadata=metadata or {},
                span_id=span_id,
                trace_id=trace_id,
            )
        )


class EvaluationStatus(str, RichEnum):
    """Status of an evaluation run"""

    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class EvaluationMetric(BaseModel):
    """Individual evaluation metric result"""

    name: str
    value: Union[float, int, str, bool]
    metric_type: str = Field(
        description='Type of metric (accuracy, f1, bleu, rouge, etc.)'
    )
    description: Optional[str] = None
    threshold: Optional[float] = Field(
        default=None, description='Pass/fail threshold if applicable'
    )
    passed: Optional[bool] = Field(
        default=None, description='Whether metric passed threshold'
    )


class EvaluationDatapoint(BaseModel):
    """Individual evaluation datapoint"""

    input_data: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    actual_output: Optional[Dict[str, Any]] = None
    metrics: List[EvaluationMetric] = Field(default_factory=list)
    latency: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationCallMetadata(BaseModel):
    """Metadata for a single evaluation run with enhanced tracing integration."""

    evaluation_id: str = Field(description='Unique identifier for this evaluation')
    evaluator_name: str = Field(
        description="Name of the evaluator (e.g., 'gpt-4-judge', 'rouge-evaluator')"
    )
    evaluator_type: str = Field(
        description='Type of evaluator (llm_judge, rule_based, metric_based, human)'
    )
    dataset_name: Optional[str] = Field(
        default=None, description='Name of the evaluation dataset'
    )
    dataset_size: int = Field(description='Number of datapoints evaluated')

    # Status and timing
    status: EvaluationStatus = EvaluationStatus.PENDING
    latency: float = Field(description='Total evaluation time in seconds')
    timestamp: datetime = Field(default_factory=datetime.now)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Results
    overall_metrics: List[EvaluationMetric] = Field(default_factory=list)
    datapoint_results: List[EvaluationDatapoint] = Field(default_factory=list)

    # Summary statistics
    total_datapoints: int = 0
    successful_datapoints: int = 0
    failed_datapoints: int = 0
    average_latency_per_datapoint: Optional[float] = None

    # Cost and resource usage
    cost_estimate: Optional[float] = Field(
        default=None, description='Estimated cost of evaluation'
    )
    tokens_used: Optional[int] = Field(
        default=None, description='Total tokens used if LLM-based'
    )

    # Error handling
    error: Optional[str] = Field(
        default=None, description='Error message if evaluation failed'
    )
    error_details: Dict[str, Any] = Field(default_factory=dict)

    # Tracing context
    span_id: Optional[str] = Field(default=None, description='Associated span ID')
    trace_id: Optional[str] = Field(default=None, description='Associated trace ID')

    # Configuration
    evaluator_config: Dict[str, Any] = Field(
        default_factory=dict, description='Evaluator configuration'
    )
    evaluation_criteria: List[str] = Field(
        default_factory=list, description='Evaluation criteria used'
    )


class EvaluationExecutionMetadata(BaseExecutionMetadata):
    """Metadata for evaluation execution tracking."""

    # Evaluation-specific fields
    evaluation_calls: List[EvaluationCallMetadata] = Field(default_factory=list)
    number_of_evaluations: int = 0
    total_datapoints_evaluated: int = 0
    total_evaluation_time: float = 0.0
    failed_evaluations: int = 0

    # Aggregated metrics
    unique_evaluators: set = Field(default_factory=set)
    unique_datasets: set = Field(default_factory=set)
    total_cost_estimate: float = 0.0
    total_tokens_used: int = 0

    # Performance statistics
    average_evaluation_latency: float = 0.0
    average_datapoint_latency: float = 0.0

    def add_evaluation_call(
        self,
        evaluation_id: str,
        evaluator_name: str,
        evaluator_type: str,
        dataset_size: int,
        latency: float,
        overall_metrics: List[EvaluationMetric] = None,
        datapoint_results: List[EvaluationDatapoint] = None,
        dataset_name: str = None,
        cost_estimate: float = None,
        tokens_used: int = None,
        error: str = None,
        span_id: str = None,
        trace_id: str = None,
        evaluator_config: Dict[str, Any] = None,
        **kwargs,
    ):
        """Add an evaluation call to the metadata."""

        # Calculate derived metrics
        successful_datapoints = len(
            [dp for dp in (datapoint_results or []) if dp.error is None]
        )
        failed_datapoints = dataset_size - successful_datapoints
        avg_latency_per_datapoint = latency / dataset_size if dataset_size > 0 else 0

        call = EvaluationCallMetadata(
            evaluation_id=evaluation_id,
            evaluator_name=evaluator_name,
            evaluator_type=evaluator_type,
            dataset_name=dataset_name,
            dataset_size=dataset_size,
            status=EvaluationStatus.FAILED if error else EvaluationStatus.COMPLETED,
            latency=latency,
            overall_metrics=overall_metrics or [],
            datapoint_results=datapoint_results or [],
            total_datapoints=dataset_size,
            successful_datapoints=successful_datapoints,
            failed_datapoints=failed_datapoints,
            average_latency_per_datapoint=avg_latency_per_datapoint,
            cost_estimate=cost_estimate,
            tokens_used=tokens_used,
            error=error,
            span_id=span_id,
            trace_id=trace_id,
            evaluator_config=evaluator_config or {},
            **kwargs,
        )

        self.evaluation_calls.append(call)
        self.number_of_evaluations += 1
        self.total_datapoints_evaluated += dataset_size
        self.total_evaluation_time += latency

        if error:
            self.failed_evaluations += 1

        # Update aggregated metrics
        self.unique_evaluators.add(evaluator_name)
        if dataset_name:
            self.unique_datasets.add(dataset_name)

        if cost_estimate:
            self.total_cost_estimate += cost_estimate

        if tokens_used:
            self.total_tokens_used += tokens_used

        # Update performance statistics
        if self.number_of_evaluations > 0:
            self.average_evaluation_latency = (
                self.total_evaluation_time / self.number_of_evaluations
            )

        if self.total_datapoints_evaluated > 0:
            self.average_datapoint_latency = (
                self.total_evaluation_time / self.total_datapoints_evaluated
            )


class LLMExecutionMetadata(EvaluationExecutionMetadata):
    """Metadata for tracking LLM execution with detailed metrics and statistics."""

    number_of_calls: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_llm_latency: float = 0.0
    average_latency: float = 0.0
    llm_calls: List[LLMCallMetadata] = Field(default_factory=list)
    total_cost_estimate: float = 0.0
    unique_models: Set[str] = Field(default_factory=set)
    unique_providers: Set[str] = Field(default_factory=set)
    failed_calls: int = 0
    retry_count: int = 0

    def add_llm_call(
        self,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency: float,
        prompt_length: Optional[int] = None,
        response_length: Optional[int] = None,
        cost_estimate: Optional[float] = None,
        error: Optional[str] = None,
        span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """Add a single LLM call entry and update metrics."""
        call = LLMCallMetadata(
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency=latency,
            prompt_length=prompt_length,
            response_length=response_length,
            cost_estimate=cost_estimate,
            error=error,
            span_id=span_id,
            trace_id=trace_id,
        )

        self.llm_calls.append(call)
        self.number_of_calls += 1
        self.total_tokens += call.total_tokens
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_llm_latency += latency

        self.average_latency = self.total_llm_latency / self.number_of_calls
        self.unique_models.add(model)
        self.unique_providers.add(provider)

        if cost_estimate:
            self.total_cost_estimate += cost_estimate

        if error:
            self.failed_calls += 1


class RetrieverCallMetadata(BaseModel):
    """Metadata for a single retrieval call."""

    id_: str
    file_name: str
    text: str
    score: Optional[float] = None
    latency: float
    timestamp: datetime = Field(default_factory=datetime.now)


class KnowledgeExecutionMetadata(BaseExecutionMetadata):
    """Knowledge/RAG execution metadata."""

    number_of_calls: int = 0
    total_tokens: int = 0
    llm_calls: List[LLMCallMetadata] = Field(default_factory=list)
    retrieved_calls: List[RetrieverCallMetadata] = Field(default_factory=list)

    def add_retrieval_call(self, context: List[Dict[str, Any]], latency: float):
        for content in context:
            call = RetrieverCallMetadata(
                id_=content.get('id_', 'Unknown'),
                file_name=content.get('file_name', 'Unknown'),
                text=content.get('text', 'Unknown'),
                score=content.get('score', None),
                latency=latency,
            )
            self.retrieved_calls.append(call)

    def add_llm_call(
        self,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency: float,
    ):
        call = LLMCallMetadata(
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency=latency,
        )
        self.llm_calls.append(call)
        self.number_of_calls += 1
        self.total_tokens += call.total_tokens


class DBExecutionMetadata(BaseExecutionMetadata):
    """Database execution metadata."""

    rows_affected: int = 0
    query_time: float = 0.0
    query_params: Dict[str, Any] = Field(default_factory=dict)
