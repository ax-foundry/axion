import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

from axion._core.asyncio import SemaphoreExecutor
from axion._core.logging import get_logger
from axion._core.metadata.schema import ToolMetadata
from axion._core.tracing import init_tracer
from axion._core.utils import Timer
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric
from axion.metrics.cache import AnalysisCache
from axion.metrics.internals.judges import (
    BatchFaithfulnessJudge,
    BatchRecallJudge,
    BatchRelevancyJudge,
    BatchUtilizationJudge,
    ChunkUsefulnessJudge,
    ContextSufficiencyJudge,
    FaithfulnessJudge,
    RecallJudge,
    StatementExtractor,
    UtilizationJudge,
)
from axion.metrics.internals.schema import (
    AnalysisMetadata,
    ContextSufficiencyVerdict,
    EvaluationMode,
    JudgedClaim,
    JudgedContextChunk,
    JudgedGroundTruthStatement,
    RAGAnalyzerResult,
)

logger = get_logger(__name__)
T = TypeVar('T')


class JudgeType(Enum):
    """Registry of available judge types for RAG analysis."""

    STATEMENT_EXTRACTOR = StatementExtractor
    RELEVANCY = BatchRelevancyJudge
    USEFULNESS = ChunkUsefulnessJudge
    FAITHFULNESS = FaithfulnessJudge
    RECALL = RecallJudge
    BATCH_FAITHFULNESS = BatchFaithfulnessJudge
    BATCH_RECALL = BatchRecallJudge
    UTILIZATION = UtilizationJudge
    BATCH_UTILIZATION = BatchUtilizationJudge
    CONTEXT_SUFFICIENCY = ContextSufficiencyJudge


@dataclass
class ExecutionStrategy:
    """Configuration for batch-or-granular execution strategy."""

    items: List[Any]
    batch_judge: JudgeType
    granular_judge: JudgeType
    batch_payload: Dict[str, Any]
    granular_payload_factory: Callable[[Any], Dict[str, Any]]


class RAGAnalyzer:
    """
    Orchestrates comprehensive RAG evaluation by coordinating multiple judges
    to assess claims, context chunks, and ground truth statements.

    Features:
    - Intelligent caching to avoid redundant computations
    - Batch processing with automatic granular fallback
    - Controlled concurrency for optimal performance
    - Detailed execution metadata for observability
    - Automatic cost tracking per execution in metadata
    """

    DEFAULT_COMPONENTS = {
        'claim_relevancy',
        'claim_faithfulness',
        'chunk_relevancy',
        'chunk_usefulness',
        'chunk_utilization',
        'context_sufficiency',
        'gt',
    }

    def __init__(
        self,
        mode: EvaluationMode = EvaluationMode.GRANULAR,
        max_concurrent_llm_calls: int = 5,
        force_granular: bool = False,
        allow_fallback: bool = True,
        **kwargs,
    ):
        """
        Initialize the RAG analyzer with configurable execution strategy.

        Args:
            mode: Level of analysis detail (GRANULAR or SUMMARY)
            max_concurrent_llm_calls: Concurrency limit for parallel judge calls
            force_granular: Bypass batch judges and use concurrent granular calls
            allow_fallback: Allow fallback to granular when batch calls fail
            **kwargs: Additional judge configuration (e.g., shared model, tracer)
        """
        self.mode = mode
        self.force_granular = force_granular
        self.allow_fallback = allow_fallback
        self.tracer = init_tracer(
            'llm', self._get_tool_metadata(), kwargs.get('tracer')
        )
        self.judges = self._initialize_judges(**kwargs)
        self.executor = SemaphoreExecutor(max_concurrent=max_concurrent_llm_calls)

    def _get_tool_metadata(self) -> ToolMetadata:
        """Get tool metadata for tracer initialization."""
        return ToolMetadata(
            name=self.__class__.__name__,
            description='RAG Analyzer Engine',
            owner='AXION',
            version='1.0.0',
        )

    def _initialize_judges(self, **kwargs) -> Dict[JudgeType, BaseMetric]:
        """Initialize all judge instances with shared configuration."""
        judge_kwargs = {'model': kwargs['model']} if 'model' in kwargs else {}
        final_kwargs = {**judge_kwargs, **kwargs}
        return {
            judge_type: judge_type.value(**final_kwargs) for judge_type in JudgeType
        }

    def get_judge(self, judge_type: JudgeType) -> BaseMetric:
        """Access a specific judge instance (useful for cost estimation)."""
        return self.judges[judge_type]

    @staticmethod
    def _track_judge_usage(
        judge_type: JudgeType, metadata: AnalysisMetadata, count: int = 1
    ):
        """
        Track judge usage count in metadata.

        Note: Cost tracking happens after execution via _update_costs_from_judges()
        since cost_estimate is calculated during execution.

        Args:
            judge_type: The type of judge being used
            metadata: The metadata object to update
            count: Number of times the judge is being called (default: 1)
        """
        metadata.judge_usage[judge_type.name] = (
            metadata.judge_usage.get(judge_type.name, 0) + count
        )

    def _update_costs_from_judges(self, metadata: AnalysisMetadata):
        """
        Update cost metadata from judges after execution.

        This must be called after all judge executions since cost_estimate
        is calculated during the execute() call.

        Args:
            metadata: The metadata object to update with cost information
        """
        cost_estimate = 0.0
        cost_by_judge = {}

        for judge_type_name, usage_count in metadata.judge_usage.items():
            # Convert string name back to enum
            try:
                judge_type = JudgeType[judge_type_name]
                judge = self.get_judge(judge_type)
                unit_cost = getattr(judge, 'cost_estimate', None)

                if unit_cost is not None and usage_count > 0:
                    judge_cost = unit_cost * usage_count
                    cost_by_judge[judge_type_name] = judge_cost
                    cost_estimate += judge_cost
            except (KeyError, AttributeError) as e:
                logger.debug(f'Could not get cost for judge {judge_type_name}: {e}')
                continue

        metadata.cost_estimate = cost_estimate
        metadata.cost_by_judge = cost_by_judge

    async def execute(
        self,
        item: DatasetItem,
        cache: Optional[AnalysisCache] = None,
        required_components: Optional[List[str]] = None,
    ) -> RAGAnalyzerResult:
        """
        Execute comprehensive RAG analysis on a dataset item.

        Returns a structured result containing judged claims, chunks, and ground truth
        statements, along with execution metadata.
        """
        metadata = AnalysisMetadata(mode=self.mode)
        required = (
            set(required_components) if required_components else self.DEFAULT_COMPONENTS
        )

        with Timer() as timer:
            try:
                claims, chunks, ground_truths, sufficiency = await asyncio.gather(
                    self._analyze_claims(item, required, metadata, cache),
                    self._analyze_chunks(item, required, metadata, cache),
                    self._analyze_ground_truth(item, required, metadata, cache),
                    self._analyze_context_sufficiency(item, required, metadata, cache),
                )

                metadata.execution_time = timer.elapsed_time

                # Update costs from judges after all executions are complete
                self._update_costs_from_judges(metadata)

                return RAGAnalyzerResult(
                    judged_claims=claims,
                    judged_context_chunks=chunks,
                    judged_ground_truth_statements=ground_truths,
                    context_sufficiency_verdict=sufficiency,
                    metadata=metadata,
                )
            except Exception as e:
                logger.error(
                    f'RAG analysis failed for item {item.id}: {e}', exc_info=True
                )
                metadata.execution_time = timer.elapsed_time

                # Still try to capture costs even on failure
                self._update_costs_from_judges(metadata)

                return RAGAnalyzerResult(metadata=metadata, error=str(e))

    async def _analyze_claims(
        self,
        item: DatasetItem,
        required: Set[str],
        metadata: AnalysisMetadata,
        cache: Optional[AnalysisCache],
    ) -> List[JudgedClaim]:
        """Extract and evaluate claims from the model's output."""
        if not self._should_analyze('claim', required):
            return []

        claims = await self._cached_computation(
            item.id,
            'rag_extracted_claims',
            lambda: self._extract_claims(item, metadata),
            cache,
            metadata,
        )

        if not claims:
            return []

        relevancy_verdicts = await self._get_claim_relevancy(
            item, claims, required, metadata, cache
        )
        faithfulness_verdicts = await self._get_claim_faithfulness(
            item, claims, required, metadata, cache
        )

        return self._assemble_claims(claims, relevancy_verdicts, faithfulness_verdicts)

    async def _get_claim_relevancy(
        self,
        item: DatasetItem,
        claims: List[str],
        required: Set[str],
        metadata: AnalysisMetadata,
        cache: Optional[AnalysisCache],
    ) -> List[Any]:
        """Evaluate whether claims are relevant to the query."""
        if 'claim_relevancy' not in required:
            return []

        return await self._cached_computation(
            item.id,
            'rag_relevancy_judgments',
            lambda: self._judge_relevancy(item.query, claims, metadata),
            cache,
            metadata,
        )

    async def _get_claim_faithfulness(
        self,
        item: DatasetItem,
        claims: List[str],
        required: Set[str],
        metadata: AnalysisMetadata,
        cache: Optional[AnalysisCache],
    ) -> List[Any]:
        """Evaluate whether claims are faithful to retrieved context."""
        if 'claim_faithfulness' not in required:
            return []

        context = '\n'.join(item.retrieved_content or [])
        strategy = ExecutionStrategy(
            items=claims,
            batch_judge=JudgeType.BATCH_FAITHFULNESS,
            granular_judge=JudgeType.FAITHFULNESS,
            batch_payload={'claims': claims, 'evidence': context},
            granular_payload_factory=lambda claim: {
                'claim': claim,
                'evidence': context,
            },
        )

        return await self._cached_computation(
            item.id,
            'rag_faithfulness_judgments',
            lambda: self._execute_with_strategy(strategy, metadata),
            cache,
            metadata,
        )

    def _assemble_claims(
        self,
        claims: List[str],
        relevancy_verdicts: List[Any],
        faithfulness_verdicts: List[Any],
    ) -> List[JudgedClaim]:
        """Combine claims with their evaluation verdicts."""
        judged_claims = []

        for i, claim_text in enumerate(claims):
            claim = JudgedClaim(claim_text=claim_text)
            reasons = []

            # Attach relevancy verdict if available
            if i < len(relevancy_verdicts) and hasattr(
                relevancy_verdicts[i], 'verdict'
            ):
                verdict = relevancy_verdicts[i]
                claim.relevancy_verdict = verdict.verdict
                if verdict.reason:
                    reasons.append(f'Relevancy: {verdict.reason}')

            # Attach faithfulness verdict if available
            if i < len(faithfulness_verdicts) and not isinstance(
                faithfulness_verdicts[i], Exception
            ):
                verdict = faithfulness_verdicts[i]
                claim.faithfulness_verdict = verdict.verdict
                if verdict.reason:
                    reasons.append(f'Faithfulness: {verdict.reason}')

            claim.reason = ' | '.join(reasons) if reasons else None
            judged_claims.append(claim)

        return judged_claims

    async def _analyze_chunks(
        self,
        item: DatasetItem,
        required: Set[str],
        metadata: AnalysisMetadata,
        cache: Optional[AnalysisCache],
    ) -> List[JudgedContextChunk]:
        """Evaluate retrieved context chunks for relevancy, usefulness, and utilization."""
        if not self._should_analyze('chunk', required):
            return []

        chunks = item.retrieved_content or []
        if not chunks:
            return []

        relevancy, usefulness, utilization = await asyncio.gather(
            self._get_chunk_relevancy(item, chunks, required, metadata, cache),
            self._get_chunk_usefulness(item, chunks, required, metadata, cache),
            self._get_chunk_utilization(item, chunks, required, metadata, cache),
        )

        judged_chunks = []
        for text, is_relevant, is_useful, is_utilized in zip(
            chunks, relevancy, usefulness, utilization
        ):
            if hasattr(is_relevant, 'verdict'):
                # Handle structured verdict (e.g., from BatchRelevancyJudge)
                is_relevant_to_query = is_relevant.verdict == 'yes'
            elif isinstance(is_relevant, bool):
                # Handle the explicit False return on failure/skip (which is the bool that fails)
                is_relevant_to_query = is_relevant
            elif isinstance(is_relevant, Exception):
                # Handle granular execution exception fallback (safely defaults to False/Unknown)
                is_relevant_to_query = False
                logger.warning(
                    f'Chunk relevancy failed for chunk: {text[:50]}... Exception: {is_relevant}'
                )
            else:
                # Default safety fallback
                is_relevant_to_query = False

            judged_chunks.append(
                JudgedContextChunk(
                    chunk_text=text,
                    is_relevant_to_query=is_relevant_to_query,
                    is_useful_for_expected_output=is_useful,
                    is_utilized_in_answer=is_utilized,
                )
            )

        return judged_chunks

    async def _get_chunk_relevancy(
        self,
        item: DatasetItem,
        chunks: List[str],
        required: Set[str],
        metadata: AnalysisMetadata,
        cache: Optional[AnalysisCache],
    ) -> List[bool]:
        """Evaluate whether chunks are relevant to the query."""
        if 'chunk_relevancy' not in required:
            return [False] * len(chunks)

        return await self._cached_computation(
            item.id,
            'rag_chunk_relevancy_verdicts',
            lambda: self._judge_relevancy(item.query, chunks, metadata),
            cache,
            metadata,
        )

    async def _get_chunk_usefulness(
        self,
        item: DatasetItem,
        chunks: List[str],
        required: Set[str],
        metadata: AnalysisMetadata,
        cache: Optional[AnalysisCache],
    ) -> List[bool]:
        """Evaluate whether chunks are useful for generating expected output."""
        if 'chunk_usefulness' not in required or not item.expected_output:
            return [False] * len(chunks)

        return await self._cached_computation(
            item.id,
            'rag_chunk_usefulness_verdicts',
            lambda: self._judge_chunk_usefulness(
                chunks, item.expected_output, metadata
            ),
            cache,
            metadata,
        )

    async def _get_chunk_utilization(
        self,
        item: DatasetItem,
        chunks: List[str],
        required: Set[str],
        metadata: AnalysisMetadata,
        cache: Optional[AnalysisCache],
    ) -> List[Optional[bool]]:
        """Evaluate whether chunks were actually utilized in the generated answer."""
        if 'chunk_utilization' not in required or not item.actual_output:
            return [None] * len(chunks)

        return await self._cached_computation(
            item.id,
            'rag_chunk_utilization_verdicts',
            lambda: self._judge_chunk_utilization(chunks, item.actual_output, metadata),
            cache,
            metadata,
        )

    async def _analyze_ground_truth(
        self,
        item: DatasetItem,
        required: Set[str],
        metadata: AnalysisMetadata,
        cache: Optional[AnalysisCache],
    ) -> List[JudgedGroundTruthStatement]:
        """Evaluate ground truth statements against retrieved context."""
        if 'gt' not in required:
            return []

        return await self._cached_computation(
            item.id,
            'rag_judged_gt',
            lambda: self._judge_ground_truth(item, metadata),
            cache,
            metadata,
        )

    async def _analyze_context_sufficiency(
        self,
        item: DatasetItem,
        required: Set[str],
        metadata: AnalysisMetadata,
        cache: Optional[AnalysisCache],
    ) -> Optional[ContextSufficiencyVerdict]:
        """Evaluate whether the retrieved context is sufficient to answer the query."""
        if 'context_sufficiency' not in required:
            return None

        return await self._cached_computation(
            item.id,
            'rag_context_sufficiency',
            lambda: self._judge_context_sufficiency(item, metadata),
            cache,
            metadata,
        )

    async def _judge_ground_truth(
        self, item: DatasetItem, metadata: AnalysisMetadata
    ) -> List[JudgedGroundTruthStatement]:
        """Extract ground truth statements and verify them against context."""
        gt_statements = await self._extract_claims(item, metadata, from_expected=True)
        if not gt_statements:
            return []

        context = '\n'.join(item.retrieved_content or [])
        strategy = ExecutionStrategy(
            items=gt_statements,
            batch_judge=JudgeType.BATCH_RECALL,
            granular_judge=JudgeType.RECALL,
            batch_payload={'statements': gt_statements, 'context': context},
            granular_payload_factory=lambda s: {
                'ground_truth_statement': s,
                'context': context,
            },
        )

        recall_verdicts = await self._execute_with_strategy(strategy, metadata)
        is_supported_list = self._parse_results(
            recall_verdicts,
            lambda result: getattr(result, 'is_supported', False),
            metadata,
        )

        return [
            JudgedGroundTruthStatement(
                statement_text=statement, is_supported_by_context=is_supported
            )
            for statement, is_supported in zip(gt_statements, is_supported_list)
        ]

    async def _judge_context_sufficiency(
        self, item: DatasetItem, metadata: AnalysisMetadata
    ) -> Optional[ContextSufficiencyVerdict]:
        """Evaluate whether the context is sufficient to answer the query."""
        if not item.query or not item.retrieved_content:
            return None

        context = '\n'.join(item.retrieved_content)

        metadata.llm_calls['granular_attempts'] += 1
        self._track_judge_usage(JudgeType.CONTEXT_SUFFICIENCY, metadata)

        try:
            result = await self.judges[JudgeType.CONTEXT_SUFFICIENCY].execute(
                query=item.query, context=context
            )
            metadata.llm_calls['granular_success'] += 1

            return ContextSufficiencyVerdict(
                is_sufficient=result.is_sufficient, reasoning=result.reasoning
            )
        except Exception as e:
            logger.error(f'Context sufficiency judgment failed: {e}')
            metadata.llm_calls['failed_judgments'] += 1
            return None

    async def _extract_claims(
        self, item: DatasetItem, metadata: AnalysisMetadata, from_expected: bool = False
    ) -> List[str]:
        """Extract atomic statements from text output."""
        source_text = item.expected_output if from_expected else item.actual_output
        if not source_text:
            return []

        metadata.llm_calls['granular_attempts'] += 1
        self._track_judge_usage(JudgeType.STATEMENT_EXTRACTOR, metadata)

        try:
            result = await self.judges[JudgeType.STATEMENT_EXTRACTOR].execute(
                actual_output=source_text
            )
            metadata.llm_calls['granular_success'] += 1
            return result.statements
        except Exception as e:
            logger.error(f'Statement extraction failed: {e}')
            metadata.llm_calls['failed_judgments'] += 1
            return []

    async def _judge_relevancy(
        self, query: str, statements: List[str], metadata: AnalysisMetadata
    ) -> List[Any]:
        """Batch evaluate relevancy of statements to a query."""
        if not statements:
            return []

        metadata.llm_calls['batch_attempts'] += 1
        self._track_judge_usage(JudgeType.RELEVANCY, metadata)

        try:
            result = await self.judges[JudgeType.RELEVANCY].execute(
                input_query=query, statements=statements
            )
            metadata.llm_calls['batch_success'] += 1
            return result.verdicts
        except Exception as e:
            logger.error(f'Batch relevancy judgment failed: {e}')
            metadata.llm_calls['failed_judgments'] += 1
            return [False] * len(statements)

    async def _judge_chunk_usefulness(
        self, chunks: List[str], expected_output: str, metadata: AnalysisMetadata
    ) -> List[bool]:
        """Evaluate whether chunks are useful for generating expected output."""
        if not chunks:
            return []

        results = await self._execute_granular(
            items=chunks,
            judge_type=JudgeType.USEFULNESS,
            payload_factory=lambda chunk: {
                'context_chunk': chunk,
                'expected_output': expected_output,
            },
            metadata=metadata,
        )

        return self._parse_results(
            results, lambda result: getattr(result, 'is_useful', False), metadata
        )

    async def _judge_chunk_utilization(
        self, chunks: List[str], actual_output: str, metadata: AnalysisMetadata
    ) -> List[bool]:
        """Evaluate whether chunks were actually utilized in the generated answer."""
        if not chunks:
            return []

        strategy = ExecutionStrategy(
            items=chunks,
            batch_judge=JudgeType.BATCH_UTILIZATION,
            granular_judge=JudgeType.UTILIZATION,
            batch_payload={'chunks': chunks, 'answer': actual_output},
            granular_payload_factory=lambda chunk: {
                'context_chunk': chunk,
                'answer': actual_output,
            },
        )

        results = await self._execute_with_strategy(strategy, metadata)
        return self._parse_results(
            results, lambda result: getattr(result, 'is_utilized', False), metadata
        )

    async def _execute_with_strategy(
        self, strategy: ExecutionStrategy, metadata: AnalysisMetadata
    ) -> List[Any]:
        """
        Execute evaluation using batch-or-granular strategy.

        Attempts batch processing first for efficiency, falling back to
        concurrent granular processing if batch fails or is disabled.
        """
        if not strategy.items:
            return []

        # Try batch processing first
        if not self.force_granular:
            batch_result = await self._try_batch_execution(strategy, metadata)
            if batch_result is not None:
                return batch_result

        # Fall back to granular processing
        return await self._execute_granular(
            items=strategy.items,
            judge_type=strategy.granular_judge,
            payload_factory=strategy.granular_payload_factory,
            metadata=metadata,
        )

    async def _try_batch_execution(
        self, strategy: ExecutionStrategy, metadata: AnalysisMetadata
    ) -> Optional[List[Any]]:
        """Attempt batch execution, returning None if it fails."""
        batch_judge = self.judges.get(strategy.batch_judge)
        if not batch_judge:
            return None

        metadata.llm_calls['batch_attempts'] += 1
        self._track_judge_usage(strategy.batch_judge, metadata)

        try:
            result = await batch_judge.execute(**strategy.batch_payload)
            metadata.llm_calls['batch_success'] += 1
            logger.info(
                f'Batch {strategy.batch_judge.name} succeeded for {len(strategy.items)} items'
            )
            return result.verdicts
        except Exception as e:
            logger.warning(
                f'Batch {strategy.batch_judge.name} failed: {e}. '
                f'{"Falling back to granular." if self.allow_fallback else "No fallback allowed."}'
            )
            if not self.allow_fallback:
                raise
            return None

    async def _execute_granular(
        self,
        items: List[Any],
        judge_type: JudgeType,
        payload_factory: Callable[[Any], Dict[str, Any]],
        metadata: AnalysisMetadata,
    ) -> List[Any]:
        """Execute multiple granular evaluations concurrently with semaphore control."""
        if not items:
            return []

        judge = self.judges[judge_type]
        logger.info(f'Executing granular {judge_type.name} for {len(items)} items')

        metadata.llm_calls['granular_attempts'] += len(items)
        self._track_judge_usage(judge_type, metadata, count=len(items))

        tasks = [
            self.executor.run(judge.execute, **payload_factory(item)) for item in items
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for r in results if not isinstance(r, Exception))
        metadata.llm_calls['granular_success'] += successful

        return results

    async def _cached_computation(
        self,
        item_id: str,
        cache_key: str,
        compute_func: Callable,
        cache: Optional[AnalysisCache],
        metadata: AnalysisMetadata,
    ) -> Any:
        """Execute computation with caching support."""
        if not cache:
            return await compute_func()

        async with cache.get_lock(item_id, cache_key):
            if cache.has(item_id, cache_key):
                metadata.cache_hits += 1
                return cache.get(item_id, cache_key)

            result = await compute_func()
            cache.set(item_id, cache_key, result)
            return result

    def _parse_results(
        self, results: List[Any], parser: Callable[[Any], T], metadata: AnalysisMetadata
    ) -> List[T]:
        """Parse granular results, handling exceptions gracefully."""
        parsed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f'Granular judgment {i} failed: {result}')
                metadata.llm_calls['failed_judgments'] += 1
                parsed.append(False)  # Default value on failure
            else:
                parsed.append(parser(result))
        return parsed

    @staticmethod
    def _should_analyze(prefix: str, required: Set[str]) -> bool:
        """Check if any required component starts with the given prefix."""
        return any(component.startswith(prefix) for component in required)
