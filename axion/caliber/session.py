"""
CalibrationSession for CaliberHQ workflow.

Central orchestrator that manages the 3-step workflow: Upload, Annotate, Evaluate.
Optional for notebooks (use components directly), required for web UI (state persistence).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from axion._core.uuid import uuid7
from axion.caliber.analysis import MisalignmentAnalysis, MisalignmentAnalyzer
from axion.caliber.annotation import AnnotationManager
from axion.caliber.evaluation import EvaluationRunner
from axion.caliber.example_selector import (
    ExampleSelector,
    SelectionResult,
    SelectionStrategy,
)
from axion.caliber.models import (
    Annotation,
    AnnotationState,
    CalibrationSessionData,
    EvaluationConfig,
    EvaluationResult,
    SessionState,
    UploadedRecord,
    UploadResult,
)
from axion.caliber.pattern_discovery import (
    AnnotatedItem,
    ClusteringMethod,
    PatternDiscovery,
    PatternDiscoveryResult,
)
from axion.caliber.prompt_optimizer import OptimizedPrompt, PromptOptimizer
from axion.caliber.upload import UploadHandler


class CalibrationSession:
    """
    Manages the CaliberHQ calibration workflow.

    Central orchestrator for the 3-step process:
    1. UPLOAD - Load data with LLM judge outputs
    2. REVIEW & LABEL - Human annotation with Accept/Reject + optional notes
    3. BUILD EVAL - Run evaluation, compute metrics, analyze misalignments

    **Optional for notebook** (use components directly like UploadHandler, AnnotationManager)
    **Required for web UI** (state persistence between requests)

    Example (Python API):
        >>> session = CalibrationSession()
        >>>
        >>> # Step 1: Upload
        >>> session.upload_csv("data.csv")
        >>> # or
        >>> session.upload_records([{"id": "1", "query": "...", ...}])
        >>>
        >>> # Step 2: Annotate
        >>> session.annotate("1", score=1, notes="Good response")
        >>> session.annotate("2", score=0, notes="Hallucinated")
        >>>
        >>> # Step 3: Evaluate
        >>> result = await session.evaluate(
        ...     criteria="Score 1 if accurate, 0 otherwise",
        ...     model_name="gpt-4o"
        ... )
        >>>
        >>> # Optional: Analysis
        >>> analysis = await session.analyze_misalignments()
        >>> optimized = await session.optimize_prompt()

    Example (Web API - each method returns JSON-serializable data):
        >>> session = CalibrationSession.from_dict(saved_state_dict)
        >>> session.upload_records(records_from_request)
        >>> return session.to_dict()  # Save state between requests
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize a new CalibrationSession.

        Args:
            session_id: Optional session identifier. Auto-generated if not provided.
        """
        self._session_id = session_id or str(uuid7())
        self._state = SessionState.UPLOAD
        self._records: List[UploadedRecord] = []
        self._upload_result: Optional[UploadResult] = None
        self._annotation_manager: Optional[AnnotationManager] = None
        self._evaluation_config: Optional[EvaluationConfig] = None
        self._evaluation_result: Optional[EvaluationResult] = None
        self._created_at = datetime.now(timezone.utc)
        self._updated_at = datetime.now(timezone.utc)

    @property
    def session_id(self) -> str:
        """Session identifier."""
        return self._session_id

    @property
    def state(self) -> SessionState:
        """Current workflow state."""
        return self._state

    @property
    def records(self) -> List[UploadedRecord]:
        """Uploaded records."""
        return self._records

    # =========================================================================
    # Step 1: Upload
    # =========================================================================

    def upload_csv(
        self,
        path: Union[str, Path],
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> UploadResult:
        """
        Upload data from a CSV file.

        Args:
            path: Path to the CSV file
            column_mapping: Optional column name mapping

        Returns:
            UploadResult with loaded records
        """
        handler = UploadHandler()
        result = handler.from_csv(path, column_mapping)
        return self._process_upload_result(result)

    def upload_dataframe(self, df: 'pd.DataFrame') -> UploadResult:
        """
        Upload data from a pandas DataFrame.

        Args:
            df: DataFrame with records

        Returns:
            UploadResult with loaded records
        """
        handler = UploadHandler()
        result = handler.from_dataframe(df)
        return self._process_upload_result(result)

    def upload_records(self, records: List[Dict[str, Any]]) -> UploadResult:
        """
        Upload data from a list of dictionaries.

        Args:
            records: List of record dictionaries

        Returns:
            UploadResult with loaded records
        """
        handler = UploadHandler()
        result = handler.from_records(records)
        return self._process_upload_result(result)

    def _process_upload_result(self, result: UploadResult) -> UploadResult:
        """Process upload result and transition state."""
        self._records = result.records
        self._upload_result = result
        self._annotation_manager = AnnotationManager(self._records)
        self._state = SessionState.ANNOTATE
        self._updated_at = datetime.now(timezone.utc)
        return result

    # =========================================================================
    # Step 2: Annotate
    # =========================================================================

    def annotate(
        self,
        record_id: str,
        score: int,
        notes: Optional[str] = None,
    ) -> Annotation:
        """
        Add or update annotation for a record.

        Args:
            record_id: ID of the record to annotate
            score: Human score (0 for reject, 1 for accept)
            notes: Optional annotation notes

        Returns:
            The created/updated Annotation

        Raises:
            RuntimeError: If not in annotate state
            ValueError: If record_id is not found
        """
        if self._state not in (SessionState.ANNOTATE, SessionState.EVALUATE):
            raise RuntimeError(
                f'Cannot annotate in state {self._state}. Upload data first.'
            )

        if self._annotation_manager is None:
            raise RuntimeError('No records uploaded. Call upload first.')

        annotation = self._annotation_manager.annotate(record_id, score, notes)
        self._updated_at = datetime.now(timezone.utc)
        return annotation

    def get_annotation_state(self) -> AnnotationState:
        """
        Get current annotation state.

        Returns:
            AnnotationState with progress and annotations
        """
        if self._annotation_manager is None:
            raise RuntimeError('No records uploaded. Call upload first.')
        return self._annotation_manager.get_state()

    def get_record_for_annotation(self, index: int) -> Optional[UploadedRecord]:
        """
        Get record at specified index for annotation.

        Args:
            index: Index of the record (0-based)

        Returns:
            UploadedRecord if index is valid, None otherwise
        """
        if self._annotation_manager is None:
            return None
        return self._annotation_manager.get_record(index)

    def get_next_unannotated(self) -> Optional[UploadedRecord]:
        """
        Get the next unannotated record.

        Returns:
            Next UploadedRecord to annotate, or None if all annotated
        """
        if self._annotation_manager is None:
            return None
        return self._annotation_manager.get_next_unannotated()

    def is_annotation_complete(self) -> bool:
        """Check if all records have been annotated."""
        if self._annotation_manager is None:
            return False
        return self._annotation_manager.is_complete()

    # =========================================================================
    # Step 3: Evaluate
    # =========================================================================

    async def evaluate(
        self,
        criteria: str,
        model_name: str = 'gpt-4o',
        llm_provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> EvaluationResult:
        """
        Run LLM-as-judge evaluation.

        Args:
            criteria: The LLM-as-judge evaluation criteria
            model_name: LLM model name (default: gpt-4o)
            llm_provider: LLM provider (optional)
            system_prompt: System prompt for evaluator (optional)
            on_progress: Progress callback (current, total)

        Returns:
            EvaluationResult with metrics and individual results
        """
        if self._annotation_manager is None or not self._records:
            raise RuntimeError('No records uploaded. Call upload first.')

        if not self._annotation_manager.completed_count:
            raise RuntimeError('No annotations found. Annotate records first.')

        config = EvaluationConfig(
            model_name=model_name,
            llm_provider=llm_provider,
            criteria=criteria,
            system_prompt=system_prompt,
        )
        self._evaluation_config = config

        runner = EvaluationRunner(config)
        annotations = {
            rid: ann for rid, ann in self._annotation_manager._annotations.items()
        }

        result = await runner.run(self._records, annotations, on_progress)
        self._evaluation_result = result
        self._state = SessionState.COMPLETE
        self._updated_at = datetime.now(timezone.utc)

        return result

    def evaluate_sync(
        self,
        criteria: str,
        model_name: str = 'gpt-4o',
        llm_provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> EvaluationResult:
        """
        Synchronous wrapper for evaluate().

        Args:
            criteria: The LLM-as-judge evaluation criteria
            model_name: LLM model name
            llm_provider: LLM provider
            system_prompt: System prompt
            on_progress: Progress callback

        Returns:
            EvaluationResult with metrics and individual results
        """
        from axion._core.asyncio import run_async_function

        async def _run():
            return await self.evaluate(
                criteria, model_name, llm_provider, system_prompt, on_progress
            )

        return run_async_function(_run)

    def get_evaluation_result(self) -> Optional[EvaluationResult]:
        """Get the evaluation result if available."""
        return self._evaluation_result

    # =========================================================================
    # Analysis (post-evaluation)
    # =========================================================================

    async def analyze_misalignments(
        self,
        max_examples: int = 10,
        model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ) -> MisalignmentAnalysis:
        """
        Analyze misalignment patterns.

        Args:
            max_examples: Max examples per category
            model_name: LLM model (defaults to evaluation model)
            llm_provider: LLM provider

        Returns:
            MisalignmentAnalysis with patterns and recommendations
        """
        if self._evaluation_result is None:
            raise RuntimeError('No evaluation result. Run evaluate() first.')

        if self._evaluation_config is None:
            raise RuntimeError('No evaluation config. Run evaluate() first.')

        analyzer = MisalignmentAnalyzer(
            model_name=model_name or self._evaluation_config.model_name,
            llm_provider=llm_provider or self._evaluation_config.llm_provider,
            max_examples=max_examples,
        )

        # Convert evaluation records to dict format
        results = [
            {
                'record_id': r.record_id,
                'human_score': r.human_score,
                'llm_score': r.llm_score,
                'llm_reasoning': r.llm_reasoning,
                'query': self._get_record_field(r.record_id, 'query'),
                'actual_output': self._get_record_field(r.record_id, 'actual_output'),
            }
            for r in self._evaluation_result.records
        ]

        return await analyzer.analyze(results, self._evaluation_config.criteria)

    async def optimize_prompt(
        self,
        max_examples: int = 10,
        model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ) -> OptimizedPrompt:
        """
        Optimize the evaluation criteria based on misalignments.

        Args:
            max_examples: Max examples per category
            model_name: LLM model (defaults to evaluation model)
            llm_provider: LLM provider

        Returns:
            OptimizedPrompt with improved criteria
        """
        if self._evaluation_result is None:
            raise RuntimeError('No evaluation result. Run evaluate() first.')

        if self._evaluation_config is None:
            raise RuntimeError('No evaluation config. Run evaluate() first.')

        optimizer = PromptOptimizer(
            model_name=model_name or self._evaluation_config.model_name,
            llm_provider=llm_provider or self._evaluation_config.llm_provider,
            max_examples=max_examples,
        )

        # Convert evaluation records to dict format
        results = [
            {
                'record_id': r.record_id,
                'human_score': r.human_score,
                'llm_score': r.llm_score,
                'llm_reasoning': r.llm_reasoning,
                'query': self._get_record_field(r.record_id, 'query'),
                'actual_output': self._get_record_field(r.record_id, 'actual_output'),
            }
            for r in self._evaluation_result.records
        ]

        return await optimizer.optimize(
            results,
            self._evaluation_config.criteria,
            self._evaluation_config.system_prompt or '',
        )

    async def discover_patterns(
        self,
        method: ClusteringMethod = ClusteringMethod.LLM,
        model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ) -> PatternDiscoveryResult:
        """
        Discover patterns in annotation notes.

        Args:
            method: Clustering method (LLM, BERTOPIC, or HYBRID)
            model_name: LLM model (defaults to evaluation model)
            llm_provider: LLM provider

        Returns:
            PatternDiscoveryResult with discovered patterns
        """
        if self._annotation_manager is None:
            raise RuntimeError('No annotations. Upload and annotate first.')

        # Build annotations dict for pattern discovery
        annotations = {}
        for rid, ann in self._annotation_manager._annotations.items():
            record = self._annotation_manager.get_record_by_id(rid)
            annotations[rid] = AnnotatedItem(
                record_id=rid,
                score=ann.score,
                notes=ann.notes,
                timestamp=ann.timestamp.isoformat() if ann.timestamp else None,
                query=record.query if record else None,
                actual_output=record.actual_output if record else None,
            )

        config_model = (
            self._evaluation_config.model_name if self._evaluation_config else None
        )
        config_provider = (
            self._evaluation_config.llm_provider if self._evaluation_config else None
        )

        discovery = PatternDiscovery(
            model_name=model_name or config_model,
            llm_provider=llm_provider or config_provider,
        )

        return await discovery.discover(annotations, method)

    def select_examples(
        self,
        count: int = 6,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
        eval_results: Optional[List[Dict[str, Any]]] = None,
        patterns: Optional[List] = None,
        seed: Optional[int] = None,
    ) -> SelectionResult:
        """
        Select few-shot examples for calibration.

        Args:
            count: Number of examples to select
            strategy: Selection strategy
            eval_results: Evaluation results (for MISALIGNMENT_GUIDED)
            patterns: Discovered patterns (for PATTERN_AWARE)
            seed: Random seed for reproducibility

        Returns:
            SelectionResult with selected examples
        """
        if self._annotation_manager is None:
            raise RuntimeError('No annotations. Upload and annotate first.')

        records = [r.model_dump() for r in self._records]
        annotations = self._annotation_manager.get_annotations_dict()

        selector = ExampleSelector(seed=seed)
        return selector.select(
            records,
            annotations,
            count=count,
            strategy=strategy,
            eval_results=eval_results,
            patterns=patterns,
        )

    def _get_record_field(self, record_id: str, field: str) -> Optional[str]:
        """Get a field value from a record by ID."""
        for record in self._records:
            if record.id == record_id:
                return getattr(record, field, None)
        return None

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize session state to a dictionary.

        Returns:
            Dict representation for persistence
        """
        data = {
            'session_id': self._session_id,
            'state': self._state.value,
            'created_at': self._created_at.isoformat(),
            'updated_at': self._updated_at.isoformat(),
        }

        if self._upload_result:
            data['upload_result'] = self._upload_result.model_dump()

        if self._annotation_manager:
            data['annotation_state'] = self._annotation_manager.to_dict()

        if self._evaluation_config:
            data['evaluation_config'] = self._evaluation_config.model_dump()

        if self._evaluation_result:
            data['evaluation_result'] = self._evaluation_result.model_dump()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationSession':
        """
        Restore session from a dictionary.

        Args:
            data: Serialized state from to_dict()

        Returns:
            Restored CalibrationSession instance
        """
        session = cls(session_id=data.get('session_id'))
        session._state = SessionState(data.get('state', 'upload'))
        session._created_at = datetime.fromisoformat(
            data.get('created_at', datetime.now(timezone.utc).isoformat())
        )
        session._updated_at = datetime.fromisoformat(
            data.get('updated_at', datetime.now(timezone.utc).isoformat())
        )

        # Restore upload result
        if 'upload_result' in data:
            session._upload_result = UploadResult(**data['upload_result'])
            session._records = session._upload_result.records

        # Restore annotation manager
        if 'annotation_state' in data and session._records:
            session._annotation_manager = AnnotationManager.from_dict(
                data['annotation_state'], session._records
            )

        # Restore evaluation config
        if 'evaluation_config' in data:
            session._evaluation_config = EvaluationConfig(**data['evaluation_config'])

        # Restore evaluation result
        if 'evaluation_result' in data:
            session._evaluation_result = EvaluationResult(**data['evaluation_result'])

        return session

    @classmethod
    def from_state(cls, state: CalibrationSessionData) -> 'CalibrationSession':
        """
        Restore session from a CalibrationSessionData model.

        Args:
            state: CalibrationSessionData model

        Returns:
            Restored CalibrationSession instance
        """
        return cls.from_dict(state.model_dump())

    def to_state(self) -> CalibrationSessionData:
        """
        Convert to CalibrationSessionData model.

        Returns:
            CalibrationSessionData for persistence
        """
        annotation_state = None
        if self._annotation_manager:
            state = self._annotation_manager.get_state()
            annotation_state = state

        return CalibrationSessionData(
            session_id=self._session_id,
            state=self._state,
            upload_result=self._upload_result,
            annotation_state=annotation_state,
            evaluation_result=self._evaluation_result,
            created_at=self._created_at,
            updated_at=self._updated_at,
        )
