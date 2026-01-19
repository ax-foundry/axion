from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import Field

from axion._core.schema import (
    AIMessage,
    BaseMessage,
    EmbeddingRunnable,
    HumanMessage,
    LLMRunnable,
    PromptValue,
    RichBaseModel,
    RichEnum,
    RichSerializer,
    ToolCall,
    ToolMessage,
)
from axion.dataset import DatasetItem

__all__ = [
    # Core base classes
    'RichBaseModel',
    'RichSerializer',
    'RichEnum',
    # Message types
    'BaseMessage',
    'HumanMessage',
    'AIMessage',
    'ToolMessage',
    'ToolCall',
    # Protocols
    'LLMRunnable',
    'EmbeddingRunnable',
    # Utilities
    'PromptValue',
    # Evaluation models
    'MetricScore',
    'ComponentResult',
    'TestResult',
    'EvaluationResult',
    'ErrorConfig',
]


def strftime() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


class MetricScore(RichBaseModel):
    """
    Standardized data model for a single metric evaluation result.
    Captures the computed score, the logic behind it, thresholds used,
    and any metadata useful for debugging or reporting.
    """

    id: Optional[str] = Field(
        default=None,
        description='Unique identifier for this metric evaluation run or record.',
    )

    name: str = Field(
        description="The name of the metric (e.g., 'faithfulness', 'precision', 'coherence')."
    )

    score: Optional[float] = Field(
        default=np.nan,
        description='Final computed score for the metric. Can be NaN if the score is undefined or not computable.',
    )

    threshold: Optional[float] = Field(
        default=None,
        description='Optional threshold used for pass/fail or alerting logic.',
    )

    explanation: Optional[str] = Field(
        default=None,
        description='Optional explanation or justification of the score. May include notes on input/output behavior, LLM reasoning, or failure mode.',
    )

    signals: Optional[Any] = Field(
        default=None,
        description='A dictionary of dynamically generated signals providing a granular breakdown of the metric.',
    )

    passed: Optional[bool] = Field(
        default=None,
        description='Boolean flag indicating if the score passed the threshold (if a threshold is defined).',
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default={},
        description='Optional structured metadata (e.g., token usage, evaluator version, raw model output).',
    )

    version: Optional[str] = Field(
        default=None,
        description="Version identifier of the metric implementation or schema (e.g., '1.0.0').",
    )

    timestamp: Optional[str] = Field(
        default_factory=strftime, description='Timestamp when the metric was evaluated.'
    )

    source: Optional[str] = Field(default=None, description='Metric Source.')

    cost_estimate: Optional[float] = Field(
        default=None, description='Cost estimate of executing the metric.'
    )

    # Fields for Hierarchical Structure
    parent: Optional[str] = Field(
        default=None, description='Name of the parent node in a hierarchy.'
    )

    type: Optional[str] = Field(
        default='metric', description="Type of the node, e.g., 'component' or 'metric'."
    )

    weight: Optional[float] = Field(
        default=None,
        description='Weight of this node relative to its siblings under the same parent.',
    )


class ComponentResult(RichBaseModel):
    """
    Represents the aggregated result of a ComponentNode execution.

    This model captures the output from a hierarchical component in the
    evaluation tree. It provides a structured, type-safe alternative to
    raw dictionaries, enabling consistency, validation, and introspection.

    A `ComponentResult` may contain results from child components
    (nested ComponentResults) as well as leaf metrics (MetricScore).
    """

    score: float = Field(
        ...,
        description=(
            'The aggregated score for this component, computed from child '
            'scores according to the specified aggregation strategy.'
        ),
    )

    children: Dict[str, Union['ComponentResult', MetricScore]] = Field(
        default_factory=dict,
        description=(
            'Mapping of child node names to their results. '
            'Each child may itself be a ComponentResult (nested structure) '
            'or a MetricScore (leaf evaluation).'
        ),
    )

    weights: List[float] = Field(
        ...,
        description=(
            'Normalized weights applied to each child during aggregation. '
            'Must align with the order of children defined at execution time.'
        ),
    )

    cost_estimate: float = Field(
        0.0,
        description=(
            'Estimated computational cost accumulated during execution '
            '(e.g., API usage cost or compute time).'
        ),
    )

    component_name: str = Field(
        ...,
        description=(
            'The name of the component node. Typically corresponds to '
            'the logical unit or behavior being evaluated.'
        ),
    )

    aggregation_strategy: str = Field(
        ...,
        description=(
            'The name of the aggregation strategy used to combine child scores '
            "(e.g., 'WeightedAverage', 'MaxScore', etc.)."
        ),
    )


# Pydantic forward reference resolution for recursive type
ComponentResult.model_rebuild()


@dataclass
class TestResult:
    """
    Represents the result of evaluating a single test case using one or more evaluation metrics.

    Attributes:
        test_case (DatasetItem):
            The input test case containing query, expected output, and other context.
            This forms the basis for which all metrics are applied.

        score_results (List[MetricScore]):
            A list of evaluation results returned from applying different metrics to this test case.
            Each MetricScore includes a score, explanation, and threshold comparison.

        metadata (Optional[Dict[str, Any]]):
            Optional metadata for storing extra context such as timestamps, evaluator info,
            experiment variant, evaluation notes, or model config parameters.
    """

    # Prevent pytest from collecting this as a test class
    __test__ = False

    test_case: Optional[DatasetItem]
    score_results: List[MetricScore]
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """
    Captures the full result of an evaluation run across multiple test cases and metrics.

    Attributes:
        run_id (str):
            A unique identifier for this specific evaluation run. Typically generated per execution.

        evaluation_name (Optional[str]):
            Optional name for the experiment or test campaign (e.g., "Lead Scoring v2 A/B").

        timestamp (str):
            ISO-formatted timestamp indicating when the evaluation was run.
            Can be used for sorting or audit logging.

        results (List[TestResult]):
            A list of TestResult objects, each representing the evaluation output
            for a single test case across one or more metrics.

        summary (Dict[str, Any]):
            Summary of the TestResult objects, representing the evaluation output
            across each metric.

        metadata (Dict[str, Any]):
            Arbitrary metadata such as configuration info, evaluator identity, model version,
            dataset name, or custom flags for internal use.
    """

    run_id: str
    evaluation_name: Optional[str]
    timestamp: str
    results: List[TestResult]
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(
        self,
        by_alias: bool = True,
        id_as_index: bool = False,
        include_test_case: bool = True,
        include_run_metadata: bool = True,
        column_order: Optional[List[str]] = None,
        rename_columns: bool = True,
    ) -> pd.DataFrame:
        """
        Flattens the entire evaluation result into a single pandas DataFrame.

        Args:
            by_alias (bool): Whether to use field aliases in the output.
            id_as_index (bool): If True, sets the test_case `id` as the DataFrame index.
            include_test_case (bool): Whether to include test_case fields in the output.
            include_run_metadata (bool): Whether to include run-level metadata.
            column_order (list): Output column ordering.
            rename_columns (bool): Rename columns to match model_arena format.

        Returns:
            pd.DataFrame: Flattened view of all metrics with test case and run context.
        """

        default_column_order = [
            'id',
            'name',
            'parent',
            'score',
            'type',
            'weight',
            'passed',
            'cost_estimate',
            'explanation',
            'source',
        ]
        column_order = column_order or default_column_order

        run_metadata = {}
        if include_run_metadata:
            run_metadata = {
                'run_id': self.run_id,
                'evaluation_name': self.evaluation_name,
                'timestamp': self.timestamp,
            }

        def _extract_test_case_data(test_case: Any) -> Dict[str, Any]:
            """Robustly extract test_case data from multiple possible formats."""
            if not include_test_case or not test_case:
                return {}

            extractors = [
                lambda tc: tc.model_dump(by_alias=by_alias),
                lambda tc: tc.dict(by_alias=by_alias),
                dict,
                lambda tc: getattr(tc, '__dict__', {}),
                vars,
            ]
            for extractor in extractors:
                try:
                    data = extractor(test_case)
                    if data:
                        break
                except Exception:
                    continue
            else:
                data = {}

            # Ensure id is captured if available
            if 'id' not in data and hasattr(test_case, 'id'):
                try:
                    data['id'] = test_case.id
                except Exception:
                    pass
            return data

        all_rows = []
        for result in self.results:
            test_case_data = _extract_test_case_data(result.test_case)

            for score_row in result.score_results:
                score_data = score_row.model_dump(by_alias=by_alias)

                # Preserve test_case id if score_data overwrote it with None
                test_case_id = test_case_data.get('id')
                row_data = {**test_case_data, **score_data, **run_metadata}
                if test_case_id is not None and row_data.get('id') is None:
                    row_data['id'] = test_case_id

                all_rows.append(row_data)

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        # Ensure preferred column order
        existing_cols = df.columns.tolist()
        final_cols = [col for col in column_order if col in existing_cols]
        final_cols += [col for col in existing_cols if col not in final_cols]
        df = df[final_cols]

        # Rename for model_arena convention
        if rename_columns:
            df = df.rename(
                columns={
                    'name': 'metric_name',
                    'score': 'metric_score',
                    'type': 'metric_type',
                }
            )
            df['evaluation_metadata'] = [self.metadata] * len(df)

        if id_as_index and 'id' in df.columns:
            df = df.set_index('id')

        return df

    def to_latency_plot(
        self,
        col_name: str = 'latency',
        id_col: str = 'id',
        bins: int = 30,
        show_legend: bool = True,
        show_stats_panel: bool = True,
        figsize: Tuple[int, int] = (16, 9),
        return_plot: bool = False,
        show_plot: bool = True,
        output_path: Optional[str] = None,
        plot_title: str = 'Latency Distribution',
        color_palette: Optional[Dict[str, str]] = None,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Optional[Any], Optional[Any]]]:
        """
        Analyzes and visualizes latency distribution.

        Args:
            col_name: Name of the column containing latency values.
            id_col: Unique identifier for test cases (used to deduplicate latency).
            bins: Number of histogram bins.
            show_legend: If True, show plot legend,
            show_stats_panel:  If True, show stats panel,
            figsize: Size of the matplotlib figure.
            return_plot: If True, returns (stats_df, fig, ax). If False, returns stats_df.
            show_plot: Whether to render the plot using plt.show() (or display in NB).
            output_path: If provided, saves the plot to this file path.
            plot_title: Descriptive name for the latency plot title (default: "Latency Distribution")
            color_palette: Custom colors for the LatencyAnalyzer.

        Returns:
            pd.DataFrame or Tuple[pd.DataFrame, Figure, Axes]
        """
        import matplotlib.pyplot as plt

        from axion.reporting.latency import LatencyAnalyzer

        df = self.to_dataframe()

        if col_name not in df.columns:
            available = ', '.join(df.columns[:5]) + '...'
            raise ValueError(
                f"Column '{col_name}' not found in dataframe. Available: {available}"
            )

        # Deduplicate: Latency is typically 1 per test case, but the DF has 1 row per metric.
        # We group by ID and take the first non-null latency.
        if id_col in df.columns:
            latency_series = df.groupby(id_col)[col_name].first()
        else:
            latency_series = df[col_name]

        # Clean NaNs
        latency_series = latency_series.dropna()

        if len(latency_series) == 0:
            raise ValueError(
                f"No valid data found in column '{col_name}' after deduplication."
            )

        analyzer = LatencyAnalyzer(
            latency_series,
            name='Latency',
            plot_title=plot_title,
            color_palette=color_palette,
        )

        # Calculate Stats
        stats_df = analyzer.statistics_table()

        # Generate Plot
        fig, ax = analyzer.plot_distribution(
            bins=bins,
            figsize=figsize,
            show_legend=show_legend,
            show_stats_panel=show_legend,
        )

        if output_path:
            fig.savefig(output_path, bbox_inches='tight', dpi=300)

        if show_plot:
            try:
                plt.show()
            except Exception:
                pass

        if return_plot:
            return stats_df, fig, ax

        # Close figure if not returning/showing to save memory
        if not show_plot and not return_plot:
            plt.close(fig)

        return stats_df

    def to_scorecard(
        self,
        llm: Optional[LLMRunnable] = None,
        metric_definitions: dict = None,
        explanation_callback: callable = None,
        instruction: Optional[str] = None,
        max_concurrent: int = 10,
        output_path: Optional[str] = None,
        display_in_notebook: bool = False,
        return_html: bool = False,
        return_styled_df: bool = True,
        id_col: str = 'metric_name',
        parent_col: str = 'parent',
        value_cols: List[str] = None,
        group_meta_cols: List[str] = None,
    ) -> Union[str, pd.DataFrame, None]:
        """
        Generates a hierarchical scorecard report using the evaluation results.

        This method creates a visual performance breakdown. It can display the report interactively
        in a notebook, save it as an HTML file, or return the styled object/HTML string for custom use.

        Args:
            llm (Optional[LLMRunnable]): Custom LLM instance to use for generating qualitative explanations.
            metric_definitions (dict): Dictionary mapping metric names to static descriptions or templates.
            explanation_callback (callable): Custom function `f(name, score, type)` to generate explanations manually.
            instruction (Optional[str]): System prompt override for the explanation generation LLM.
            max_concurrent (int): Maximum number of parallel LLM calls for batch processing explanations.
            output_path (Optional[str]): File path to save the generated HTML report.
            display_in_notebook (bool): If True, renders the styled dataframe (or HTML) directly in Jupyter/IPython.
            return_html (bool): If True, returns the raw HTML string of the report.
            return_styled_df (bool): If True, returns the pandas Styler object for further customization.
            id_col (str): Column name representing the unique node identifier (default: 'metric_name').
            parent_col (str): Column name representing the parent node identifier (default: 'parent').
            value_cols (List[str]): List of columns to aggregate values for (e.g., ['metric_score', 'weight']).
            group_meta_cols (List[str]): List of metadata columns to include in grouping (e.g., ['metric_type']).

        Returns:
            Union[str, Styler, None]:
                - The HTML string if `return_html=True`.
                - The pandas Styler object if `return_styled_df=True`.
                - None otherwise (default behavior is just to display or save).
        """
        from axion.reporting.scorecard import ScoreCard

        dataframe = self.to_dataframe()

        # Sanity check for required hierarchy columns
        if id_col not in dataframe.columns or parent_col not in dataframe.columns:
            # Fallback check for standard column names
            if 'metric_name' in dataframe.columns and 'parent' in dataframe.columns:
                id_col = 'metric_name'
                parent_col = 'parent'
            else:
                raise ValueError(
                    f"Scorecard requires '{id_col}' and '{parent_col}' columns in the dataframe."
                )

        score_card = ScoreCard(
            dataframe,
            id_col=id_col,
            parent_col=parent_col,
            value_cols=value_cols,
            group_meta_cols=group_meta_cols,
            llm=llm,
            instruction=instruction,
            explanation_callback=explanation_callback,
            metric_definitions=metric_definitions,
            max_concurrent=max_concurrent,
        )

        score_card.build_scorecard()

        # Handle File Export / HTML Return
        html_content = None
        if output_path or return_html:
            html_content = score_card.generate_html_report(output_path=output_path)

        # Handle Notebook Display / Styler Return
        styled_df = None
        if display_in_notebook or return_styled_df:
            # Generate the styled dataframe object
            styled_df = score_card.generate_styled_dataframe()

            if display_in_notebook:
                try:
                    from IPython.display import display

                    display(styled_df)
                except ImportError:
                    pass

        if return_styled_df:
            return styled_df

        if return_html:
            return html_content

        return None

    def publish_to_observability(
        self,
        loader: Optional[Any] = None,
        observation_id_field: Optional[str] = 'observation_id',
        flush: bool = True,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Publish evaluation scores to an observability platform.

        Uses a trace loader to publish scores. By default, uses LangfuseTraceLoader.

        Args:
            loader: A trace loader instance (e.g., LangfuseTraceLoader, OpikTraceLoader).
                If None, creates a new LangfuseTraceLoader using environment variables.
            observation_id_field: Field name on DatasetItem containing the
                observation/span ID. If provided, scores attach to that specific
                observation within the trace. If None, scores attach to the trace itself.
                Default: 'observation_id'.
            flush: Whether to flush the client after uploading. Defaults to True.
            tags: Optional list of tags to attach to all scores as metadata.
                Falls back to LANGFUSE_TAGS env var if not provided.

        Note:
            Environment cannot be set when pushing scores to existing traces.
            To set environment, configure it at client initialization when creating
            traces (via LANGFUSE_ENVIRONMENT or LANGFUSE_TRACING_ENVIRONMENT env vars
            or the environment parameter in LangfuseTracer).

        Returns:
            Dict with counts: {'uploaded': N, 'skipped': M}
                - uploaded: Number of scores successfully pushed
                - skipped: Number of scores skipped (missing trace_id or invalid score)

        Example:
            >>> from axion._core.tracing.loaders import LangfuseTraceLoader
            >>>
            >>> # Using default Langfuse loader
            >>> stats = result.publish_to_observability()
            >>>
            >>> # Using explicit loader with tags
            >>> loader = LangfuseTraceLoader()
            >>> stats = result.publish_to_observability(
            ...     loader=loader,
            ...     tags=['prod', 'v1.0']
            ... )
            >>>
            >>> # Attach scores to traces only (no observation)
            >>> stats = result.publish_to_observability(observation_id_field=None)
        """
        if loader is None:
            from axion._core.tracing.loaders import LangfuseTraceLoader

            loader = LangfuseTraceLoader()

        return loader.push_scores(
            evaluation_result=self,
            observation_id_field=observation_id_field,
            flush=flush,
            tags=tags,
        )

    def publish_as_experiment(
        self,
        loader: Optional[Any] = None,
        dataset_name: Optional[str] = None,
        run_name: Optional[str] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
        flush: bool = True,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Publish evaluation results to Langfuse as a dataset experiment.

        This method creates a complete experiment in Langfuse with a dataset,
        dataset items, experiment runs, and scores. Unlike `publish_to_observability()`,
        it does not require existing traces - it creates everything from scratch.

        Args:
            loader: A LangfuseTraceLoader instance. If None, creates a new one
                using environment variables.
            dataset_name: Name for the Langfuse dataset. Defaults to
                evaluation_name or generates one based on run_id.
            run_name: Name for the experiment run. Defaults to
                "{dataset_name}-{run_id}" pattern.
            run_metadata: Optional metadata to attach to the experiment run.
            flush: Whether to flush the client after uploading. Defaults to True.
            tags: Optional list of tags to attach to all scores as metadata.

        Returns:
            Dict with statistics:
                - dataset_name: Name of the created/used dataset
                - run_name: Name of the experiment run
                - items_created: Number of dataset items created
                - runs_created: Number of experiment runs created
                - scores_uploaded: Number of scores attached
                - scores_skipped: Number of scores skipped (None/NaN values)
                - errors: List of error messages encountered

        Example:
            >>> from axion import evaluation_runner
            >>> from axion.metrics import Faithfulness, AnswerRelevancy
            >>>
            >>> # Run evaluation
            >>> results = evaluation_runner(
            ...     evaluation_inputs=dataset,
            ...     scoring_config=config,
            ...     evaluation_name="RAG Evaluation",
            ... )
            >>>
            >>> # Upload to Langfuse as experiment
            >>> stats = results.publish_as_experiment(
            ...     dataset_name="my-rag-dataset",
            ...     run_name="experiment-v1",
            ...     tags=["production"]
            ... )
            >>>
            >>> print(f"Uploaded {stats['scores_uploaded']} scores to {stats['dataset_name']}")
        """
        if loader is None:
            from axion._core.tracing.loaders import LangfuseTraceLoader

            loader = LangfuseTraceLoader()

        return loader.upload_experiment(
            evaluation_result=self,
            dataset_name=dataset_name,
            run_name=run_name,
            run_metadata=run_metadata,
            flush=flush,
            tags=tags,
        )


@dataclass
class ErrorConfig:
    """
    Configuration class for controlling error handling during metric execution.

    Attributes:
        ignore_errors (bool):
            If True, any exceptions raised during metric execution will be caught and suppressed.
            The metric will return a placeholder result (e.g., None or NaN) instead of failing the entire evaluation.
            Use this to allow evaluations to proceed even if some metrics occasionally fail.

        skip_on_missing_params (bool):
            If True, metrics will be skipped entirely when required input fields are missing from the data.
            This is useful when running multiple metrics over heterogeneous data where not all fields are always present.
            If False, the metric will raise an error if required inputs are missing.
    """

    ignore_errors: bool = True
    skip_on_missing_params: bool = False
