import uuid
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from axion._core.config import Config
from axion._core.error import InvalidConfig
from axion._core.logging import get_logger
from axion._core.types import TraceGranularity
from axion._core.utils import Timer
from axion.dataset import Dataset, DatasetItem, format_input
from axion.eval_tree._tree import TreeMixin
from axion.eval_tree.aggregation import (
    AGGREGATION_STRATEGIES,
    WeightedAverage,
)
from axion.eval_tree.component import ComponentNode
from axion.eval_tree.metric import MetricNode
from axion.eval_tree.node import Node
from axion.metrics import metric_registry
from axion.runners import MetricRunner
from axion.runners.summary import BaseSummary, HierarchicalSummary
from axion.runners.utils import input_to_dataset
from axion.schema import (
    ComponentResult,
    EvaluationResult,
    MetricScore,
    TestResult,
)
from axion.utils import lazy_import

logger = get_logger(__name__)


class EvalTree(TreeMixin):
    """
    Hierarchical scoring model that builds a tree from a configuration and
    executes metrics using an optimized two-phase batch process.
    """

    _name = 'model'
    _STRATEGY_KEY = '_strategy'
    _EXCLUDE_FAILED_KEY = '_exclude_failed'
    _CLASS_KEY = 'class'

    description: str = 'Dynamic hierarchical scoring system that builds a tree from config and executes metrics.'
    dataset_name: str = 'eval_tree_dataset'

    def __init__(
        self,
        config: Union[Config, Dict[str, Any], str],
        max_concurrent: int = 5,
        summary_generator: Optional[BaseSummary] = None,
        enable_internal_caching: bool = True,
        trace_granularity: TraceGranularity = TraceGranularity.SEPARATE,
    ):
        self.config: Config = config if isinstance(config, Config) else Config(config)
        self.max_concurrent = max_concurrent
        self.enable_internal_caching = enable_internal_caching
        self.trace_granularity = trace_granularity
        self.nodes: Dict[str, Node] = {}
        self.root_node: Optional[ComponentNode] = None
        self.summary_generator = summary_generator or HierarchicalSummary()
        self._elapsed_time: Optional[float] = None
        self._summary: Optional[Dict] = None

        self._build_tree()
        self._set_weights()

    def _build_tree(self) -> None:
        model_cfg = self.config.get(self._name, {})
        if not model_cfg:
            raise InvalidConfig(
                f"""
                No model configuration found.
                Must have a key named '{self._name}' storing model hierarchy and metric weights.
                Example:
                {{
                    "model": {{
                        "COMPONENT1": {{
                            "metric1": 0.5,
                            "metric2": 0.5
                        }}
                    }}
                }}
                """
            )

        self.root_node = ComponentNode(self._name)
        self.nodes[self._name] = self.root_node
        self._build_node_children(self.root_node, model_cfg)

    def _build_node_children(
        self, parent: ComponentNode, section: Dict[str, Any]
    ) -> None:
        for child_name, child_cfg in section.items():
            if child_name in [self._STRATEGY_KEY, self._EXCLUDE_FAILED_KEY]:
                continue

            if isinstance(child_cfg, dict):
                strategy_name = child_cfg.get(self._STRATEGY_KEY, 'weighted_average')
                strategy_cls = AGGREGATION_STRATEGIES.get(
                    strategy_name, WeightedAverage
                )
                exclude_failed = child_cfg.get(self._EXCLUDE_FAILED_KEY, True)
                component = ComponentNode(
                    child_name,
                    parent=parent,
                    aggregation_strategy=strategy_cls(),
                    exclude_failed_metrics=exclude_failed,
                )
                self.nodes[child_name] = component
                self._build_node_children(component, child_cfg)
            else:
                metric_weight = child_cfg
                metric_config = self.config.get(f'metric.{child_name}')
                if not metric_config:
                    logger.warning(f"Metric config '{child_name}' not found. Skipping.")
                    continue

                # Handle both dict-based config and pre-instantiated metric objects
                metric_instance: Any
                if isinstance(metric_config, dict):
                    metric_instance = self._create_metric_instance(
                        child_name, metric_config
                    )
                else:
                    metric_instance = metric_config

                metric_node = MetricNode(
                    child_name, metric_instance, parent=parent, weight=metric_weight
                )
                self.nodes[child_name] = metric_node

    def _create_metric_instance(self, name: str, cfg: Dict[str, Any]) -> Any:
        class_path = cfg.get(self._CLASS_KEY)
        kwargs = {k: v for k, v in cfg.items() if k != self._CLASS_KEY}
        if class_path:
            try:
                metric_class = lazy_import(class_path)
                return metric_class(**kwargs)
            except (ImportError, TypeError) as e:
                raise InvalidConfig(
                    f"Failed to create metric '{name}' from class '{class_path}': {e}"
                )
        metric_class = metric_registry.get(name)
        if not metric_class:
            raise InvalidConfig(f"Metric '{name}' not found in registry.")
        return metric_class(**kwargs)

    def _set_weights(self) -> None:
        """Assigns weights from the 'weights' config section to component nodes."""
        weights_cfg = self.config.get('weights', {})
        for node_name, weight in weights_cfg.items():
            if node_name in self.nodes and isinstance(
                self.nodes[node_name], ComponentNode
            ):
                self.nodes[node_name].weight = float(weight)

    async def execute(
        self, input_data: Union[DatasetItem, Dict[str, Any]]
    ) -> TestResult:
        """Executes the evaluation for a single data item."""
        input_item = format_input(input_data)
        result_wrapper = await self.batch_execute([input_item], show_progress=False)
        return result_wrapper.results[0]

    async def batch_execute(
        self,
        evaluation_inputs: Union[List[DatasetItem], Dataset, pd.DataFrame],
        evaluation_name: str = 'EvalTree Evaluation',
        show_progress: bool = True,
    ) -> EvaluationResult:
        """
        Executes the evaluation using an optimized two-phase process:
        1. Batch Calculation: Runs all leaf metrics across the dataset.
        2. In-Memory Aggregation: Computes hierarchical scores from the results.
        """
        dataset = input_to_dataset(evaluation_inputs, self.dataset_name)

        with Timer() as timer:
            # Batch Calculation of all leaf metrics
            leaf_nodes = [
                node for node in self.nodes.values() if isinstance(node, MetricNode)
            ]
            master_runner = MetricRunner(
                metrics=leaf_nodes,
                max_concurrent=self.max_concurrent,
                enable_internal_caching=self.enable_internal_caching,
                summary_generator=None,  # Handles this later
                trace_granularity=self.trace_granularity,
            )
            metric_results: List[TestResult] = await master_runner.execute_batch(
                dataset, show_progress=show_progress
            )

            # Create a fast lookup cache for the aggregation phase
            results_cache: Dict[tuple[str, str], MetricScore] = {
                (str(res.test_case.id), score.name): score
                for res in metric_results
                for score in res.score_results
            }

            # In-Memory Aggregation for each data item
            final_test_results: List[TestResult] = []
            for item in dataset:
                item_id = str(item.id)
                full_breakdown = self._aggregate_scores_for_item(
                    self.root_node, item_id, results_cache
                )
                flat_scores = self._to_flat_list(full_breakdown, item)
                final_test_results.append(
                    TestResult(test_case=item, score_results=flat_scores)
                )

        self._elapsed_time = timer.elapsed_time
        if self.summary_generator and self._elapsed_time is not None:
            self._summary = self.summary_generator.execute(
                final_test_results, self._elapsed_time
            )

        return EvaluationResult(
            run_id=f'evaltree_run_{uuid.uuid4().hex}',
            evaluation_name=evaluation_name,
            timestamp=Timer.current_timestamp(),
            results=final_test_results,
            summary=self._summary,
        )

    def _aggregate_scores_for_item(
        self,
        node: Node,
        item_id: str,
        cache: Dict[tuple[str, str], MetricScore],
    ) -> Union[MetricScore, ComponentResult]:
        """
        Recursively calculates scores for a single item using pre-computed results.
        This is a synchronous, in-memory operation.
        """
        if isinstance(node, MetricNode):
            # Retrieve pre-computed score for a metric node
            score = cache.get((item_id, node.name))
            if not score:
                return MetricScore(
                    id=item_id,
                    name=node.name,
                    score=np.nan,
                    explanation='Metric not run',
                )
            # Apply bias
            raw_score = score.score
            score.score = np.nan if raw_score is None else float(raw_score) + node.bias
            return score

        if isinstance(node, ComponentNode):
            # Aggregate scores from child nodes
            child_results = [
                self._aggregate_scores_for_item(child, item_id, cache)
                for child in node.children
            ]
            child_scores = np.array([res.score for res in child_results], dtype=float)
            initial_weights = np.array([c.weight for c in node.children], dtype=float)

            valid_mask = ~np.isnan(child_scores)

            # Determine the weights of only the valid children for normalization
            valid_weights = initial_weights[valid_mask]
            total_valid_weight = np.sum(valid_weights)

            # Create the final, normalized weight array for the output
            final_weights = np.zeros_like(initial_weights)
            if total_valid_weight > 0:
                final_weights[valid_mask] = valid_weights / total_valid_weight
            elif np.any(valid_mask):  # Handle case where all valid weights are 0
                num_valid = np.sum(valid_mask)
                final_weights[valid_mask] = 1.0 / num_valid

            # Determine scores and weights to pass to the aggregation strategy
            scores_for_agg, weights_for_agg = child_scores, initial_weights
            if node.exclude_failed_metrics:
                scores_for_agg = child_scores[valid_mask]
                weights_for_agg = initial_weights[valid_mask]
                if scores_for_agg.size == 0:
                    return ComponentResult(
                        score=np.nan,
                        children={},
                        weights=[],
                        component_name=node.name,
                        aggregation_strategy=node.aggregation_strategy_name,
                    )

            # The aggregation strategy handles its own normalization internally
            combined_score = node.aggregation_strategy.aggregate(
                scores_for_agg, weights_for_agg
            )

            return ComponentResult(
                score=combined_score,
                children={
                    child.name: res for child, res in zip(node.children, child_results)
                },
                weights=final_weights.tolist(),  # Pass the final normalized weights
                component_name=node.name,
                aggregation_strategy=node.aggregation_strategy_name,
            )
        raise TypeError(f'Unknown node type: {type(node)}')

    def _to_flat_list(
        self, breakdown: Union[ComponentResult, MetricScore], data_item: DatasetItem
    ) -> List[MetricScore]:
        """Converts the nested result from aggregation into a flat list."""
        flat_list = []
        item_id = str(data_item.id)

        def _traverse(
            node_result: Union[ComponentResult, MetricScore],
            parent: Optional[Node],
            weight: Optional[float] = None,
        ):
            if isinstance(node_result, MetricScore):
                node = self.nodes.get(node_result.name)
                if node:
                    node_result.parent = parent.name if parent else None
                    # Use the passed-in normalized weight
                    node_result.weight = weight if weight is not None else node.weight
                    node_result.type = 'metric'
                flat_list.append(node_result)
                return

            if isinstance(node_result, ComponentResult):
                node = self.nodes.get(node_result.component_name)
                component_score = MetricScore(
                    id=item_id,
                    name=node.name,
                    score=node_result.score,
                    type='component',
                    parent=parent.name if parent else None,
                    weight=weight
                    if weight is not None
                    else (node.weight if node else 1.0),
                )
                flat_list.append(component_score)

                if node:
                    # Map children to their calculated normalized weights for traversal
                    for i, child_node in enumerate(node.children):
                        child_result = node_result.children.get(child_node.name)
                        if child_result:
                            child_weight = (
                                node_result.weights[i]
                                if i < len(node_result.weights)
                                else None
                            )
                            _traverse(child_result, parent=node, weight=child_weight)

        _traverse(breakdown, parent=None)
        return flat_list

    def get_metric_summary(self) -> Dict[str, Any]:
        return {
            'total_nodes': len(self.nodes),
            'metric_nodes': sum(isinstance(n, MetricNode) for n in self.nodes.values()),
            'component_nodes': sum(
                isinstance(n, ComponentNode) for n in self.nodes.values()
            ),
            'hierarchy_depth': self._calculate_depth(),
            'metrics': {
                name: {
                    'class': node.metric.__class__.__name__,
                    'parent': node.parent.name if node.parent else None,
                }
                for name, node in self.nodes.items()
                if isinstance(node, MetricNode)
            },
        }

    def _calculate_depth(self) -> int:
        """Calculate depth of tree"""
        if not self.root_node:
            return 0

        def _depth(node: Node, current: int = 0) -> int:
            if not hasattr(node, 'children') or not node.children:
                return current
            return max(_depth(child, current + 1) for child in node.children)

        return _depth(self.root_node)

    def get_node(self, name: str) -> Optional[Node]:
        """Get node by name"""
        return self.nodes.get(name)

    @property
    def elapsed_time(self) -> Union[float, None]:
        """Execution Time"""
        return self._elapsed_time

    @property
    def summary(self) -> Union[Dict[str, Any], None]:
        """Model Summary"""
        return self._summary

    @classmethod
    def display(cls):
        """Display Usage Documentation"""
        from IPython.display import HTML, display

        from axion.docs.eval_tree import (
            documentation_template,
            python_template,
            yaml_template,
        )
        from axion.docs.render import create_multi_usage_modal_card

        evaluation_runner_card = create_multi_usage_modal_card(
            key='eval_tree',
            title='EvalTree',
            description=cls.description,
            documentation_templates=[(documentation_template, '📖 Documentation')],
            usage_templates=[
                (python_template, '▶️ Python Example'),
                (yaml_template, '📋️ YAML Example'),
            ],
            max_width='1350px',
        )
        display(HTML(evaluation_runner_card))

    def __repr__(self) -> str:
        root = self.root_node.name if self.root_node else None
        return f'EvalTree(nodes={len(self.nodes)}, root={root})'
