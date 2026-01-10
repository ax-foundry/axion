import json
import uuid
from typing import Any, Dict, List, Optional, Union

from axion.eval_tree.metric import MetricNode
from axion.eval_tree.node import Node
from axion.schema import EvaluationResult, MetricScore, TestResult


class TreeMixin:
    """Mixin for generating eval tree."""

    name = 'Eval Tree'

    def _calculate_normalized_weights(
        self, node: Optional[Node] = None
    ) -> Dict[str, float]:
        """Calculate normalized weights for all nodes in the tree."""
        node = node or self.root_node
        normalized_weights = {}

        def _calculate_recursive(current_node: Node, parent_name: Optional[str] = None):
            # Get siblings (nodes with the same parent)
            if parent_name is None:
                # Root node - no siblings, weight is 1.0
                normalized_weights[current_node.name] = 1.0
            else:
                parent_node = next(
                    (n for n in self._get_all_nodes() if n.name == parent_name), None
                )
                if parent_node and hasattr(parent_node, 'children'):
                    siblings = parent_node.children
                    raw_sibling_weights = [
                        getattr(child, 'weight', 1.0) for child in siblings
                    ]
                    sibling_weights = [
                        w if w is not None else 1.0 for w in raw_sibling_weights
                    ]

                    total_weight = sum(sibling_weights)
                    raw_current_weight = getattr(current_node, 'weight', 1.0)
                    current_weight = (
                        raw_current_weight if raw_current_weight is not None else 1.0
                    )

                    if total_weight > 0:
                        normalized_weights[current_node.name] = (
                            current_weight / total_weight
                        )
                    else:
                        normalized_weights[current_node.name] = (
                            1.0 / len(siblings) if siblings else 1.0
                        )
                else:
                    normalized_weights[current_node.name] = 1.0

            # Recursively calculate for children
            if hasattr(current_node, 'children'):
                for child in current_node.children:
                    _calculate_recursive(child, current_node.name)

        _calculate_recursive(node)
        return normalized_weights

    def _get_all_nodes(self) -> List[Node]:
        """Get all nodes in the tree."""
        if hasattr(self, 'nodes') and isinstance(self.nodes, dict):
            return list(self.nodes.values())

        # Fallback: traverse tree manually
        nodes = []

        def _collect_nodes(node):
            nodes.append(node)
            if hasattr(node, 'children'):
                for child in node.children:
                    _collect_nodes(child)

        if hasattr(self, 'root_node') and self.root_node:
            _collect_nodes(self.root_node)
        return nodes

    def _generate_tree_data(self, node: Optional[Node] = None) -> Dict[str, Any]:
        node = node or self.root_node  # type: ignore[assignment]

        # Calculate normalized weights for all nodes
        normalized_weights = self._calculate_normalized_weights()

        node_data: Dict[str, Any] = {
            'name': node.name,  # type: ignore[union-attr]
            'type': 'metric' if isinstance(node, MetricNode) else 'component',
            'weight': getattr(node, 'weight', 1.0) or 1.0,  # type: ignore[union-attr]
            'normalized_weight': normalized_weights.get(node.name, 1.0),  # type: ignore[union-attr]
            'children': [],
            'score': None,
        }

        if isinstance(node, MetricNode):
            node_data.update(
                {
                    'source': node.metric.__class__.__name__,
                    'threshold': getattr(node.metric, 'threshold', None),
                }
            )

        for child in node.children:  # type: ignore[union-attr]
            node_data['children'].append(self._generate_tree_data(child))
        return node_data

    @staticmethod
    def _merge_results_with_tree(
        tree_data: Dict[str, Any],
        flat_results: List[MetricScore],
        summary_data: Optional[Dict[str, Any]] = None,
        is_aggregate_view: bool = False,
    ) -> Dict[str, Any]:
        """
        Merge scoring results and summary stats into the tree data structure.
        """
        if not flat_results:
            return tree_data

        results_map = {row.name: row for row in flat_results}

        def merge_node(node_dict: Dict[str, Any]) -> None:
            node_name = node_dict.get('name')
            result_row = results_map.get(node_name)

            if result_row:
                node_dict['score'] = result_row.score
                node_dict['cost_estimate'] = result_row.cost_estimate
                if node_dict.get('type') == 'metric':
                    node_dict['explanation'] = result_row.explanation
                    node_dict['passed'] = result_row.passed
                    # Add signals data only if not in aggregate view
                    if (
                        not is_aggregate_view
                        and hasattr(result_row, 'signals')
                        and result_row.signals
                    ):
                        node_dict['signals'] = result_row.signals

            if summary_data and node_name in summary_data:
                node_dict['summary'] = summary_data[node_name]

            for child in node_dict.get('children', []):
                merge_node(child)

        merge_node(tree_data)
        return tree_data

    def visualize_html(
        self, results: Optional[Union[EvaluationResult, TestResult]] = None
    ) -> str:
        """
        Generate the standalone HTML for the interactive tree visualization.
        Handles both single test results and batch evaluation results (visualizing the first item).
        """
        container_id = f'hsm-container-{uuid.uuid4().hex}'
        tree_data = self._generate_tree_data()
        flat_scores = None
        summary_data = None
        test_case_data = None

        if isinstance(results, EvaluationResult):
            if results.results:
                flat_scores = results.results[0].score_results
                summary_data = results.summary
        elif isinstance(results, TestResult):
            flat_scores = results.score_results
            test_case_data = results.test_case

        if flat_scores:
            # Detect if this is an aggregate view (multiple test runs)
            is_aggregate_view = (
                isinstance(results, EvaluationResult) and len(results.results) > 1
            )
            tree_data = self._merge_results_with_tree(
                tree_data, flat_scores, summary_data, is_aggregate_view
            )

        json_tree_data = json.dumps(tree_data)

        # Extract test case information for display
        test_case_info = ''
        evaluation_stats = ''

        if test_case_data:
            query = getattr(test_case_data, 'query', None) or getattr(
                test_case_data, 'question', None
            )
            actual_output = (
                getattr(test_case_data, 'actual_output', None)
                or getattr(test_case_data, 'response', None)
                or getattr(test_case_data, 'answer', None)
            )
            retrieved_content = (
                getattr(test_case_data, 'retrieved_content', None)
                or getattr(test_case_data, 'context', None)
                or getattr(test_case_data, 'retrieved_contexts', None)
            )

            cards = []
            if query:
                preview = (query[:80] + '...') if len(query) > 80 else query
                cards.append(f"""
                <div class="content-card">
                    <div class="card-header">
                        <span class="card-icon">📝</span>
                        <span class="card-title">Query</span>
                    </div>
                    <div class="card-preview">{preview}</div>
                    <button class="expand-btn" onclick="expandCard('query-{container_id}', this)">Expand</button>
                    <div id="query-{container_id}" class="card-full-content">{query}</div>
                </div>
                """)

            if actual_output:
                preview = (
                    (actual_output[:80] + '...')
                    if len(actual_output) > 80
                    else actual_output
                )
                cards.append(f"""
                <div class="content-card">
                    <div class="card-header">
                        <span class="card-icon">💬</span>
                        <span class="card-title">Response</span>
                    </div>
                    <div class="card-preview">{preview}</div>
                    <button class="expand-btn" onclick="expandCard('response-{container_id}', this)">Expand</button>
                    <div id="response-{container_id}" class="card-full-content">{actual_output}</div>
                </div>
                """)

            if retrieved_content:
                # Handle different formats of retrieved content
                content_display = str(retrieved_content)
                if isinstance(retrieved_content, list):
                    content_display = ' | '.join(
                        str(item) for item in retrieved_content
                    )
                preview = (
                    (content_display[:80] + '...')
                    if len(content_display) > 80
                    else content_display
                )
                cards.append(f"""
                <div class="content-card">
                    <div class="card-header">
                        <span class="card-icon">📚</span>
                        <span class="card-title">Context</span>
                    </div>
                    <div class="card-preview">{preview}</div>
                    <button class="expand-btn" onclick="expandCard('context-{container_id}', this)">Expand</button>
                    <div id="context-{container_id}" class="card-full-content">{content_display}</div>
                </div>
                """)

            if cards:
                test_case_info = f'<div class="test-case-info"><div class="cards-container">{"".join(cards)}</div></div>'

        elif isinstance(results, EvaluationResult):
            total_runs = len(results.results)
            evaluation_name = results.evaluation_name or 'Evaluation'

            evaluation_stats = f"""
            <div class="evaluation-stats">
                <div class="stats-row">
                    <div class="stat-item">
                        <span class="stat-value">{total_runs}</span>
                        <span class="stat-label">Test Cases</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">{evaluation_name}</span>
                        <span class="stat-label">Experiment</span>
                    </div>
                </div>
            </div>
            """

        L, R = '{', '}'

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tree Visualization</title>
    <style>
        /* CSS Reset to avoid notebook style conflicts */
        #{container_id} {{
            all: initial;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: #333;
        }}
        #{container_id} * {{
            all: revert;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: #f0f2f5; color: #333; padding: 20px;
            display: flex; justify-content: center; align-items: flex-start;
            min-height: 100vh; margin: 0;
        }}
        #{container_id} {{
            width: 100%; max-width: 1400px; background: #ffffff;
            border-radius: 16px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        #{container_id} .hsm-header {{
            text-align: center; padding: 20px; position: relative;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        #{container_id} .hsm-header h1 {{
            margin: 0 0 10px 0; font-size: 1.8em; font-weight: 300;
        }}
        #{container_id} .metrics-summary {{
            display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;
        }}
        #{container_id} .metric-item {{ text-align: center; }}
        #{container_id} .metric-value {{ font-size: 1.5em; font-weight: 600; margin-bottom: 3px; }}
        #{container_id} .metric-label {{ font-size: 0.8em; opacity: 0.8; }}
        #{container_id} .test-case-info {{
            margin-top: 15px; padding: 15px; background: rgba(255,255,255,0.1);
            border-radius: 6px; max-width: 900px; margin-left: auto; margin-right: auto;
        }}
        #{container_id} .cards-container {{
            display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;
        }}
        #{container_id} .content-card {{
            flex: 1; min-width: 250px; max-width: 300px;
            background: rgba(255,255,255,0.15); border-radius: 8px; padding: 12px;
            border: 1px solid rgba(255,255,255,0.2); transition: all 0.2s ease;
        }}
        #{container_id} .content-card:hover {{
            background: rgba(255,255,255,0.2); transform: translateY(-1px);
        }}
        #{container_id} .card-header {{
            display: flex; align-items: center; gap: 8px; margin-bottom: 8px;
        }}
        #{container_id} .card-icon {{
            font-size: 1.1em;
        }}
        #{container_id} .card-title {{
            font-weight: 600; font-size: 0.85em; color: rgba(255,255,255,0.95);
        }}
        #{container_id} .card-preview {{
            font-size: 0.75em; line-height: 1.4; color: rgba(255,255,255,0.8);
            margin-bottom: 10px; min-height: 40px;
        }}
        #{container_id} .expand-btn {{
            background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.3);
            color: white; padding: 4px 8px; border-radius: 4px; cursor: pointer;
            font-size: 0.7em; font-weight: 500; width: 100%;
            transition: all 0.2s ease;
        }}
        #{container_id} .expand-btn:hover {{
            background: rgba(255,255,255,0.3);
        }}
        #{container_id} .card-full-content {{
            display: none; margin-top: 10px; padding: 10px;
            background: rgba(0,0,0,0.2); border-radius: 4px;
            font-size: 0.7em; line-height: 1.4; color: rgba(255,255,255,0.9);
            max-height: 200px; overflow-y: auto; word-wrap: break-word;
        }}
        #{container_id} .card-full-content.visible {{
            display: block;
        }}
        #{container_id} .compact-toggle {{
            position: absolute; top: 15px; right: 15px; background: rgba(255,255,255,0.2);
            border: 1px solid rgba(255,255,255,0.3); color: white; padding: 6px 12px;
            border-radius: 4px; cursor: pointer; font-size: 0.8em; font-weight: 500;
        }}
        #{container_id} .compact-toggle:hover {{
            background: rgba(255,255,255,0.3);
        }}
        #{container_id} .evaluation-stats {{
            margin-top: 15px; padding: 15px; background: rgba(255,255,255,0.1);
            border-radius: 6px; max-width: 500px; margin-left: auto; margin-right: auto;
        }}
        #{container_id} .stats-row {{
            display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;
        }}
        #{container_id} .stat-item {{
            text-align: center;
        }}
        #{container_id} .stat-value {{
            display: block; font-size: 1.2em; font-weight: 600; margin-bottom: 3px;
        }}
        #{container_id} .stat-label {{
            display: block; font-size: 0.75em; opacity: 0.8;
        }}
        #{container_id} .tree-container {{ padding: 30px; }}
        #{container_id} .tree {{ list-style-type: none; padding: 0; margin: 0; position: relative; }}
        #{container_id} .tree .children {{
            list-style-type: none; padding-left: 40px; margin-left: 20px;
            border-left: 2px dashed #dbe0e6;
        }}
        #{container_id} .tree-node {{ padding: 10px 0; position: relative; }}
        #{container_id} .tree-node::before {{
            content: ''; position: absolute; top: 32px; left: -22px;
            width: 20px; height: 2px; background-color: #dbe0e6; z-index: 1;
        }}
        #{container_id} .tree > .tree-node::before {{ display: none; }}
        #{container_id} .node-card {{
            display: flex; align-items: flex-start; background: #fff;
            border: 1px solid #e9ecef; border-radius: 12px; padding: 20px;
            position: relative; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease; cursor: pointer;
        }}
        #{container_id} .node-card:hover {{
            transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
            border-color: #667eea;
        }}
        #{container_id} .node-info {{ flex-grow: 1; }}
        #{container_id} .node-name {{
            font-size: 1.3em; font-weight: 600; color: #2c3e50; margin: 0 0 8px 0;
            display: flex; align-items: center; gap: 8px;
        }}
        #{container_id} .node-type-label {{
            font-size: 0.75em; font-weight: 500; color: #889ab2;
            background: #f8f9fa; padding: 2px 8px; border-radius: 12px;
        }}
        #{container_id} .weight-info {{
            margin-top: 8px; display: flex; gap: 15px; flex-wrap: wrap;
        }}
        #{container_id} .weight-item {{
            display: flex; align-items: center; gap: 4px; font-size: 0.85em;
        }}
        #{container_id} .weight-label {{
            color: #666; font-weight: 500;
        }}
        #{container_id} .weight-value {{
            color: #333; font-weight: 600;
            background: #f0f2f5; padding: 2px 6px; border-radius: 4px;
        }}
        #{container_id} .normalized-weight {{
            color: #667eea; font-weight: 700;
        }}
        #{container_id} .score-display {{
            font-size: 1.8em; font-weight: 700; color: #667eea; margin-left: 20px;
            text-align: center; min-width: 80px;
        }}
        #{container_id} .node-card.component {{ border-left: 5px solid #764ba2; }}
        #{container_id} .node-card.metric {{ border-left: 5px solid #00b894; }}
        #{container_id} .node-card.metric.failed {{ border-left-color: #d63031; }}
        #{container_id} .node-card.metric.failed .score-display {{ color: #d63031; }}
        #{container_id} .toggle-button {{
            position: absolute; left: -52px; top: 22px; width: 20px; height: 20px;
            background-color: #fff; border: 2px solid #dbe0e6; border-radius: 50%;
            cursor: pointer; display: flex; justify-content: center; align-items: center;
            font-size: 16px; color: #889ab2; transition: all 0.2s ease; z-index: 2;
        }}
        #{container_id} .toggle-button:hover {{
            background-color: #667eea; border-color: #667eea; color: white;
        }}
        #{container_id} .children {{ display: none; }}
        #{container_id} .tree-node.expanded > .children {{ display: block; }}
        #{container_id} .tree-node.expanded > .toggle-button::before {{ content: '−'; }}
        #{container_id} .tree-node:not(.expanded) > .toggle-button::before {{ content: '+'; }}

        /* Enhanced Summary Styles */
        #{container_id} .node-summary-info {{
            margin-top: 12px; padding-top: 12px; border-top: 1px solid #f0f0f0;
        }}
        #{container_id} .summary-row {{
            display: flex; gap: 20px; margin-bottom: 8px; font-size: 0.85em;
            align-items: center; flex-wrap: wrap;
        }}
        #{container_id} .summary-item {{
            display: flex; align-items: center; gap: 4px; color: #555;
        }}
        #{container_id} .summary-item .label {{
            font-weight: 500; color: #666;
        }}
        #{container_id} .summary-item .value {{
            font-weight: 600; color: #333;
        }}
        #{container_id} .performance-tier {{
            padding: 2px 8px; border-radius: 8px; font-size: 0.75em;
            font-weight: 600; text-transform: uppercase;
        }}
        #{container_id} .tier-excellent {{ background: #d4edda; color: #155724; }}
        #{container_id} .tier-good {{ background: #cce5ff; color: #004085; }}
        #{container_id} .tier-fair {{ background: #fff3cd; color: #856404; }}
        #{container_id} .tier-poor {{ background: #f8d7da; color: #721c24; }}
        #{container_id} .distribution-chart {{
            font-family: 'Courier New', monospace; background: #f8f9fa;
            padding: 4px 8px; border-radius: 4px; font-size: 0.8em; color: #666;
        }}
        #{container_id} .trend-indicator {{
            font-size: 1.2em; font-weight: bold;
        }}
        #{container_id} .stats-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 12px; margin-top: 8px;
        }}
        #{container_id} .stat-box {{
            background: #f8f9fa; padding: 8px; border-radius: 6px; text-align: center;
            border: 1px solid #e9ecef;
        }}
        #{container_id} .stat-box .stat-value {{
            font-weight: 700; font-size: 1.1em; color: #495057;
        }}
        #{container_id} .stat-box .stat-label {{
            font-size: 0.75em; color: #6c757d; margin-top: 2px;
        }}

        /* Signals Styles */
        #{container_id} .signals-section {{
            margin-top: 15px; padding-top: 15px; border-top: 1px solid #e9ecef;
        }}
        #{container_id} .signals-header {{
            display: flex; align-items: center; justify-content: space-between;
            margin-bottom: 12px;
        }}
        #{container_id} .signals-title {{
            font-size: 0.95em; font-weight: 600; color: #495057;
            display: flex; align-items: center; gap: 6px;
        }}
        #{container_id} .signals-toggle {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none; color: white;
            padding: 6px 16px; border-radius: 6px; cursor: pointer;
            font-size: 0.8em; font-weight: 600; transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(102, 126, 234, 0.2);
            display: flex; align-items: center; gap: 6px;
        }}
        #{container_id} .signals-toggle:hover {{
            background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
        }}
        #{container_id} .signals-toggle:active {{
            transform: translateY(0);
            box-shadow: 0 2px 4px rgba(102, 126, 234, 0.2);
        }}
        #{container_id} .signals-content {{
            display: none; max-height: 400px; overflow-y: auto;
        }}
        #{container_id} .signals-content.visible {{
            display: block;
        }}
        #{container_id} .signals-group {{
            margin-bottom: 16px; border: 1px solid #e9ecef;
            border-radius: 6px; overflow: hidden;
        }}
        #{container_id} .signals-group-header {{
            background: #f8f9fa; padding: 10px 12px; border-bottom: 1px solid #e9ecef;
            cursor: pointer; display: flex; align-items: center; justify-content: space-between;
            transition: background-color 0.2s ease;
        }}
        #{container_id} .signals-group-header:hover {{
            background: #e9ecef;
        }}
        #{container_id} .signals-group-title {{
            font-size: 0.85em; font-weight: 600; color: #495057;
        }}
        #{container_id} .signals-group-score {{
            font-size: 0.8em; font-weight: 600; color: #6c757d;
            background: white; padding: 2px 8px; border-radius: 12px;
        }}
        #{container_id} .signals-group-content {{
            display: none; padding: 12px;
        }}
        #{container_id} .signals-group-content.visible {{
            display: block;
        }}
        #{container_id} .signal-item {{
            display: flex; align-items: center; justify-content: space-between;
            padding: 6px 0; border-bottom: 1px solid #f1f3f4;
        }}
        #{container_id} .signal-item:last-child {{
            border-bottom: none;
        }}
        #{container_id} .signal-name {{
            font-size: 0.8em; color: #495057; flex: 1;
        }}
        #{container_id} .signal-value {{
            font-size: 0.75em; color: #6c757d; margin-right: 12px;
            background: #f8f9fa; padding: 2px 6px; border-radius: 4px;
            font-family: 'Courier New', monospace;
        }}
        #{container_id} .signal-score {{
            font-size: 0.8em; font-weight: 600; min-width: 40px; text-align: right;
        }}
        #{container_id} .signal-score.good {{ color: #28a745; }}
        #{container_id} .signal-score.bad {{ color: #dc3545; }}
        #{container_id} .signal-expand-btn {{
            background: none; border: none; color: #6c757d; cursor: pointer;
            font-size: 12px; padding: 2px; transition: color 0.2s ease;
        }}
        #{container_id} .signal-expand-btn:hover {{
            color: #495057;
        }}
        #{container_id} .signal-details {{
            display: none; margin-top: 8px; padding: 8px;
            background: #f8f9fa; border-radius: 4px; font-size: 0.75em;
        }}
        #{container_id} .signal-details.visible {{
            display: block;
        }}
        #{container_id} .signal-details-item {{
            margin: 4px 0; color: #6c757d;
        }}

        /* Collapsible sections */
        #{container_id} .details-toggle {{
            background: none; border: none; color: #667eea; cursor: pointer;
            font-size: 0.8em; font-weight: 500; padding: 4px 0;
            text-decoration: underline;
        }}
        #{container_id} .details-toggle:hover {{ color: #5a6fd8; }}
        #{container_id} .detailed-stats {{ display: none; margin-top: 12px; }}
        #{container_id} .detailed-stats.visible {{ display: block; }}

        /* Performance indicators */
        #{container_id} .performance-indicator {{
            display: inline-block; width: 8px; height: 8px; border-radius: 50%;
            margin-right: 6px;
        }}
        #{container_id} .perf-excellent {{ background: #28a745; }}
        #{container_id} .perf-good {{ background: #17a2b8; }}
        #{container_id} .perf-fair {{ background: #ffc107; }}
        #{container_id} .perf-poor {{ background: #dc3545; }}

        /* Compact Mode Styles */
        #{container_id}.compact-mode .node-card {{
            padding: 12px;
        }}
        #{container_id}.compact-mode .node-name {{
            font-size: 1.1em; margin-bottom: 4px;
        }}
        #{container_id}.compact-mode .weight-info {{
            margin-top: 4px;
        }}
        #{container_id}.compact-mode .weight-item {{
            font-size: 0.75em;
        }}
        #{container_id}.compact-mode .node-summary-info {{
            margin-top: 8px; padding-top: 8px;
        }}
        #{container_id}.compact-mode .summary-row {{
            font-size: 0.75em; gap: 15px;
        }}
        #{container_id}.compact-mode .score-display {{
            font-size: 1.5em; min-width: 60px;
        }}
        #{container_id}.compact-mode .tree .children {{
            padding-left: 30px; margin-left: 15px;
        }}
        #{container_id}.compact-mode .tree-node {{
            padding: 5px 0;
        }}
        #{container_id}.compact-mode .signals-section {{
            margin-top: 10px; padding-top: 10px;
        }}

    </style>
</head>
<body>
    <div id="{container_id}">
        <div class="hsm-header">
            <button class="compact-toggle" onclick="toggleCompactMode()">Compact View</button>
            <h1>{self.name}</h1>
            <div class="metrics-summary">
                <div class="metric-item">
                    <div class="metric-value overall-score">0.000</div>
                    <div class="metric-label">Overall Score</div>
                </div>
            </div>
            {test_case_info}
            {evaluation_stats}
        </div>
        <div class="tree-container"><ul class="tree"></ul></div>
    </div>

    <script>
    (function() {{
        const container = document.getElementById('{container_id}');
        if (!container) return;

        const treeData = {json_tree_data};
        const treeRoot = container.querySelector('.tree');
        const overallScoreEl = container.querySelector('.overall-score');

        function formatNumber(num, decimals = 2) {{
            if (num === null || num === undefined || isNaN(num)) return 'N/A';
            return num.toFixed(decimals);
        }}

        function formatPercentage(num, decimals = 0) {{
            if (num === null || num === undefined || isNaN(num)) return 'N/A';
            return (num * 100).toFixed(decimals) + '%';
        }}

        function createWeightInfo(nodeData) {{
            const weight = nodeData.weight;
            const normalizedWeight = nodeData.normalized_weight;

            if (weight === null || weight === undefined) {{
                return '';
            }}

            return `
                <div class="weight-info">
                    <div class="weight-item">
                        <span class="weight-label">Weight:</span>
                        <span class="weight-value normalized-weight">${L}formatPercentage(normalizedWeight, 1){R}</span>
                    </div>
                </div>
            `;
        }}

        function createEnhancedSummaryInfo(nodeData) {{
            const summary = nodeData.summary;
            if (!summary) return '';

            const isMetric = nodeData.type === 'metric';

            // Main summary row with key metrics
            let summaryHtml = `
                <div class="summary-row">
                    <div class="summary-item">
                        <span class="label">Avg:</span>
                        <span class="value">${L}formatNumber(summary.avg_score){R}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Pass:</span>
                        <span class="value">${L}formatPercentage(summary.pass_rate){R}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Range:</span>
                        <span class="value">${L}formatNumber(summary.min_score){R}-${{formatNumber(summary.max_score)}}</span>
                    </div>
                </div>
            `;

            // Second row with distribution and counts
            summaryHtml += `
                <div class="summary-row">
                    <div class="summary-item">
                        <span class="label">Distribution:</span>
                        <span class="distribution-chart">${{summary.distribution}}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Success:</span>
                        <span class="value">${{summary.valid_count}}/${{summary.total_runs}} (${{formatPercentage(summary.success_rate)}})</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Std Dev:</span>
                        <span class="value">${L}formatNumber(summary.std_dev, 3){R}</span>
                    </div>
                </div>
            `;

            // Collapsible detailed stats for metrics
            if (isMetric && summary.percentiles) {{
                const detailsId = `details-${{Math.random().toString(36).substr(2, 9)}}`;
                summaryHtml += `
                    <button class="details-toggle" onclick="toggleDetails('${{detailsId}}')">
                        View Detailed Statistics ▼
                    </button>
                    <div id="${{detailsId}}" class="detailed-stats">
                        <div class="stats-grid">
                            <div class="stat-box">
                                <div class="stat-value">${L}formatNumber(summary.median_score){R}</div>
                                <div class="stat-label">Median</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-value">${L}formatNumber(summary.percentiles.p25){R}</div>
                                <div class="stat-label">25th %ile</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-value">${L}formatNumber(summary.percentiles.p75){R}</div>
                                <div class="stat-label">75th %ile</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-value">${L}formatNumber(summary.percentiles.p95){R}</div>
                                <div class="stat-label">95th %ile</div>
                            </div>
                        </div>
                    </div>
                `;
            }}

            return summaryHtml;
        }}

        function createSignalsSection(nodeData) {{
            const signals = nodeData.signals;
            if (!signals || Object.keys(signals).length === 0) return '';

            const signalsId = `signals-${{Math.random().toString(36).substr(2, 9)}}`;

            let signalsHtml = `
                <div class="signals-section">
                    <div class="signals-header">
                        <div class="signals-title">
                            🔍 Signals Analysis
                        </div>
                        <button class="signals-toggle" onclick="toggleSignals('${{signalsId}}')">
                            <span>Show Details</span>
                            <span style="font-size: 0.9em;">▼</span>
                        </button>
                    </div>
                    <div id="${{signalsId}}" class="signals-content">
            `;

            // Process signals groups
            Object.entries(signals).forEach(([groupKey, groupSignals]) => {{
                if (!Array.isArray(groupSignals) || groupSignals.length === 0) return;

                const groupId = `group-${{Math.random().toString(36).substr(2, 9)}}`;
                const isOverall = groupKey === 'overall';
                const groupTitle = isOverall ? 'Overall Metrics' : groupKey.replace(/_/g, ' ').replace(/^./, str => str.toUpperCase());

                // Calculate group score if available
                let groupScore = '';
                if (!isOverall && groupSignals.length === 1 && groupSignals[0].score !== undefined) {{
                    const score = groupSignals[0].score;
                    const scoreClass = score > 0.5 ? 'good' : 'bad';
                    groupScore = `<span class="signals-group-score ${{scoreClass}}">${{formatNumber(score, 2)}}</span>`;
                }}

                signalsHtml += `
                    <div class="signals-group">
                        <div class="signals-group-header" onclick="toggleSignalsGroup('${{groupId}}')">
                            <span class="signals-group-title">${{groupTitle}}</span>
                            ${{groupScore}}
                        </div>
                        <div id="${{groupId}}" class="signals-group-content">
                `;

                // Process individual signals
                groupSignals.forEach((signal, index) => {{
                    const signalId = `signal-${{Math.random().toString(36).substr(2, 9)}}`;
                    const scoreClass = signal.score > 0.5 ? 'good' : 'bad';
                    const hasDescription = signal.description && signal.description !== `Signal: ${{signal.name}}`;

                    signalsHtml += `
                        <div class="signal-item">
                            <div class="signal-name">${{signal.name}}</div>
                            <div class="signal-value">${{signal.value}}</div>
                            <div class="signal-score ${{scoreClass}}">${{formatNumber(signal.score, 2)}}</div>
                            ${{hasDescription ? `<button class="signal-expand-btn" onclick="toggleSignalDetails('${{signalId}}')">ℹ️</button>` : ''}}
                        </div>
                        ${{hasDescription ? `
                            <div id="${{signalId}}" class="signal-details">
                                <div class="signal-details-item"><strong>Description:</strong> ${{signal.description}}</div>
                            </div>
                        ` : ''}}
                    `;
                }});

                signalsHtml += `
                        </div>
                    </div>
                `;
            }});

            signalsHtml += `
                    </div>
                </div>
            `;

            return signalsHtml;
        }}

        function createTreeNode(nodeData) {{
            const nodeLi = document.createElement('li');
            nodeLi.className = 'tree-node';
            const hasChildren = nodeData.children && nodeData.children.length > 0;

            const nodeCard = document.createElement('div');
            nodeCard.className = `node-card ${{nodeData.type}}`;
            if (nodeData.type === 'metric' && nodeData.passed === false) {{
                nodeCard.classList.add('failed');
            }}

            const nodeInfo = document.createElement('div');
            nodeInfo.className = 'node-info';

            // Enhanced node header
            const typeLabel = nodeData.type === 'component' ? 'Component' :
                             (nodeData.source || 'Metric');

            nodeInfo.innerHTML = `
                <div class="node-name">
                    ${{nodeData.name}}
                    <span class="node-type-label">${{typeLabel}}</span>
                </div>
                ${{createWeightInfo(nodeData)}}
            `;

            // Add enhanced summary info if available
            const summaryInfo = createEnhancedSummaryInfo(nodeData);
            if (summaryInfo) {{
                const summaryDiv = document.createElement('div');
                summaryDiv.className = 'node-summary-info';
                summaryDiv.innerHTML = summaryInfo;
                nodeInfo.appendChild(summaryDiv);
            }}

            // Add signals section for metric nodes (handled by Python logic)
            if (nodeData.type === 'metric' && nodeData.signals) {{
                const signalsInfo = createSignalsSection(nodeData);
                if (signalsInfo) {{
                    const signalsDiv = document.createElement('div');
                    signalsDiv.innerHTML = signalsInfo;
                    nodeInfo.appendChild(signalsDiv);
                }}
            }}

            nodeCard.appendChild(nodeInfo);

            // Score display - UPDATED to prioritize average score
            const scoreDisplay = document.createElement('div');
            scoreDisplay.className = 'score-display';

            // Prioritize summary average score if available, then fall back to individual score
            let displayScore = null;
            if (nodeData.summary && nodeData.summary.avg_score !== null && nodeData.summary.avg_score !== undefined) {{
                displayScore = nodeData.summary.avg_score;
            }} else if (nodeData.score !== null && nodeData.score !== undefined) {{
                displayScore = nodeData.score;
            }}

            if (displayScore !== null) {{
                scoreDisplay.textContent = displayScore.toFixed(3);

                // Add performance color based on score
                if (nodeData.summary && nodeData.summary.performance_tier) {{
                    scoreDisplay.style.color = {{
                        'excellent': '#28a745',
                        'good': '#17a2b8',
                        'fair': '#ffc107',
                        'poor': '#dc3545'
                    }}[nodeData.summary.performance_tier] || '#667eea';
                }}
            }} else {{
                scoreDisplay.textContent = 'N/A';
                scoreDisplay.style.color = '#999';
            }}
            nodeCard.appendChild(scoreDisplay);
            nodeLi.appendChild(nodeCard);

            if (hasChildren) {{
                if(nodeData.name === 'model') nodeLi.classList.add('expanded');
                const toggleButton = document.createElement('button');
                toggleButton.className = 'toggle-button';
                toggleButton.addEventListener('click', (e) => {{
                    e.stopPropagation();
                    nodeLi.classList.toggle('expanded');
                }});
                nodeLi.appendChild(toggleButton);

                const childrenUl = document.createElement('ul');
                childrenUl.className = 'children';
                nodeData.children.forEach(child => childrenUl.appendChild(createTreeNode(child)));
                nodeLi.appendChild(childrenUl);
            }}
            return nodeLi;
        }}

        // Global function for toggling detailed stats
        window.toggleDetails = function(detailsId) {{
            const element = document.getElementById(detailsId);
            if (element) {{
                element.classList.toggle('visible');
                const button = element.previousElementSibling;
                if (button && button.classList.contains('details-toggle')) {{
                    const isVisible = element.classList.contains('visible');
                    button.textContent = isVisible ?
                        'Hide Detailed Statistics ▲' :
                        'View Detailed Statistics ▼';
                }}
            }}
        }}

        // Global function for toggling signals section
        window.toggleSignals = function(signalsId) {{
            const element = document.getElementById(signalsId);
            if (element) {{
                element.classList.toggle('visible');
                const button = element.parentElement.querySelector('.signals-toggle');
                if (button) {{
                    const isVisible = element.classList.contains('visible');
                    const textSpan = button.querySelector('span:first-child');
                    const arrowSpan = button.querySelector('span:last-child');
                    if (textSpan && arrowSpan) {{
                        textSpan.textContent = isVisible ? 'Hide Details' : 'Show Details';
                        arrowSpan.textContent = isVisible ? '▲' : '▼';
                    }}
                }}
            }}
        }}

        // Global function for toggling signals groups
        window.toggleSignalsGroup = function(groupId) {{
            const element = document.getElementById(groupId);
            if (element) {{
                element.classList.toggle('visible');
            }}
        }}

        // Global function for toggling signal details
        window.toggleSignalDetails = function(signalId) {{
            const element = document.getElementById(signalId);
            if (element) {{
                element.classList.toggle('visible');
            }}
        }}

        // Global function for expanding card content
        window.expandCard = function(contentId, button) {{
            const content = document.getElementById(contentId);
            if (content) {{
                const isVisible = content.classList.contains('visible');
                if (isVisible) {{
                    content.classList.remove('visible');
                    button.textContent = 'Expand';
                }} else {{
                    content.classList.add('visible');
                    button.textContent = 'Collapse';
                }}
            }}
        }}

        // Global function for toggling compact mode
        window.toggleCompactMode = function() {{
            const container = document.getElementById('{container_id}');
            const button = container.querySelector('.compact-toggle');
            if (container.classList.contains('compact-mode')) {{
                container.classList.remove('compact-mode');
                button.textContent = 'Compact View';
            }} else {{
                container.classList.add('compact-mode');
                button.textContent = 'Detailed View';
            }}
        }}

        function initializeTree() {{
            if (!treeData) return;
            treeRoot.innerHTML = '';
            treeRoot.appendChild(createTreeNode(treeData));

            let overallScore = 'N/A';
            // Prefer the summary average score for the root node if available (batch mode)
            if (treeData.summary && treeData.summary.avg_score !== null && treeData.summary.avg_score !== undefined) {{
                overallScore = treeData.summary.avg_score.toFixed(3);
            }}
            // Fallback to the single execution score
            else if (treeData.score !== null && treeData.score !== undefined) {{
                overallScore = treeData.score.toFixed(3);
            }}
            overallScoreEl.textContent = overallScore;
        }}

        initializeTree();
    }})();
    </script>
</body>
</html>
        """
        return html_content

    def visualize(self, results: Optional[Union[EvaluationResult, TestResult]] = None):
        """Displays the tree visualization in a Jupyter Notebook or saves to HTML."""
        try:
            from IPython.display import HTML, display

            display(HTML(self.visualize_html(results)))
        except ImportError:
            filename = 'tree.html'
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.visualize_html(results))
            print(f"Visualization saved to '{filename}'")
