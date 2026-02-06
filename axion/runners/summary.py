import math
import re
import shutil
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from axion.schema import TestResult


def normalize_metric_name(name: str) -> str:
    """Normalize metric names for display."""
    name = name.replace('_', ' ')
    name = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip().upper()


class BaseSummary:
    """Abstract base class for creating post-execution summary displays."""

    def execute(self, results: List['TestResult'], total_time: float) -> Dict[str, Any]:
        """
        Processes the results and displays a summary.
        Subclasses must implement this method.
        """
        raise NotImplementedError


class SimpleSummary(BaseSummary):
    """
    Summary focused on high-level KPIs and business impact.
    Clean dashboard format.
    """

    COLORS = {
        'success': '\033[92m',
        'warning': '\033[93m',
        'critical': '\033[91m',
        'info': '\033[94m',
        'accent': '\033[95m',
        'bold': '\033[1m',
        'reset': '\033[0m',
    }

    def _is_valid_score(self, score) -> bool:
        """Check if a score is valid (not None or NaN)."""
        if score is None:
            return False
        if isinstance(score, float) and math.isnan(score):
            return False
        return True

    def execute(
        self,
        results: List['TestResult'],
        total_time: float,
        total_task_runs: int = None,
    ) -> Dict[str, Any]:
        """
        Execute simple summary.

        Args:
            results: List of test results to analyze
            total_time: Total execution time in seconds
            total_task_runs: Optional total number of task runs
        """
        if not results:
            print('📊 No evaluation data available.')
            return {}

        # Collect valid scores only
        all_scores: List[float] = []
        for result in results:
            for score in result.score_results:
                if self._is_valid_score(score.score) and score.score is not None:
                    all_scores.append(score.score)

        # Calculate KPIs
        overall_performance = statistics.mean(all_scores) if all_scores else 0.0
        consistency = 1.0 - statistics.stdev(all_scores) if len(all_scores) > 1 else 1.0

        # Format display values
        table_width = 58
        time_text = time.strftime('%M:%S', time.gmtime(total_time))

        # Executive dashboard
        print(f'\n{self.COLORS["bold"]}{self.COLORS["accent"]}')
        print('╔' + '═' * table_width + '╗')
        print('║' + ' EVALUATION REPORT '.center(table_width) + '║')
        print('╠' + '═' * table_width + '╣')

        # Performance line
        perf_content = f' Performance Score: {overall_performance:.1%}'
        print(f'║{perf_content}{" " * (table_width - len(perf_content))}║')

        # Consistency line
        consistency_content = f' Consistency Index: {consistency:.1%}'
        print(
            f'║{consistency_content}{" " * (table_width - len(consistency_content))}║'
        )

        # Task runs line (only if provided)
        if total_task_runs is not None:
            runs_content = f' Task Runs: {total_task_runs:,}'
            print(f'║{runs_content}{" " * (table_width - len(runs_content))}║')

        # Samples line
        samples_content = f' Samples Analyzed: {len(results):,}'
        print(f'║{samples_content}{" " * (table_width - len(samples_content))}║')

        # Time line
        time_content = f' Execution Time: {time_text}'
        print(f'║{time_content}{" " * (table_width - len(time_content))}║')

        print('╚' + '═' * table_width + '╝')
        print(f'{self.COLORS["reset"]}\n')

        return {}


class MetricSummary(BaseSummary):
    """
    Detailed Metric summary with detailed analytics, performance insights,
    and executive-style reporting.
    """

    # Enhanced color palette
    COLORS = {
        'excellent': '\033[92m',  # Bright green
        'good': '\033[96m',  # Cyan
        'warning': '\033[93m',  # Yellow
        'poor': '\033[91m',  # Red
        'accent': '\033[95m',  # Magenta
        'header': '\033[94m',  # Blue
        'bold': '\033[1m',
        'dim': '\033[2m',
        'reset': '\033[0m',
    }

    # Performance thresholds
    THRESHOLDS = {'excellent': 0.90, 'good': 0.75, 'warning': 0.60}

    def __init__(self, show_distribution: bool = True):
        """
        Initialize the professional summary.

        Args:
            show_distribution: Whether to show score distribution charts
        """
        self.show_distribution = show_distribution
        self.terminal_width = self._get_terminal_width()

    def _get_terminal_width(self) -> int:
        """Get terminal width with fallback."""
        try:
            return min(120, shutil.get_terminal_size().columns)
        except:
            return 100

    def _is_valid_score(self, score) -> bool:
        """Check if a score is valid (not None or NaN)."""
        if score is None:
            return False
        if isinstance(score, float) and math.isnan(score):
            return False
        return True

    def _get_performance_color(self, score: float) -> str:
        """Get color based on performance score."""
        if score >= self.THRESHOLDS['excellent']:
            return self.COLORS['excellent']
        elif score >= self.THRESHOLDS['good']:
            return self.COLORS['good']
        elif score >= self.THRESHOLDS['warning']:
            return self.COLORS['warning']
        else:
            return self.COLORS['poor']

    def _create_bar_chart(self, value: float, max_value: float, width: int = 20) -> str:
        """Create a horizontal bar chart."""
        if max_value == 0:
            return '░' * width

        filled_width = int((value / max_value) * width)
        color = self._get_performance_color(value)

        bar = color + '█' * filled_width + self.COLORS['reset']
        bar += self.COLORS['dim'] + '░' * (width - filled_width) + self.COLORS['reset']
        return bar

    def _create_trend_indicator(self, scores: List[float]) -> str:
        """Create a trend indicator based on score progression."""
        if len(scores) < 2:
            return '─'

        # Simple trend: compare first half vs second half
        mid = len(scores) // 2
        first_half_avg = statistics.mean(scores[:mid]) if mid > 0 else scores[0]
        second_half_avg = statistics.mean(scores[mid:])

        diff = second_half_avg - first_half_avg

        if diff > 0.05:
            return f'{self.COLORS["excellent"]}↗{self.COLORS["reset"]}'  # Improving
        elif diff < -0.05:
            return f'{self.COLORS["poor"]}↘{self.COLORS["reset"]}'  # Declining
        else:
            return f'{self.COLORS["good"]}→{self.COLORS["reset"]}'  # Stable

    def _create_distribution_histogram(
        self, scores: List[float], width: int = 30
    ) -> str:
        """Create a mini histogram of score distribution."""
        if not scores:
            return ' ' * width

        bins = [0] * 10  # 10 bins for 0.0-1.0 range
        for score in scores:
            bin_idx = min(int(score * 10), 9)
            bins[bin_idx] += 1

        max_bin = max(bins) if bins else 1
        if max_bin == 0:
            max_bin = 1

        chars = ' ▁▂▃▄▅▆▇█'
        histogram = ''
        for count in bins:
            char_idx = min(int((count / max_bin) * (len(chars) - 1)), len(chars) - 1)
            histogram += chars[char_idx]

        return histogram

    def _calculate_metric_insights(self, metric_stats: Dict) -> Dict[str, Any]:
        """Calculate advanced insights for metrics."""
        insights = {}

        for name, data in metric_stats.items():
            scores = data['valid_scores']  # Use only valid scores
            if not scores:
                continue

            try:
                std = statistics.stdev(scores) if len(scores) > 1 else 0
            except AttributeError:
                std = 0

            insight = {
                'avg': statistics.mean(scores),
                'median': statistics.median(scores),
                'std_dev': std,
                'min': min(scores),
                'max': max(scores),
                'pass_rate': (
                    data['passed'] / data['valid_count']
                    if data['valid_count'] > 0
                    else 0
                ),
                'reliability': 1 - std,
                'trend': self._create_trend_indicator(scores),
                'scores': scores,
                'valid_count': data['valid_count'],
                'total_count': data['total_count'],
                'success_rate': (
                    data['valid_count'] / data['total_count']
                    if data['total_count'] > 0
                    else 0
                ),
                'passed': data['passed'],
            }

            insights[name] = insight

        return insights

    def _calculate_analysis_insights(self, analysis_metrics: Dict) -> Dict[str, Any]:
        """Calculate insights for analysis/classification metrics (no numeric scores)."""
        insights = {}

        for name, data in analysis_metrics.items():
            if data['total_count'] == 0:
                continue

            insights[name] = {
                'total_count': data['total_count'],
                'successful_count': data['successful_count'],
                'success_rate': (
                    data['successful_count'] / data['total_count']
                    if data['total_count'] > 0
                    else 0
                ),
                'category': data.get('category', 'analysis'),
            }

        return insights

    def _is_successful_metric(self, score_result) -> bool:
        """
        Check if a metric execution was successful.

        A metric is considered successful if:
        - It has a valid numeric score (for SCORE metrics), OR
        - It is an analysis/classification metric that ran without error
        """
        # Check for execution error in explanation
        explanation = getattr(score_result, 'explanation', '') or ''
        if explanation.startswith('Error executing metric'):
            return False

        # Analysis and classification metrics are successful if they ran without error
        metric_category = getattr(score_result, 'metric_category', 'score')
        if metric_category in ('analysis', 'classification'):
            return True

        # For score metrics, check if the score is valid
        return self._is_valid_score(score_result.score)

    def _calculate_overall_stats(
        self, results: List['TestResult'], total_task_runs: int = None
    ) -> Dict[str, Any]:
        """Calculate overall statistics across all test results."""
        total_expected_metrics = 0
        total_successful_metrics = 0

        # Count metrics per result
        for result in results:
            total_expected_metrics += len(result.score_results)
            for score in result.score_results:
                if self._is_successful_metric(score):
                    total_successful_metrics += 1

        stats = {
            'total_expected_metrics': total_expected_metrics,
            'total_successful_metrics': total_successful_metrics,
            'metric_success_rate': (
                total_successful_metrics / total_expected_metrics
                if total_expected_metrics > 0
                else 0
            ),
        }

        # Only include task runs if provided
        if total_task_runs is not None:
            stats['total_task_runs'] = total_task_runs

        return stats

    def _print_summary_box_enhanced(
        self, overall_stats: Dict[str, Any], time_str: str, header_width: int
    ):
        """Print enhanced summary box with metric success rates and optional task runs."""
        # Top border
        print(
            f'{self.COLORS["accent"]}╔{"═" * (header_width - 2)}╗{self.COLORS["reset"]}'
        )

        # Title line - centered
        title_text = 'SUMMARY'
        title_padding = (header_width - len(title_text) - 2) // 2
        remaining_padding = header_width - len(title_text) - title_padding - 2
        print(
            f'{self.COLORS["accent"]}║{" " * title_padding}{self.COLORS["bold"]}{title_text}{self.COLORS["reset"]}{" " * remaining_padding}{self.COLORS["accent"]}║{self.COLORS["reset"]}'
        )

        # Separator
        print(
            f'{self.COLORS["accent"]}╠{"═" * (header_width - 2)}╣{self.COLORS["reset"]}'
        )

        # Task Runs line (only if total_task_runs is provided)
        if 'total_task_runs' in overall_stats:
            task_runs_text = f'{overall_stats["total_task_runs"]:,} task runs'
            task_runs_base = f' Task Runs: {task_runs_text}'
            task_runs_padding = header_width - len(task_runs_base) - 3
            print(
                f'{self.COLORS["accent"]}║{self.COLORS["reset"]}  Task Runs: {self.COLORS["bold"]}{task_runs_text}{self.COLORS["reset"]}{" " * task_runs_padding}{self.COLORS["accent"]}║{self.COLORS["reset"]}'
            )

        # Metric Success Rate line
        success_rate = overall_stats['metric_success_rate']
        success_text = f'{overall_stats["total_successful_metrics"]}/{overall_stats["total_expected_metrics"]} ({success_rate:.1%})'
        success_base = f' Metrics Success: {success_text}'
        success_padding = header_width - len(success_base) - 3
        success_color = self._get_performance_color(success_rate)
        print(
            f'{self.COLORS["accent"]}║{self.COLORS["reset"]}  Metrics Success: {success_color}{self.COLORS["bold"]}{success_text}{self.COLORS["reset"]}{" " * success_padding}{self.COLORS["accent"]}║{self.COLORS["reset"]}'
        )

        # Execution Time line
        time_base = f' Execution Time: {time_str}'
        time_padding = header_width - len(time_base) - 3
        print(
            f'{self.COLORS["accent"]}║{self.COLORS["reset"]}  Execution Time: {self.COLORS["bold"]}{time_str}{self.COLORS["reset"]}{" " * time_padding}{self.COLORS["accent"]}║{self.COLORS["reset"]}'
        )

        # Analysis Complete line
        complete_base = ' Analysis Complete: ✓'
        complete_padding = header_width - len(complete_base) - 3
        print(
            f'{self.COLORS["accent"]}║{self.COLORS["reset"]}  Analysis Complete: {self.COLORS["excellent"]}{self.COLORS["bold"]}✓{self.COLORS["reset"]}{" " * complete_padding}{self.COLORS["accent"]}║{self.COLORS["reset"]}'
        )

        # Bottom border
        print(
            f'{self.COLORS["accent"]}╚{"═" * (header_width - 2)}╝{self.COLORS["reset"]}\n'
        )

    def _print_header(self, overall_stats: Dict[str, Any], total_time: float):
        """Print professional header section."""
        time_str = time.strftime('%H:%M:%S', time.gmtime(total_time))

        # Dynamic header based on terminal width
        header_width = min(self.terminal_width - 4, 80)

        print(f'\n{self.COLORS["header"]}{self.COLORS["bold"]}')
        print('=' * header_width)
        print('EVALUATION REPORT'.center(header_width))
        print('=' * header_width)
        print(f'{self.COLORS["reset"]}')
        self._print_summary_box_enhanced(
            overall_stats, time_str, header_width=header_width
        )
        print(f'{self.COLORS["dim"]}{"─" * 50}{self.COLORS["reset"]}\n')

    def _print_metric_details(self, insights: Dict):
        """Print detailed metric analysis."""
        print(
            f'{self.COLORS["header"]}{self.COLORS["bold"]}📋 DETAILED METRIC ANALYSIS{self.COLORS["reset"]}'
        )
        print(f'{self.COLORS["dim"]}{"─" * 50}{self.COLORS["reset"]}\n')

        for name, insight in insights.items():
            color = self._get_performance_color(insight['avg'])

            # format name
            name = normalize_metric_name(name)
            # Metric header
            print(
                f'{color}{self.COLORS["bold"]}▶ {name}{self.COLORS["reset"]} {insight["trend"]}'
            )

            # Performance bar
            avg_score = insight['avg']
            if avg_score <= 1.0:
                bar = self._create_bar_chart(avg_score, 1.0, 25)
                print(
                    f'  Average Performance: {bar} {color}{avg_score:.2f}{self.COLORS["reset"]}'
                )
            else:
                print(
                    f'  Average Performance: {color}{avg_score:.2f}{self.COLORS["reset"]}'
                )

            # Key metrics in columns
            print(
                '  📊 Statistics: '
                + f'Median: {insight["median"]:.2f} | '
                + f'Range: {insight["min"]:.2f}-{insight["max"]:.2f} | '
                + f'Std Dev: {insight["std_dev"]:.2f}'
            )

            # Metric success rate (successful runs / total expected runs)
            success_rate = insight['success_rate']
            success_color = self._get_performance_color(success_rate)
            print(
                f'  🔧 Metric Success: {success_color}'
                + f'{insight["valid_count"]}/{insight["total_count"]} ({success_rate:.1%}){self.COLORS["reset"]} '
                + 'metrics ran successfully'
            )

            # Pass rate (passed / successful runs)
            print(
                f'  ✅ Pass Rate:      {self._get_performance_color(insight["pass_rate"])}'
                + f'{insight["pass_rate"]:.1%}{self.COLORS["reset"]} '
                + f'({insight["passed"]} passed out of {insight["valid_count"]} successful)'
            )

            # Distribution if enabled
            if self.show_distribution:
                scores = insight.get('scores', [])
                try:
                    histogram = self._create_distribution_histogram(scores)
                except:
                    histogram = 'NULL'
                print(f'  📈 Distribution: [{histogram}] (0.0 ──── 1.0)')

            print()

    def _print_analysis_metrics(self, insights: Dict):
        """Print analysis/classification metrics section (metrics without numeric pass/fail)."""
        if not insights:
            return

        print(
            f'{self.COLORS["header"]}{self.COLORS["bold"]}🔍 ANALYSIS METRICS{self.COLORS["reset"]}'
        )
        print(f'{self.COLORS["dim"]}{"─" * 50}{self.COLORS["reset"]}\n')

        for name, insight in insights.items():
            name = normalize_metric_name(name)
            success_rate = insight['success_rate']
            success_color = self._get_performance_color(success_rate)
            category = insight.get('category', 'analysis')

            # Category-specific descriptions
            if category == 'classification':
                category_desc = 'Classification (categorical labels, no numeric score)'
            else:
                category_desc = 'Analysis (qualitative insights, no numeric score)'

            print(
                f'{self.COLORS["accent"]}{self.COLORS["bold"]}▶ {name}{self.COLORS["reset"]}'
            )
            print(
                f'  🔧 Execution Success: {success_color}'
                + f'{insight["successful_count"]}/{insight["total_count"]} ({success_rate:.1%}){self.COLORS["reset"]} '
                + 'ran successfully'
            )
            print(
                f'  📝 Category: {self.COLORS["dim"]}{category_desc}{self.COLORS["reset"]}'
            )
            print()

    def execute(
        self,
        results: List['TestResult'],
        total_time: float,
        total_task_runs: int = None,
    ) -> Dict[str, Any]:
        """
        Execute comprehensive metric summary.

        Args:
            results: List of test results to analyze
            total_time: Total execution time in seconds
            total_task_runs: Optional total number of task runs (if different from len(results))
        """
        if not results:
            print(
                f'{self.COLORS["warning"]}⚠️  No results to summarize.{self.COLORS["reset"]}'
            )
            return {}

        # Calculate overall statistics
        overall_stats = self._calculate_overall_stats(results, total_task_runs)

        # Aggregate metric statistics - only count valid scores
        metric_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'valid_scores': [],
                'passed': 0,
                'valid_count': 0,
                'total_count': 0,
            }
        )

        # Track analysis/classification metrics separately
        analysis_metrics: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                'total_count': 0,
                'successful_count': 0,
                'category': 'analysis',
            }
        )

        for result in results:
            for score in result.score_results:
                metric_category = getattr(score, 'metric_category', 'score')

                # Handle analysis and classification metrics separately
                if metric_category in ('analysis', 'classification'):
                    analysis_metrics[score.name]['total_count'] += 1
                    analysis_metrics[score.name]['category'] = metric_category
                    # These metrics are successful if they ran without error
                    # (explanation doesn't indicate an execution failure)
                    explanation = getattr(score, 'explanation', '') or ''
                    if not explanation.startswith('Error executing metric'):
                        analysis_metrics[score.name]['successful_count'] += 1
                    continue

                metric_stats[score.name]['total_count'] += 1

                if self._is_valid_score(score.score):
                    metric_stats[score.name]['valid_scores'].append(score.score)
                    metric_stats[score.name]['valid_count'] += 1
                    if hasattr(score, 'passed') and score.passed:
                        metric_stats[score.name]['passed'] += 1

        # Calculate insights
        insights = self._calculate_metric_insights(metric_stats)
        analysis_insights = self._calculate_analysis_insights(analysis_metrics)

        # Display sections
        self._print_header(overall_stats, total_time)
        self._print_metric_details(insights)
        self._print_analysis_metrics(analysis_insights)

        # Footer
        print(f'\n{self.COLORS["dim"]}{"─" * 60}{self.COLORS["reset"]}')
        print(
            f'{self.COLORS["accent"]}📊 Report generated at {time.strftime("%Y-%m-%d %H:%M:%S")}{self.COLORS["reset"]}'
        )
        print(f'{self.COLORS["dim"]}{"─" * 60}{self.COLORS["reset"]}\n')
        return insights


class HierarchicalSummary:
    """
    Generates summary with rich insights for each node in
    the EvalTree across a batch of results.
    """

    @staticmethod
    def _is_valid_score(score) -> bool:
        """Check if a score is valid (not None or NaN)."""
        if score is None:
            return False
        if isinstance(score, float) and math.isnan(score):
            return False
        return True

    @staticmethod
    def _create_distribution_histogram(scores: List[float], bins: int = 10) -> str:
        """Create a mini histogram of score distribution."""
        if not scores:
            return ' ' * bins

        # Create bins for 0.0-1.0 range
        bin_counts = [0] * bins
        for score in scores:
            # Clip score to [0, 1] range and find appropriate bin
            clip_score = max(0, min(1, score))
            bin_idx = min(int(clip_score * bins), bins - 1)
            bin_counts[bin_idx] += 1

        max_count = max(bin_counts) if bin_counts else 1
        if max_count == 0:
            max_count = 1

        # Use block characters for visualization
        chars = ' ▁▂▃▄▅▆▇█'
        histogram = ''
        for count in bin_counts:
            char_idx = min(int((count / max_count) * (len(chars) - 1)), len(chars) - 1)
            histogram += chars[char_idx]

        return histogram

    @staticmethod
    def _calculate_percentiles(scores: List[float]) -> Dict[str, float]:
        """Calculate key percentiles for the scores."""
        if not scores:
            return {'p25': 0, 'p75': 0, 'p90': 0, 'p95': 0}

        return {
            'p25': np.percentile(scores, 25),
            'p75': np.percentile(scores, 75),
            'p90': np.percentile(scores, 90),
            'p95': np.percentile(scores, 95),
        }

    @staticmethod
    def _calculate_consistency_score(scores: List[float]) -> float:
        """Calculate consistency as 1 - coefficient of variation."""
        if len(scores) <= 1:
            return 1.0

        mean_score = statistics.mean(scores)
        if mean_score == 0:
            return 1.0

        std_dev = statistics.stdev(scores)
        cv = std_dev / mean_score  # Coefficient of variation
        return max(0, 1 - cv)  # Convert to consistency score

    @staticmethod
    def _get_performance_tier(score: float) -> str:
        """Categorize performance into tiers."""
        if score >= 0.90:
            return 'excellent'
        elif score >= 0.75:
            return 'good'
        elif score >= 0.60:
            return 'fair'
        else:
            return 'poor'

    def execute(self, results: List, total_time: float) -> Dict[str, Any]:
        """Processes the hierarchical results and creates enhanced summary for each node."""
        node_stats = defaultdict(
            lambda: {
                'scores': [],
                'passes': [],
                'costs': [],
                'node_type': None,
                'source': None,
            }
        )

        if not results:
            return {}

        total_runs = len(results)

        # Collect data for each node
        for result in results:
            for row in result.score_results:
                node_name = row.name
                node_stats[node_name]['node_type'] = row.type
                if row.source:
                    node_stats[node_name]['source'] = row.source

                if self._is_valid_score(row.score):
                    node_stats[node_name]['scores'].append(row.score)
                if row.passed is not None:
                    node_stats[node_name]['passes'].append(row.passed)
                if row.cost_estimate is not None:
                    node_stats[node_name]['costs'].append(row.cost_estimate)

        summary = {}
        for name, data in node_stats.items():
            scores = np.array(data['scores'])
            passes = data['passes']
            costs = np.array(data['costs'])

            valid_count = len(scores)
            pass_count = sum(passes)

            # Basic statistics
            avg_score = float(np.mean(scores)) if valid_count > 0 else 0.0
            median_score = float(np.median(scores)) if valid_count > 0 else 0.0
            std_dev = float(np.std(scores)) if valid_count > 1 else 0.0
            min_score = float(np.min(scores)) if valid_count > 0 else 0.0
            max_score = float(np.max(scores)) if valid_count > 0 else 0.0

            # Enhanced insights
            percentiles = self._calculate_percentiles(scores.tolist())
            consistency = self._calculate_consistency_score(scores.tolist())
            performance_tier = self._get_performance_tier(avg_score)

            # Create distribution and trend
            distribution = self._create_distribution_histogram(scores.tolist())
            # trend = self._create_trend_indicator(scores.tolist())

            # Calculate rates
            pass_rate = pass_count / len(passes) if passes else 0.0
            success_rate = valid_count / total_runs if total_runs > 0 else 0.0

            # Cost analysis
            avg_cost = float(np.mean(costs)) if len(costs) > 0 else 0.0
            total_cost = float(np.sum(costs)) if len(costs) > 0 else 0.0

            summary[name] = {
                # Core metrics (existing)
                'avg_score': avg_score,
                'median_score': median_score,
                'std_dev': std_dev,
                'min_score': min_score,
                'max_score': max_score,
                'pass_rate': pass_rate,
                'avg_cost': avg_cost,
                'total_cost': total_cost,
                'valid_count': valid_count,
                'total_runs': total_runs,
                'success_rate': success_rate,
                'total_time': total_time,
                # Enhanced insights (new)
                'percentiles': percentiles,
                'consistency_score': consistency,
                'performance_tier': performance_tier,
                'distribution': distribution,
                # 'trend_indicator': trend,
                'coefficient_variation': std_dev / avg_score if avg_score > 0 else 0,
                'score_range': max_score - min_score if valid_count > 0 else 0,
                'quartile_range': percentiles['p75'] - percentiles['p25']
                if valid_count > 0
                else 0,
                # Metadata
                'node_type': data['node_type'],
                'source': data['source'],
                'raw_scores': scores.tolist() if valid_count > 0 else [],
                # Summary text for display
                'summary_text': self._generate_summary_text(
                    consistency, performance_tier, valid_count, total_runs
                ),
            }

        return summary

    @staticmethod
    def _generate_summary_text(
        consistency: float, performance_tier: str, valid_count: int, total_runs: int
    ) -> str:
        """Generate a human-readable summary text for the node."""

        # Performance description
        perf_desc = {
            'excellent': 'Outstanding',
            'good': 'Strong',
            'fair': 'Moderate',
            'poor': 'Needs Improvement',
        }.get(performance_tier, 'Unknown')

        # Consistency description
        if consistency >= 0.9:
            consistency_desc = 'Very Consistent'
        elif consistency >= 0.7:
            consistency_desc = 'Consistent'
        elif consistency >= 0.5:
            consistency_desc = 'Moderate Variance'
        else:
            consistency_desc = 'High Variance'

        # Success rate description
        if valid_count == total_runs:
            reliability_desc = 'Fully Reliable'
        elif valid_count / total_runs >= 0.9:
            reliability_desc = 'Highly Reliable'
        elif valid_count / total_runs >= 0.7:
            reliability_desc = 'Mostly Reliable'
        else:
            reliability_desc = 'Unreliable'

        return f'{perf_desc} • {consistency_desc} • {reliability_desc}'
