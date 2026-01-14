from typing import Tuple

import numpy as np
import pandas as pd


class LatencyAnalyzer:
    """
    Latency analysis for AXION for visualizing and summarizing latency metrics.
    """

    def __init__(
        self,
        latency: pd.Series,
        name: str = 'Latency',
        plot_title: str = 'Latency Distribution',
        color_palette: dict = None,
    ):
        """
        Initialize the LatencyAnalyzer with latency data.

        Args:
            latency: Pandas Series containing latency values (in seconds)
            name: Descriptive name for the latency metric (default: "Latency")
            plot_title: Descriptive name for the latency plot title (default: "Latency Distribution")
            color_palette: Optional dictionary to customize colors. Available keys:
                - 'primary': Main text and consistency indicator (default: '#1e3a8a')
                - 'secondary': Histogram bars (default: '#3b82f6')
                - 'accent': Mean line (default: '#8b5cf6')
                - 'success': Median line (default: '#10b981')
                - 'warning': P95 line (default: '#f59e0b')
                - 'danger': P99 line (default: '#ef4444')
                - 'background': Figure background (default: '#f8fafc')
                - 'grid': Grid lines (default: '#e2e8f0')
                - 'text': Primary text (default: '#1e293b')
                - 'text_light': Secondary text (default: '#64748b')
                - 'shadow': Drop shadow color (default: '#1e293b')
        """
        self.latency = latency
        self.name = name
        self.plot_title = plot_title

        # Default color palette
        self.default_colors = {
            'primary': '#1e3a8a',  # Deep navy
            'secondary': '#3b82f6',  # Bright blue
            'accent': '#8b5cf6',  # Purple
            'success': '#10b981',  # Emerald
            'warning': '#f59e0b',  # Amber
            'danger': '#ef4444',  # Red
            'background': '#f8fafc',  # Soft gray
            'grid': '#e2e8f0',  # Light gray
            'text': '#1e293b',  # Dark slate
            'text_light': '#64748b',  # Medium slate
            'shadow': '#1e293b',  # Shadow color
        }

        # Merge user colors with defaults
        if color_palette:
            self.colors = {**self.default_colors, **color_palette}
        else:
            self.colors = self.default_colors

    def plot_distribution(
        self,
        bins: int = 30,
        figsize: Tuple[int, int] = (16, 9),
        show_legend: bool = True,
        show_stats_panel: bool = True,
    ):
        """
        Generate a latency distribution plot.

        Args:
            bins: Number of histogram bins (default: 30)
            figsize: Figure size as (width, height) tuple (default: (16, 9))
            show_legend: Whether to display the consistency legend (default: True)
            show_stats_panel: Whether to display the statistics panel (default: True)

        Returns:
            Tuple of (figure, axes) objects for further customization if needed
        """
        # Import visualization libraries only when needed
        try:
            import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]  # type: ignore[import-untyped]
            import seaborn as sns  # pyright: ignore[reportMissingImports]  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                'Visualization dependencies not installed. '
                'Install with: pip install matplotlib seaborn'
            ) from e

        # Ultra-modern styling with custom parameters
        sns.set_theme(style='white', context='notebook')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'Helvetica']

        # Use instance colors
        colors = self.colors

        # Create figure with custom background
        fig = plt.figure(figsize=figsize, facecolor=colors['background'])
        ax = fig.add_subplot(111, facecolor='white')

        # Calculate comprehensive statistics
        stats = {
            'mean': self.latency.mean(),
            'median': self.latency.median(),
            'std': self.latency.std(),
            'min': self.latency.min(),
            'max': self.latency.max(),
            'p25': self.latency.quantile(0.25),
            'p75': self.latency.quantile(0.75),
            'p90': self.latency.quantile(0.90),
            'p95': self.latency.quantile(0.95),
            'p99': self.latency.quantile(0.99),
            'count': len(self.latency),
        }

        # Create histogram data
        counts, bin_edges = np.histogram(self.latency, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Normalize to percentage
        counts_pct = (counts / len(self.latency)) * 100

        # Create bars with 3D drop shadow effect (with error handling)
        try:
            from matplotlib.patches import Rectangle

            for i, (center, count, left, right) in enumerate(
                zip(bin_centers, counts_pct, bin_edges[:-1], bin_edges[1:])
            ):
                bar_width = (right - left) * 0.95
                bar_left = center - bar_width / 2

                shadow_offset_x = (
                    stats['max'] - stats['min']
                ) * 0.008  # 0.8% of x-range
                shadow_offset_y = ax.get_ylim()[1] * 0.003  # 0.3% of y-range

                shadow = Rectangle(
                    (bar_left + shadow_offset_x, -shadow_offset_y),
                    bar_width,
                    count + shadow_offset_y,
                    facecolor=colors['shadow'],  # Use configurable shadow color
                    edgecolor='none',
                    alpha=0.25,
                    zorder=1,
                )
                ax.add_patch(shadow)

                # Base bar
                main_bar = ax.bar(  # noqa: F841
                    center,
                    count,
                    width=bar_width,
                    color=colors['secondary'],  # Bright blue
                    alpha=0.85,
                    edgecolor='white',
                    linewidth=1.8,
                    zorder=2,
                )

                # Add subtle top highlight for extra depth
                if count > 0:
                    highlight = Rectangle(
                        (bar_left, count * 0.92),
                        bar_width,
                        count * 0.08,
                        facecolor='white',
                        edgecolor='none',
                        alpha=0.2,
                        zorder=3,
                    )
                    ax.add_patch(highlight)
        except ImportError:
            # Fallback to simple bar plot if Rectangle is not available
            ax.bar(
                bin_centers,
                counts_pct,
                width=(bin_edges[1] - bin_edges[0]) * 0.95,
                color=colors['secondary'],
                alpha=0.85,
                edgecolor='white',
                linewidth=1.8,
                zorder=2,
            )

        # Calculate KDE using numpy (Gaussian kernel)
        def gaussian_kde(data, x_range, bandwidth=None):
            """Simple Gaussian KDE implementation using numpy"""
            data = np.asarray(data)

            # Silverman's rule of thumb for bandwidth
            if bandwidth is None:
                n = len(data)
                if len(data) <= 1:
                    std = 0
                else:
                    std = np.std(data, ddof=1)
                bandwidth = 1.06 * std * n ** (-1 / 5)

            # Calculate KDE at each point in x_range
            kde_values = np.zeros_like(x_range)

            if bandwidth == 0 or np.isnan(bandwidth) or np.isinf(bandwidth):
                unique_val = data[0]  # All values are the same
                closest_idx = np.argmin(np.abs(x_range - unique_val))

                # Calculate spacing safely
                if len(x_range) > 1:
                    spacing = x_range[1] - x_range[0]
                    if spacing > 0:
                        kde_values[closest_idx] = 1.0 / spacing
                    else:
                        kde_values[closest_idx] = 1.0
                else:
                    kde_values[closest_idx] = 1.0
            else:
                for i, x in enumerate(x_range):
                    # Gaussian kernel
                    diff = (x - data) / bandwidth
                    kernel = np.exp(-0.5 * diff**2) / (bandwidth * np.sqrt(2 * np.pi))
                    kde_values[i] = np.sum(kernel) / len(data)

            return kde_values

        x_range = np.linspace(stats['min'], stats['max'], 500)
        kde_values = gaussian_kde(self.latency, x_range)
        # Scale KDE to match histogram scale
        kde_scaled = (
            kde_values
            * len(self.latency)
            * (bin_edges[1] - bin_edges[0])
            / len(self.latency)
            * 100
        )

        # Plot KDE with gradient underneath
        ax.plot(
            x_range,
            kde_scaled,
            color=colors['primary'],
            linewidth=3.5,
            alpha=0.9,
            zorder=3,
            label='Density Curve',
        )
        ax.fill_between(
            x_range, 0, kde_scaled, color=colors['secondary'], alpha=0.15, zorder=1
        )

        line_styles = {
            'median': {
                'color': colors['success'],
                'style': '-',
                'width': 3,
                'label': 'Median',
                'priority': 1,
            },
            'mean': {
                'color': colors['accent'],
                'style': '--',
                'width': 3,
                'label': 'Mean',
                'priority': 2,
            },
            'p95': {
                'color': colors['warning'],
                'style': '-.',
                'width': 3,
                'label': 'P95',
                'priority': 3,
            },
            'p99': {
                'color': colors['danger'],
                'style': ':',
                'width': 3,
                'label': 'P99',
                'priority': 4,
            },
        }

        # Calculate positions to avoid overlap
        stat_values = {name: stats[name] for name in line_styles.keys()}
        sorted_stats = sorted(stat_values.items(), key=lambda x: x[1])

        # Assign vertical positions based on proximity
        positions = {}
        y_levels = [0.92, 0.82, 0.72, 0.62]  # Multiple height levels

        for idx, (stat_name, value) in enumerate(sorted_stats):
            # Check if close to previous value (within 10% of range)
            if idx > 0:
                prev_value = sorted_stats[idx - 1][1]
                value_range = stats['max'] - stats['min']
                if abs(value - prev_value) < (value_range * 0.10):
                    # Use alternating heights for close values
                    positions[stat_name] = y_levels[idx % len(y_levels)]
                else:
                    positions[stat_name] = y_levels[0]  # Default high position
            else:
                positions[stat_name] = y_levels[0]

        # Draw lines and annotations
        for stat_name, style in line_styles.items():
            value = stats[stat_name]
            line = ax.axvline(  # noqa: F841
                value,
                color=style['color'],
                linestyle=style['style'],
                linewidth=style['width'],
                alpha=0.85,
                zorder=4,
            )

            # Add floating annotation at calculated position
            y_pos = ax.get_ylim()[1] * positions[stat_name]
            ax.annotate(
                f"{style['label']}\n{value:.2f}s",
                xy=(value, y_pos),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center',
                va='top',
                fontsize=9,
                fontweight='600',
                color=style['color'],
                bbox=dict(
                    boxstyle='round,pad=0.4',
                    facecolor='white',
                    edgecolor=style['color'],
                    linewidth=2,
                    alpha=0.95,
                ),
                zorder=5,
            )

        if show_stats_panel:
            # Create a stunning stats panel with multiple sections
            panel_text = (
                f"DISTRIBUTION METRICS\n"
                f"{'─' * 28}\n"
                f"Samples      {stats['count']:>12,}\n"
                f"\n"
                f"CENTRAL TENDENCY\n"
                f"{'─' * 28}\n"
                f"Mean         {stats['mean']:>12.2f}s\n"
                f"Median       {stats['median']:>12.2f}s\n"
                f"Std Dev      {stats['std']:>12.2f}s\n"
                f"\n"
                f"PERCENTILES\n"
                f"{'─' * 28}\n"
                f"P25          {stats['p25']:>12.2f}s\n"
                f"P50          {stats['median']:>12.2f}s\n"
                f"P75          {stats['p75']:>12.2f}s\n"
                f"P90          {stats['p90']:>12.2f}s\n"
                f"P95          {stats['p95']:>12.2f}s\n"
                f"P99          {stats['p99']:>12.2f}s\n"
                f"\n"
                f"RANGE\n"
                f"{'─' * 28}\n"
                f"Min          {stats['min']:>12.2f}s\n"
                f"Max          {stats['max']:>12.2f}s\n"
                f"Span         {stats['max'] - stats['min']:>12.2f}s"
            )

            ax.text(
                0.985,
                0.97,
                panel_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                fontfamily='monospace',
                color=colors['text'],
                bbox=dict(
                    boxstyle='round,pad=1.0',
                    facecolor='white',
                    edgecolor=colors['grid'],
                    linewidth=2.5,
                    alpha=0.97,
                ),
                zorder=10,
            )
        if show_legend and not pd.isnull(stats['std']):
            # Coefficient of variation
            cv = stats['std'] / stats['mean']

            # Use neutral color instead of judgmental colors
            perf_color = colors['primary']  # Consistent navy blue
            perf_symbol = '●'  # Solid circle

            # Provide descriptive text without judgment
            if cv < 0.2:
                perf_text = 'LOW VARIANCE'
            elif cv < 0.4:
                perf_text = 'MODERATE VARIANCE'
            elif cv < 0.6:
                perf_text = 'HIGH VARIANCE'
            else:
                perf_text = 'VERY HIGH VARIANCE'

            ax.text(
                0.015,
                0.97,
                f'CONSISTENCY\n{perf_symbol} {perf_text}\nCV: {cv:.2%}',
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='top',
                horizontalalignment='left',
                fontweight='bold',
                color=perf_color,
                bbox=dict(
                    boxstyle='round,pad=0.7',
                    facecolor='white',
                    edgecolor=perf_color,
                    linewidth=3,
                    alpha=0.95,
                ),
                zorder=10,
            )

        title = ax.text(  # noqa: F841
            0.015,
            1.08,
            self.plot_title,
            transform=ax.transAxes,
            fontsize=26,
            fontweight='bold',
            color=colors['primary'],
            verticalalignment='top',
        )

        subtitle = ax.text(  # noqa: F841
            0.015,
            1.03,
            f'Performance analysis across {stats["count"]:,} test cases',
            transform=ax.transAxes,
            fontsize=13,
            color=colors['text_light'],
            style='italic',
            verticalalignment='top',
        )

        ax.set_xlabel(
            'Latency (seconds)',
            fontsize=15,
            fontweight='700',
            labelpad=12,
            color=colors['text'],
        )
        ax.set_ylabel(
            'Distribution (%)',
            fontsize=15,
            fontweight='700',
            labelpad=12,
            color=colors['text'],
        )

        ax.grid(
            True,
            alpha=0.25,
            linestyle='-',
            linewidth=0.8,
            color=colors['grid'],
            zorder=0,
        )
        ax.set_axisbelow(True)

        # Add subtle horizontal lines at key percentages
        for pct in [2.5, 5, 7.5, 10]:
            ax.axhline(
                pct,
                color=colors['grid'],
                linewidth=0.5,
                alpha=0.2,
                linestyle=':',
                zorder=0,
            )

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(colors['text_light'])
            ax.spines[spine].set_linewidth(2)

        ax.tick_params(
            axis='both',
            which='major',
            labelsize=11,
            length=8,
            width=2,
            colors=colors['text'],
            labelcolor=colors['text'],
        )

        # Format y-axis to show percentage sign
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        return fig, ax

    def statistics_table(self) -> pd.DataFrame:
        """
        Generate and return a comprehensive statistics table for latency data.

        Returns:
            pd.DataFrame: Single-row DataFrame with statistics, indexed by metric name
        """
        stats_dict = self.calculate_statistics()

        # Create DataFrame with metric name as index
        df = pd.DataFrame([stats_dict], index=[self.name])

        return df

    def calculate_statistics(self) -> dict:
        """
        Calculate latency statistics and return as a dictionary.

        Returns:
            dict: Dictionary containing all latency statistics
        """
        return {
            'count': len(self.latency),
            'mean': self.latency.mean(),
            'p50': np.percentile(self.latency, 50),
            'p90': np.percentile(self.latency, 90),
            'p95': np.percentile(self.latency, 95),
            'p99': np.percentile(self.latency, 99),
            'min': self.latency.min(),
            'max': self.latency.max(),
        }
