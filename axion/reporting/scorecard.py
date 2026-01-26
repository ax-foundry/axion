import asyncio
import base64
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import Field

from axion._core.asyncio import SemaphoreExecutor, run_async_function
from axion._core.logging import get_logger
from axion._core.schema import LLMRunnable, RichBaseModel
from axion._handlers.llm.handler import LLMHandler
from axion._handlers.utils import camel_to_snake
from axion.reporting.explanation_templates import ExplanationTemplates

logger = get_logger(__name__)


class MetricExplanationInput(RichBaseModel):
    metric_name: str = Field(description='The name of the metric being evaluated.')
    score: float = Field(description='The numeric score (0.0 to 1.0).')
    description: Optional[str] = Field(
        None, description='The technical definition of the metric.'
    )


class MetricExplanationOutput(RichBaseModel):
    explanation: str = Field(
        description='A 1-2 sentence business-friendly explanation of what this score means.'
    )


class MetricExplanationHandler(LLMHandler):
    description = 'Generates a business-friendly explanation for a metric score.'
    instruction = 'You are an expert analyst. Given a metric name, its score, and its technical description, write a concise, business-friendly explanation of what this performance level means. Avoid technical jargon where possible.'

    as_structured_llm = True
    input_model = MetricExplanationInput
    output_model = MetricExplanationOutput

    examples = [
        (
            MetricExplanationInput(
                metric_name='MeanReciprocalRank',
                score=0.42,
                description='Calculates the reciprocal rank of the first relevant result found.',
            ),
            MetricExplanationOutput(
                explanation='On average, the first correct answer appeared at position 2.4 in search results (MRR: 0.42), showing moderate ranking quality.'
            ),
        ),
        (
            MetricExplanationInput(
                metric_name='CriteriaCorrectness',
                score=0.70,
                description="Evaluates responses based on user-defined criteria, with support for single-turn and multi-turn conversations.'",
            ),
            MetricExplanationOutput(
                explanation='70.0% of user-defined correctness criteria (specific facts, required logic, constraints) were satisfied, using aspect-based evaluation to verify accuracy requirements.'
            ),
        ),
    ]

    def __init__(
        self,
        instruction: Optional[str] = None,
        llm: Optional[LLMRunnable] = None,
    ):
        super().__init__(llm=llm)
        if instruction:
            self.instruction = instruction


class MetricExplainer:
    """
    Handles the generation of business-friendly explanations for metrics.
    Uses SemaphoreExecutor for batch processing.
    """

    def __init__(
        self,
        metric_definitions: dict = None,
        explanation_callback: Optional[Callable] = None,
        instruction: Optional[str] = None,
        llm: Optional[LLMRunnable] = None,
        max_concurrent: int = 10,
    ):
        self.metric_definitions = metric_definitions if metric_definitions else {}
        self.explanation_callback = explanation_callback
        self.max_concurrent = max_concurrent
        self.llm = llm
        if self.llm:
            self.llm_handler = MetricExplanationHandler(
                instruction=instruction, llm=llm
            )
        else:
            self.llm_handler = None

        self._set_metric_descriptions()

    def _set_metric_descriptions(self) -> None:
        from axion.metrics import metric_registry

        self.registry_descriptions = metric_registry.get_metric_descriptions()

    def _try_static_explanation(self, metric_name, score, type_label) -> Optional[str]:
        if self.explanation_callback:
            try:
                custom_expl = self.explanation_callback(metric_name, score, type_label)
                if custom_expl:
                    return custom_expl
            except Exception:
                pass

        if metric_name in self.metric_definitions:
            return self._format_template(self.metric_definitions[metric_name], score)

        manual_override = ExplanationTemplates.get_manual_explanation(
            metric_name, score
        )
        if manual_override:
            return manual_override

        return None

    @staticmethod
    def _get_fallback_explanation(metric_name, score):
        category_expl = ExplanationTemplates.get_category_explanation(
            metric_name, score
        )
        if category_expl:
            return category_expl
        return f'{metric_name} performance metric. Currently tracking at {score:.1%}.'

    def generate_explanation(self, metric_name, score, type_label):
        """Synchronous single-item generation (Legacy/Fallback)."""
        static = self._try_static_explanation(metric_name, score, type_label)
        if static:
            return static

        description = self._get_registry_description(metric_name)
        if self.llm:
            try:
                input_data = MetricExplanationInput(
                    metric_name=metric_name, score=score, description=description
                )
                result = run_async_function(self.llm_handler.execute, input_data)
                return result.explanation
            except Exception as e:
                logger.info(f'LLM Generation failed for {metric_name}: {e}')

        return self._get_fallback_explanation(metric_name, score)

    def batch_generate_explanations(self, inputs: List[Dict[str, Any]]) -> List[str]:
        """
        Processes a list of explanation requests in parallel using SemaphoreExecutor.
        """
        results: List[Any] = [None] * len(inputs)
        llm_inputs = []  # Stores tuple (original_index, input_model)

        # First pass: Resolve static templates and prepare LLM inputs
        for i, item in enumerate(inputs):
            metric_name = item['metric_name']
            score = item['score']
            type_label = item['type_label']

            static_expl = self._try_static_explanation(metric_name, score, type_label)

            if static_expl:
                results[i] = static_expl
            elif self.llm:
                desc = self._get_registry_description(metric_name)
                input_model = MetricExplanationInput(
                    metric_name=metric_name, score=score, description=desc
                )
                llm_inputs.append((i, input_model))
            else:
                results[i] = self._get_fallback_explanation(metric_name, score)

        # Run all LLM tasks concurrently
        if llm_inputs:
            logger.debug(f'Running {len(llm_inputs)} explanation tasks in parallel...')

            async def _run_batch():
                semaphore = SemaphoreExecutor(max_concurrent=self.max_concurrent)
                tasks = []
                for _, input_model in llm_inputs:
                    tasks.append(semaphore.run(self.llm_handler.execute, input_model))
                try:
                    from tqdm.asyncio import tqdm

                    pbar = tqdm(total=len(tasks), desc='Generating Explanations')

                    async def _wrap(task):
                        try:
                            return await task
                        finally:
                            pbar.update(1)

                    wrapped_tasks = [_wrap(t) for t in tasks]
                    try:
                        return await asyncio.gather(
                            *wrapped_tasks, return_exceptions=True
                        )
                    finally:
                        pbar.close()
                except ImportError:
                    return await asyncio.gather(*tasks, return_exceptions=True)

            # Block here until all async tasks complete
            llm_results = run_async_function(_run_batch)

            for (original_index, _), result in zip(llm_inputs, llm_results):
                if isinstance(result, Exception):
                    logger.warning(
                        f'Batch LLM failed for index {original_index}: {result}'
                    )
                    item = inputs[original_index]
                    results[original_index] = self._get_fallback_explanation(
                        item['metric_name'], item['score']
                    )
                else:
                    results[original_index] = result.explanation

        for i, res in enumerate(results):
            if res is None:
                item = inputs[i]
                results[i] = self._get_fallback_explanation(
                    item['metric_name'], item['score']
                )

        return results

    def _get_registry_description(self, metric_name):
        reg_key = self._lookup(metric_name)
        return self.registry_descriptions.get(reg_key, None)

    @staticmethod
    def _lookup(metric_name: str) -> str:
        reg_key = camel_to_snake(metric_name)
        if 'criteria' in reg_key:
            reg_key = 'answer_criteria'
        return reg_key

    @staticmethod
    def _format_template(template, score):
        try:
            return template.format(score=score)
        except:
            return template


class ScoreCard:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        id_col: str = 'metric_name',
        parent_col: str = 'parent',
        group_meta_cols: list = None,
        value_cols: list = None,
        metric_definitions: dict = None,
        explanation_callback: Optional[Callable] = None,
        instruction: Optional[str] = None,
        llm: Optional[LLMRunnable] = None,
        max_concurrent: int = 10,
    ):
        """
        Initializes the ScoreCard processor.

        Args:
            instruction (str, optional): Custom system prompt for the explanation generation LLM.
            llm (Any, optional): Custom LLM instance to use for explanation generation.
        """
        self.raw_dataframe = dataframe.copy()

        # Configuration mapping
        self.id_col = id_col
        self.parent_col = parent_col

        # Set defaults if not provided
        self.group_meta_cols = group_meta_cols if group_meta_cols else ['metric_type']
        self.value_cols = value_cols if value_cols else ['metric_score', 'weight']

        # Initialize the explainer class with the provided definitions/callback
        self.explainer = MetricExplainer(
            metric_definitions,
            explanation_callback,
            instruction=instruction,
            llm=llm,
            max_concurrent=max_concurrent,
        )

        # Internal state
        self.aggregated_dataframe = None
        self.hierarchical_dataframe = None
        self.parent_map = {}

    def _get_lineage(self, item_name):
        """
        Recursively finds the path from root to the specific node.
        """
        path = [item_name]
        current_node = item_name

        # Safety counter to prevent infinite loops (circular dependencies)
        loop_guard = 0
        max_loops = 20

        while loop_guard < max_loops:
            parent_node = self.parent_map.get(current_node)

            # Stop if parent is NaN or doesn't exist in the map (Root reached)
            if pd.isna(parent_node) or parent_node not in self.parent_map:
                break

            path.append(parent_node)
            current_node = parent_node
            loop_guard += 1

        return path[::-1]  # Reverse to go Root -> Leaf

    def build_scorecard(self):
        """
        Main execution method to transform raw data into a hierarchical MultiIndex dataframe.
        """
        self._aggregate_data()
        self._construct_hierarchy()
        self._format_multiindex()
        return self.hierarchical_dataframe

    def _aggregate_data(self):
        """
        Groups by ID and Parent (plus extra meta cols) to get averages of value columns.
        """
        # distinct keys to group by
        group_keys = [self.id_col, self.parent_col] + self.group_meta_cols

        self.aggregated_dataframe = (
            self.raw_dataframe.groupby(group_keys, dropna=False)[self.value_cols]
            .agg(
                {'metric_score': 'mean', 'weight': 'max'}
            )  # take max for weight for failure issues
            .reset_index()
        )

    def _construct_hierarchy(self):
        """
        Creates the parent mapping and applies the lineage logic.
        """
        if self.aggregated_dataframe is None:
            raise ValueError(
                'Aggregated dataframe is missing. Run _aggregate_data first.'
            )

        # Create mapping of Item -> Parent
        self.parent_map = dict(
            zip(
                self.aggregated_dataframe[self.id_col],
                self.aggregated_dataframe[self.parent_col],
            )
        )

        # Calculate lineage path for every row
        self.aggregated_dataframe['lineage_path'] = self.aggregated_dataframe[
            self.id_col
        ].apply(self._get_lineage)

    def _format_multiindex(self):
        """
        Flattens the lineage paths into separate columns (Level_1, Level_2...)
        and sets the MultiIndex.
        """
        if self.aggregated_dataframe is None:
            raise ValueError('Hierarchy construction missing.')

        # Determine the maximum depth of the tree
        max_depth = self.aggregated_dataframe['lineage_path'].apply(len).max()

        # Dynamically generate level column names
        level_columns = []
        for i in range(max_depth):
            col_name = f'Level_{i + 1}'
            level_columns.append(col_name)

            # Extract the item at index i, or fill with empty string if path is shorter
            self.aggregated_dataframe[col_name] = self.aggregated_dataframe[
                'lineage_path'
            ].apply(lambda x: x[i] if i < len(x) else '')

        # Sort to ensure visual hierarchy and set index
        self.hierarchical_dataframe = self.aggregated_dataframe.sort_values(
            by=level_columns
        )
        self.hierarchical_dataframe = self.hierarchical_dataframe.set_index(
            level_columns
        )

    def generate_styled_dataframe(self) -> pd.DataFrame:
        """
        Generates a styled pandas DataFrame with the same visual appearance as the HTML report.

        Returns:
            pd.io.formats.style.Styler: Styled DataFrame ready for display in Jupyter or export
        """
        if self.hierarchical_dataframe is None:
            raise ValueError('Scorecard data not ready. Run build_scorecard() first.')

        tree_df = self._create_tree_visualization_df()

        # Apply the same styling as HTML output
        styled_df = self._apply_dataframe_styling(tree_df)

        return styled_df

    def generate_html_report(self, output_path: str = None) -> str:
        """
        Generates a fancy HTML scorecard with sparklines and explanations.

        Args:
            output_path (str, optional): If provided, saves the HTML to this file path.

        Returns:
            str: The raw HTML string.
        """
        if self.hierarchical_dataframe is None:
            raise ValueError('Scorecard data not ready. Run build_scorecard() first.')

        tree_df = self._create_tree_visualization_df()
        styled_html = self._apply_html_styling(tree_df)

        full_html = self._wrap_html_template(styled_html)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_html)

        return full_html

    def _create_tree_visualization_df(self) -> pd.DataFrame:
        """Transforms the MultiIndex dataframe into a flattened Tree View dataframe."""
        df = self.hierarchical_dataframe
        if df is None:
            raise ValueError(
                'hierarchical_dataframe is not set. Call aggregate() first.'
            )
        normalized_weights = self._calculate_normalized_weights(df)

        # Check if weight column should be included (all NaN means exclude)
        has_valid_weights = df['weight'].notna().any()

        records = []
        explanation_inputs = []  # Queue for batch processing

        type_col = 'metric_type'
        if self.group_meta_cols:
            if 'metric_type' in self.group_meta_cols:
                type_col = 'metric_type'
            else:
                type_col = self.group_meta_cols[0]

        # Build a lookup map for node types to check parent types efficiently
        node_type_map = {}
        if (
            self.aggregated_dataframe is not None
            and type_col in self.aggregated_dataframe.columns
        ):
            node_type_map = dict(
                zip(
                    self.aggregated_dataframe[self.id_col],
                    self.aggregated_dataframe[type_col].astype(str).str.lower(),
                )
            )

        for idx, row in df.iterrows():
            levels = idx if isinstance(idx, tuple) else (idx,)
            current_level_depth = 0
            metric_name = ''
            for i, val in enumerate(levels):
                if val and val != '':
                    current_level_depth = i
                    metric_name = val

            raw_type = str(row.get(type_col, '')).lower()
            parent_type = ''
            parent_name = ''
            if current_level_depth > 0:
                parent_name = levels[current_level_depth - 1]
                parent_type = node_type_map.get(parent_name, '')

            if metric_name == 'model':
                type_label = 'CATEGORY'
                icon = '🏆'
            elif 'metric' in raw_type:
                type_label = 'METRIC'
                icon = '◆'
            elif 'component' in parent_type and parent_name != 'model':
                type_label = 'SUB-COMPONENT'
                icon = '▶'
            elif 'sub-component' in raw_type or 'sub_component' in raw_type:
                type_label = 'SUB-COMPONENT'
                icon = '▶'
            elif 'component' in raw_type:
                type_label = 'COMPONENT'
                icon = '📊'
            else:
                if current_level_depth == 0:
                    type_label = 'CATEGORY'
                    icon = '🏆'
                elif current_level_depth == 1:
                    type_label = 'COMPONENT'
                    icon = '📊'
                elif current_level_depth == 2:
                    type_label = 'SUB-COMPONENT'
                    icon = '▶'
                else:
                    type_label = 'METRIC'
                    icon = '◆'

            indent_spaces = ' ' * (current_level_depth * 4)
            display_name = f'{indent_spaces}{icon} {metric_name}'

            # Generate Visuals (Fast)
            score_dist = self._get_real_score_distribution(metric_name)
            try:
                dist_sparkline = self._create_distribution_sparkline(
                    score_dist, row['metric_score']
                )
            except ImportError:
                dist_sparkline = ''
                logger.warning(
                    'matplotlib required for sparklines. Install with: pip install matplotlib'
                )

            # Store data for final dataframe - conditionally include weight columns
            record = {
                'Hierarchy': display_name,
                'Type': type_label,
                'Score': row['metric_score'],
                'Distribution': dist_sparkline,
                'raw_idx': idx,
            }

            # Only add weight columns if there are valid weights
            if has_valid_weights:
                record['Weight'] = row['weight']
                record['Normalized Weight'] = normalized_weights.get(idx, 0.0)

            records.append(record)

            # Store data for Explanation Batching
            explanation_inputs.append(
                {
                    'metric_name': metric_name,
                    'score': row['metric_score'],
                    'type_label': type_label,
                }
            )

        # This will run all LLM calls in parallel and wait once
        explanations = self.explainer.batch_generate_explanations(explanation_inputs)

        for i, record in enumerate(records):
            record['Explanation'] = explanations[i]

        return pd.DataFrame(records)

    @staticmethod
    def _calculate_normalized_weights(df):
        """
        Calculates the normalized weight contribution of each node to the root.
        Recursive multiplication of weights up the hierarchy chain.
        """
        normalized = {}

        for idx, row in df.iterrows():
            current_weight = row['weight']

            # Check parents dynamically
            levels = idx if isinstance(idx, tuple) else (idx,)

            # Iterate backwards from current level up to root
            # If current is at index 3, we check parents at 2, 1, 0
            active_indices = [i for i, v in enumerate(levels) if v != '']
            if not active_indices:
                normalized[idx] = current_weight
                continue

            current_depth = active_indices[-1]  # The index of the item itself

            # Multiply by parents
            for d in range(current_depth - 1, -1, -1):
                # Construct parent index: Keep 0..d, fill rest with ''
                parent_list = list(levels)
                # Reset levels after d to ''
                for k in range(d + 1, len(levels)):
                    parent_list[k] = ''

                parent_idx = tuple(parent_list)

                if parent_idx in df.index:
                    current_weight *= df.loc[parent_idx, 'weight']

            normalized[idx] = current_weight

        return normalized

    def _get_real_score_distribution(self, metric_name):
        """
        Fetches the actual array of scores from the raw dataframe for a given metric.
        This allows the sparkline to reflect the real data distribution (variance)
        rather than a simulated Gaussian.
        """
        # Determine which column holds the score (defaulting to the first value col)
        score_col = self.value_cols[0]

        # Safety check if column exists
        if score_col not in self.raw_dataframe.columns:
            return np.array([0.0])

        # Filter the raw dataframe
        mask = self.raw_dataframe[self.id_col] == metric_name
        scores = self.raw_dataframe.loc[mask, score_col].values

        # Remove NaNs if any
        scores = scores[~np.isnan(scores)]

        # Fallback for empty data
        if len(scores) == 0:
            return np.array([0.0])

        return scores

    @staticmethod
    def _create_distribution_sparkline(scores, mean_score):
        """Generates a base64 encoded PNG sparkline."""

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(1.6, 0.5), dpi=100)
        ax = fig.add_subplot(111)

        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        # Use auto-binning or fixed bins depending on sample size
        bins = 20 if len(scores) > 20 else 10

        n, bins, patches = ax.hist(
            scores,
            bins=bins,
            range=(0, 1),
            color='#4a7ba7',
            alpha=0.7,
            edgecolor='#2c5f8d',
            linewidth=0.5,
        )

        ax.axvline(mean_score, color='#d32f2f', linewidth=2, linestyle='--', alpha=0.9)

        ax.set_xlim(0, 1)
        if len(n) > 0:
            ax.set_ylim(0, max(n) * 1.1)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.margins(0)

        buffer = BytesIO()
        plt.savefig(
            buffer,
            format='png',
            dpi=100,
            bbox_inches='tight',
            pad_inches=0,
            transparent=True,
        )
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)

        return f'<img src="data:image/png;base64,{image_base64}" style="width: 120px; height: 40px; display: block; margin: 0 auto;"/>'

    @staticmethod
    def _apply_dataframe_styling(tree_df) -> pd.DataFrame:
        """
        Applies pandas styling to match the HTML report's visual appearance.
        This creates a styled DataFrame that can be displayed in Jupyter or exported.
        """

        def get_row_style(s):
            """Apply row-level background colors based on Type"""
            type_val = tree_df.loc[s.name, 'Type']
            base_colors = {
                'CATEGORY': '#1e3a5f',
                'COMPONENT': '#2c5f8d',
                'SUB-COMPONENT': '#4a7ba7',
                'METRIC': '#e3f2fd',
            }
            bg_color = base_colors.get(type_val, '#e3f2fd')
            text_color = 'white' if type_val != 'METRIC' else 'black'
            font_weight = 'bold' if type_val != 'METRIC' else 'normal'
            return [
                f'background-color: {bg_color}; color: {text_color}; font-weight: {font_weight};'
            ] * len(s)

        # Determine which columns to display based on what's in tree_df
        display_cols = ['Hierarchy', 'Score']
        if 'Weight' in tree_df.columns:
            display_cols.append('Weight')
        if 'Normalized Weight' in tree_df.columns:
            display_cols.append('Normalized Weight')
        display_cols.extend(['Distribution', 'Explanation'])

        # Select display columns (excluding raw_idx)
        display_df = tree_df[display_cols]

        # Build format dict dynamically
        format_dict = {'Score': '{:.2f}'}
        if 'Weight' in display_cols:
            format_dict['Weight'] = '{:.2f}'
        if 'Normalized Weight' in display_cols:
            format_dict['Normalized Weight'] = '{:.3f}'

        styled = (
            display_df.style.format(format_dict)
            .apply(get_row_style, axis=1)
            .hide(axis='index')
            .set_table_styles(
                [
                    # Header styling
                    {
                        'selector': 'th',
                        'props': [
                            ('background-color', '#0d1b2a'),
                            ('color', 'white'),
                            ('padding', '12px'),
                            ('text-align', 'center'),
                            ('font-weight', 'bold'),
                            ('font-size', '14px'),
                            ('border', '1px solid #0d1b2a'),
                        ],
                    },
                    # Cell styling
                    {
                        'selector': 'td',
                        'props': [
                            ('padding', '10px'),
                            ('border', '1px solid #dee2e6'),
                            ('text-align', 'center'),
                            ('vertical-align', 'middle'),
                        ],
                    },
                    # First column (Hierarchy) special styling
                    {
                        'selector': 'td:first-child',
                        'props': [
                            ('text-align', 'left'),
                            ('font-family', 'monospace'),
                            ('white-space', 'pre'),
                            ('padding-left', '15px'),
                        ],
                    },
                    # Last column (Explanation) special styling
                    {
                        'selector': 'td:last-child',
                        'props': [
                            ('text-align', 'left'),
                            ('max-width', '400px'),
                            ('padding', '10px 15px'),
                        ],
                    },
                    # Table overall styling
                    {
                        'selector': 'table',
                        'props': [
                            ('border-collapse', 'collapse'),
                            ('width', '100%'),
                            ('margin', '20px 0'),
                            ('box-shadow', '0 2px 8px rgba(0,0,0,0.1)'),
                        ],
                    },
                    # Hover effect for rows
                    {
                        'selector': 'tr:hover',
                        'props': [
                            ('background-color', '#f5f8fa'),
                            ('transition', 'background-color 0.2s ease'),
                        ],
                    },
                ]
            )
        )

        return styled

    @staticmethod
    def _apply_html_styling(tree_df):
        """Applies pandas styling to the dataframe for HTML export."""

        def get_row_style(s):
            type_val = tree_df.loc[s.name, 'Type']
            base_colors = {
                'CATEGORY': '#1e3a5f',
                'COMPONENT': '#2c5f8d',
                'SUB-COMPONENT': '#4a7ba7',
                'METRIC': '#e3f2fd',
            }
            bg_color = base_colors.get(type_val, '#e3f2fd')
            text_color = 'white' if type_val != 'METRIC' else 'black'
            font_weight = 'bold' if type_val != 'METRIC' else 'normal'
            return [
                f'background-color: {bg_color}; color: {text_color}; font-weight: {font_weight};'
            ] * len(s)

        # Determine which columns to display
        display_cols = ['Hierarchy', 'Score']
        if 'Weight' in tree_df.columns:
            display_cols.append('Weight')
        if 'Normalized Weight' in tree_df.columns:
            display_cols.append('Normalized Weight')
        display_cols.extend(['Distribution', 'Explanation'])

        # Build format dict dynamically
        format_dict = {'Score': '{:.2f}'}
        if 'Weight' in display_cols:
            format_dict['Weight'] = '{:.2f}'
        if 'Normalized Weight' in display_cols:
            format_dict['Normalized Weight'] = '{:.3f}'

        styled = (
            tree_df[display_cols]
            .style.format(format_dict)
            .apply(get_row_style, axis=1)
            .hide(axis='index')
            .set_table_styles(
                [
                    {
                        'selector': 'th',
                        'props': [
                            ('background-color', '#0d1b2a'),
                            ('color', 'white'),
                            ('padding', '12px'),
                        ],
                    },
                    {
                        'selector': 'td',
                        'props': [('padding', '10px'), ('border', '1px solid #dee2e6')],
                    },
                    {
                        'selector': 'td:first-child',
                        'props': [
                            ('text-align', 'left'),
                            ('font-family', 'monospace'),
                            ('white-space', 'pre'),
                        ],
                    },
                ]
            )
        )
        return styled

    @staticmethod
    def _wrap_html_template(styled_df) -> str:
        """Wraps the table in a full HTML page."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Scorecard Report</title>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; padding: 40px; background: #f0f2f5; }}
                .container {{ background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); max-width: 1400px; margin: 0 auto; }}
                h1 {{ color: #1e3a5f; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Evaluation Scorecard Report</h1>
                {styled_df.to_html()}
            </div>
        </body>
        </html>
        """
