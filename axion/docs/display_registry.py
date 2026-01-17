import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


class RegistryItem(Protocol):
    """Protocol for items that can be displayed in a registry."""

    @property
    def config(self) -> Any:
        """Return the configuration object for this registry item."""
        ...


@dataclass
class DisplayConfig:
    """Configuration for how to display registry items."""

    # Basic display properties
    title: str = 'Registry'
    icon: str = '📊'
    description: str = 'Available items in registry'

    # Card display configuration
    show_badges: bool = True
    show_tags: bool = True
    show_filters: bool = True
    show_stats: bool = True

    # Button configuration
    show_details_button: bool = True
    show_usage_button: bool = True
    details_button_text: str = 'ℹ️ More Details'
    usage_button_text: str = '💡 Usage Example'

    # Grid configuration
    min_card_width: str = '300px'
    max_width: str = '1250px'

    # Code theme configuration
    code_theme: str = 'light'  # "light" or "dark"


@dataclass
class CardField:
    """Configuration for a field displayed on a card."""

    key: str
    label: str
    icon: str = ''
    field_type: str = 'text'  # text, list, badge, code
    color_scheme: Optional[str] = None  # For badges: primary, success, warning, danger
    show_count: bool = False  # For lists, show count in parentheses


class BaseRegistryExtractor(ABC):
    """Base class for extracting display information from registry items."""

    @abstractmethod
    def get_title(self, item: Any) -> str:
        """Get the display title for an item."""
        pass

    @abstractmethod
    def get_subtitle(self, item: Any) -> str:
        """Get the subtitle/key for an item."""
        pass

    @abstractmethod
    def get_description(self, item: Any) -> str:
        """Get the description for an item."""
        pass

    @abstractmethod
    def get_tags(self, item: Any) -> List[str]:
        """Get tags for filtering."""
        pass

    @abstractmethod
    def get_fields(self, item: Any) -> Dict[str, Any]:
        """Get all fields to display on the card."""
        pass

    @abstractmethod
    def get_usage_example(self, key: str, item: Any) -> str:
        """Generate usage example code."""
        pass

    @abstractmethod
    def get_details_content(self, key: str, item: Any) -> str:
        """Generate detailed information content."""
        pass


class MetricRegistryExtractor(BaseRegistryExtractor):
    """Extractor for metric registry items."""

    def get_title(self, item: Any) -> str:
        return item.config.name

    def get_subtitle(self, item: Any) -> str:
        return item.config.key

    def get_description(self, item: Any) -> str:
        return item.config.description

    def get_tags(self, item: Any) -> List[str]:
        return item.config.tags

    def get_fields(self, item: Any) -> Dict[str, Any]:
        config = item.config
        return {
            'threshold': {
                'value': config.default_threshold,
                'type': 'threshold_badge',
                'label': 'Threshold',
            },
            'score_range': {
                'value': f'{config.score_range[0]}-{config.score_range[1]}',
                'type': 'badge',
                'label': 'Range',
                'color': 'primary',
            },
            'required_fields': {
                'value': config.required_fields,
                'type': 'field_list',
                'label': 'Required Fields',
                'field_class': 'required',
            },
            'optional_fields': {
                'value': getattr(config, 'optional_fields', []),
                'type': 'field_list',
                'label': 'Optional Fields',
                'field_class': 'optional',
            },
        }

    def get_usage_example(self, key: str, item: Any) -> str:
        class_name = ''.join(word.capitalize() for word in key.split('_'))

        return f"""# Usage example for {key}
from axion.metrics import {class_name}

# Initialize the metric
metric = {class_name}()

# Evaluate with your data
from axion.dataset import DatasetItem

data_item = DatasetItem(
    query = "What is the infield fly rule in baseball?",
    actual_output = "The infield fly rule prevents the defense from intentionally dropping a fly ball to turn a double play.",
    expected_output = "The infield fly rule protects baserunners by declaring the batter out on certain easy pop-ups.",
    retrieved_content = ["The infield fly rule prevents unfair advantage.", "Applies with runners on first and second."],
    latency = 2.13
)
result = await metric.execute(data_item)
print(result.pretty())"""

    def get_multi_turn_usage_example(self, key: str, item: Any) -> str:
        class_name = ''.join(word.capitalize() for word in key.split('_'))

        return f"""# Usage example for {key}
from axion.metrics import {class_name}
from axion.dataset import DatasetItem
from axion._core.schema import HumanMessage, AIMessage, ToolMessage, ToolCall
from axion.dataset_schema import MultiTurnConversation

# Initialize the metric
metric = {class_name}()

data_item = DatasetItem(
    conversation = MultiTurnConversation(
            messages=[
                HumanMessage(content="What's the weather like in San Francisco? My name is Bob."),
                AIMessage(tool_calls=[ToolCall(id="call_123", name="get_weather", args={{"city": "San Francisco"}})]),
                ToolMessage(tool_call_id="call_123", content='{{"temperature": "68F", "condition": "Sunny"}}'),
                AIMessage(content="Hi Bob, the weather in San Francisco is currently 68°F and Sunny.")
            ]
        )
)
result = await metric.execute(data_item)
print(result.pretty())"""

    def get_tool_usage_example(self, key: str, item: Any) -> str:
        class_name = ''.join(word.capitalize() for word in key.split('_'))

        return f"""# Usage example for {key}
from axion.metrics import {class_name}

# Initialize the metric
metric = {class_name}()

# Evaluate with your data
from axion.dataset import DatasetItem
from axion._core.schema import ToolCall

data_item = DatasetItem(
    tools_called=[
            ToolCall(name="get_weather", args={{"city": "San Francisco"}}),
            ToolCall(name="format_response", args={{"data": "weather_data"}})
        ],
        expected_tools=[
            ToolCall(name="get_weather", args={{"city": "San Francisco"}})
        ]
)
result = await metric.execute(data_item)
print(result.pretty())"""

    def get_criteria_usage_example(self, key: str, item: Any) -> str:
        class_name = ''.join(word.capitalize() for word in key.split('_'))

        return f"""# Usage example for {key}
from axion.metrics import {class_name}

# Initialize the metric
metric = {class_name}()

# Evaluate with your data
from axion.dataset import DatasetItem

data_item = DatasetItem(
    query = "What is the infield fly rule in baseball?",
    actual_output = "The infield fly rule prevents the defense from intentionally dropping a fly ball to turn a double play.",
    additional_input={{"Complete": "Explain the purpose and conditions of the infield fly rule."}} # Metric defaults to "Complete"
)
result = await metric.execute(data_item)
print(result.pretty())"""

    def get_citation_usage_example(self, key: str, item: Any) -> str:
        class_name = ''.join(word.capitalize() for word in key.split('_'))

        return f"""# Usage example for {key}
from axion.metrics import {class_name}

# Initialize the metric
metric = {class_name}()

# Evaluate with your data
from axion.dataset import DatasetItem

data_item = DatasetItem(
    query = "What is the infield fly rule in baseball?",
    actual_output = "The infield fly rule prevents the defense from intentionally dropping a fly ball to turn a double play. For more details, refer to: - [Infield Fly Rule](https://www.mlb.com/glossary/rules/infield-fly) [Official MLB Rules](https://www.mlb.com/official-rules)",
)
result = await metric.execute(data_item)
print(result.pretty())"""

    def get_retrieval_example(self, key: str, item: Any) -> str:
        class_name = ''.join(word.capitalize() for word in key.split('_'))

        return f"""# Usage example for {key}
from axion.metrics import {class_name}

# Initialize the metric
metric = {class_name}()

# Evaluate with your data
from axion.dataset import DatasetItem

data_item = DatasetItem(
    actual_ranking = [{{"id": "doc1", "relevance": 0.9}}, {{"id": "doc2", "relevance": 0.8}}],
    expected_reference = [{{"id": "doc1"}}, {{"id": "doc3"}}],
)
result = await metric.execute(data_item)
print(result.pretty())"""

    def get_details_content(self, key: str, item: Any) -> str:
        config = item.config
        return f"""
        <div class="details-content">
            <h4>Configuration Details</h4>
            <ul>
                <li><strong>Key:</strong> <code>{config.key}</code></li>
                <li><strong>Default Threshold:</strong> {config.default_threshold}</li>
                <li><strong>Score Range:</strong> {config.score_range[0]} - {
            config.score_range[1]
        }</li>
                <li><strong>Required Fields:</strong> {len(config.required_fields)}</li>
                <li><strong>Optional Fields:</strong> {
            len(getattr(config, 'optional_fields', []))
        }</li>
            </ul>

            <h4>Field Details</h4>
            <div class="field-details">
                <h5>Required:</h5>
                <ul>
                    {
            ''.join(
                f'<li><code>{field}</code></li>' for field in config.required_fields
            )
        }
                </ul>

                {
            f'''<h5>Optional:</h5>
                <ul>
                    {''.join(f'<li><code>{field}</code></li>' for field in config.optional_fields)}
                </ul>'''
            if getattr(config, 'optional_fields', [])
            else ''
        }
            </div>

            <h4>Usage Notes</h4>
            <p>This metric evaluates: {config.description}</p>
        </div>
        """


class GenericRegistryDisplay:
    """Generic registry display that can work with any type of registry."""

    def __init__(
        self,
        extractor: BaseRegistryExtractor,
        display_config: Optional[DisplayConfig] = None,
    ):
        self.extractor = extractor
        self.config = display_config or DisplayConfig()

    def _generate_header_html(self, registry: Dict) -> str:
        """Generate the header section."""
        total_items = len(registry)
        all_tags = set()

        for item in registry.values():
            tags = self.extractor.get_tags(item)
            all_tags.update(tags)

        stats_html = ''
        if self.config.show_stats:
            stats_html = f"""
            <div class="registry-stats">
                <span class="stat-item">
                    <span class="stat-number">{total_items}</span>
                    <span class="stat-label">Total Items</span>
                </span>
                <span class="stat-item">
                    <span class="stat-number">{len(all_tags)}</span>
                    <span class="stat-label">Unique Tags</span>
                </span>
            </div>
            """

        return f"""
        <div class="registry-container">
            <div class="registry-header">
                <div class="registry-icon">{self.config.icon}</div>
                <div class="header-content">
                    <h2 class="registry-title">{self.config.title}</h2>
                    {stats_html}
                </div>
            </div>
        """

    def _generate_filter_html(self, registry: Dict) -> str:
        """Generate the filter section."""
        all_tags = set()
        for item in registry.values():
            all_tags.update(self.extractor.get_tags(item))

        filter_tags_html = ''
        for tag in sorted(all_tags):
            filter_tags_html += f'<button class="filter-tag" onclick="filterByTag(\'{tag}\')">{tag}</button>'

        return f"""
        <div class="filter-section">
            <div class="filter-label">Filter by tags:</div>
            <div class="filter-tags">
                {filter_tags_html}
                <button class="filter-tag filter-clear" onclick="clearFilter()">Clear All</button>
            </div>
        </div>
        """

    def _generate_grid_html(self, registry: Dict) -> str:
        """Generate the main grid of cards."""
        if not registry:
            return """
            <div class="metrics-grid">
                <div class="empty-state">
                    <div class="empty-icon">📦</div>
                    <p class="empty-text">No items registered.</p>
                    <p class="empty-subtext">Register some items to see them here.</p>
                </div>
            </div>
            """

        cards_html = ''
        for key, item in sorted(registry.items()):
            cards_html += self._generate_card_html(key, item)

        return f"""
        <div class="metrics-grid">
            {cards_html}
        </div>
        </div>
        """

    def _generate_card_html(self, key: str, item: Any) -> str:
        """Generate HTML for a single card."""
        title = self.extractor.get_title(item)
        subtitle = self.extractor.get_subtitle(item)
        description = self.extractor.get_description(item)
        tags = self.extractor.get_tags(item)
        fields = self.extractor.get_fields(item)

        # Generate tags
        tags_html = ''
        tag_data = ' '.join(tags)
        for tag in tags:
            tags_html += f'<span class="metric-tag tag-{tag}">{tag}</span>'

        # Generate badges
        badges_html = ''
        if self.config.show_badges:
            for field_key, field_info in fields.items():
                if field_info.get('type') in ['badge', 'threshold_badge']:
                    badge_class = self._get_badge_class(field_info)
                    badges_html += f'<span class="{badge_class}">{field_info["label"]}: {field_info["value"]}</span>'

        # Generate field sections
        field_sections_html = ''
        for field_key, field_info in fields.items():
            if field_info.get('type') == 'field_list':
                field_sections_html += self._generate_field_section_html(field_info)

        # Generate footer buttons
        footer_buttons = ''
        if self.config.show_details_button:
            footer_buttons += f'<button class="info-btn" onclick="showItemDetails(\'{key}\')">{self.config.details_button_text}</button>'
        if self.config.show_usage_button:
            footer_buttons += f'<button class="use-btn" onclick="showUsageExample(\'{key}\')">{self.config.usage_button_text}</button>'

        return f"""
        <div class="metric-card" data-tags="{tag_data}">
            <div class="metric-card-header">
                <div class="metric-header-top">
                    <h3 class="metric-title">{title}</h3>
                    <div class="metric-badges">{badges_html}</div>
                </div>
                <div class="metric-key-container">
                    <code class="metric-key">{subtitle}</code>
                    <button class="copy-btn" onclick="copyToClipboard('{subtitle}')" title="Copy key">📋</button>
                </div>
            </div>

            <div class="metric-card-body">
                <p class="metric-description">{description}</p>
                <div class="fields-container">{field_sections_html}</div>

                {f'<div class="tags-section"><div class="tags-header"><span class="tags-icon">🏷️</span><span class="tags-title">Tags</span></div><div class="tags-container">{tags_html}</div></div>' if self.config.show_tags and tags else ''}

                <div class="metric-footer">{footer_buttons}</div>
            </div>
        </div>
        """

    def _generate_field_section_html(self, field_info: Dict[str, Any]) -> str:
        """Generate HTML for a field section."""
        fields_html = ''
        field_class = field_info.get('field_class', 'default')
        icon = (
            '🔴'
            if field_class == 'required'
            else '🔵'
            if field_class == 'optional'
            else '📋'
        )

        for field in field_info['value']:
            fields_html += f'<li class="field-item {field_class}-field">{field}</li>'

        count_text = (
            f' ({len(field_info["value"])})'
            if field_info.get('show_count', True)
            else ''
        )

        return f"""
        <div class="fields-section">
            <div class="fields-header">
                <span class="fields-icon">{icon}</span>
                <span class="fields-title">{field_info['label']}{count_text}</span>
            </div>
            <ul class="fields-list">{fields_html}</ul>
        </div>
        """

    def _get_badge_class(self, field_info: Dict[str, Any]) -> str:
        """Get the CSS class for a badge."""
        if field_info.get('type') == 'threshold_badge':
            threshold = float(field_info['value'])
            if threshold <= 0.3:
                return 'threshold-badge threshold-low'
            elif threshold >= 0.7:
                return 'threshold-badge threshold-high'
            else:
                return 'threshold-badge threshold-medium'
        else:
            return 'score-badge'

    def _simple_display(self, registry: Dict) -> None:
        """Simple text fallback display."""
        print('=' * 80)
        print(
            f'{self.config.icon} {self.config.title.upper()} - {len(registry)} Available Items'
        )
        print('=' * 80)

        if not registry:
            print('... No items registered ...')
            return

        for i, (key, item) in enumerate(sorted(registry.items()), 1):
            title = self.extractor.get_title(item)
            description = self.extractor.get_description(item)
            tags = self.extractor.get_tags(item)

            print(f'\n{i:2d}. {title}')
            print(f'    Key: {key}')
            print(f'    Description: {description}')
            print(f'    Tags: {", ".join(tags)}')
            print('-' * 60)

    def _get_base_styles(self) -> str:
        """Get the base CSS styles (same as before but more modular)."""
        return (
            """
        <style>
            .registry-container {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                max-width: """
            + self.config.max_width
            + """;
                margin: 0 auto;
                background: #ffffff;
                border: 1px solid #e1e5e9;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }

            .registry-header {
                background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
                color: white;
                padding: 24px 32px;
                border-bottom: 1px solid #e1e5e9;
            }

            .registry-icon {
                width: 40px; height: 40px; margin-right: 16px;
                background: rgba(255,255,255,0.2);
                border-radius: 6px; display: flex;
                align-items: center; justify-content: center;
                font-size: 20px; float: left;
            }

            .header-content { margin-left: 56px; }

            .registry-title {
                font-size: 24px; font-weight: 600;
                margin: 0 0 8px 0; color: white;
            }

            .registry-stats {
                display: flex; gap: 24px; margin-top: 12px;
                font-size: 14px; opacity: 0.9;
            }

            .stat-item {
                display: flex; align-items: center; gap: 6px;
            }

            .stat-number {
                font-weight: 600; font-size: 16px;
            }

            .filter-section {
                padding: 20px 32px;
                background: #f8f9fa;
                border-bottom: 1px solid #e1e5e9;
            }

            .filter-label {
                font-weight: 600; color: #495057; margin-bottom: 12px;
                font-size: 14px;
            }

            .filter-tags {
                display: flex; flex-wrap: wrap; gap: 8px;
            }

            .filter-tag {
                padding: 6px 12px; border: 1px solid #dee2e6;
                background: white; border-radius: 4px;
                cursor: pointer; transition: all 0.15s;
                font-size: 13px; font-weight: 500;
                color: #495057;
            }

            .filter-tag:hover {
                background: #e9ecef; border-color: #adb5bd;
            }

            .filter-tag.active {
                background: #007bff; color: white; border-color: #007bff;
            }

            .filter-clear {
                background: #dc3545 !important; color: white !important;
                border-color: #dc3545 !important;
            }

            .filter-clear:hover {
                background: #c82333 !important; border-color: #bd2130 !important;
            }

            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax("""
            + self.config.min_card_width
            + """, 1fr));
                gap: 0;
                padding: 32px;
                background: #f8f9fa;
            }


            .metric-card {
                background: white;
                border: 1px solid #e1e5e9;
                border-radius: 6px;
                margin-bottom: 20px;
                overflow: hidden;
                transition: box-shadow 0.15s ease;
            }

            .metric-card:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }

            .metric-card-header {
                padding: 20px;
                background: #ffffff;
                border-bottom: 1px solid #e1e5e9;
            }

            .metric-header-top {
                display: flex; justify-content: space-between; align-items: flex-start;
                margin-bottom: 12px;
            }

            .metric-title {
                font-weight: 600; font-size: 16px; color: #2d3748;
                margin: 0; line-height: 1.3;
            }

            .metric-badges {
                display: flex; flex-direction: column; gap: 4px; align-items: flex-end;
            }

            .score-badge, .threshold-badge {
                font-size: 11px; padding: 3px 8px; border-radius: 3px;
                font-weight: 600; white-space: nowrap;
            }

            .score-badge {
                background: #e3f2fd; color: #1565c0;
            }

            .threshold-low { background: #e8f5e8; color: #2e7d32; }
            .threshold-medium { background: #fff3e0; color: #f57c00; }
            .threshold-high { background: #ffebee; color: #d32f2f; }

            .metric-key-container {
                display: flex; align-items: center; gap: 8px;
            }

            .metric-key {
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
                font-size: 12px; color: #6c757d;
                background: #f8f9fa; padding: 4px 8px;
                border-radius: 3px; border: 1px solid #e9ecef;
                flex: 1;
            }

            .copy-btn {
                border: 1px solid #dee2e6; background: white;
                padding: 4px 8px; border-radius: 3px; cursor: pointer;
                font-size: 11px; color: #6c757d;
                transition: all 0.15s;
            }

            .copy-btn:hover {
                background: #f8f9fa; border-color: #adb5bd;
            }

            .metric-card-body {
                padding: 20px;
                background: white;
            }

            .metric-description {
                color: #495057; font-size: 14px;
                line-height: 1.5; margin-bottom: 20px;
                padding: 12px 16px;
                background: #f8f9fa;
                border-radius: 4px;
                border-left: 3px solid #007bff;
            }

            .fields-container {
                margin-bottom: 20px;
            }

            .fields-section {
                margin-bottom: 16px;
            }

            .fields-header {
                display: flex; align-items: center; gap: 6px;
                font-size: 13px; font-weight: 600; color: #495057;
                margin-bottom: 8px;
            }

            .fields-icon {
                font-size: 12px;
            }

            .fields-list {
                list-style: none; padding: 0; margin: 0;
                display: flex; flex-wrap: wrap; gap: 6px;
            }

            .field-item {
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
                font-size: 11px; padding: 4px 8px;
                border-radius: 3px; font-weight: 500;
                border: 1px solid;
            }

            .required-field {
                background: #fff5f5; color: #c53030; border-color: #fed7d7;
            }

            .optional-field {
                background: #ebf8ff; color: #3182ce; border-color: #bee3f8;
            }

            .default-field {
                background: #f7fafc; color: #4a5568; border-color: #e2e8f0;
            }

            .tags-section {
                margin-bottom: 20px;
            }

            .tags-header {
                display: flex; align-items: center; gap: 6px;
                font-size: 13px; font-weight: 600; color: #495057;
                margin-bottom: 8px;
            }

            .tags-container {
                display: flex; flex-wrap: wrap; gap: 6px;
            }

            .metric-tag {
                font-size: 11px; padding: 4px 8px; border-radius: 12px;
                font-weight: 500;
                background: #e9ecef; color: #495057;
                border: 1px solid #dee2e6;
            }

            .tag-agent { background: #fff3cd; color: #856404; border-color: #ffeaa7; }
            .tag-knowledge { background: #d1ecf1; color: #0c5460; border-color: #bee5eb; }
            .tag-retrieval { background: #e2e3ff; color: #4c63d2; border-color: #c3c6ff; }

            .metric-footer {
                display: flex; gap: 8px; padding-top: 16px;
                border-top: 1px solid #f1f3f4;
            }

            .info-btn, .use-btn {
                flex: 1; padding: 8px 12px;
                border: 1px solid #dee2e6;
                background: white; border-radius: 4px; cursor: pointer;
                font-size: 12px; font-weight: 500;
                transition: all 0.15s;
                color: #495057;
            }

            .info-btn:hover {
                background: #f8f9fa; border-color: #adb5bd;
            }

            .use-btn:hover {
                background: #007bff; color: white; border-color: #007bff;
            }

            .empty-state {
                grid-column: 1 / -1;
                display: flex; flex-direction: column;
                align-items: center; justify-content: center;
                padding: 60px 20px;
                background: white;
                border: 2px dashed #dee2e6;
                border-radius: 6px;
                color: #6c757d;
            }

            .empty-icon {
                font-size: 48px; margin-bottom: 16px; opacity: 0.5;
            }

            .empty-text {
                font-size: 18px; font-weight: 600; margin-bottom: 8px;
            }

            .empty-subtext {
                font-size: 14px; opacity: 0.7;
            }

            /* Modal Styles */
            .metric-modal .modal-backdrop {
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: rgba(0,0,0,0.6); z-index: 10000;
                display: flex; align-items: center; justify-content: center;
                animation: fadeIn 0.2s ease;
            }
            .metric-modal .modal-content {
                background: white; border-radius: 8px; max-width: 700px; width: 90%;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3); max-height: 80vh; overflow-y: auto;
                animation: slideIn 0.2s ease;
            }
            .metric-modal .modal-header {
                padding: 20px; border-bottom: 1px solid #e1e5e9;
                display: flex; justify-content: space-between; align-items: center;
            }
            .metric-modal .modal-header h3 {
                margin: 0; color: #2d3748; font-size: 18px; font-weight: 600;
            }
            .metric-modal .modal-close {
                background: none; border: none; font-size: 24px; cursor: pointer;
                color: #6c757d; padding: 0; width: 30px; height: 30px;
                display: flex; align-items: center; justify-content: center;
            }
            .metric-modal .modal-close:hover { color: #495057; }
            .metric-modal .modal-body {
                padding: 20px; color: #4a5568; line-height: 1.6;
            }
            .metric-modal .modal-footer {
                padding: 20px; border-top: 1px solid #e1e5e9;
                display: flex; justify-content: flex-end; gap: 10px;
            }
            .metric-modal .btn-secondary {
                padding: 8px 16px; background: #6c757d; color: white;
                border: none; border-radius: 4px; cursor: pointer; font-weight: 500;
            }
            .metric-modal .btn-secondary:hover { background: #5a6268; }
            .details-content h4 { color: #2d3748; margin: 20px 0 10px 0; }
            .details-content h5 { color: #4a5568; margin: 15px 0 8px 0; }
            .details-content ul { margin: 0 0 15px 20px; }
            .details-content code { background: #f8f9fa; padding: 2px 4px; border-radius: 3px; }
            @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
            @keyframes slideIn { from { transform: translateY(-20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
        </style>
        """
        )

    def _generate_javascript(self) -> str:
        """Generate the JavaScript for interactivity."""

        # Define code themes
        if self.config.code_theme == 'dark':
            code_style = 'background: #1e1e1e; color: #d4d4d4; padding: 16px; border-radius: 4px; overflow-x: auto; margin: 0; border: 1px solid #3c3c3c;'
        else:  # light theme (default)
            code_style = 'background: #f8f9fa; padding: 16px; border-radius: 4px; overflow-x: auto; margin: 0; border: 1px solid #e1e5e9;'

        return f"""
        <script>
            // Store registry data for JavaScript access
            window.registryData = {{}};

            function setRegistryData(data) {{
                window.registryData = data;
            }}

            function filterByTag(tag) {{
                const cards = document.querySelectorAll('.metric-card');
                const filterBtns = document.querySelectorAll('.filter-tag');

                // Reset filter buttons
                filterBtns.forEach(btn => btn.classList.remove('active'));
                document.querySelector(`[onclick="filterByTag('${{tag}}')"]`).classList.add('active');

                cards.forEach(card => {{
                    const cardTags = card.getAttribute('data-tags');
                    if (cardTags.includes(tag)) {{
                        card.style.display = 'block';
                        card.style.opacity = '1';
                    }} else {{
                        card.style.display = 'none';
                    }}
                }});
            }}

            function clearFilter() {{
                const cards = document.querySelectorAll('.metric-card');
                const filterBtns = document.querySelectorAll('.filter-tag');

                filterBtns.forEach(btn => btn.classList.remove('active'));
                cards.forEach(card => {{
                    card.style.display = 'block';
                    card.style.opacity = '1';
                }});
            }}

            function copyToClipboard(text) {{
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(text).then(() => {{
                        const btn = event.target;
                        const originalText = btn.textContent;
                        btn.textContent = '✅';
                        btn.style.background = '#28a745';
                        setTimeout(() => {{
                            btn.textContent = originalText;
                            btn.style.background = '';
                        }}, 1500);
                    }}).catch(() => {{
                        fallbackCopyTextToClipboard(text);
                    }});
                }} else {{
                    fallbackCopyTextToClipboard(text);
                }}
            }}

            function fallbackCopyTextToClipboard(text) {{
                const textArea = document.createElement("textarea");
                textArea.value = text;
                textArea.style.position = "fixed";
                textArea.style.left = "-999999px";
                textArea.style.top = "-999999px";
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();

                try {{
                    document.execCommand('copy');
                    const btn = event.target;
                    const originalText = btn.textContent;
                    btn.textContent = '✅';
                    setTimeout(() => {{
                        btn.textContent = originalText;
                    }}, 1500);
                }} catch (err) {{
                    console.error('Fallback: Oops, unable to copy', err);
                }}

                document.body.removeChild(textArea);
            }}

            function showItemDetails(key) {{
                closeExistingModals();

                const item = window.registryData[key];
                if (!item) {{
                    console.error('Item not found:', key);
                    return;
                }}

                const modal = document.createElement('div');
                modal.className = 'metric-modal';
                modal.innerHTML = `
                    <div class="modal-backdrop">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h3>Details: ${{key}}</h3>
                                <button class="modal-close" onclick="closeModal(this)">&times;</button>
                            </div>
                            <div class="modal-body">
                                ${{item.detailsContent}}
                            </div>
                            <div class="modal-footer">
                                <button class="btn-secondary" onclick="closeModal(this)">Close</button>
                            </div>
                        </div>
                    </div>
                `;

                document.body.appendChild(modal);

                modal.querySelector('.modal-backdrop').addEventListener('click', function(e) {{
                    if (e.target === this) {{
                        closeModal(modal);
                    }}
                }});
            }}

            function showUsageExample(key) {{
                closeExistingModals();

                const item = window.registryData[key];
                if (!item) {{
                    console.error('Item not found:', key);
                    return;
                }}

                const modal = document.createElement('div');
                modal.className = 'metric-modal';
                modal.innerHTML = `
                    <div class="modal-backdrop">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h3>Usage Example: ${{key}}</h3>
                                <button class="modal-close" onclick="closeModal(this)">&times;</button>
                            </div>
                            <div class="modal-body">
                                <pre style="{code_style}"><code>${{item.usageExample}}</code></pre>
                            </div>
                            <div class="modal-footer">
                                <button class="btn-secondary" onclick="copyExampleCode('${{key}}')">Copy Code</button>
                                <button class="btn-secondary" onclick="closeModal(this)">Close</button>
                            </div>
                        </div>
                    </div>
            `   ;

                document.body.appendChild(modal);

                modal.querySelector('.modal-backdrop').addEventListener('click', function(e) {{
                    if (e.target === this) {{
                        closeModal(modal);
                    }}
                }});
            }}

            function copyExampleCode(key) {{
                const item = window.registryData[key];
                if (item && item.usageExample) {{
                    copyToClipboard(item.usageExample);
                }}
            }}

            function closeModal(element) {{
                const modal = element.closest('.metric-modal');
                if (modal) {{
                    modal.style.opacity = '0';
                    setTimeout(() => {{
                        if (modal.parentNode) {{
                            modal.parentNode.removeChild(modal);
                        }}
                    }}, 200);
                }}
            }}

            function closeExistingModals() {{
                const existingModals = document.querySelectorAll('.metric-modal');
                existingModals.forEach(modal => {{
                    if (modal.parentNode) {{
                        modal.parentNode.removeChild(modal);
                    }}
                }});
            }}

            document.addEventListener('keydown', function(e) {{
                if (e.key === 'Escape') {{
                    closeExistingModals();
                }}
            }});
        </script>
        """

    def display(self, registry: Dict) -> None:
        """Display the registry in a rich HTML format."""
        try:
            from IPython.display import HTML, display
        except ImportError:
            print(
                'IPython is not installed. Please install it to use the rich display feature.'
            )
            self._simple_display(registry)
            return

        js_data = {}
        for key, item in registry.items():
            # based on tags
            if hasattr(item, 'config'):
                if 'answer_criteria' in item.config.key:
                    example_script = self.extractor.get_criteria_usage_example(
                        key, item
                    )
                elif 'citation' in item.config.tags:
                    example_script = self.extractor.get_citation_usage_example(
                        key, item
                    )
                elif 'multi_turn' in item.config.tags:
                    example_script = self.extractor.get_multi_turn_usage_example(
                        key, item
                    )
                elif 'tool' in item.config.tags:
                    example_script = self.extractor.get_tool_usage_example(key, item)
                elif 'retrieval' in item.config.tags:
                    example_script = self.extractor.get_retrieval_example(key, item)
                else:
                    example_script = self.extractor.get_usage_example(key, item)
            else:
                example_script = self.extractor.get_usage_example(key, item)
            js_data[key] = {
                'usageExample': example_script,
                'detailsContent': self.extractor.get_details_content(key, item),
            }

        html = self._get_base_styles()
        html += self._generate_header_html(registry)

        if self.config.show_filters:
            html += self._generate_filter_html(registry)

        html += self._generate_grid_html(registry)
        html += self._generate_javascript()

        # Add the data injection
        html += f"""
        <script>
            setRegistryData({json.dumps(js_data)});
        </script>
        """

        display(HTML(html))


class APIRegistryExtractor(BaseRegistryExtractor):
    """Extractor for API registry items."""

    def get_title(self, item: Any) -> str:
        return getattr(item, 'name', item.__class__.__name__)

    def get_subtitle(self, item: Any) -> str:
        return item.__class__.__name__

    def get_description(self, item: Any) -> str:
        return getattr(item, '__doc__', 'API endpoint') or 'API endpoint'

    def get_tags(self, item: Any) -> List[str]:
        return getattr(item, 'tags', ['api'])

    def get_fields(self, item: Any) -> Dict[str, Any]:
        return {
            'methods': {
                'value': getattr(item, 'methods', ['GET']),
                'type': 'field_list',
                'label': 'HTTP Methods',
                'field_class': 'default',
            },
            'auth_required': {
                'value': 'Yes' if getattr(item, 'auth_required', False) else 'No',
                'type': 'badge',
                'label': 'Auth Required',
                'color': (
                    'warning' if getattr(item, 'auth_required', False) else 'success'
                ),
            },
        }

    def get_usage_example(self, key: str, item: Any) -> str:
        class_name = item.__class__.__name__
        return f"""# Usage example for {key}
from axion.api import {class_name}

# Initialize the API
api = {class_name}()

# Make a request
response = api.request(data)
print(response)"""

    def get_details_content(self, key: str, item: Any) -> str:
        return f"""
        <div class="details-content">
            <h4>API Details</h4>
            <ul>
                <li><strong>Class:</strong> <code>{item.__class__.__name__}</code></li>
                <li><strong>Methods:</strong> {', '.join(getattr(item, 'methods', ['GET']))}</li>
                <li><strong>Auth Required:</strong> {'Yes' if getattr(item, 'auth_required', False) else 'No'}</li>
            </ul>

            <h4>Description</h4>
            <p>{self.get_description(item)}</p>
        </div>
        """


class APIRunnerRegistryExtractor(BaseRegistryExtractor):
    """Extractor for APIRunner registry items."""

    def get_title(self, item: Any) -> str:
        # Get the class name without 'Runner' suffix for cleaner display
        class_name = item.__name__
        if class_name.endswith('Runner'):
            class_name = class_name[:-6]  # Remove 'Runner'
        return class_name

    def get_subtitle(self, item: Any) -> str:
        # Get the registered key from the registry
        return getattr(item, '_registry_key', item.__name__)

    def get_description(self, item: Any) -> str:
        # Get docstring or create a description based on the class
        doc = item.__doc__ or ''
        if doc:
            # Get first line of docstring
            return doc.split('\n')[0].strip('"""').strip()
        else:
            # Create description from class name
            try:
                class_name = item._registry_key
            except:
                class_name = self.get_title(item)
            return f'Runner for {class_name} API'

    def get_tags(self, item: Any) -> List[str]:
        # Determine tags based on the executor type
        tags = ['api']

        class_name = item.__name__.lower()
        if 'miaw' in class_name:
            tags.append('agent')
        elif 'agent' in class_name:
            tags.append('agent')
        elif 'template' in class_name or 'prompt' in class_name:
            tags.extend(['knowledge'])

        return tags

    def get_fields(self, item: Any) -> Dict[str, Any]:
        # Analyze the class to extract field information
        fields = {}

        # Check if it has batch processing
        has_batch = hasattr(item, 'execute_batch') or 'execute_batch' in [
            method for method in dir(item)
        ]

        # Check required methods
        required_methods = []
        optional_methods = []

        if hasattr(item, 'execute') or 'execute' in dir(item):
            required_methods.append('execute')
        if has_batch:
            optional_methods.append('execute_batch')
        if hasattr(item, 'shutdown') or 'shutdown' in dir(item):
            optional_methods.append('shutdown')

        # Get init parameters by inspecting the __init__ method
        init_params = []
        try:
            import inspect

            sig = inspect.signature(item.__init__)
            for param_name, param in sig.parameters.items():
                if param_name not in [
                    'self',
                    'credentials',
                    'max_concurrent',
                ]:
                    init_params.append(param_name)
        except:
            pass

        fields['required_methods'] = {
            'value': required_methods,
            'type': 'field_list',
            'label': 'Required Methods',
            'field_class': 'required',
        }

        if optional_methods:
            fields['optional_methods'] = {
                'value': optional_methods,
                'type': 'field_list',
                'label': 'Optional Methods',
                'field_class': 'optional',
            }

        if init_params:
            fields['init_params'] = {
                'value': init_params,
                'type': 'field_list',
                'label': 'Init Parameters',
                'field_class': 'default',
            }

        # Add capability badges
        fields['async_support'] = {
            'value': 'Yes' if has_batch else 'No',
            'type': 'badge',
            'label': 'Async Batch',
            'color': 'success' if has_batch else 'warning',
        }

        fields['concurrency'] = {
            'value': 'Supported',
            'type': 'badge',
            'label': 'Concurrency',
            'color': 'primary',
        }

        return fields

    def get_usage_example(self, key: str, item: Any) -> str:
        class_name = item.__name__

        config_mapping = {
            'miaw': {
                'domain': '...',
                'org_id': '...',
                'deployment_name': '...',
            },
            'agent_api': {
                'domain': '...',
                'org_id': '...',
                'deployment_name': '...',
            },
            'prompt_template': {
                'domain': '...',
                'token': '...',
                'retriever_name': '...',
                'input_key': '...',
                'prompt_template_name': '...',
                'application_name': 'PromptBuilderPreview',
            },
            'default': {'your credentials': '...'},
        }

        return f"""# Usage example for {key}
from axion.runners import APIRunner

# Method 1: Use with APIRunner
config = {{
    '{key}': {config_mapping.get(key, config_mapping.get('default'))}
}}

runner = APIRunner(config=config)

# Single query
response = runner.execute('{key}', "Your query here")
print(response.actual_output)

# Batch queries (async)
queries = ["Query 1", "Query 2", "Query 3"]
responses = await runner.execute_batch('{key}', queries)

# Method 2: Direct usage
from axion.runners import {class_name}

api_runner = {class_name}(
    config=config['{key}'],
)
response = api_runner.execute("Your query here")
print(response.actual_output)
"""

    def get_details_content(self, key: str, item: Any) -> str:
        class_name = item.__name__

        # Get method signatures
        methods_info = ''
        try:
            import inspect

            for method_name in ['execute', 'execute_batch', 'shutdown']:
                if hasattr(item, method_name):
                    try:
                        method = getattr(item, method_name)
                        sig = inspect.signature(method)
                        methods_info += f'<li><code>{method_name}{sig}</code></li>'
                    except:
                        methods_info += f'<li><code>{method_name}(...)</code></li>'
        except:
            pass

        # Get inheritance info
        inheritance_info = ''
        if hasattr(item, '__bases__'):
            bases = [
                base.__name__ for base in item.__bases__ if base.__name__ != 'object'
            ]
            if bases:
                inheritance_info = (
                    f'<li><strong>Inherits from:</strong> {", ".join(bases)}</li>'
                )

        return f"""
        <div class="details-content">
            <h4>Executor Details</h4>
            <ul>
                <li><strong>Class:</strong> <code>{class_name}</code></li>
                <li><strong>Registry Key:</strong> <code>{key}</code></li>
                {inheritance_info}
                <li><strong>Module:</strong> <code>{item.__module__}</code></li>
            </ul>

            <h4>Available Methods</h4>
            <ul>
                {methods_info}
            </ul>

            <h4>Features</h4>
            <ul>
                <li>✅ Synchronous execution via <code>execute()</code></li>
                <li>✅ Asynchronous batch processing via <code>execute_batch()</code></li>
                <li>✅ Automatic concurrency control with semaphores</li>
                <li>✅ Built-in logging and error handling</li>
                <li>✅ Graceful shutdown with <code>shutdown()</code></li>
                <li>✅ Standardized response format (<code>APIResponseData</code>)</li>
            </ul>

            <h4>Response Format</h4>
            <p>All executors return <code>APIResponseData</code> objects with:</p>
            <ul>
                <li><code>actual_output</code> - The main response text</li>
                <li><code>retrieved_content</code> - Retrieved context (if applicable)</li>
                <li><code>latency</code> - Response time in seconds</li>
                <li><code>trace</code> - Debug information</li>
                <li><code>status</code> - Execution status</li>
                <li><code>timestamp</code> - Response timestamp</li>
            </ul>

            <h4>Integration</h4>
            <p>This executor is registered with the <code>@APIRunner.register('{key}')</code> decorator
            and can be used both standalone and through the <code>APIRunner</code> class.</p>
        </div>
        """


def create_metric_display():
    """Create a metric registry display."""
    extractor = MetricRegistryExtractor()
    config = DisplayConfig(
        title='Metrics Registry', icon='📊', description='Available evaluation metrics'
    )
    return GenericRegistryDisplay(extractor, config)


def create_api_runner_display():
    """Create an API Executor registry display."""
    extractor = APIRunnerRegistryExtractor()
    config = DisplayConfig(
        title='API Runner Registry',
        icon='🔌',
        description='Available API runner for different services',
        show_badges=True,
        show_tags=True,
        show_filters=True,
        details_button_text='📖 Details',
        usage_button_text='⚡ Usage Examples',
    )
    return GenericRegistryDisplay(extractor, config)


# Enhanced usage function that adds registry keys to classes
def prepare_api_runner_registry(registry: Dict) -> Dict:
    """Prepare the APIRunner registry by adding registry keys to classes."""
    prepared_registry = {}
    for key, executor_class in registry.items():
        # Add the registry key as an attribute for easier access
        executor_class._registry_key = key
        prepared_registry[key] = executor_class
    return prepared_registry


def display_table(registry: Dict) -> None:
    """Prints a formatted table of all registered metrics."""
    print(f'📊 Found {len(registry)} available metrics:\n')

    # Enhanced table with more columns
    header = f'{"Key":<25} | {"Name":<25} | {"Threshold":<9} | {"Range":<8} | {"Req Fields":<12} | {"Opt Fields":<12} | {"Tags"}'
    print(header)
    print('-' * len(header))

    if not registry:
        print('... No metrics registered ...')
        return

    for key, metric_class in sorted(registry.items()):
        config = metric_class.config

        # Format fields for display
        req_fields = f'({len(config.required_fields)})'
        opt_fields = (
            f'({len(config.optional_fields)})' if config.optional_fields else '(0)'
        )
        score_range = f'{config.score_range[0]}-{config.score_range[1]}'
        tags = ', '.join(config.tags[:2])  # Show first 2 tags
        if len(config.tags) > 2:
            tags += '...'

        print(
            f'{config.key[:24]:<25} | {config.name[:24]:<25} | {config.default_threshold:<9} | {score_range:<8} | {req_fields:<12} | {opt_fields:<12} | {tags}'
        )


class MetricRunnerRegistryExtractor(BaseRegistryExtractor):
    """Extractor for MetricRunner registry items (executor classes)."""

    def __init__(
        self, show_helper_methods: bool = False, show_init_params: bool = False
    ):
        self.show_helper_methods = show_helper_methods
        self.show_init_params = show_init_params
        self.show_default_threshold = False

    def get_title(self, item: Any) -> str:
        # Get the class name without 'Executor' suffix for cleaner display
        class_name = item.__name__
        if class_name.endswith('Runner'):
            class_name = class_name[:-6]  # Remove 'Executor'
        return class_name

    def get_subtitle(self, item: Any) -> str:
        # Get the registered key from the registry
        return getattr(item, '_registry_key', item.__name__)

    def get_description(self, item: Any) -> str:
        # Get docstring or create a description based on the class
        doc = item.__doc__ or ''
        if doc:
            # Get first line of docstring
            return doc.split('\n')[0].strip('"""').strip()
        else:
            # Create description from class name and _name attribute
            framework_name = getattr(item, '_name', 'Unknown')
            return f'Executes metrics from {framework_name}'

    def get_tags(self, item: Any) -> List[str]:
        # Determine tags based on the executor type
        tags = ['knowledge', 'agent']

        # Add framework-specific tags
        framework_name = getattr(item, '_name', '').lower()
        if framework_name:
            tags.append(framework_name)

        # Add capability tags based on class analysis
        class_name = item.__name__.lower()
        if 'axion' in framework_name or 'axion' in class_name:
            tags.extend([])
        elif 'ragas' in framework_name:
            tags.extend([])
        elif 'deepeval' in framework_name:
            tags.extend([])

        return tags

    def get_fields(self, item: Any) -> Dict[str, Any]:
        # Analyze the class to extract field information
        fields = {}

        # Get framework information
        framework_name = getattr(item, '_name', 'Unknown')

        # Check for async support
        has_async = hasattr(item, 'execute')

        # Get required methods
        required_methods = ['execute']
        optional_methods = []

        # Check for specific methods
        if hasattr(item, '_create_error_score'):
            optional_methods.append('_create_error_score')
        if hasattr(item, '_prepare_retrieved_content'):
            optional_methods.append('_prepare_retrieved_content')
        if hasattr(item, '_has_passed'):
            optional_methods.append('_has_passed')

        # Get init parameters by inspecting the __init__ method
        init_params = []
        try:
            import inspect

            sig = inspect.signature(item.__init__)
            for param_name, param in sig.parameters.items():
                if param_name not in ['self', 'metric', 'threshold', 'kwargs']:
                    init_params.append(param_name)
        except:
            pass

        fields['framework'] = {
            'value': framework_name.title(),
            'type': 'badge',
            'label': 'Module',
            'color': 'primary',
        }

        fields['async_support'] = {
            'value': 'Yes' if has_async else 'No',
            'type': 'badge',
            'label': 'Async Support',
            'color': 'success' if has_async else 'warning',
        }

        fields['required_methods'] = {
            'value': required_methods,
            'type': 'field_list',
            'label': 'Required Methods',
            'field_class': 'required',
        }

        if self.show_helper_methods and optional_methods:
            fields['helper_methods'] = {
                'value': optional_methods,
                'type': 'field_list',
                'label': 'Helper Methods',
                'field_class': 'optional',
            }

        if self.show_init_params and init_params:
            fields['init_params'] = {
                'value': init_params,
                'type': 'field_list',
                'label': 'Init Parameters',
                'field_class': 'default',
            }

        # Add threshold support
        default_threshold = getattr(item, 'DEFAULT_THRESHOLD', None)
        if default_threshold is not None and self.show_default_threshold:
            fields['default_threshold'] = {
                'value': str(default_threshold),
                'type': 'threshold_badge',
                'label': 'Threshold',
            }

        return fields

    def get_usage_example(self, key: str, item: Any) -> str:
        class_name = item.__name__
        framework_name = getattr(item, '_name', 'unknown')

        # Create framework-specific example
        if framework_name == 'axion':
            metric_example = (
                'from axion.metrics import AnswerRelevancy\nmetric = AnswerRelevancy()'
            )
        elif framework_name == 'ragas':
            metric_example = 'from ragas.metrics.collections import Faithfulness\nfrom axion.integrations.models import LiteLLMRagas\nmetric = Faithfulness(llm=LiteLLMRagas()) # optional: LiteLLMRagas()'
        elif framework_name == 'deepeval':
            metric_example = 'from deepeval.metrics import AnswerRelevancyMetric\nfrom axion.integrations.models import LiteLLMDeepEval\nmetric = AnswerRelevancyMetric(model=LiteLLMDeepEval()) # optional: LiteLLMDeepEval()'
        else:
            metric_example = '# Your metric instance\nmetric = YourMetric()'

        return f"""# Usage example for {key} runner
from axion.runners import MetricRunner, {class_name}
from axion.dataset import DatasetItem

# Method 1: Use with MetricRunner
{metric_example}

metrics = [metric]  # List of metrics
runner = MetricRunner(
    metrics=metrics,
    max_concurrent=5,
)
# Execute batch evaluation
data_item = DatasetItem(
    query="What is the infield fly rule in baseball?",
    actual_output="The infield fly rule prevents the defense from intentionally dropping a fly ball to turn a double play.",
    expected_output="The infield fly rule protects baserunners by declaring the batter out on certain easy pop-ups.",
    retrieved_content=["The infield fly rule is designed to prevent unfair advantage by the defense."],
)
evaluation_inputs = [data_item]
results = await runner.execute_batch(evaluation_inputs)

# Method 2: Direct runner usage (Execute single evaluation)
metric_cls = {class_name}(metric=metric, threshold=0.7)
result = await metric_cls.execute(data_item)
print(f"Score: {{result.score}}, Explanation: {{result.explanation}}")"""

    def get_details_content(self, key: str, item: Any) -> str:
        class_name = item.__name__
        framework_name = getattr(item, '_name', 'Unknown')
        default_threshold = getattr(item, 'DEFAULT_THRESHOLD', 'Not set')

        # Get method signatures
        methods_info = ''
        try:
            import inspect

            for method_name in [
                'execute',
                '_create_error_score',
                '_prepare_retrieved_content',
                '_has_passed',
            ]:
                if hasattr(item, method_name):
                    try:
                        method = getattr(item, method_name)
                        if callable(method):
                            sig = inspect.signature(method)
                            methods_info += f'<li><code>{method_name}{sig}</code></li>'
                    except:
                        methods_info += f'<li><code>{method_name}(...)</code></li>'
        except:
            pass

        # Get inheritance info
        inheritance_info = ''
        if hasattr(item, '__bases__'):
            bases = [
                base.__name__ for base in item.__bases__ if base.__name__ != 'object'
            ]
            if bases:
                inheritance_info = (
                    f'<li><strong>Inherits from:</strong> {", ".join(bases)}</li>'
                )

        # Framework-specific notes
        framework_notes = {
            'axion': 'Native AXION metrics with async support and rich response objects.',
            'ragas': 'RAG-focused metrics using SingleTurnSample format for retrieval evaluation.',
            'deepeval': 'LLM evaluation metrics using LLMTestCase format with comprehensive testing.',
        }

        framework_note = framework_notes.get(
            framework_name, 'Custom metric runner implementation.'
        )

        return f"""
        <div class="details-content">
            <h4>Runner Details</h4>
            <ul>
                <li><strong>Class:</strong> <code>{class_name}</code></li>
                <li><strong>Registry Key:</strong> <code>{key}</code></li>
                <li><strong>Module:</strong> {framework_name.title()}</li>
                # <li><strong>Default Threshold:</strong> {default_threshold}</li>
                {inheritance_info}
                <li><strong>Module:</strong> <code>{item.__module__}</code></li>
            </ul>

            <h4>Available Methods</h4>
            <ul>
                {methods_info}
            </ul>

            <h4>Framework Integration</h4>
            <p>{framework_note}</p>

            <h4>Features</h4>
            <ul>
                <li>✅ Asynchronous metric execution via <code>execute()</code></li>
                <li>✅ Automatic error handling with fallback scores</li>
                <li>✅ Threshold-based pass/fail evaluation</li>
                <li>✅ Structured logging and debugging</li>
                <li>✅ Standardized response format (<code>MetricScore</code>)</li>
                <li>✅ Content preparation utilities</li>
                <li>✅ Configurable thresholds per metric</li>
            </ul>

            <h4>MetricScore Response Format</h4>
            <p>All Runners return <code>MetricScore</code> objects with:</p>
            <ul>
                <li><code>id</code> - Unique identifier for the evaluation</li>
                <li><code>name</code> - Name of the metric that was executed</li>
                <li><code>score</code> - Computed score (0.0 to 1.0 typically)</li>
                <li><code>threshold</code> - Threshold used for evaluation</li>
                <li><code>explanation</code> - Human-readable explanation</li>
                <li><code>passed</code> - Whether the score passed the threshold</li>
                <li><code>metadata</code> - Additional structured data</li>
                <li><code>source</code> - Framework source identifier</li>
                <li><code>timestamp</code> - When the evaluation occurred</li>
            </ul>

            <h4>Integration with MetricRunner</h4>
            <p>This runner is registered with <code>@MetricRunner.register('{key}')</code>
            and integrates seamlessly with the MetricRunner orchestration system for batch processing.</p>

            <h4>Error Handling</h4>
            <p>Includes robust error handling that creates fallback <code>MetricScore</code> objects
            with NaN scores and detailed error explanations when metric execution fails.</p>
        </div>
        """


def create_metric_runner_display(
    show_helper_methods: bool = False, show_init_params: bool = False
):
    """Create a Metric Runner registry display."""
    extractor = MetricRunnerRegistryExtractor(show_helper_methods, show_init_params)
    config = DisplayConfig(
        title='Metric Runner Registry',
        icon='⚙️',
        description='Available metric execution engines for different frameworks',
        show_badges=True,
        show_tags=True,
        show_filters=True,
        details_button_text='📖 Technical Details',
        usage_button_text='⚡ Usage Examples',
    )
    return GenericRegistryDisplay(extractor, config)


def prepare_metric_runner_registry(registry: Dict) -> Dict:
    """Prepare the MetricRunner registry by adding registry keys to classes."""
    prepared_registry = {}
    for key, runner_class in registry.items():
        # Add the registry key as an attribute for easier access
        runner_class._registry_key = key
        prepared_registry[key] = runner_class
    return prepared_registry


class LLMRegistryExtractor(BaseRegistryExtractor):
    """Extractor for LLM Registry provider items."""

    def get_title(self, item: Any) -> str:
        # Get the class name without 'Provider' suffix for cleaner display
        class_name = item.__name__
        if class_name.endswith('Provider'):
            class_name = class_name[:-8]  # Remove 'Provider'
        return class_name

    def get_subtitle(self, item: Any) -> str:
        # Get the registered key from the registry
        return getattr(item, '_registry_key', item.__name__)

    def get_description(self, item: Any) -> str:
        # Get docstring or create a description based on the class
        doc = item.__doc__ or ''
        if doc:
            # Get first line of docstring
            return doc.split('\n')[0].strip('"""').strip()
        else:
            # Create description from class name
            provider_name = self.get_title(item)
            return f'Provider for {provider_name} language models and embeddings'

    def get_tags(self, item: Any) -> List[str]:
        # Determine tags based on the provider type
        tags = ['llm', 'provider']

        class_name = item.__name__.lower()
        if 'openai' in class_name or 'client' in class_name:
            tags.extend(['openai'])
        elif 'anthropic' in class_name:
            tags.extend(['anthropic', 'claude'])
        elif 'gemini' in class_name:
            tags.extend(['gemini', 'google'])
        elif 'vertex' in class_name:
            tags.extend(['vertex_ai', 'google', 'gcp'])
        elif 'gateway' in class_name or 'proxy' in class_name:
            tags.extend(['gateway', 'proxy'])
        elif 'llama' in class_name or 'llamaindex' in class_name:
            tags.extend(['llamaindex'])
        elif 'huggingface' in class_name:
            tags.extend(['huggingface'])
        elif 'local' in class_name:
            tags.extend(['local', 'selfhosted'])

        return tags

    def get_fields(self, item: Any) -> Dict[str, Any]:
        # Analyze the class to extract field information
        fields: Dict[str, Any] = {}

        # Check available methods
        has_llm = hasattr(item, 'create_llm')
        has_embedding = hasattr(item, 'create_embedding_model')
        has_cost_estimate = hasattr(item, 'estimate_cost')

        # Get init parameters by inspecting the __init__ method
        required_params = []
        optional_params = []

        try:
            import inspect

            sig = inspect.signature(item.__init__)
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue

                param_info = param_name
                if param.default != inspect.Parameter.empty:
                    optional_params.append(param_info)
                else:
                    required_params.append(param_info)
        except:
            pass

        # Determine provider type and capabilities
        class_name = item.__name__.lower()
        provider_type = 'Unknown'
        if 'openai' in class_name:
            provider_type = 'OpenAI Compatible'
        elif 'anthropic' in class_name:
            provider_type = 'Anthropic'
        elif 'gemini' in class_name:
            provider_type = 'Google Gemini'
        elif 'vertex' in class_name:
            provider_type = 'Google Vertex AI'
        elif 'gateway' in class_name:
            provider_type = 'Gateway/Proxy'
        elif 'llama' in class_name:
            provider_type = 'LlamaIndex'
        elif 'huggingface' in class_name:
            provider_type = 'HuggingFace'

        fields['provider_type'] = {
            'value': provider_type,
            'type': 'badge',
            'label': 'Provider Type',
            'color': 'primary',
        }

        # Model support badges
        fields['llm_support'] = {
            'value': 'Yes' if has_llm else 'No',
            'type': 'badge',
            'label': 'LLM Support',
            'color': 'success' if has_llm else 'danger',
        }

        # Check SUPPORTS_EMBEDDINGS attribute first, fall back to hasattr
        supports_embeddings = getattr(item, 'SUPPORTS_EMBEDDINGS', has_embedding)
        fields['embedding_support'] = {
            'value': 'Yes' if supports_embeddings else 'No',
            'type': 'badge',
            'label': 'Embedding Support',
            'color': 'success' if supports_embeddings else 'warning',
        }

        # Cost estimation support
        if has_cost_estimate:
            fields['cost_estimation'] = {
                'value': 'Supported',
                'type': 'badge',
                'label': 'Cost Estimation',
                'color': 'success',
            }

        # Parameters
        if required_params:
            fields['required_params'] = {
                'value': required_params,
                'type': 'field_list',
                'label': 'Required Parameters',
                'field_class': 'required',
            }

        if optional_params:
            fields['optional_params'] = {
                'value': optional_params,
                'type': 'field_list',
                'label': 'Optional Parameters',
                'field_class': 'optional',
            }

        # Special configuration notes
        special_configs = []
        if 'gateway' in class_name:
            special_configs.append('api_base')
        if 'huggingface' in class_name:
            special_configs.append('device_map')
        if 'openai' in class_name:
            special_configs.append('organization_id')

        if special_configs:
            fields['special_configs'] = {
                'value': special_configs,
                'type': 'field_list',
                'label': 'Special Configs',
                'field_class': 'default',
            }

        return fields

    def get_usage_example(self, key: str, item: Any) -> str:
        class_name = item.__name__

        if 'express' in class_name.lower():
            config_example = """# Express Gateway configuration
from axion._core.environment import settings
from axion.llm_registry import LLMRegistry

# Option 1: Using global settings (recommended)
# Set these in your .env file or as environment variables
# LLM_PROVIDER="llm_gateway_express"
# API_BASE_URL="https://eng-ai-model-gateway.sfproxy.devx-preprod.aws-esvc1-useast2.aws.sfdc.cl"
# GATEWAY_API_KEY="your-api-key"

registry = LLMRegistry()
llm = registry.get_llm(model_name="gpt-5")
embedding_model = None # Embedding model not available for Express Gateway

# Option 2: Direct provider instantiation
registry = LLMRegistry(
    provider="llm_gateway_express",
    api_key="your-api-key",
    base_url="https://eng-ai-model-gateway.sfproxy.devx-preprod.aws-esvc1-useast2.aws.sfdc.cl"
)"""

        elif 'gateway' in class_name.lower():
            config_example = """# Gateway/Proxy configuration
from axion._core.environment import settings
from axion.llm_registry import LLMRegistry

# Option 1: Using global settings (recommended)
# Set these in your .env file or as environment variables
# LLM_PROVIDER="llm_gateway"
# API_BASE_URL="http://localhost:8000/v1"
# GATEWAY_API_KEY="your-api-key"

# You can also override at runtime
settings.llm_provider = "llm_gateway"

registry = LLMRegistry()
llm = registry.get_llm(model_name="gpt-4o")
embedding_model = registry.get_embedding_model(model_name="text-embedding-3-small")

# Option 2: Direct provider instantiation
registry = LLMRegistry(
    provider="llm_gateway",
    api_key="your-api-key",
    base_url="http://localhost:8000/v1"
)"""
        elif 'huggingface' in class_name.lower():
            config_example = """# HuggingFace configuration
from axion.llm_registry import LLMRegistry

# For HuggingFace models (requires transformers, torch)
# Install with: pip install -r requirements-dev.txt
registry = LLMRegistry(
    provider="huggingface",
    api_key="your-hf-token"  # Optional for public models
)

# Get models with device configuration
llm = registry.get_llm(
    model_name="microsoft/DialoGPT-medium",
    device_map="auto",
)

embedding_model = registry.get_embedding_model(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)"""
        elif 'llama' in class_name.lower():
            config_example = """# LlamaIndex configuration
from axion.llm_registry import LLMRegistry

registry = LLMRegistry(
    provider="llama_index",
    api_key="your-openai-api-key"
)

# Standard OpenAI models via LlamaIndex
llm = registry.get_llm(model_name="gpt-4o")
embedding_model = registry.get_embedding_model(model_name="text-embedding-3-small")"""
        else:
            config_example = f"""# {class_name} configuration
from axion.llm_registry import LLMRegistry

registry = LLMRegistry(
    provider="{key}",
    api_key="your-api-key"
)

llm = registry.get_llm(model_name="your-model")
embedding_model = registry.get_embedding_model(model_name="your-embedding-model")"""

        return f"""# Usage example for {key} provider
{config_example}

# Advanced usage with custom parameters
llm_with_params = registry.get_llm(
    model_name="your-model",
    temperature=0.7,
    max_tokens=1000,
)

# Query the model_name
response = llm_with_params.complete("What is artificial intelligence?")
print(response.text)"""

    def get_details_content(self, key: str, item: Any) -> str:
        class_name = item.__name__

        # Get method signatures
        methods_info = ''
        try:
            import inspect

            for method_name in [
                'create_llm',
                'create_embedding_model',
                'estimate_cost',
            ]:
                if hasattr(item, method_name):
                    try:
                        method = getattr(item, method_name)
                        sig = inspect.signature(method)
                        methods_info += f'<li><code>{method_name}{sig}</code></li>'
                    except:
                        methods_info += f'<li><code>{method_name}(...)</code></li>'
        except:
            pass

        # Get inheritance info
        inheritance_info = ''
        if hasattr(item, '__bases__'):
            bases = [
                base.__name__ for base in item.__bases__ if base.__name__ != 'object'
            ]
            if bases:
                inheritance_info = (
                    f'<li><strong>Inherits from:</strong> {", ".join(bases)}</li>'
                )

        # Provider-specific details
        provider_notes = {
            'llm_gateway': 'Routes requests through a local gateway/proxy server, useful for enterprise deployments with custom routing.',
            'llama_index': 'Native LlamaIndex provider for seamless integration with LlamaIndex workflows and applications.',
            'huggingface': 'Provides access to open-source models from HuggingFace Hub, supporting local inference.',
            'client': 'Direct OpenAI client connection for standard API access without additional frameworks.',
        }

        provider_note = provider_notes.get(
            key, 'Custom provider implementation for specialized model access.'
        )

        # Supported models info
        supported_models_info = ''
        if 'openai' in class_name.lower() or 'gateway' in class_name.lower():
            supported_models_info = """
            <h4>Commonly Supported Models</h4>
            <h5>Language Models:</h5>
            <ul>
                <li><code>gpt-4o</code> - Latest GPT-4 Optimized model</li>
                <li><code>gpt-4o-mini</code> - Faster, cost-effective GPT-4 variant</li>
                <li><code>gpt-4</code> - Standard GPT-4 model</li>
                <li><code>gpt-3.5-turbo</code> - Fast and efficient model</li>
                <li><code>o1</code> - Advanced reasoning model</li>
            </ul>
            <h5>Embedding Models:</h5>
            <ul>
                <li><code>text-embedding-3-small</code> - Compact embedding model</li>
                <li><code>text-embedding-3-large</code> - High-dimensional embeddings</li>
                <li><code>text-embedding-ada-002</code> - Legacy embedding model</li>
            </ul>
            """
        elif 'huggingface' in class_name.lower():
            supported_models_info = """
            <h4>HuggingFace Model Examples</h4>
            <h5>Language Models:</h5>
            <ul>
                <li><code>microsoft/DialoGPT-medium</code> - Conversational AI</li>
                <li><code>google/flan-t5-large</code> - Instruction-tuned model</li>
                <li><code>bigscience/bloom-560m</code> - Multilingual model</li>
            </ul>
            <h5>Embedding Models:</h5>
            <ul>
                <li><code>sentence-transformers/all-MiniLM-L6-v2</code> - General purpose</li>
                <li><code>sentence-transformers/all-mpnet-base-v2</code> - High quality</li>
            </ul>
            """

        return f"""
        <div class="details-content">
            <h4>Provider Details</h4>
            <ul>
                <li><strong>Class:</strong> <code>{class_name}</code></li>
                <li><strong>Registry Key:</strong> <code>{key}</code></li>
                {inheritance_info}
                <li><strong>Module:</strong> <code>{item.__module__}</code></li>
            </ul>

            <h4>Available Methods</h4>
            <ul>
                {methods_info}
            </ul>

            <h4>Provider Description</h4>
            <p>{provider_note}</p>

            <h4>Features</h4>
            <ul>
                <li>✅ Language model creation via <code>create_llm()</code></li>
                <li>✅ Embedding model creation via <code>create_embedding_model()</code></li>
                <li>✅ Cost estimation via <code>estimate_cost()</code></li>
                <li>✅ Flexible configuration with kwargs support</li>
                <li>✅ Integration with global Settings class</li>
                <li>✅ Automatic API key management</li>
                <li>✅ Provider-specific optimizations</li>
            </ul>

            {supported_models_info}

            <h4>Configuration</h4>
            <p>This provider can be configured through:</p>
            <ul>
                <li><strong>Global Settings:</strong> Set <code>settings.llm_provider = "{key}"</code></li>
                <li><strong>Direct instantiation:</strong> <code>LLMRegistry(provider="{key}", ...)</code></li>
                <li><strong>Environment variables:</strong> <code>LLM_PROVIDER={key}</code></li>
                <li><strong>Method-level override:</strong> <code>registry.get_llm(provider="{key}")</code></li>
            </ul>

            <h4>Cost Estimation</h4>
            <p>Uses the centralized <code>LLMCostEstimator</code> with model-specific pricing:</p>
            <ul>
                <li>Supports popular models like GPT-4o, GPT-3.5-Turbo, O1</li>
                <li>Calculates costs per 1,000 tokens for prompt and completion</li>
                <li>Falls back to default pricing for unknown models</li>
                <li>Pricing based on published OpenAI rates</li>
            </ul>
        </div>
        """


def create_llm_registry_display():
    """Create an LLM Registry display."""
    extractor = LLMRegistryExtractor()
    config = DisplayConfig(
        title='LLM Provider Registry',
        icon='🤖',
        description='Available language model and embedding providers',
        show_badges=True,
        show_tags=True,
        show_filters=True,
        details_button_text='🔧 Provider Details',
        usage_button_text='📝 Usage Examples',
        code_theme='light',  # Good for code examples
    )
    return GenericRegistryDisplay(extractor, config)


def prepare_llm_registry(registry_dict: Dict) -> Dict:
    """Prepare the LLM Registry by adding registry keys to provider classes."""
    prepared_registry = {}
    for key, provider_class in registry_dict.items():
        # Add the registry key as an attribute for easier access
        provider_class._registry_key = key
        prepared_registry[key] = provider_class
    return prepared_registry


class SettingsDisplayExtractor(BaseRegistryExtractor):
    """Extractor for displaying Settings configuration."""

    def get_title(self, item: Any) -> str:
        return 'Global Settings'

    def get_subtitle(self, item: Any) -> str:
        return 'axion._core.environment.settings'

    def get_description(self, item: Any) -> str:
        return 'Global configuration for LLM providers, models, and API settings'

    def get_tags(self, item: Any) -> List[str]:
        return ['configuration', 'global', 'settings']

    def get_fields(self, item: Any) -> Dict[str, Any]:
        return {
            'llm_provider': {
                'value': item.llm_provider,
                'type': 'badge',
                'label': 'LLM Provider',
                'color': 'primary',
            },
            'embedding_provider': {
                'value': item.embedding_provider,
                'type': 'badge',
                'label': 'Embedding Provider',
                'color': 'primary',
            },
            'llm_model': {
                'value': item.llm_model,
                'type': 'badge',
                'label': 'Default LLM Model',
                'color': 'success',
            },
            'embedding_model': {
                'value': item.embedding_model,
                'type': 'badge',
                'label': 'Default Embedding Model',
                'color': 'success',
            },
            'api_configured': {
                'value': 'Yes' if item.api_key else 'No',
                'type': 'badge',
                'label': 'API Key Set',
                'color': 'success' if item.api_key else 'warning',
            },
            'base_url_configured': {
                'value': 'Yes' if item.base_url else 'No',
                'type': 'badge',
                'label': 'Base URL Set',
                'color': 'success' if item.base_url else 'warning',
            },
        }

    def get_usage_example(self, key: str, item: Any) -> str:
        return """# Global Settings Configuration
from axion.llm_registry import Settings

# Method 1: Direct attribute setting
Settings.llm_provider = "llm_gateway"
Settings.embedding_provider = "llm_gateway"
Settings.llm_model = "gpt-4o"
Settings.embedding_model = "text-embedding-3-small"
Settings.api_key = "your-api-key"
Settings.base_url = "http://localhost:8000/v1"

# Method 2: Environment variables (set before import)
import os
os.environ["LLM_PROVIDER"] = "llm_gateway"
os.environ["EMBEDDING_PROVIDER"] = "llm_gateway"
os.environ["LLM_MODEL"] = "gpt-4o"
os.environ["EMBEDDING_MODEL"] = "text-embedding-3-small"
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"

# Then use with LLMRegistry
from axion.llm_registry import LLMRegistry

registry = LLMRegistry()
llm = registry.get_llm()  # Uses Settings.llm_model
embedding = registry.get_embedding_model()  # Uses Settings.embedding_model

# Override at call time
llm_override = registry.get_llm(model="gpt-3.5-turbo", provider="llama_index")"""

    def get_details_content(self, key: str, item: Any) -> str:
        return f"""
        <div class="details-content">
            <h4>Current Configuration</h4>
            <ul>
                <li><strong>LLM Provider:</strong> <code>{item.llm_provider}</code></li>
                <li><strong>Embedding Provider:</strong> <code>{item.embedding_provider}</code></li>
                <li><strong>LLM Model:</strong> <code>{item.llm_model}</code></li>
                <li><strong>Embedding Model:</strong> <code>{item.embedding_model}</code></li>
                <li><strong>API Key:</strong> {'✅ Configured' if item.api_key else '❌ Not Set'}</li>
                <li><strong>Base URL:</strong> {item.base_url or '❌ Not Set'}</li>
            </ul>

            <h4>Configuration Precedence</h4>
            <p>Settings are resolved in the following order (highest to lowest priority):</p>
            <ol>
                <li><strong>Direct method arguments:</strong> <code>registry.get_llm(model="gpt-4", provider="openai")</code></li>
                <li><strong>Settings class attributes:</strong> <code>Settings.llm_model = "gpt-4"</code></li>
                <li><strong>Environment variables:</strong> <code>LLM_MODEL=gpt-4</code></li>
                <li><strong>Hardcoded defaults:</strong> Built-in fallback values</li>
            </ol>

            <h4>Environment Variables</h4>
            <ul>
                <li><code>LLM_PROVIDER</code> - Default LLM provider name</li>
                <li><code>EMBEDDING_PROVIDER</code> - Default embedding provider name</li>
                <li><code>LLM_MODEL</code> - Default language model</li>
                <li><code>EMBEDDING_MODEL</code> - Default embedding model</li>
                <li><code>OPENAI_API_KEY</code> - API key for OpenAI-compatible services</li>
                <li><code>OPENAI_BASE_URL</code> - Base URL for API endpoints</li>
            </ul>

            <h4>Usage Patterns</h4>
            <ul>
                <li><strong>Global Configuration:</strong> Set once, use everywhere</li>
                <li><strong>Provider Locking:</strong> Lock registry to specific provider</li>
                <li><strong>Per-call Overrides:</strong> Override settings for specific calls</li>
                <li><strong>Environment-based:</strong> Configure via environment variables</li>
            </ul>
        </div>
        """


def create_settings_display():
    """Create a display for the Settings configuration."""
    extractor = SettingsDisplayExtractor()
    config = DisplayConfig(
        title='LLM Registry Settings',
        icon='⚙️',
        description='Global configuration and environment settings',
        show_badges=True,
        show_tags=False,
        show_filters=False,
        details_button_text='🔧 Configuration Details',
        usage_button_text='📝 Usage Examples',
    )
    return GenericRegistryDisplay(extractor, config)


class TracerRegistryExtractor(BaseRegistryExtractor):
    """Extractor for TracerRegistry items."""

    def get_title(self, item: Any) -> str:
        # Get class name, remove 'Tracer' suffix for cleaner display
        class_name = item.__name__
        if class_name.endswith('Tracer'):
            class_name = class_name[:-6]
        return class_name

    def get_subtitle(self, item: Any) -> str:
        return getattr(item, '_registry_key', item.__name__)

    def get_description(self, item: Any) -> str:
        doc = item.__doc__ or ''
        if doc:
            # Get first paragraph of docstring
            lines = doc.strip().split('\n\n')[0].split('\n')
            return ' '.join(line.strip() for line in lines if line.strip())
        return 'Tracer implementation for observability'

    def get_tags(self, item: Any) -> List[str]:
        tags = ['tracer']
        class_name = item.__name__.lower()

        if 'noop' in class_name:
            tags.append('testing')
        elif 'logfire' in class_name:
            tags.extend(['opentelemetry', 'observability'])
        elif 'langfuse' in class_name:
            tags.extend(['llm', 'observability'])
        elif 'opik' in class_name:
            tags.extend(['llm', 'comet'])

        return tags

    def get_fields(self, item: Any) -> Dict[str, Any]:
        # Analyze available methods
        core_methods = []
        optional_methods = []

        for method in ['span', 'async_span', 'start', 'complete', 'fail']:
            if hasattr(item, method):
                core_methods.append(method)

        for method in ['flush', 'shutdown', 'log_llm_call', 'log_evaluation']:
            if hasattr(item, method):
                optional_methods.append(method)

        return {
            'core_methods': {
                'value': core_methods,
                'type': 'field_list',
                'label': 'Core Methods',
                'field_class': 'required',
            },
            'optional_methods': {
                'value': optional_methods,
                'type': 'field_list',
                'label': 'Optional Methods',
                'field_class': 'optional',
            },
        }

    def get_usage_example(self, key: str, item: Any) -> str:
        return f"""# Usage example for {key} tracer
from axion._core.tracing import configure_tracing, Tracer

# Configure tracing mode
configure_tracing(tracing_mode='{key}')

# Create a tracer instance
tracer = Tracer('llm')

# Use spans for tracing operations
with tracer.span('my-operation') as span:
    span.set_input({{'query': 'Hello'}})
    # ... your code ...
    span.set_output({{'response': 'Hi!'}})

# Flush traces before exiting
tracer.flush()
"""

    def get_details_content(self, key: str, item: Any) -> str:
        class_name = item.__name__
        doc = item.__doc__ or 'No documentation available.'

        return f"""
<h3>{class_name}</h3>
<p><strong>Registry Key:</strong> <code>{key}</code></p>
<h4>Description</h4>
<p>{doc}</p>
<h4>Configuration</h4>
<p>Set <code>TRACING_MODE={key}</code> in your environment or use:</p>
<pre><code>configure_tracing(tracing_mode='{key}')</code></pre>
"""


def create_tracer_registry_display():
    """Create a Tracer Registry display."""
    extractor = TracerRegistryExtractor()
    config = DisplayConfig(
        title='Tracer Registry',
        icon='🔍',
        description='Available tracing providers for observability',
        show_badges=True,
        show_tags=True,
        show_filters=True,
        details_button_text='📖 Tracer Details',
        usage_button_text='⚡ Usage Examples',
    )
    return GenericRegistryDisplay(extractor, config)


def prepare_tracer_registry(registry: Dict) -> Dict:
    """Prepare the TracerRegistry by adding registry keys to tracer classes."""
    prepared_registry = {}
    for key, tracer_class in registry.items():
        tracer_class._registry_key = key
        prepared_registry[key] = tracer_class
    return prepared_registry
