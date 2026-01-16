import re
from typing import Any, Callable, Optional, Union

from axion.docs.render.code_usage_button import UsageButton
from axion.docs.render.documentation_button import DocumentationButton


class ModalButtonCard:
    """Card view that displays documentation and usage buttons side by side."""

    def __init__(
        self,
        card_title: str = '',
        card_description: str = '',
        card_background: str = '#ffffff',
        card_border_color: str = '#e5e7eb',
        card_shadow: str = '0 4px 6px rgba(0, 0, 0, 0.05)',
        max_width: str = '600px',
    ):
        """
        Initialize the card view.

        Args:
            card_title: Title to display at the top of the card
            card_description: Description text below the title
            card_background: Background color of the card
            card_border_color: Border color of the card
            card_shadow: CSS box-shadow for the card
            max_width: Maximum width of the card (default: 600px)
        """
        self.card_title = card_title
        self.card_description = card_description
        self.card_background = card_background
        self.card_border_color = card_border_color
        self.card_shadow = card_shadow
        self.max_width = max_width

    def render(
        self,
        key: str,
        item: Any = None,
        documentation: Optional[Union[str, Callable[[str, Any], str]]] = None,
        usage_template: Optional[Callable[[str, Any], str]] = None,
        doc_button_text: str = '📖 Documentation',
        usage_button_text: str = '💡 Usage Example',
        title_override: Optional[str] = None,
        description_override: Optional[str] = None,
    ) -> str:
        """
        Render the card with documentation and/or usage buttons.

        Args:
            key: Unique identifier for the item
            item: The registry item
            documentation: Documentation content or template function
            usage_template: Usage template function
            doc_button_text: Text for documentation button
            usage_button_text: Text for usage button
            title_override: Override the card title for this specific render
            description_override: Override the card description for this specific render

        Returns:
            Complete HTML string with card, buttons, styles, and JavaScript
        """
        card_id = f'card_{key.replace("-", "_").replace(".", "_")}'
        title = (
            title_override
            or self.card_title
            or key.replace('_', ' ').replace('-', ' ').title()
        )
        description = description_override or self.card_description

        # Generate button components separately with unique keys
        buttons_html = []
        all_styles = []
        all_scripts = []

        if documentation is not None:
            # Create documentation button with unique key
            doc_key = f'{key}_doc'
            doc_button = DocumentationButton(documentation)
            doc_html = doc_button.render(doc_key, item, doc_button_text)

            button_html = self._extract_button_html(doc_html)
            styles = self._extract_styles(doc_html)
            scripts = self._extract_scripts(doc_html)

            buttons_html.append(button_html)
            if styles:
                all_styles.append(styles)
            if scripts:
                all_scripts.append(scripts)

        if usage_template is not None:
            # Create usage button with unique key
            usage_key = f'{key}_usage'
            usage_button = UsageButton(usage_template)
            usage_html = usage_button.render(usage_key, item, usage_button_text)

            button_html = self._extract_button_html(usage_html)
            styles = self._extract_styles(usage_html)
            scripts = self._extract_scripts(usage_html)

            buttons_html.append(button_html)
            if styles:
                all_styles.append(styles)
            if scripts:
                all_scripts.append(scripts)

        # Combine all styles and scripts (remove duplicates)
        combined_styles = self._deduplicate_styles(all_styles)
        combined_scripts = self._deduplicate_scripts(all_scripts)

        return f"""
        <div class="modal-card" id="{card_id}">
            <div class="modal-card-header">
                <h3 class="modal-card-title">{title}</h3>
                {f'<p class="modal-card-description">{description}</p>' if description else ''}
            </div>
            <div class="modal-card-actions">
                {''.join(buttons_html)}
            </div>
        </div>

        <style>
            {self._get_card_styles()}
            {combined_styles}
        </style>

        {combined_scripts}
        """

    def _deduplicate_styles(self, styles_list):
        """Remove duplicate CSS rules from multiple style blocks."""
        seen_rules = set()
        unique_styles = []

        for styles in styles_list:
            if styles and styles not in seen_rules:
                seen_rules.add(styles)
                unique_styles.append(styles)

        return '\n'.join(unique_styles)

    def _deduplicate_scripts(self, scripts_list):
        """Remove duplicate JavaScript functions from multiple script blocks."""
        seen_functions = set()
        unique_scripts = []

        for scripts in scripts_list:
            if scripts:
                # Extract function names to avoid duplicates
                function_names = re.findall(r'function\s+(\w+)', scripts)
                script_key = '_'.join(sorted(function_names))

                if script_key not in seen_functions:
                    seen_functions.add(script_key)
                    unique_scripts.append(scripts)

        return '\n'.join(unique_scripts)

    def _get_card_styles(self) -> str:
        """Get card-specific CSS styles."""
        return f"""
            .modal-card {{
                background: {self.card_background};
                border: 1px solid {self.card_border_color};
                border-radius: 12px;
                padding: 24px;
                margin: 16px 0;
                box-shadow: {self.card_shadow};
                transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: {self.max_width};
            }}

            .modal-card:hover {{
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
                transform: translateY(-2px);
            }}

            .modal-card-header {{
                margin-bottom: 20px;
            }}

            .modal-card-title {{
                margin: 0 0 8px 0;
                color: #111827;
                font-size: 20px;
                font-weight: 700;
                line-height: 1.3;
            }}

            .modal-card-description {{
                margin: 0;
                color: #6b7280;
                font-size: 14px;
                line-height: 1.5;
            }}

            .modal-card-actions {{
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
                align-items: center;
            }}

            .modal-card-actions .modal-button-container {{
                margin: 0;
            }}

            .modal-card-actions .modal-btn {{
                font-size: 13px;
                padding: 8px 16px;
                min-width: 140px;
                justify-content: center;
                display: flex;
                align-items: center;
                gap: 6px;
            }}

            /* Responsive design */
            @media (max-width: 640px) {{
                .modal-card {{
                    margin: 12px 0;
                    padding: 20px;
                }}

                .modal-card-actions {{
                    flex-direction: column;
                    align-items: stretch;
                }}

                .modal-card-actions .modal-btn {{
                    width: 100%;
                    min-width: auto;
                }}
            }}

            /* Card container for multiple cards */
            .modal-cards-container {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                padding: 20px 0;
            }}

            .modal-cards-container .modal-card {{
                margin: 0;
            }}

            /* Dark mode support */
            @media (prefers-color-scheme: dark) {{
                .modal-card {{
                    background: #1f2937;
                    border-color: #374151;
                    color: #f9fafb;
                }}

                .modal-card-title {{
                    color: #f9fafb;
                }}

                .modal-card-description {{
                    color: #9ca3af;
                }}
            }}
        """

    def _extract_button_html(self, full_html: str) -> str:
        """Extract just the button container HTML from the full modal HTML."""
        # Look for the button container div
        match = re.search(
            r'(<div class="modal-button-container">.*?</div>)', full_html, re.DOTALL
        )
        return match.group(1) if match else ''

    def _extract_scripts(self, full_html: str) -> str:
        """Extract script tags from the full modal HTML."""
        # Find all script blocks
        scripts = re.findall(r'<script>(.*?)</script>', full_html, re.DOTALL)
        return '\n'.join(f'<script>{script}</script>' for script in scripts)

    def _extract_styles(self, full_html: str) -> str:
        """Extract style tags from the full modal HTML."""
        # Find all style blocks
        styles = re.findall(r'<style>(.*?)</style>', full_html, re.DOTALL)
        return '\n'.join(styles)

    def render_multiple(
        self,
        items: list,
        documentation_template: Optional[Callable[[str, Any], str]] = None,
        usage_template: Optional[Callable[[str, Any], str]] = None,
        container_class: str = 'modal-cards-container',
    ) -> str:
        """
        Render multiple cards in a grid container.

        Args:
            items: List of dictionaries with 'key', 'item', 'title', 'description' keys
            documentation_template: Template function for documentation
            usage_template: Template function for usage examples
            container_class: CSS class for the container

        Returns:
            HTML string with multiple cards in a grid
        """
        cards_html = []
        all_styles = []
        all_scripts = []

        for item_config in items:
            key = item_config.get('key', '')
            item = item_config.get('item')
            title = item_config.get(
                'title', key.replace('_', ' ').replace('-', ' ').title()
            )
            description = item_config.get('description', '')

            card_html = self.render(
                key=key,
                item=item,
                documentation=documentation_template,
                usage_template=usage_template,
                title_override=title,
                description_override=description,
            )

            # Extract card div and collect styles/scripts
            card_div = self._extract_card_div(card_html)
            styles = self._extract_styles(card_html)
            scripts = self._extract_scripts(card_html)

            cards_html.append(card_div)
            if styles:
                all_styles.append(styles)
            if scripts:
                all_scripts.append(scripts)

        # Combine all styles and scripts
        combined_styles = self._deduplicate_styles(all_styles)
        combined_scripts = self._deduplicate_scripts(all_scripts)

        return f"""
        <div class="{container_class}">
            {''.join(cards_html)}
        </div>

        <style>
            {combined_styles}
        </style>

        {combined_scripts}
        """

    def _extract_card_div(self, full_html: str) -> str:
        """Extract just the card div from the full HTML."""
        # Find the card div
        match = re.search(
            r'(<div class="modal-card"[^>]*>.*?</div>)', full_html, re.DOTALL
        )
        return match.group(1) if match else ''


class MultiUsageModalButtonCard(ModalButtonCard):
    """Extended card view that supports multiple usage templates."""

    def render_with_multiple_usage(
        self,
        key: str,
        item: Any = None,
        usage_templates: list[tuple[Callable[[str, Any], str], str]] = None,
        title_override: Optional[str] = None,
        description_override: Optional[str] = None,
    ) -> str:
        """
        Render the card with multiple usage buttons.

        Args:
            key: Unique identifier for the item
            item: The registry item
            usage_templates: List of tuples (template_function, button_text)
            title_override: Override the card title for this specific render
            description_override: Override the card description for this specific render

        Returns:
            Complete HTML string with card, buttons, styles, and JavaScript
        """
        card_id = f'card_{key.replace("-", "_").replace(".", "_")}'
        title = (
            title_override
            or self.card_title
            or key.replace('_', ' ').replace('-', ' ').title()
        )
        description = description_override or self.card_description

        buttons_html = []
        all_styles = []
        all_scripts = []

        if usage_templates:
            for i, (template_func, button_text) in enumerate(usage_templates):
                usage_key = f'{key}_usage_{i + 1}'
                usage_button = UsageButton(template_func)
                usage_html = usage_button.render(usage_key, item, button_text)

                button_html = self._extract_button_html(usage_html)
                styles = self._extract_styles(usage_html)
                scripts = self._extract_scripts(usage_html)

                buttons_html.append(button_html)
                if styles:
                    all_styles.append(styles)
                if scripts:
                    all_scripts.append(scripts)

        # Combine all styles and scripts
        combined_styles = self._deduplicate_styles(all_styles)
        combined_scripts = self._deduplicate_scripts(all_scripts)

        return f"""
        <div class="modal-card" id="{card_id}">
            <div class="modal-card-header">
                <h3 class="modal-card-title">{title}</h3>
                {f'<p class="modal-card-description">{description}</p>' if description else ''}
            </div>
            <div class="modal-card-actions">
                {''.join(buttons_html)}
            </div>
        </div>

        <style>
            {self._get_card_styles()}
            {combined_styles}
        </style>

        {combined_scripts}
        """


def create_multi_usage_modal_card(
    key: str,
    item: Any = None,
    title: str = '',
    description: str = '',
    documentation_templates: list[tuple[Callable[[str, Any], str], str]] = None,
    usage_templates: list[tuple[Callable[[str, Any], str], str]] = None,
    max_width: str = '1350px',
) -> str:
    """
    Convenience function to create a modal card with multiple documentation and usage templates.

    Args:
        key: Unique identifier
        item: Registry item
        title: Card title
        description: Card description
        documentation_templates: List of tuples (template_function, button_text) for documentation
        usage_templates: List of tuples (template_function, button_text) for usage examples
        max_width: Maximum width of the card

    Returns:
        HTML string for the card with multiple documentation and usage buttons

    Example:
        create_multi_usage_modal_card(
            key="custom_metrics",
            title="Building Custom Metrics",
            description="Choose the right template for your evaluation needs.",
            documentation_templates=[
                (documentation_template, 'Documentation')
            ],
            usage_templates=[
                (single_turn_metric_template, "Single-Turn Metric Template"),
                (multi_turn_metric_template, "Multi-Turn Conversation Template"),
                (heuristic_metric_template, "Heuristic Template"),
                (yaml_metric_template, "YAML Driven Template")
            ],
            max_width='1350px'
        )
    """
    card = MultiUsageModalButtonCard(
        card_title=title or key.replace('_', ' ').replace('-', ' ').title(),
        card_description=description,
        max_width=max_width,
    )

    # Build the complete render with both documentation and usage templates
    card_id = f'card_{key.replace("-", "_").replace(".", "_")}'
    display_title = title or key.replace('_', ' ').replace('-', ' ').title()

    buttons_html = []
    all_styles = []
    all_scripts = []

    # Process documentation templates
    if documentation_templates:
        for i, (template_func, button_text) in enumerate(documentation_templates):
            doc_key = f'{key}_doc_{i + 1}'
            doc_button = DocumentationButton(template_func)
            doc_html = doc_button.render(doc_key, item, button_text)

            button_html = card._extract_button_html(doc_html)
            styles = card._extract_styles(doc_html)
            scripts = card._extract_scripts(doc_html)

            buttons_html.append(button_html)
            if styles:
                all_styles.append(styles)
            if scripts:
                all_scripts.append(scripts)

    # Process usage templates
    if usage_templates:
        for i, (template_func, button_text) in enumerate(usage_templates):
            usage_key = f'{key}_usage_{i + 1}'
            usage_button = UsageButton(template_func)
            usage_html = usage_button.render(usage_key, item, button_text)

            button_html = card._extract_button_html(usage_html)
            styles = card._extract_styles(usage_html)
            scripts = card._extract_scripts(usage_html)

            buttons_html.append(button_html)
            if styles:
                all_styles.append(styles)
            if scripts:
                all_scripts.append(scripts)

    # Combine all styles and scripts
    combined_styles = card._deduplicate_styles(all_styles)
    combined_scripts = card._deduplicate_scripts(all_scripts)

    return f"""
    <div class="modal-card" id="{card_id}">
        <div class="modal-card-header">
            <h3 class="modal-card-title">{display_title}</h3>
            {f'<p class="modal-card-description">{description}</p>' if description else ''}
        </div>
        <div class="modal-card-actions">
            {''.join(buttons_html)}
        </div>
    </div>

    <style>
        {card._get_card_styles()}
        {combined_styles}
    </style>

    {combined_scripts}
    """


def create_modal_card(
    key: str,
    item: Any = None,
    title: str = '',
    description: str = '',
    documentation: Optional[Union[str, Callable[[str, Any], str]]] = None,
    usage_template: Optional[Callable[[str, Any], str]] = None,
    max_width: str = '600px',
) -> str:
    """
    Fixed convenience function to create a modal card quickly.

    Args:
        key: Unique identifier
        item: Registry item
        title: Card title
        description: Card description
        documentation: Documentation content or template function
        usage_template: Usage template function (different from documentation)
        max_width: Maximum width of the card

    Returns:
        HTML string for the card
    """
    card = ModalButtonCard(
        card_title=title or key.replace('_', ' ').replace('-', ' ').title(),
        card_description=description,
        max_width=max_width,
    )

    return card.render(
        key=key,
        item=item,
        documentation=documentation,  # This should be the documentation template
        usage_template=usage_template,  # This should be the usage template
    )
