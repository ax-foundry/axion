import re
from typing import Any, Callable, Union

from axion.docs.render.base_modal import BaseModalButton


class DocumentationButton(BaseModalButton):
    """Documentation button that can be embedded in any HTML."""

    def __init__(
        self,
        documentation: Union[str, Callable[[str, Any], str]],
        button_background_color: str = '#3b82f6',  # Blue
        button_hover_color: str = '#2563eb',  # Darker Blue
        action_primary_color: str = '#3b82f6',  # Match button color
        action_primary_hover: str = '#2563eb',  # Match hover color
    ):
        """
        Initialize with documentation content or template function.

        Args:
            documentation: Either a markdown string or function that takes (key, item) and returns markdown string
            button_background_color: Hex color of the button background
            button_hover_color: Hex color when hovering over button
            action_primary_color: Color for primary action buttons
            action_primary_hover: Hover color for primary action buttons
        """
        super().__init__(
            button_background_color=button_background_color,
            button_hover_color=button_hover_color,
            modal_type='documentation',
            action_primary_color=action_primary_color,
            action_primary_hover=action_primary_hover,
        )
        self.documentation = documentation

    def render(
        self, key: str, item: Any = None, button_text: str = '📖 Documentation'
    ) -> str:
        """
        Render the documentation button with all necessary HTML/CSS/JS.

        Args:
            key: Unique identifier for the item
            item: The registry item (optional, used if documentation is a function)
            button_text: Text to display on the button

        Returns:
            Complete HTML string with button, styles, and JavaScript
        """
        sanitized_key = self._sanitize_key(key)

        # Get documentation content
        if callable(self.documentation):
            doc_content = self.documentation(key, item)
        else:
            doc_content = self.documentation

        # Convert markdown to HTML
        html_content = self._markdown_to_html(doc_content)

        modal_title = f'Documentation: {key}'
        modal_body = f'<div class="doc-content">{html_content}</div>'
        modal_actions = f'<button class="action-btn action-btn-secondary" onclick="closeModal_{sanitized_key}()">Close</button>'

        additional_styles = self._get_documentation_styles()

        return self._render_modal_structure(
            key=key,
            sanitized_key=sanitized_key,
            button_text=button_text,
            modal_title=modal_title,
            modal_body_content=modal_body,
            modal_actions=modal_actions,
            additional_styles=additional_styles,
        )

    def _get_documentation_styles(self) -> str:
        """Get documentation-specific CSS styles."""
        return """
            .doc-content {
                color: #374151;
                line-height: 1.7;
            }

            .doc-content h1 {
                font-size: 28px;
                font-weight: 700;
                color: #111827;
                margin: 0 0 24px 0;
                border-bottom: 2px solid #e5e7eb;
                padding-bottom: 12px;
            }

            .doc-content h2 {
                font-size: 24px;
                font-weight: 600;
                color: #111827;
                margin: 32px 0 16px 0;
                border-bottom: 1px solid #e5e7eb;
                padding-bottom: 8px;
            }

            .doc-content h3 {
                font-size: 20px;
                font-weight: 600;
                color: #111827;
                margin: 28px 0 14px 0;
            }

            .doc-content h4 {
                font-size: 18px;
                font-weight: 600;
                color: #374151;
                margin: 24px 0 12px 0;
            }

            .doc-content h5, .doc-content h6 {
                font-size: 16px;
                font-weight: 600;
                color: #374151;
                margin: 20px 0 10px 0;
            }

            .doc-content p {
                margin: 16px 0;
                color: #4b5563;
                line-height: 1.7;
            }

            .doc-content ul, .doc-content ol {
                margin: 16px 0;
                padding-left: 24px;
                color: #4b5563;
            }

            .doc-content li {
                margin: 8px 0;
                line-height: 1.6;
            }

            .doc-content code {
                background: #f3f4f6;
                padding: 3px 8px;
                border-radius: 4px;
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
                font-size: 0.9em;
                color: #dc2626;
                border: 1px solid #e5e7eb;
            }

            .doc-content pre {
                background: #f9fafb;
                padding: 24px;
                border-radius: 8px;
                overflow-x: auto;
                margin: 20px 0;
                border: 1px solid #e5e7eb;
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
                font-size: 14px;
                line-height: 1.6;
                color: #374151;
            }

            .doc-content pre code {
                background: none;
                padding: 0;
                border: none;
                color: inherit;
                font-size: inherit;
            }

            .doc-content blockquote {
                border-left: 4px solid #3b82f6;
                background: #eff6ff;
                margin: 20px 0;
                padding: 16px 24px;
                border-radius: 0 6px 6px 0;
            }

            .doc-content blockquote p {
                margin: 0;
                color: #1e40af;
                font-style: italic;
            }

            .doc-content table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                border: 1px solid #e5e7eb;
                border-radius: 6px;
                overflow: hidden;
            }

            .doc-content th, .doc-content td {
                border: 1px solid #e5e7eb;
                padding: 12px 16px;
                text-align: left;
            }

            .doc-content th {
                background: #f9fafb;
                font-weight: 600;
                color: #111827;
            }

            .doc-content strong {
                font-weight: 600;
                color: #111827;
            }

            .doc-content em {
                font-style: italic;
                color: #4b5563;
            }

            .doc-content a {
                color: #3b82f6;
                text-decoration: none;
                border-bottom: 1px solid transparent;
                transition: border-color 0.15s;
            }

            .doc-content a:hover {
                border-bottom-color: #3b82f6;
            }
        """

    def _markdown_to_html(self, markdown: str) -> str:
        """Convert basic markdown to HTML."""
        html = markdown

        # Headers
        html = self._replace_headers(html)

        # Code blocks (must be before inline code)
        html = self._replace_code_blocks(html)

        # Inline code
        html = self._replace_inline_code(html)

        # Bold and italic
        html = self._replace_bold_italic(html)

        # Lists
        html = self._replace_lists(html)

        # Blockquotes
        html = self._replace_blockquotes(html)

        # Links
        html = self._replace_links(html)

        # Paragraphs
        html = self._replace_paragraphs(html)

        return html

    def _replace_headers(self, text: str) -> str:
        """Replace markdown headers with HTML."""
        for i in range(6, 0, -1):
            pattern = r'^' + '#' * i + r' (.+)$'
            replacement = f'<h{i}>\\1</h{i}>'
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        return text

    def _replace_code_blocks(self, text: str) -> str:
        """Replace markdown code blocks with HTML."""
        # Fenced code blocks
        pattern = r'```(\w+)?\n(.*?)\n```'
        replacement = r'<pre><code>\2</code></pre>'
        text = re.sub(pattern, replacement, text, flags=re.DOTALL)

        # Indented code blocks
        pattern = r'^(    .+)$'
        replacement = r'<pre><code>\1</code></pre>'
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

        return text

    def _replace_inline_code(self, text: str) -> str:
        """Replace inline code with HTML."""
        pattern = r'`([^`]+)`'
        replacement = r'<code>\1</code>'
        return re.sub(pattern, replacement, text)

    def _replace_bold_italic(self, text: str) -> str:
        """Replace bold and italic formatting."""
        # Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)

        # Italic
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)

        return text

    def _replace_lists(self, text: str) -> str:
        """Replace markdown lists with HTML."""
        lines = text.split('\n')
        result = []
        in_ul = False
        in_ol = False

        for line in lines:
            # Unordered list
            if re.match(r'^[-*+] ', line):
                if not in_ul:
                    if in_ol:
                        result.append('</ol>')
                        in_ol = False
                    result.append('<ul>')
                    in_ul = True
                content = re.sub(r'^[-*+] ', '', line)
                result.append(f'<li>{content}</li>')
            # Ordered list
            elif re.match(r'^\d+\. ', line):
                if not in_ol:
                    if in_ul:
                        result.append('</ul>')
                        in_ul = False
                    result.append('<ol>')
                    in_ol = True
                content = re.sub(r'^\d+\. ', '', line)
                result.append(f'<li>{content}</li>')
            else:
                if in_ul:
                    result.append('</ul>')
                    in_ul = False
                if in_ol:
                    result.append('</ol>')
                    in_ol = False
                result.append(line)

        # Close any open lists
        if in_ul:
            result.append('</ul>')
        if in_ol:
            result.append('</ol>')

        return '\n'.join(result)

    def _replace_blockquotes(self, text: str) -> str:
        """Replace markdown blockquotes with HTML."""
        lines = text.split('\n')
        result = []
        in_blockquote = False

        for line in lines:
            if line.startswith('> '):
                if not in_blockquote:
                    result.append('<blockquote>')
                    in_blockquote = True
                content = line[2:]  # Remove '> '
                result.append(f'<p>{content}</p>')
            else:
                if in_blockquote:
                    result.append('</blockquote>')
                    in_blockquote = False
                result.append(line)

        if in_blockquote:
            result.append('</blockquote>')

        return '\n'.join(result)

    def _replace_links(self, text: str) -> str:
        """Replace markdown links with HTML."""
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        replacement = r'<a href="\2" target="_blank">\1</a>'
        return re.sub(pattern, replacement, text)

    def _replace_paragraphs(self, text: str) -> str:
        """Wrap text in paragraph tags."""
        lines = text.split('\n')
        result = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith('<'):
                result.append(f'<p>{line}</p>')
            else:
                result.append(line)

        return '\n'.join(result)
