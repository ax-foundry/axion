from typing import Any, Callable

from axion.docs.render.base_modal import BaseModalButton


class UsageButton(BaseModalButton):
    """Usage button that can be embedded in any HTML."""

    def __init__(
        self,
        template: Callable[[str, Any], str],
        button_background_color: str = '#2D5F8A',  # Navy light
        button_hover_color: str = '#1E3A5F',  # Deep navy
        action_primary_color: str = '#2D5F8A',  # Match button color for consistency
        action_primary_hover: str = '#1E3A5F',  # Match hover color
        use_dark_theme: bool = False,
    ):
        """
        Initialize with a custom template function.

        Args:
            template: Function that takes (key, item) and returns usage string
            button_background_color: Hex color of the button background
            button_hover_color: Hex color when hovering over button
            action_primary_color: Color for primary action buttons (Copy button)
            action_primary_hover: Hover color for primary action buttons
        """
        super().__init__(
            button_background_color=button_background_color,
            button_hover_color=button_hover_color,
            modal_type='usage',
            action_primary_color=action_primary_color,
            action_primary_hover=action_primary_hover,
        )
        self.template = template
        self.use_dark_theme = use_dark_theme

    def render(self, key: str, item: Any, button_text: str = '💡 Usage Example') -> str:
        """
        Render the usage button with all necessary HTML/CSS/JS.

        Args:
            key: Unique identifier for the item
            item: The registry item
            button_text: Text to display on the button

        Returns:
            Complete HTML string with button, styles, and JavaScript
        """
        sanitized_key = self._sanitize_key(key)
        usage_content = self.template(key, item)

        modal_title = f'Usage Example: {key}'
        modal_body = f'<pre class="usage-code"><code>{usage_content}</code></pre>'
        modal_actions = f"""
            <button class="action-btn action-btn-primary" onclick="copyUsageCode_{sanitized_key}()">📋 Copy Code</button>
            <button class="action-btn action-btn-secondary" onclick="closeModal_{sanitized_key}()">Close</button>
        """

        additional_styles = self._get_usage_styles()
        additional_javascript = self._get_copy_javascript(sanitized_key, usage_content)

        return self._render_modal_structure(
            key=key,
            sanitized_key=sanitized_key,
            button_text=button_text,
            modal_title=modal_title,
            modal_body_content=modal_body,
            modal_actions=modal_actions,
            additional_styles=additional_styles,
            additional_javascript=additional_javascript,
        )

    def _get_usage_styles(self) -> str:
        """Get usage-specific CSS styles with theme support."""
        if self.use_dark_theme:
            # Dark theme styles
            return """
                .usage-code {
                    background: #1e2428;
                    color: #e8eaed;
                    padding: 24px;
                    border-radius: 8px;
                    overflow-x: auto;
                    margin: 0;
                    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
                    font-size: 14px;
                    line-height: 1.6;
                    border: 1px solid rgba(74, 144, 201, 0.25);
                    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
                }

                .usage-code code {
                    color: inherit;
                    background: none;
                    padding: 0;
                    border: none;
                    font-size: inherit;
                }

                /* Dark theme syntax highlighting colors */
                .usage-code .comment {
                    color: #7F8C8D;
                    font-style: italic;
                }

                .usage-code .keyword {
                    color: #4A90C9;
                    font-weight: 600;
                }

                .usage-code .string {
                    color: #10b981;
                }

                .usage-code .number {
                    color: #f59e0b;
                }

                .usage-code .function {
                    color: #4A90C9;
                }
            """
        else:
            # Light theme styles (default)
            return """
                .usage-code {
                    background: #f8fafc;
                    color: #1e293b;
                    padding: 24px;
                    border-radius: 8px;
                    overflow-x: auto;
                    margin: 0;
                    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
                    font-size: 14px;
                    line-height: 1.6;
                    border: 1px solid #e2e8f0;
                    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
                }

                .usage-code code {
                    color: inherit;
                    background: none;
                    padding: 0;
                    border: none;
                    font-size: inherit;
                }

                /* Light theme syntax highlighting colors */
                .usage-code .comment {
                    color: #7F8C8D;
                    font-style: italic;
                }

                .usage-code .keyword {
                    color: #1E3A5F;
                    font-weight: 600;
                }

                .usage-code .string {
                    color: #10b981;
                }

                .usage-code .number {
                    color: #d97706;
                }

                .usage-code .function {
                    color: #2D5F8A;
                }
            """

    def _get_copy_javascript(self, sanitized_key: str, usage_content: str) -> str:
        """Get copy functionality JavaScript."""
        escaped_content = self._escape_for_js(usage_content)

        return f"""
            function copyUsageCode_{sanitized_key}() {{
                const code = `{escaped_content}`;

                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(code).then(() => {{
                        showCopySuccess_{sanitized_key}();
                    }}).catch(() => {{
                        fallbackCopy_{sanitized_key}(code);
                    }});
                }} else {{
                    fallbackCopy_{sanitized_key}(code);
                }}
            }}

            function showCopySuccess_{sanitized_key}() {{
                const btn = document.querySelector('.action-btn-primary');
                if (btn) {{
                    const originalText = btn.textContent;
                    const originalBg = btn.style.background;
                    btn.textContent = '✅ Copied!';
                    btn.style.background = '#10b981';
                    setTimeout(() => {{
                        btn.textContent = originalText;
                        btn.style.background = originalBg;
                    }}, 2000);
                }}
            }}

            function fallbackCopy_{sanitized_key}(text) {{
                const textArea = document.createElement('textarea');
                textArea.value = text;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                textArea.style.top = '-999999px';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();

                try {{
                    document.execCommand('copy');
                    showCopySuccess_{sanitized_key}();
                }} catch (err) {{
                    console.error('Copy failed:', err);
                    alert('Copy failed. Please select and copy the text manually.');
                }}

                document.body.removeChild(textArea);
            }}
        """
