import re
from typing import Any


class BaseModalButton:
    """Base class for modal buttons that can be embedded in HTML."""

    def __init__(
        self,
        button_background_color: str = '#1E3A5F',
        button_hover_color: str = '#0F2440',
        modal_type: str = 'generic',
        action_primary_color: str = '#1E3A5F',  # Match button color for consistency
        action_primary_hover: str = '#0F2440',  # Match hover color
        action_secondary_color: str = '#7F8C8D',  # Muted for secondary actions
        action_secondary_hover: str = '#34495E',  # Secondary text for hover
    ):
        """
        Initialize the base modal button.

        Args:
            button_background_color: Hex color of the button background
            button_hover_color: Hex color when hovering over button
            modal_type: Type identifier for styling purposes
        """
        self.button_background_color = button_background_color
        self.button_hover_color = button_hover_color
        self.modal_type = modal_type
        self.action_primary_color = action_primary_color
        self.action_primary_hover = action_primary_hover
        self.action_secondary_color = action_secondary_color
        self.action_secondary_hover = action_secondary_hover

    def _sanitize_key(self, key: str) -> str:
        """Sanitize key for use in JavaScript function names."""
        return re.sub(r'[^a-zA-Z0-9_]', '_', key)

    def _escape_for_js(self, content: str) -> str:
        """Escape content for safe JavaScript embedding."""
        return (
            content.replace('\\', '\\\\')
            .replace('`', '\\`')
            .replace('$', '\\$')
            .replace('"', '\\"')
            .replace('\n', '\\n')
            .replace('\r', '\\r')
        )

    def _get_base_styles(self) -> str:
        """Get shared CSS styles for all modal types."""
        return f"""
            .modal-button-container {{
                display: inline-block;
                margin: 2px;
            }}

            .modal-btn {{
                padding: 10px 18px;
                background: {self.button_background_color};
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06);
                position: relative;
                overflow: hidden;
            }}

            .modal-btn:hover {{
                background: {self.button_hover_color};
                transform: translateY(-1px);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.06);
            }}

            .modal-btn:active {{
                transform: translateY(0);
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }}

            .modal-btn::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }}

            .modal-btn:hover::before {{
                left: 100%;
            }}

            .base-modal {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                backdrop-filter: blur(4px);
                z-index: 10000;
                display: flex;
                align-items: center;
                justify-content: center;
                animation: fadeIn 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                padding: 20px;
                box-sizing: border-box;
            }}

            .base-modal-content {{
                background: white;
                border-radius: 12px;
                max-width: 800px;
                width: 100%;
                max-height: 90vh;
                overflow: hidden;
                box-shadow: 0 25px 50px rgba(0, 0, 0, 0.25);
                animation: slideIn 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                display: flex;
                flex-direction: column;
            }}

            .base-modal-header {{
                padding: 24px 32px 20px;
                border-bottom: 1px solid #e2e8f0;
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: #f8fafc;
                flex-shrink: 0;
            }}

            .base-modal-header h3 {{
                margin: 0;
                color: #1E3A5F;
                font-size: 20px;
                font-weight: 700;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }}

            .base-modal-close {{
                background: none;
                border: none;
                font-size: 24px;
                cursor: pointer;
                color: #7F8C8D;
                padding: 8px;
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 8px;
                transition: all 0.15s ease;
            }}

            .base-modal-close:hover {{
                background: #e2e8f0;
                color: #34495E;
            }}

            .base-modal-body {{
                padding: 32px;
                overflow-y: auto;
                flex: 1;
                line-height: 1.6;
            }}

            .base-modal-actions {{
                display: flex;
                gap: 12px;
                padding: 20px 32px 24px;
                border-top: 1px solid #e2e8f0;
                background: #f8fafc;
                flex-shrink: 0;
            }}

            .action-btn {{
                padding: 10px 16px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                transition: all 0.15s ease;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            }}

            .action-btn-primary {{
                background: {self.action_primary_color};
                color: white;
            }}

            .action-btn-primary:hover {{
                background: {self.action_primary_hover};
            }}

            .action-btn-secondary {{
                background: {self.action_secondary_color};
                color: white;
            }}

            .action-btn-secondary:hover {{
                background: {self.action_secondary_hover};
            }}

            @keyframes fadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}

            @keyframes slideIn {{
                from {{
                    transform: translateY(-20px) scale(0.95);
                    opacity: 0;
                }}
                to {{
                    transform: translateY(0) scale(1);
                    opacity: 1;
                }}
            }}

            @media (max-width: 768px) {{
                .base-modal {{
                    padding: 10px;
                }}

                .base-modal-content {{
                    max-height: 95vh;
                    border-radius: 8px;
                }}

                .base-modal-header,
                .base-modal-body,
                .base-modal-actions {{
                    padding-left: 20px;
                    padding-right: 20px;
                }}
            }}
        """

    def _get_base_javascript(self, key: str, sanitized_key: str) -> str:
        """Get shared JavaScript functionality."""
        return f"""
            function closeModal_{sanitized_key}() {{
                const modal = document.querySelector('.base-modal[data-key="{sanitized_key}"]');
                if (modal) {{
                    modal.style.opacity = '0';
                    modal.style.transform = 'scale(0.95)';
                    setTimeout(() => {{
                        if (modal.parentNode) {{
                            modal.parentNode.removeChild(modal);
                        }}
                    }}, 300);
                }}
            }}

            function setupModalEvents_{sanitized_key}(modal) {{
                // Close on backdrop click
                modal.addEventListener('click', function(e) {{
                    if (e.target === modal) {{
                        closeModal_{sanitized_key}();
                    }}
                }});

                // Close on Escape key
                const escapeHandler = function(e) {{
                    if (e.key === 'Escape') {{
                        closeModal_{sanitized_key}();
                        document.removeEventListener('keydown', escapeHandler);
                    }}
                }};
                document.addEventListener('keydown', escapeHandler);
            }}
        """

    def render(self, key: str, item: Any = None, button_text: str = '🔍 View') -> str:
        """
        Base render method - should be overridden by subclasses.

        Args:
            key: Unique identifier for the item
            item: The registry item (optional)
            button_text: Text to display on the button

        Returns:
            Complete HTML string with button, styles, and JavaScript
        """
        raise NotImplementedError('Subclasses must implement the render method')

    def _render_modal_structure(
        self,
        key: str,
        sanitized_key: str,
        button_text: str,
        modal_title: str,
        modal_body_content: str,
        modal_actions: str,
        additional_styles: str = '',
        additional_javascript: str = '',
    ) -> str:
        """
        Render the complete modal structure.

        Args:
            key: Original key
            sanitized_key: Sanitized key for JavaScript
            button_text: Button display text
            modal_title: Modal header title
            modal_body_content: HTML content for modal body
            modal_actions: HTML content for modal actions
            additional_styles: Additional CSS styles
            additional_javascript: Additional JavaScript functions

        Returns:
            Complete HTML string
        """
        escaped_body = self._escape_for_js(modal_body_content)
        escaped_actions = self._escape_for_js(modal_actions)

        return f"""
        <div class="modal-button-container">
            <button class="modal-btn" onclick="showModal_{sanitized_key}()">{button_text}</button>
        </div>

        <style>
            {self._get_base_styles()}
            {additional_styles}
        </style>

        <script>
            {self._get_base_javascript(key, sanitized_key)}

            function showModal_{sanitized_key}() {{
                const modal = document.createElement('div');
                modal.className = 'base-modal';
                modal.setAttribute('data-key', '{sanitized_key}');
                modal.innerHTML = `
                    <div class="base-modal-content">
                        <div class="base-modal-header">
                            <h3>{modal_title}</h3>
                            <button class="base-modal-close" onclick="closeModal_{sanitized_key}()">&times;</button>
                        </div>
                        <div class="base-modal-body">
                            {escaped_body}
                        </div>
                        <div class="base-modal-actions">
                            {escaped_actions}
                        </div>
                    </div>
                `;

                document.body.appendChild(modal);
                setupModalEvents_{sanitized_key}(modal);
            }}

            {additional_javascript}
        </script>
        """
