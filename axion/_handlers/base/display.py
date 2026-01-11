import json
from datetime import datetime
from typing import Dict, List, Union

from pydantic import BaseModel


def display_pydantic(
    obj: Union[BaseModel, List[BaseModel], Dict[str, BaseModel]],
    title: str = 'Model Output',
    str_limit: int = 1000,
    **kwargs,
):
    """
    Generates a fancy HTML visualization of Pydantic model instances.

    Usage -- use to show input and output pydantic -- title can be name

    Args:
        obj: A Pydantic model instance, list of instances, or dictionary of instances
        title: Title to display at the top of the visualization
        str_limit: Max character limit for visualization

    Returns:
        IPython.display.HTML: A rendered HTML visualization
    """

    import json

    from IPython.display import HTML, display

    # Helper function to determine the CSS class for a given value
    def get_value_class(value):
        if isinstance(value, (dict, list, BaseModel)):
            return 'complex-value'
        elif isinstance(value, str):
            return 'string-value'
        elif isinstance(value, bool):
            return 'boolean-value'
        elif isinstance(value, (int, float)):
            return 'number-value'
        elif value is None:
            return 'null-value'
        return 'default-value'

    # Recursive function to format complex nested structures
    def format_complex_value(value, level=0):
        if level > 10:  # Prevent infinite recursion
            return (
                "<span class='recursion-limit'>Maximum recursion depth reached</span>"
            )

        indent = '  ' * level

        if isinstance(value, list):
            if not value:
                return '[]'

            items = []
            for item in value:
                if isinstance(item, BaseModel):
                    item = json.dumps(item.model_dump(exclude_none=True), indent=2)
                formatted_item = format_complex_value(item, level + 1)
                items.append(f'{indent}  {formatted_item}')

            return '[\n' + ',\n'.join(items) + f'\n{indent}]'

        elif isinstance(value, dict):
            if not value:
                return '{}'

            items = []
            for key, val in value.items():
                if isinstance(item, BaseModel):
                    item = json.dumps(item.model_dump(exclude_none=True), indent=2)
                formatted_val = format_complex_value(val, level + 1)
                items.append(f'{indent}  "{key}": {formatted_val}')

            return '{\n' + ',\n'.join(items) + f'\n{indent}}}'

        elif isinstance(value, BaseModel):
            return f"<span class='model-reference'>{value.__class__.__name__} instance</span>"
        elif isinstance(value, str):
            return f'"{value}"'
        elif value is None:
            return 'null'
        else:
            return str(value)

    # Helper function to format a value for display
    def format_value(value):
        if isinstance(value, BaseModel):
            return f"<span class='model-reference'>{value.__class__.__name__} instance</span>"
        elif isinstance(value, (dict, list)):
            try:
                # First try advanced recursive formatting for complex nested structures
                formatted = format_complex_value(value)
                if len(formatted) > str_limit:
                    formatted = formatted[:str_limit] + '...'
                return f"<pre class='code-block'>{formatted}</pre>"
            except:
                # Fallback to simple JSON formatting
                try:
                    formatted = json.dumps(
                        value, default=lambda o: f'<{o.__class__.__name__}>', indent=2
                    )
                    if len(formatted) > str_limit:
                        formatted = formatted[:str_limit] + '...'
                    return f"<pre class='code-block'>{formatted}</pre>"
                except:
                    return "<pre class='code-block error-value'>Error formatting complex value</pre>"
        elif isinstance(value, str):
            if len(value) > str_limit:
                formatted = value[:str_limit] + '...'
                return f"<div class='str-value'>{formatted}</div>"
            return f"<div class='str-value'>{value}</div>"
        elif value is None:
            return "<span class='null-value'>null</span>"
        return str(value)

    # Create cards for each model instance
    def create_model_card(instance, instance_name=None):
        model_class = instance.__class__
        model_name = model_class.__name__
        model_doc = model_class.__doc__ or ''
        model_doc = model_doc.strip()

        card_html = f"""
        <div class="topic-card">
            <div class="topic-card-header">
                <div class="model-icon-title">
                    <div class="model-icon">
                        <div class="model-icon-inner">{model_name[0]}</div>
                    </div>
                    <div>
                        <h3 class="topic-card-name">{model_name}</h3>
                        {f'<p class="instance-name">{instance_name}</p>' if instance_name else ''}
                    </div>
                </div>
            </div>
            <div class="topic-card-body">
        """

        # Get all attributes and their values
        attributes = {}
        for field_name, field_value in instance:
            attributes[field_name] = field_value

        # Generate HTML for each attribute
        if attributes:
            for field_name, field_value in attributes.items():
                value_class = get_value_class(field_value)
                formatted_value = format_value(field_value)
                field_color = f'hsl({hash(field_name) % 360}, 70%, 94%)'

                card_html += f"""
                <div class="instruction-item" style="border-left-color: {field_color};">
                    <span class="item-name instruction-name">{field_name}</span>
                    <div class="field-value {value_class}">{formatted_value}</div>
                </div>
                """
        else:
            card_html += '<p class="empty-note">No attributes available</p>'

        card_html += """
            </div>
        </div>
        """
        return card_html

    # Generate all cards
    cards_html = ''

    if isinstance(obj, BaseModel):
        cards_html += create_model_card(obj)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, BaseModel):
                cards_html += create_model_card(item, f'Item {i}')
    elif isinstance(obj, dict):
        for key, item in obj.items():
            if isinstance(item, BaseModel):
                cards_html += create_model_card(item, key)

    # CSS styles
    styles = """
    <style>
        .pydantic-container {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1200px;
            margin: 20px auto;
            padding: 30px;
            background-color: #f8fafc;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.08);
            position: relative;
            overflow: hidden;
            animation: fadeIn 0.6s ease-out;
        }

        .pydantic-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 6px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
            z-index: 1;
        }

        .pydantic-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 16px;
        }

        .pydantic-title {
            font-size: 22px;
            font-weight: 700;
            color: #1e293b;
            margin: 0;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .pydantic-count {
            background-color: #3b82f6;
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 0.5px;
            box-shadow: 0 2px 5px rgba(59, 130, 246, 0.2);
        }

        .topics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 25px;
        }

        .topic-card {
            background-color: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            transition: transform 0.2s, box-shadow 0.2s;
            margin-bottom: 25px;
            border: 1px solid #e2e8f0;
        }

        .topic-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }

        .topic-card-header {
            padding: 20px;
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            color: white;
        }

        .model-icon-title {
            display: flex;
            align-items: center;
        }

        .model-icon {
            width: 42px;
            height: 42px;
            margin-right: 16px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        .model-icon-inner {
            font-size: 24px;
        }

        .topic-card-name {
            font-size: 1.25rem;
            font-weight: 700;
            margin: 0 0 10px 0;
        }

        .topic-card-description {
            font-size: 0.95rem;
            opacity: 0.9;
            margin: 0;
        }

        .instance-name {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 8px;
            padding: 3px 8px;
            background-color: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            display: inline-block;
        }

        .topic-card-body {
            padding: 20px;
        }

        .instruction-item {
            background-color: #f8fafc;
            border-left: 4px solid #f59e0b;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 0 5px 5px 0;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
            position: relative;
            overflow: hidden;
        }

        .instruction-item::after {
            content: '';
            position: absolute;
            bottom: 0;
            right: 0;
            width: 100px;
            height: 100px;
            background: radial-gradient(circle at bottom right,
                                       rgba(59, 130, 246, 0.04),
                                       transparent 70%);
            z-index: 0;
            opacity: 0.5;
        }

        .item-name {
            font-weight: 600;
            color: #334155;
            display: block;
            margin-bottom: 10px;
            font-size: 1rem;
        }

        .instruction-name {
            color: #1e293b;
            display: flex;
            align-items: center;
        }

        .instruction-name::before {
            content: '🔹';
            margin-right: 8px;
            font-size: 14px;
        }

        .item-description {
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 8px;
        }

        .empty-note {
            color: #94a3b8;
            font-style: italic;
            padding: 10px;
        }

        .field-value {
            font-family: 'Fira Code', Consolas, monospace;
            padding: 12px;
            background-color: #f1f5f9;
            border-radius: 6px;
            overflow-x: auto;
            line-height: 1.5;
            position: relative;
            z-index: 1;
        }

        .string-value {
            color: #0369a1;
        }

        .number-value {
            color: #b91c1c;
        }

        .boolean-value {
            color: #4f46e5;
        }

        .null-value {
            color: #9ca3af;
            font-style: italic;
        }

        .complex-value {
            color: #1e293b;
        }

        .model-reference {
            color: #047857;
            font-weight: bold;
        }

        .recursion-limit {
            color: #ef4444;
            font-style: italic;
        }

        .error-value {
            color: #ef4444;
            background-color: #fee2e2;
            border-left: 3px solid #ef4444;
        }

        .code-block {
            margin: 0;
            max-height: 500px;
            overflow-y: auto;
            background-color: #272822;
            color: #f8f8f2;
            padding: 12px;
            border-radius: 6px;
            font-size: 14px;
            white-space: pre;
        }

        .str-value {
            white-space: pre-wrap;
            word-break: break-word;
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .pydantic-container {
                margin: 10px;
                padding: 15px;
            }

            .topics-grid {
                grid-template-columns: 1fr;
            }

            .pydantic-header {
                flex-direction: column;
                align-items: flex-start;
            }

            .code-block {
                font-size: 12px;
                padding: 10px;
            }
        }
    </style>
    """

    # Assemble the final HTML
    count_text = ''
    if isinstance(obj, list):
        count_text = f'{len(obj)} Items'
    elif isinstance(obj, dict):
        count_text = f'{len(obj)} Items'

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        {styles}
    </head>
    <body>
        <div class="pydantic-container">
            <div class="pydantic-header">
                <h1 class="pydantic-title">{title}</h1>
                {f'<span class="pydantic-count">{count_text}</span>' if count_text else ''}
            </div>
            <div class="topics-grid">
                {cards_html}
            </div>
            <div class="prompt-footer">
                <div class="prompt-footer-left">
                    <span class="prompt-footer-icon">⚡</span>
                    <span>Powered by AXION</span>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return display(HTML(html_content))


def display_execution_metadata(execution_data):
    from IPython.display import HTML, display

    # CSS styles - based on your template
    styles = """
    <style>
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f9f9f9;
        }
        .metadata-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metadata-header {
            border-bottom: 1px solid #eaeaea;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
        .metadata-title {
            font-size: 1.5em;
            color: #2c3e50;
            margin: 0;
        }
        .metadata-id {
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .section {
            margin-bottom: 25px;
        }
        .section-title {
            font-weight: bold;
            color: #2c3e50;
            font-size: 1.2em;
            margin-bottom: 10px;
            border-bottom: 1px solid #eaeaea;
            padding-bottom: 5px;
        }
        .property-row {
            display: flex;
            margin-bottom: 8px;
            padding: 5px 0;
        }
        .property-name {
            font-weight: bold;
            width: 30%;
            color: #34495e;
        }
        .property-value {
            flex: 1;
            color: #34495e;
        }
        .trace-item {
            background-color: #f8f9fa;
            border-left: 3px solid #3498db;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        .trace-timestamp {
            color: #7f8c8d;
            font-size: 0.85em;
            margin-bottom: 5px;
        }
        .trace-type {
            font-weight: bold;
            color: #2c3e50;
        }
        .trace-message {
            margin: 5px 0;
        }
        .metadata-block {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            white-space: pre-wrap;
            margin-top: 5px;
        }
        .status-started {
            color: #f39c12;
            font-weight: bold;
        }
        .status-completed {
            color: #2ecc71;
            font-weight: bold;
        }
        .status-failed {
            color: #e74c3c;
            font-weight: bold;
        }
        .llm-call {
            background-color: #eafaf1;
            border-left: 3px solid #2ecc71;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        .retrieval-call {
            background-color: #ebf5fb;
            border-left: 3px solid #3498db;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        .db-call {
            background-color: #fef9e7;
            border-left: 3px solid #f1c40f;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
    </style>
    """

    # Helper function to format timestamps
    def format_datetime(dt):
        if not dt:
            return 'N/A'
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
            except ValueError:
                return dt
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    # Helper function to format dictionary as pretty HTML
    def format_dict(d):
        if not d:
            return 'None'

        try:
            # Format as pretty JSON if possible
            return f'<div class="metadata-block">{json.dumps(d, indent=2, default=str)}</div>'
        except:
            # Fallback for non-serializable objects
            return str(d)

    # Helper function to get status class
    def get_status_class(status):
        status = str(status).lower()
        if status in ['started']:
            return 'status-started'
        elif status in ['completed', 'success']:
            return 'status-completed'
        elif status in ['failed', 'error']:
            return 'status-failed'
        return ''

    # Start building HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        {styles}
    </head>
    <body>
        <div class="metadata-container">
            <div class="metadata-header">
                <h2 class="metadata-title">Execution Metadata</h2>
                <div class="metadata-id">ID: {execution_data.get('id', 'N/A')}</div>
                <div class="metadata-id">Name: {execution_data.get('name', 'N/A')}</div>
                <div class="metadata-id">Session: {execution_data.get('session_id', 'N/A')}</div>
            </div>
    """

    # Execution timing section
    html_content += """
            <div class="section">
                <div class="section-title">Execution Timing</div>
    """

    status_class = get_status_class(execution_data.get('status', ''))

    html_content += f"""
                <div class="property-row">
                    <div class="property-name">Status</div>
                    <div class="property-value"><span class="{status_class}">{execution_data.get('status', 'N/A')}</span></div>
                </div>
                <div class="property-row">
                    <div class="property-name">Start Time</div>
                    <div class="property-value">{format_datetime(execution_data.get('start_time'))}</div>
                </div>
                <div class="property-row">
                    <div class="property-name">End Time</div>
                    <div class="property-value">{format_datetime(execution_data.get('end_time'))}</div>
                </div>
    """

    if execution_data.get('error'):
        html_content += f"""
                <div class="property-row">
                    <div class="property-name">Error</div>
                    <div class="property-value" style="color: #e74c3c;">{execution_data.get('error')}</div>
                </div>
        """

    html_content += """
            </div>
    """

    # Tool metadata section
    if execution_data.get('tool_metadata'):
        html_content += """
            <div class="section">
                <div class="section-title">Tool Metadata</div>
        """

        for key, value in execution_data.get('tool_metadata', {}).items():
            if key == 'run_metadata' and isinstance(value, dict):
                html_content += f"""
                <div class="property-row">
                    <div class="property-name">{key.replace('_', ' ').title()}</div>
                    <div class="property-value">{format_dict(value)}</div>
                </div>
                """
            else:
                html_content += f"""
                <div class="property-row">
                    <div class="property-name">{key.replace('_', ' ').title()}</div>
                    <div class="property-value">{value}</div>
                </div>
                """

        html_content += """
            </div>
        """

    # Input/Output data section
    for data_type in ['input_data', 'output_data']:
        if execution_data.get(data_type):
            html_content += f"""
                <div class="section">
                    <div class="section-title">{data_type.replace('_', ' ').title()}</div>
                    <div class="property-value">{format_dict(execution_data.get(data_type))}</div>
                </div>
            """

    # Traces section
    if execution_data.get('traces'):
        html_content += """
            <div class="section">
                <div class="section-title">Execution Traces</div>
        """

        for trace in execution_data.get('traces', []):
            html_content += f"""
                <div class="trace-item">
                    <div class="trace-timestamp">{format_datetime(trace.get('timestamp'))}</div>
                    <div class="trace-type">{trace.get('event_type', 'N/A').replace('_', ' ').title()}</div>
                    <div class="trace-message">{trace.get('message', 'N/A')}</div>
            """

            if trace.get('metadata'):
                html_content += f"""
                    <div>{format_dict(trace.get('metadata'))}</div>
                """

            html_content += """
                </div>
            """

        html_content += """
            </div>
        """

    # LLM Calls section
    if execution_data.get('llm_calls'):
        html_content += f"""
            <div class="section">
                <div class="section-title">LLM Calls ({len(execution_data.get('llm_calls', []))})</div>
        """

        for call in execution_data.get('llm_calls', []):
            html_content += f"""
                <div class="llm-call">
                    <div class="property-row">
                        <div class="property-name">Model</div>
                        <div class="property-value">{call.get('model', 'N/A')}</div>
                    </div>
                    <div class="property-row">
                        <div class="property-name">Provider</div>
                        <div class="property-value">{call.get('provider', 'N/A')}</div>
                    </div>
                    <div class="property-row">
                        <div class="property-name">Tokens</div>
                        <div class="property-value">Prompt: {call.get('prompt_tokens', 0)} | Completion: {call.get('completion_tokens', 0)} | Total: {call.get('total_tokens', 0)}</div>
                    </div>
                    <div class="property-row">
                        <div class="property-name">Latency</div>
                        <div class="property-value">{call.get('latency', 'N/A')} seconds</div>
                    </div>
                    <div class="property-row">
                        <div class="property-name">Timestamp</div>
                        <div class="property-value">{format_datetime(call.get('timestamp'))}</div>
                    </div>
                </div>
            """

        html_content += """
            </div>
        """

    # Database Calls section (if available)
    if any(
        isinstance(item.get('rows_affected'), int)
        for item in execution_data.get('db_calls', [])
    ):
        html_content += f"""
            <div class="section">
                <div class="section-title">Database Calls ({len(execution_data.get('db_calls', []))})</div>
        """

        for call in execution_data.get('db_calls', []):
            html_content += f"""
                <div class="db-call">
                    <div class="property-row">
                        <div class="property-name">Rows Affected</div>
                        <div class="property-value">{call.get('rows_affected', 0)}</div>
                    </div>
                    <div class="property-row">
                        <div class="property-name">Query Time</div>
                        <div class="property-value">{call.get('query_time', 0.0)} seconds</div>
                    </div>
            """

            if call.get('query_params'):
                html_content += f"""
                    <div class="property-row">
                        <div class="property-name">Query Params</div>
                        <div class="property-value">{format_dict(call.get('query_params'))}</div>
                    </div>
                """

            html_content += """
                </div>
            """

        html_content += """
            </div>
        """

    # Retrieved calls section
    if execution_data.get('retrieved_calls'):
        html_content += f"""
            <div class="section">
                <div class="section-title">Retrieved Documents ({len(execution_data.get('retrieved_calls', []))})</div>
        """

        for call in execution_data.get('retrieved_calls', []):
            html_content += f"""
                <div class="retrieval-call">
                    <div class="property-row">
                        <div class="property-name">ID</div>
                        <div class="property-value">{call.get('id_', 'Unknown')}</div>
                    </div>
                    <div class="property-row">
                        <div class="property-name">File Name</div>
                        <div class="property-value">{call.get('file_name', 'N/A')}</div>
                    </div>
            """

            if call.get('text'):
                # Truncate text if too long
                text = call.get('text', '')
                if len(text) > 200:
                    displayed_text = text[:200] + '...'
                else:
                    displayed_text = text

                html_content += f"""
                    <div class="property-row">
                        <div class="property-name">Content</div>
                        <div class="property-value">{displayed_text}</div>
                    </div>
                """

            html_content += """
                </div>
            """

        html_content += """
            </div>
        """

    # Token usage summary
    html_content += f"""
            <div class="section">
                <div class="section-title">Summary</div>
                <div class="property-row">
                    <div class="property-name">Total Calls</div>
                    <div class="property-value">{execution_data.get('number_of_calls', 0)}</div>
                </div>
                <div class="property-row">
                    <div class="property-name">Total Tokens</div>
                    <div class="property-value">{execution_data.get('total_tokens', 0)}</div>
                </div>
            </div>
    """

    # Close HTML tags
    html_content += """
        </div>
    </body>
    </html>
    """

    return display(HTML(html_content))


def display_prompt(prompt: str, query: str | BaseModel):
    """
    Displays visually enhanced LLM prompts in a Jupyter notebook or compatible environment.

    Args:
        prompt (str): Raw text of the reasoning prompt
        query (Any): Query text or a Pydantic model
    """
    import json

    from IPython.display import HTML, display

    # Process query
    query_html = ''

    if hasattr(query, '__pydantic_fields_set__') or (
        hasattr(query, '__class__') and hasattr(query.__class__, 'model_fields')
    ):
        # Handle Pydantic model (both v1 and v2)
        try:
            # Try to get model as dict - works for both Pydantic v1 and v2
            query_dict = (
                query.model_dump() if hasattr(query, 'model_dump') else query.dict()
            )

            # Convert to JSON with indentation
            json_str = json.dumps(query_dict, indent=2)

            # Create visual representation for Pydantic model
            visual_html = '<div class="model-fields">'
            for key, value in query_dict.items():
                field_color = f'hsl({hash(key) % 360}, 70%, 94%)'
                visual_html += f"""
                <div class="model-field" style="background-color: {field_color}">
                    <div class="field-name">{key}</div>
                    <div class="field-value">{str(value)}</div>
                </div>
                """
            visual_html += '</div>'

            # Include both JSON and visual representation
            query_html = f"""
            <div class="json-data">
                <pre>{json_str}</pre>
            </div>
            {visual_html}
            """

        except Exception:
            # Fallback to string representation if conversion fails
            query_html = f"<p class='query-fallback'>{str(query)}</p>"
    else:
        # Handle regular string or other types
        query_html = f"<p class='query-string'>{str(query)}</p>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            /* Modern styling for the prompt display */
            .prompt-container {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 900px;
                margin: 20px auto;
                padding: 24px;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                background-color: #f8f9fa;
                position: relative;
                overflow: hidden;
            }}

            .prompt-container::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 6px;
                background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
                z-index: 1;
            }}

            /* Header styles */
            .prompt-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 20px;
                border-bottom: 2px solid #e2e8f0;
                padding-bottom: 16px;
                position: relative;
            }}

            .prompt-title-area {{
                display: flex;
                align-items: center;
            }}

            .prompt-icon {{
                width: 42px;
                height: 42px;
                margin-right: 16px;
                background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                color: white;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3);
            }}

            .prompt-icon-inner {{
                font-size: 24px;
            }}

            .prompt-title {{
                font-size: 22px;
                font-weight: 700;
                color: #1e293b;
                margin: 0;
                background: linear-gradient(90deg, #3b82f6, #8b5cf6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}

            .prompt-subtitle {{
                font-size: 14px;
                color: #475569;
                margin: 4px 0 0 0;
            }}

            .prompt-badge {{
                background-color: #3b82f6;
                color: white;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 0.5px;
                box-shadow: 0 2px 5px rgba(59, 130, 246, 0.2);
            }}

            /* Query section */
            .query-section {{
                background-color: #f1f5f9;
                border-left: 4px solid #3b82f6;
                padding: 16px;
                margin-bottom: 24px;
                border-radius: 0 12px 12px 0;
                box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08);
                position: relative;
                overflow: hidden;
            }}

            .query-section::after {{
                content: '';
                position: absolute;
                bottom: 0;
                right: 0;
                width: 150px;
                height: 150px;
                background: radial-gradient(circle at bottom right,
                                           rgba(59, 130, 246, 0.08),
                                           transparent 70%);
                z-index: 0;
                opacity: 0.5;
            }}

            .query-label {{
                font-weight: 600;
                color: #3b82f6;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
                font-size: 15px;
            }}

            .query-label::before {{
                content: '🔍';
                margin-right: 8px;
            }}

            .query-text {{
                margin: 0;
                color: #1e293b;
                position: relative;
                z-index: 1;
            }}

            /* Model fields visual representation */
            .model-fields {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 12px;
                margin-top: 16px;
            }}

            .model-field {{
                border-radius: 8px;
                padding: 12px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                position: relative;
                overflow: hidden;
                transition: transform 0.2s ease;
            }}

            .model-field:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
            }}

            .field-name {{
                font-weight: 600;
                font-size: 14px;
                margin-bottom: 6px;
                border-bottom: 1px solid rgba(0, 0, 0, 0.1);
                padding-bottom: 4px;
            }}

            .field-value {{
                font-size: 13px;
                word-break: break-word;
            }}

            /* JSON formatting */
            .json-data {{
                margin-top: 10px;
            }}

            .json-data pre {{
                background-color: #272822;
                color: #f8f8f2;
                padding: 16px;
                border-radius: 8px;
                font-family: 'Fira Code', 'Monaco', 'Consolas', monospace;
                font-size: 14px;
                line-height: 1.5;
                overflow-x: auto;
                max-height: 300px;
                overflow-y: auto;
            }}

            /* Content sections */
            .prompt-section {{
                margin-bottom: 24px;
                position: relative;
            }}

            .section-title {{
                font-weight: 600;
                margin-bottom: 10px;
                color: #1e293b;
                display: flex;
                align-items: center;
                font-size: 18px;  /* Increased font size */
                justify-content: space-between;
            }}

            .section-title::before {{
                content: '📄';
                margin-right: 8px;
                font-size: 20px;  /* Increased font size */
            }}

            .copy-indicator {{
                color: #64748b;
                font-size: 14px;
                background-color: #e2e8f0;
                padding: 4px 10px;
                border-radius: 4px;
                display: flex;
                align-items: center;
                cursor: pointer;
            }}

            .copy-indicator::before {{
                content: '📋';
                margin-right: 5px;
            }}

            /* Code block styles */
            .code-block {{
                background-color: #272822;
                color: #f8f8f2;
                padding: 25px;  /* Increased padding */
                border-radius: 10px;
                overflow-x: auto;
                margin: 10px 0;
                font-family: 'Fira Code', 'Monaco', 'Consolas', monospace;
                font-size: 16px;  /* Increased font size */
                line-height: 1.6;  /* Increased line height */
                position: relative;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                white-space: pre;
                min-height: 200px;  /* Minimum height */
                max-height: 600px;  /* Maximum height */
                overflow-y: auto;
            }}

            /* Footer */
            .prompt-footer {{
                margin-top: 24px;
                padding-top: 16px;
                border-top: 2px solid #e2e8f0;
                font-size: 13px;
                color: #64748b;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}

            .prompt-footer-left {{
                display: flex;
                align-items: center;
            }}

            .prompt-footer-right {{
                display: flex;
                align-items: center;
            }}

            .prompt-footer-icon {{
                margin-right: 8px;
                font-size: 16px;
            }}

            /* Animations */
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}

            .prompt-container {{
                animation: fadeIn 0.6s ease-out;
            }}

            /* Responsive design */
            @media (max-width: 768px) {{
                .prompt-container {{
                    margin: 10px;
                    padding: 15px;
                }}

                .model-fields {{
                    grid-template-columns: 1fr;
                }}

                .prompt-header {{
                    flex-direction: column;
                    align-items: flex-start;
                }}

                .code-block {{
                    font-size: 14px;
                    padding: 15px;
                }}
            }}
        </style>
    </head>
    <body>
        <!-- Enhanced Prompt Display -->
        <div class="prompt-container">
            <div class="prompt-header">
                <div class="prompt-title-area">
                    <div class="prompt-icon">
                        <div class="prompt-icon-inner">P</div>
                    </div>
                    <div>
                        <h3 class="prompt-title">LLM Prompt Visualizer</h3>
                        <p class="prompt-subtitle">Input configuration for language model</p>
                    </div>
                </div>
                <div>
                    <span class="prompt-badge">AI Input</span>
                </div>
            </div>

            <div class="query-section">
                <span class="query-label">Query Input</span>
                <div class="query-text">{query_html}</div>
            </div>

            <div class="prompt-section">
                <div class="section-title">
                    <span>Prompt Template</span>
                    <span class="copy-indicator" title="Select and copy to clipboard">Prompt passed to LLM</span>
                </div>
                <pre class="code-block">{prompt}</pre>
            </div>

            <div class="prompt-footer">
                <div class="prompt-footer-left">
                    <span class="prompt-footer-icon">⚡</span>
                    <span>Powered by AXION</span>
                </div>
                <div class="prompt-footer-right">
                    <span>Select the code to copy</span>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return display(HTML(html_content))
