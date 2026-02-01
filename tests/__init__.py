We need to add another option under the evaluate tab that will allow user to upload a dataset, optionally connect to an Agent, select metrics from _pytest import python
from the registry and actually run the batch evalution from axionevaluation_runner, 




Here are some reference that I did this before in an dash app (but was mocked), this will be real and hit evaluation_runner. And obviously needs to be in the current typscript/nextjs and python style now

```
import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import pandas as pd

from components.metric_registry import create_metric_registry_modal, METRIC_REGISTRY
from config import Config


def create_experiment_layout():
    """Create the redesigned experiment runner interface with horizontal flow"""

    if Config.GATEWAY_AVAILABLE:
        required_check_content = html.Div(
            "Complete required checks to begin (agent connection is optional)",
            className="run-hint-exp",
            id="run-hint-exp"
        )
    else:
        required_check_content = html.Div(
            "Access to LLM Gateway Not Available. This is currently mock output",
            className="run-hint-exp",
            id="run-hint-exp"
        )

    return html.Div(className="experiment-page", children=[
        # Header Section
        html.Div(className="experiment-header", children=[
            html.Div(className="header-content", children=[
                html.Div(className="header-icon-exp", children=[
                    html.I(className="fas fa-flask")
                ]),
                html.Div(children=[
                    html.H1("Evaluation Runner (Coming Soon)", className="experiment-title"),
                    html.P(
                        "Configure datasets, connect to agents, "
                        "select metrics, and run comprehensive batch evaluation",
                        className="experiment-subtitle")
                ])
            ])
        ]),

        # Horizontal Progress Indicator - Now with 4 steps
        html.Div(className="progress-container", children=[
            html.Div(className="progress-bar", children=[
                html.Div(className="progress-step", children=[
                    html.Div("1", className="step-number active", id="exp-step-1"),
                    html.Div(className="step-info", children=[
                        html.Div("Upload Dataset", className="step-title"),
                        html.Div("Select evaluation data", className="step-description")
                    ])
                ]),
                html.Div(className="step-connector", id="exp-connector-1"),
                html.Div(className="progress-step", children=[
                    html.Div("2", className="step-number", id="exp-step-2"),
                    html.Div(className="step-info", children=[
                        html.Div("Connect to Agent", className="step-title"),
                        html.Div("Interact with agent", className="step-description")
                    ])
                ]),
                html.Div(className="step-connector", id="exp-connector-2"),
                html.Div(className="progress-step", children=[
                    html.Div("3", className="step-number", id="exp-step-3"),
                    html.Div(className="step-info", children=[
                        html.Div("Configure Metrics", className="step-title"),
                        html.Div("Choose evaluation metrics", className="step-description")
                    ])
                ]),
                html.Div(className="step-connector", id="exp-connector-3"),
                html.Div(className="progress-step", children=[
                    html.Div("4", className="step-number", id="exp-step-4"),
                    html.Div(className="step-info", children=[
                        html.Div("Run Evaluation", className="step-title"),
                        html.Div("Execute and monitor", className="step-description")
                    ])
                ])
            ])
        ]),

        # Main Content Container
        dbc.Container(fluid=True, className="experiment-container", children=[
            dbc.Row([
                # Left Panel - Dataset Configuration
                dbc.Col(lg=6, className="config-panel", children=[
                    html.Div(className="panel-card panel-card-compact", children=[
                        html.H3([
                            html.I(className="fas fa-database"),
                            " Dataset Configuration"
                        ], className="panel-title"),

                        html.Div(className="config-group", style={"marginBottom": "1rem"}, children=[
                            html.Label([
                                html.I(className="fas fa-folder-open"),
                                "Select Dataset"
                            ], className="config-label"),
                            dcc.Dropdown(
                                id="dataset-selector",
                                options=[],
                                placeholder="Choose from uploaded datasets...",
                                className="retriever-dropdown-professional",
                                clearable=False
                            ),
                            html.Div(id="dataset-info", className="dataset-info-box")
                        ]),

                        html.Div(className="config-group", style={"marginBottom": "1.5rem"}, children=[
                            html.Label([
                                html.I(className="fas fa-upload"),
                                "Upload New Dataset"
                            ], className="config-label"),
                            dcc.Upload(
                                id='dataset-upload-exp',
                                children=dbc.Button([
                                    html.I(className="fas fa-file-csv me-2"),
                                    "Upload CSV File"
                                ], className="btn-upload-dataset", size="md"),
                                multiple=False,
                                accept='.csv'
                            ),
                            html.Div(id='dataset-upload-status-exp', className="upload-status-exp")
                        ]),

                        html.Div(className="config-group", style={"marginBottom": "0"}, children=[
                            html.Label([
                                html.I(className="fas fa-eye"),
                                "Dataset Preview"
                            ], className="config-label"),
                            html.Div(id="dataset-preview", className="preview-container-exp")
                        ])
                    ])
                ]),

                # Right Panel - Agent Connection (Optional)
                dbc.Col(lg=6, className="results-panel", children=[
                    html.Div(className="panel-card panel-card-compact", children=[
                        html.H3([
                            html.I(className="fas fa-plug"),
                            " Agent Connection ",
                            html.Span("(Optional)", style={
                                "fontSize": "0.85rem",
                                "fontWeight": "400",
                                "color": "var(--text-muted)",
                                "marginLeft": "0.5rem"
                            })
                        ], className="panel-title"),

                        html.Div(className="config-group", style={"marginBottom": "1rem"}, children=[
                            html.Label([
                                html.I(className="fas fa-robot"),
                                "Select Agent API"
                            ], className="config-label"),
                            dcc.Dropdown(
                                id="agent-api-selector",
                                options=[
                                    {'label': 'Agent API', 'value': 'agent_api'},
                                    {'label': 'MIAW API', 'value': 'miaw'},
                                    {'label': 'Prompt Template API', 'value': 'prompt_template_api'}
                                ],
                                placeholder="Choose agent API type...",
                                className="retriever-dropdown-professional",
                                clearable=True
                            )
                        ]),

                        html.Div(className="config-group", style={"marginBottom": "0"}, children=[
                            html.Label([
                                html.I(className="fas fa-key"),
                                "Agent Credentials & Configuration"
                            ], className="config-label"),
                            dcc.Textarea(
                                id="agent-config-editor",
                                className="config-textarea-exp",
                                placeholder="# Select an agent API type above to see configuration template",
                                spellCheck=False
                            )
                        ])
                    ])
                ])
            ]),

            dbc.Row([
                # Metrics Configuration - Full Width
                dbc.Col(lg=12, children=[
                    html.Div(className="panel-card panel-card-compact", style={"marginTop": "1.5rem"}, children=[
                        html.H3([
                            html.I(className="fas fa-chart-line"),
                            " Evaluation Metrics"
                        ], className="panel-title"),

                        dbc.Row([
                            dbc.Col(lg=6, children=[
                                html.Div(className="config-group", style={"marginBottom": "0"}, children=[
                                    dbc.Button([
                                        html.I(className="fas fa-cube me-2"),
                                        html.Span("Browse Metric Registry")
                                    ], id="open-metric-registry",
                                        className="btn-browse-metrics-full",
                                        size="lg"),

                                    html.Div(id="selected-metrics-preview",
                                             className="selected-metrics-preview-exp",
                                             children=[
                                                 html.Div(className="preview-empty", children=[
                                                     html.I(className="fas fa-info-circle me-2"),
                                                     html.Span("No metrics selected yet")
                                                 ])
                                             ])
                                ])
                            ]),
                            dbc.Col(lg=6, children=[
                                html.Div(className="config-group", style={"marginBottom": "0"}, children=[
                                    html.Label([
                                        html.I(className="fas fa-file-code"),
                                        "Evaluation Configuration (Optional)"
                                    ], className="config-label"),
                                    dcc.Textarea(
                                        id="config-editor",
                                        className="config-textarea-exp",
                                        placeholder="# Optional: Add custom configuration\nmodel:\n "
                                                    "name: gpt-4\n  temperature: 0.7",
                                        spellCheck=False
                                    )
                                ])
                            ])
                        ])
                    ])
                ])
            ]),

            # Full Width Run Evaluation Section
            dbc.Row([
                dbc.Col(lg=12, children=[
                    html.Div(className="panel-card", style={"marginTop": "1.5rem"}, children=[
                        html.H3([
                            html.I(className="fas fa-rocket"),
                            " Run Evaluation"
                        ], className="panel-title"),

                        # Preflight Checks
                        html.Div(className="preflight-section-horizontal", children=[
                            html.H4("Validation Checks", className="preflight-title-exp"),
                            html.Div(className="preflight-grid-horizontal", children=[
                                create_preflight_item_exp("Dataset Loaded", "dataset-check-exp", False),
                                create_preflight_item_exp("Agent Connected", "agent-check-exp", True),
                                create_preflight_item_exp("Metrics Selected", "metrics-check-exp", False),
                                create_preflight_item_exp("Ready to Run", "ready-check-exp", False)
                            ])
                        ]),

                        # Run Button
                        html.Div(className="run-section", children=[
                            dbc.Button([
                                html.I(className="fas fa-rocket me-2", id="run-icon-exp"),
                                html.Span("Launch Evaluation", id="run-text-exp")
                            ], id="run-experiment-btn",
                                className="btn-run-experiment-exp",
                                size="lg",
                                disabled=True),
                            required_check_content
                            # html.Div("Complete required checks to begin (agent connection is optional)",
                            #          className="run-hint-exp",
                            #          id="run-hint-exp")
                        ])
                    ])
                ])
            ]),

            # Progress and Results Section
            dbc.Row([
                dbc.Col(lg=12, children=[
                    # Progress Card (Initially Hidden)
                    html.Div(id="progress-section-exp",
                             className="panel-card",
                             style={"display": "none", "marginTop": "1.5rem"},
                             children=[
                                 html.H3([
                                     html.I(className="fas fa-tasks"),
                                     " Evaluation Progress"
                                 ], className="panel-title"),

                                 # Progress Stats
                                 html.Div(className="progress-stats-row", children=[
                                     html.Div(className="stat-item-exp", children=[
                                         html.Span("0%", id="progress-percentage-exp", className="stat-number"),
                                         html.Span("Complete", className="stat-label")
                                     ]),
                                     html.Div(className="stat-item-exp", children=[
                                         html.Span("0/0", id="progress-count-exp", className="stat-number"),
                                         html.Span("Items", className="stat-label")
                                     ]),
                                     html.Div(className="stat-item-exp", children=[
                                         html.Span("00:00:00", id="elapsed-time-exp", className="stat-number"),
                                         html.Span("Elapsed", className="stat-label")
                                     ])
                                 ]),

                                 # Progress Bar
                                 html.Div(className="progress-bar-container", children=[
                                     dbc.Progress(
                                         value=0,
                                         id="main-progress-exp",
                                         className="main-progress-bar-exp",
                                         striped=True,
                                         animated=True
                                     )
                                 ]),

                                 # Live Log
                                 html.Div(className="live-log-section", children=[
                                     html.Div(className="log-header-exp", children=[
                                         html.Span("Live Log", className="log-title-exp"),
                                         dbc.Switch(
                                             id="auto-scroll-log-exp",
                                             label="Auto-scroll",
                                             value=True,
                                             className="log-switch-exp"
                                         )
                                     ]),
                                     html.Div(id="live-log-exp", className="live-log-exp")
                                 ]),

                                 # Control Buttons
                                 html.Div(id="progress-controls-exp", className="progress-controls-exp", children=[
                                     dbc.Button([
                                         html.I(className="fas fa-pause me-2"),
                                         "Pause"
                                     ], id="pause-btn-exp", className="btn-control-exp"),
                                     dbc.Button([
                                         html.I(className="fas fa-stop me-2"),
                                         "Stop"
                                     ], id="stop-btn-exp", className="btn-control-exp btn-control-danger")
                                 ])
                             ]),

                    # Results Card (Initially Hidden)
                    html.Div(id="results-section-exp",
                             className="panel-card",
                             style={"display": "none", "marginTop": "1.5rem"},
                             children=[
                                 html.H3([
                                     html.I(className="fas fa-trophy"),
                                     " Evaluation Complete"
                                 ], className="panel-title"),

                                 # Results Stats
                                 html.Div(className="results-stats-row", children=[
                                     create_result_stat_exp("Evaluations", "100", "fa-list"),
                                     create_result_stat_exp("Success Rate", "97%", "fa-check"),
                                     create_result_stat_exp("Avg Score", "0.89", "fa-star"),
                                     create_result_stat_exp("Duration", "00:33:42", "fa-clock")
                                 ]),

                                 # Actions
                                 html.Div(className="results-actions-exp", children=[
                                     dbc.Button([
                                         html.I(className="fas fa-download me-2"),
                                         "Download Results"
                                     ], id="download-csv-exp", className="btn-result-action"),
                                     dbc.Button([
                                         html.I(className="fas fa-chart-bar me-2"),
                                         "View Analytics"
                                     ], id="view-analytics-exp", className="btn-result-action"),
                                     dbc.Button([
                                         html.I(className="fas fa-redo me-2"),
                                         "Run Again"
                                     ], id="run-again-btn-exp", className="btn-result-action btn-result-primary")
                                 ])
                             ])
                ])
            ])
        ]),

        # Store components
        dcc.Store(id='experiment-state', data={'status': 'idle'}),
        dcc.Store(id='experiment-results'),
        dcc.Store(id='agent-connection-store', data=None),
        dcc.Download(id="download-experiment-results"),
        dcc.Interval(
            id='experiment-interval',
            interval=1000,
            n_intervals=0,
            disabled=True
        ),

        create_metric_registry_modal()
    ])


def create_preflight_item_exp(label, item_id, passed=False):
    """Create a preflight check item"""
    icon = "fa-check-circle" if passed else "fa-exclamation-circle"
    status = "passed" if passed else "failed"

    return html.Div(className=f"preflight-item-exp {status}", id=item_id, children=[
        html.I(className=f"fas {icon} preflight-icon-exp"),
        html.Span(label, className="preflight-label-exp")
    ])


def create_result_stat_exp(label, value, icon):
    """Create a result stat display"""
    return html.Div(className="result-stat-exp", children=[
        html.I(className=f"fas {icon} result-stat-icon"),
        html.Div(className="result-stat-content", children=[
            html.Div(value, className="result-stat-value",
                     id=f"result-{label.lower().replace(' ', '-')}-exp"),
            html.Div(label, className="result-stat-label")
        ])
    ])


def register_experiment_callbacks(app):
    """Register callbacks for redesigned experiment runner"""

    from components.metric_registry import register_metric_registry_callbacks
    register_metric_registry_callbacks(app)

    # Update metrics preview
    @app.callback(
        Output('selected-metrics-preview', 'children'),
        Input('selected-metrics-store', 'data'),
        prevent_initial_call=False
    )
    def update_metrics_preview(selected_metrics):
        if not selected_metrics:
            return html.Div(className="preview-empty", children=[
                html.I(className="fas fa-info-circle me-2"),
                html.Span("No metrics selected yet")
            ])

        return [
            html.Div(className="metrics-preview-header", children=[
                html.I(className="fas fa-check-circle me-2"),
                html.Span(f"{len(selected_metrics)} metrics selected")
            ]),
            html.Div(className="metrics-preview-list", children=[
                html.Span(className="metric-chip-exp", children=[
                    html.I(className="fas fa-cube me-1"),
                    METRIC_REGISTRY.get(key, {}).get("name", key)
                ])
                for key in selected_metrics if key in METRIC_REGISTRY
            ])
        ]

    # Update dataset options
    @app.callback(
        Output('dataset-selector', 'options'),
        [Input('data-store', 'data'),
         Input('format-store', 'data')]
    )
    def update_dataset_options(data, data_format):
        options = []
        if data:
            options.append({'label': f'📊 Current Upload ({data_format}, {len(data)} rows)',
                            'value': 'uploaded_current'})
            options.append({'label': '─' * 30, 'value': 'divider1', 'disabled': True})

        options.extend([
            {'label': '📁 Stored Datasets', 'value': 'header_stored', 'disabled': True},
            {'label': '  Golden Agent Dataset v2.1', 'value': 'help_rag'},
            {'label': '  Help RAG Dataset v1.2', 'value': 'help_quality'},
        ])
        return options

    @app.callback(
        [Output('dataset-upload-status-exp', 'children'),
         Output('dataset-upload-status-exp', 'className'),
         Output('dataset-selector', 'value')],
        Input('dataset-upload-exp', 'contents'),
        State('dataset-upload-exp', 'filename'),
        prevent_initial_call=True
    )
    def handle_dataset_upload(contents, filename):
        if contents is None:
            return "", "", None

        try:
            import base64
            import io

            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)

            if filename.endswith('.csv'):
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                # Don't store the data, just validate it

                status = html.Div([
                    html.I(className="fas fa-check-circle me-2"),
                    f"Successfully uploaded {filename} ({len(df)} rows)"
                ])

                return status, "upload-status-exp success", 'uploaded_current'
            else:
                status = html.Div([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "Please upload a CSV file"
                ])
                return status, "upload-status-exp error", None

        except Exception as e:
            status = html.Div([
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"Error uploading file: {str(e)}"
            ])
            return status, "upload-status-exp error", None

    # Show dataset info and preview
    @app.callback(
        [Output('dataset-info', 'children'),
         Output('dataset-preview', 'children')],
        [Input('dataset-selector', 'value'),
         Input('data-store', 'data')],
        [State('format-store', 'data')]
    )
    def show_dataset_info(selected_dataset, uploaded_data, data_format):
        if not selected_dataset and not uploaded_data:
            return "", html.Div(className="preview-empty-state", children=[
                html.I(className="fas fa-table",
                       style={"fontSize": "2rem", "color": "#CBD5E0", "marginBottom": "0.5rem"}),
                html.Div("Select a dataset to preview", style={"color": "#A0AEC0", "fontSize": "0.9rem"})
            ])

        # If data was just uploaded but dropdown hasn't updated yet, show the uploaded data
        if uploaded_data and (not selected_dataset or selected_dataset == 'uploaded_current'):
            df = pd.DataFrame(uploaded_data)
            info = html.Div(className="dataset-info-content", children=[
                html.Div([html.Strong("Format: "), html.Span(data_format if data_format else "CSV")]),
                html.Div([html.Strong("Rows: "), html.Span(f"{len(df):,}")]),
                html.Div([html.Strong("Columns: "), html.Span(str(len(df.columns)))])
            ])

            preview_df = df.head(5)
            preview_cols = list(preview_df.columns[:4])
            preview = html.Table(className="preview-table-exp", children=[
                html.Thead([html.Tr([html.Th(col) for col in preview_cols])]),
                html.Tbody([
                    html.Tr([html.Td(str(preview_df.iloc[i][col])[:50] + "..."
                                     if len(str(preview_df.iloc[i][col])) > 50
                                     else str(preview_df.iloc[i][col]))
                             for col in preview_cols])
                    for i in range(len(preview_df))
                ])
            ])

            return info, preview

        mock_data = {
            'help_rag': {
                'rows': 247,
                'cols': 8,
                'format': 'CSV',
                'preview': [
                    ['001', 'What is Data Cloud?', 'Data Cloud is a comprehensive...', 'Data Cloud is a...'],
                    ['002', 'Explain Agentforce', 'Agentforce enables AI agents...', 'Agentforce is ...'],
                    ['003', 'How does Einstein work?', 'Einstein AI provides predictive...', 'Einstein AI is...'],
                    ['004', 'What is Tableau?', 'Tableau is a visual analytics...', 'Tableau is ...'],
                    ['005', 'Describe Slack integration', 'Slack integration allows...', 'Slack ...']
                ]
            },
            'help_quality': {
                'rows': 512,
                'cols': 6,
                'format': 'CSV',
                'preview': [
                    ['001', 'What is Data Cloud?', 'Data Cloud is a comprehensive...', 'Data Cloud is a...'],
                    ['002', 'Explain Agentforce', 'Agentforce enables AI agents...', 'Agentforce is ...'],
                    ['003', 'How does Einstein work?', 'Einstein AI provides predictive...', 'Einstein AI is...'],
                    ['004', 'What is Tableau?', 'Tableau is a visual analytics...', 'Tableau is ...'],
                    ['005', 'Describe Slack integration', 'Slack integration allows...', 'Slack ...']
                ]
            }
        }

        if selected_dataset in mock_data:
            data = mock_data[selected_dataset]
            info = html.Div(className="dataset-info-content", children=[
                html.Div([html.Strong("Format: "), html.Span(data['format'])]),
                html.Div([html.Strong("Rows: "), html.Span(f"{data['rows']:,}")]),
                html.Div([html.Strong("Columns: "), html.Span(str(data['cols']))])
            ])

            preview = html.Table(className="preview-table-exp", children=[
                html.Thead([html.Tr([html.Th("ID"), html.Th("Query"), html.Th("Expected"), html.Th("Actual")])]),
                html.Tbody([
                    html.Tr([html.Td(cell) for cell in row])
                    for row in data['preview']
                ])
            ])

            return info, preview

        return "", html.Div(className="preview-empty-state", children=[
            html.I(className="fas fa-table", style={"fontSize": "2rem", "color": "#CBD5E0", "marginBottom": "0.5rem"}),
            html.Div("No preview available", style={"color": "#A0AEC0", "fontSize": "0.9rem"})
        ])

    @app.callback(
        Output('agent-config-editor', 'placeholder'),
        Input('agent-api-selector', 'value'),
        prevent_initial_call=False
    )
    def update_agent_config_placeholder(api_type):
        config_templates = {
            'miaw': """# MIAW Agent Configuration
domain: ...
org_id: ...
deployment_name: ...""",
            'agent_api': """# Agent API Configuration
domain: ...
org_id: ...
deployment_name: ...""",
            'prompt_template_api': """# Prompt Template API Configuration
domain: ...
token: ...
retriever_name: ...
input_key: ...
prompt_template_name: ...
application_name: PromptBuilderPreview"""
        }

        if api_type in config_templates:
            return config_templates[api_type]
        else:
            return "# Select an agent API type above to see configuration template"

    # Handle agent connection
    @app.callback(
        [Output('agent-connection-store', 'data'),
         Output('agent-check-exp', 'className')],
        [Input('agent-api-selector', 'value'),
         Input('agent-config-editor', 'value')],
        prevent_initial_call=False
    )
    def handle_agent_connection(api_type, config):
        # Agent is optional, so it's always "passed" (green)
        # But we track if it's actually configured
        agent_configured = api_type is not None and config

        connection_data = {
            'api_type': api_type,
            'config': config,
            'connected': agent_configured
        }

        # Always show as passed since it's optional
        return connection_data, "preflight-item-exp passed"

    # Update preflight checks and progress steps
    @app.callback(
        [Output('dataset-check-exp', 'className'),
         Output('metrics-check-exp', 'className'),
         Output('ready-check-exp', 'className'),
         Output('run-experiment-btn', 'disabled'),
         Output('exp-step-1', 'className'),
         Output('exp-step-2', 'className'),
         Output('exp-step-3', 'className'),
         Output('exp-connector-1', 'className'),
         Output('exp-connector-2', 'className')],
        [Input('dataset-selector', 'value'),
         Input('selected-metrics-store', 'data'),
         Input('data-store', 'data'),
         Input('agent-connection-store', 'data')],
        prevent_initial_call=False
    )
    def update_checks_and_steps(dataset, metrics, uploaded_data, agent_connection):
        dataset_ok = (dataset == 'uploaded_current' and uploaded_data) or \
                     (dataset in ['help_rag', 'help_quality']) or \
                     (uploaded_data is not None and len(uploaded_data) > 0)
        metrics_ok = metrics and len(metrics) > 0
        agent_configured = agent_connection and agent_connection.get('connected', False)

        # Ready to run if dataset and metrics are OK (agent is optional)
        ready_to_run = dataset_ok and metrics_ok

        dataset_class = "preflight-item-exp passed" if dataset_ok else "preflight-item-exp failed"
        metrics_class = "preflight-item-exp passed" if metrics_ok else "preflight-item-exp failed"
        ready_class = "preflight-item-exp passed" if ready_to_run else "preflight-item-exp failed"

        # Update steps based on progress
        if ready_to_run:
            # All required steps complete
            step1 = "step-number completed"
            step2 = "step-number completed"
            step3 = "step-number completed"
            conn1 = "step-connector completed"
            conn2 = "step-connector completed"
        elif metrics_ok:
            # Dataset and metrics done, ready for final step
            step1 = "step-number completed"
            step2 = "step-number completed"
            step3 = "step-number completed"
            conn1 = "step-connector completed"
            conn2 = "step-connector completed"
        elif agent_configured or dataset_ok:
            # Dataset loaded, move to metrics (step 2 is optional so we skip to step 3)
            step1 = "step-number completed"
            step2 = "step-number completed"
            step3 = "step-number active"
            conn1 = "step-connector completed"
            conn2 = "step-connector completed"
        else:
            # Just starting
            step1 = "step-number active"
            step2 = "step-number"
            step3 = "step-number"
            conn1 = "step-connector"
            conn2 = "step-connector"

        return (dataset_class, metrics_class, ready_class, not ready_to_run,
                step1, step2, step3, conn1, conn2)

    # Start experiment
    @app.callback(
        [Output('progress-section-exp', 'style'),
         Output('experiment-interval', 'disabled'),
         Output('exp-step-4', 'className'),
         Output('exp-connector-3', 'className'),
         Output('experiment-state', 'data')],
        Input('run-experiment-btn', 'n_clicks'),
        [State('dataset-selector', 'value'),
         State('data-store', 'data'),
         State('selected-metrics-store', 'data')],
        prevent_initial_call=True
    )
    def start_experiment(n_clicks, selected_dataset, uploaded_data, selected_metrics):
        if n_clicks:
            # Calculate total based on dataset size and number of metrics
            num_metrics = len(selected_metrics) if selected_metrics else 1

            # Determine dataset size
            if selected_dataset == 'uploaded_current' and uploaded_data:
                df = pd.DataFrame(uploaded_data)
                # Get unique IDs - assume first column is ID or use row count
                if len(df.columns) > 0:
                    num_ids = df.iloc[:, 0].nunique()
                else:
                    num_ids = len(df)
            elif selected_dataset == 'help_rag':
                num_ids = 247
            elif selected_dataset == 'help_quality':
                num_ids = 512
            else:
                num_ids = 100  # Default fallback

            total_evaluations = num_metrics * num_ids

            return ({"display": "block", "marginTop": "1.5rem"}, False,
                    "step-number active", "step-connector completed",
                    {
                        'status': 'running',
                        'paused': False,
                        'start_time': datetime.now().isoformat(),
                        'total': total_evaluations,
                        'num_metrics': num_metrics,
                        'num_ids': num_ids
                    })
        return {"display": "none"}, True, "step-number", "step-connector", {'status': 'idle'}

    # Handle pause button
    @app.callback(
        [Output('experiment-state', 'data', allow_duplicate=True),
         Output('experiment-interval', 'disabled', allow_duplicate=True),
         Output('pause-btn-exp', 'children')],
        Input('pause-btn-exp', 'n_clicks'),
        State('experiment-state', 'data'),
        prevent_initial_call=True
    )
    def handle_pause(n_clicks, exp_state):
        if not n_clicks or not exp_state:
            return dash.no_update, dash.no_update, dash.no_update

        is_paused = exp_state.get('paused', False)

        if is_paused:
            # Resume experiment
            exp_state['paused'] = False
            exp_state['status'] = 'running'
            return exp_state, False, [html.I(className="fas fa-pause me-2"), "Pause"]
        else:
            # Pause experiment
            exp_state['paused'] = True
            exp_state['status'] = 'paused'
            exp_state['paused_time'] = datetime.now().isoformat()
            return exp_state, True, [html.I(className="fas fa-play me-2"), "Resume"]

    # Handle stop button
    @app.callback(
        [Output('experiment-state', 'data', allow_duplicate=True),
         Output('experiment-interval', 'disabled', allow_duplicate=True),
         Output('progress-section-exp', 'style', allow_duplicate=True),
         Output('results-section-exp', 'style', allow_duplicate=True),
         Output('live-log-exp', 'children', allow_duplicate=True)],
        Input('stop-btn-exp', 'n_clicks'),
        State('experiment-state', 'data'),
        State('live-log-exp', 'children'),
        prevent_initial_call=True
    )
    def handle_stop(n_clicks, exp_state, current_log):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        # Stop the experiment
        exp_state['status'] = 'stopped'
        exp_state['stopped'] = True
        exp_state['stop_time'] = datetime.now().isoformat()

        # Add stop message to log
        if current_log is None:
            current_log = []

        stop_entry = html.Div(className="log-entry-exp", children=[
            html.Span(f"[{datetime.now().strftime('%H:%M:%S')}]", className="log-time"),
            html.Span("[STOPPED]", className="log-status", style={"color": "#E53E3E"}),
            html.Span("Evaluation stopped by user", className="log-message")
        ])
        current_log.append(stop_entry)

        # Show results section
        return (exp_state, True,
                {"display": "none"},  # Hide progress
                {"display": "block", "marginTop": "1.5rem"},  # Show results
                current_log)

    # Update progress (modified to respect pause/stop state and use dynamic total)
    @app.callback(
        [Output('main-progress-exp', 'value'),
         Output('progress-percentage-exp', 'children'),
         Output('progress-count-exp', 'children'),
         Output('elapsed-time-exp', 'children'),
         Output('live-log-exp', 'children', allow_duplicate=True),
         Output('progress-section-exp', 'style', allow_duplicate=True),
         Output('results-section-exp', 'style', allow_duplicate=True),
         Output('result-evaluations-exp', 'children'),
         Output('result-success-rate-exp', 'children'),
         Output('result-avg-score-exp', 'children'),
         Output('result-duration-exp', 'children'),
         Output('progress-controls-exp', 'style')],
        Input('experiment-interval', 'n_intervals'),
        [State('live-log-exp', 'children'),
         State('experiment-state', 'data')],
        prevent_initial_call=True
    )
    def update_progress(n_intervals, current_log, exp_state):
        if not n_intervals or not exp_state:
            return (0, "0%", "0/0", "00:00:00", [],
                    {"display": "block", "marginTop": "1.5rem"},
                    {"display": "none"},
                    "0", "0%", "0.00", "00:00:00", {"display": "flex"})

        # Check if stopped
        if exp_state.get('stopped', False):
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
                   dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
                   dash.no_update, dash.no_update

        # Get total from experiment state or default to 100
        total = exp_state.get('total', 100)

        # Calculate completed based on total
        # Adjust speed so it completes in reasonable time regardless of total
        items_per_second = max(1, total // 300)  # Complete in 5 minutes
        completed = min(n_intervals * items_per_second, total)
        percentage = (completed / total) * 100 if total > 0 else 0

        elapsed = timedelta(seconds=n_intervals)
        elapsed_str = str(elapsed).split('.')[0]

        if current_log is None:
            current_log = []

        # Only add log entry if not paused
        if not exp_state.get('paused', False):
            # Calculate which metric/item we're on
            num_metrics = exp_state.get('num_metrics', 1)
            num_ids = exp_state.get('num_ids', total)

            current_item = (completed - 1) % num_ids + 1 if num_ids > 0 else 1
            current_metric = ((completed - 1) // num_ids) % num_metrics + 1 if num_metrics > 0 and num_ids > 0 else 1

            new_entry = html.Div(className="log-entry-exp", children=[
                html.Span(f"[{datetime.now().strftime('%H:%M:%S')}]", className="log-time"),
                html.Span("[SUCCESS]", className="log-status log-success"),
                html.Span(f"Evaluated metric {current_metric}/{num_metrics} on item {current_item}/{num_ids}",
                          className="log-message")
            ])
            current_log.append(new_entry)
            current_log = current_log[-15:]

        # Calculate final results when complete
        if percentage >= 100:
            progress_style = {"display": "none"}
            results_style = {"display": "block", "marginTop": "1.5rem"}
            controls_style = {"display": "none"}
            success_rate = 97
            avg_score = 0.89
            evaluations = total
            duration = elapsed_str
        else:
            progress_style = {"display": "block", "marginTop": "1.5rem"}
            results_style = {"display": "none"}
            controls_style = {"display": "flex"}
            success_rate = 0
            avg_score = 0
            evaluations = 0
            duration = "00:00:00"

        return (
            percentage,
            f"{percentage: .0f}%",
            f"{completed}/{total}",
            elapsed_str,
            current_log,
            progress_style,
            results_style,
            str(evaluations),
            f"{success_rate}%",
            f"{avg_score: .2f}",
            duration,
            controls_style
        )

    # Download results as CSV
    @app.callback(
        Output("download-experiment-results", "data"),
        Input("download-csv-exp", "n_clicks"),
        prevent_initial_call=True
    )
    def download_results(n_clicks):
        if n_clicks:
            # Create sample results data
            results_data = {
                'test_id': [f'TEST_{str(i).zfill(3)}' for i in range(1, 101)],
                'query': [f'What is the purpose of feature {i}?' for i in range(1, 101)],
                'expected_output': [f'Feature {i} is designed to improve system performance' for i in range(1, 101)],
                'actual_output': [f'Feature {i} enhances system capabilities and performance' for i in range(1, 101)],
                'metric_answer_relevancy': [round(0.75 + (i % 25) * 0.01, 2) for i in range(1, 101)],
                'metric_answer_correctness': [round(0.80 + (i % 20) * 0.01, 2) for i in range(1, 101)],
                'metric_faithfulness': [round(0.85 + (i % 15) * 0.01, 2) for i in range(1, 101)],
                'overall_score': [round(0.82 + (i % 18) * 0.01, 2) for i in range(1, 101)],
                'status': ['passed' if i % 10 != 0 else 'failed' for i in range(1, 101)],
                'execution_time_ms': [150 + (i * 5) for i in range(1, 101)],
                'timestamp': [f'2025-10-03 17:48:{str(i % 60).zfill(2)}' for i in range(1, 101)]
            }

            df = pd.DataFrame(results_data)

            return dcc.send_data_frame(df.to_csv,
                                       filename=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                       index=False)
        return None

    # Run again button - resets the experiment
    @app.callback(
        [Output('progress-section-exp', 'style', allow_duplicate=True),
         Output('results-section-exp', 'style', allow_duplicate=True),
         Output('experiment-interval', 'disabled', allow_duplicate=True),
         Output('experiment-interval', 'n_intervals', allow_duplicate=True),
         Output('experiment-state', 'data', allow_duplicate=True),
         Output('pause-btn-exp', 'children', allow_duplicate=True)],
        Input('run-again-btn-exp', 'n_clicks'),
        [State('dataset-selector', 'value'),
         State('data-store', 'data'),
         State('selected-metrics-store', 'data')],
        prevent_initial_call=True
    )
    def run_again(n_clicks, selected_dataset, uploaded_data, selected_metrics):
        if n_clicks:
            # Recalculate total based on current dataset and metrics
            num_metrics = len(selected_metrics) if selected_metrics else 1

            # Determine dataset size
            if selected_dataset == 'uploaded_current' and uploaded_data:
                df = pd.DataFrame(uploaded_data)
                if len(df.columns) > 0:
                    num_ids = df.iloc[:, 0].nunique()
                else:
                    num_ids = len(df)
            elif selected_dataset == 'help_rag':
                num_ids = 247
            elif selected_dataset == 'help_quality':
                num_ids = 512
            else:
                num_ids = 100  # Default fallback

            total_evaluations = num_metrics * num_ids

            # Reset everything to start a new experiment
            return ({"display": "block", "marginTop": "1.5rem"},
                    {"display": "none"},
                    False,  # Enable interval
                    0,  # Reset counter
                    {
                        'status': 'running',
                        'paused': False,
                        'start_time': datetime.now().isoformat(),
                        'total': total_evaluations,
                        'num_metrics': num_metrics,
                        'num_ids': num_ids
                    },
                    [html.I(className="fas fa-pause me-2"), "Pause"])
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    @app.callback(
        Output('experiment-state', 'data', allow_duplicate=True),
        Input('view-analytics-exp', 'n_clicks'),
        prevent_initial_call=True
    )
    def view_analytics(n_clicks):
        # Placeholder
        if n_clicks:
            return {'status': 'viewing_analytics', 'timestamp': datetime.now().isoformat()}
        return dash.no_update

    return app
````
```
from dash import dcc, html, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
import dash


from config import settings, Config

FALLBACK_REGISTRY = {
    'answer_completeness': {
        'key': 'answer_completeness',
        'name': 'Answer Completeness',
        'description': 'Evaluates the completeness of the answer using one of two '
                       'approaches: Either. Aspect-based evaluation or sub-question '
                       'based evaluation.',
        'required_fields': ['query', 'actual_output', 'expected_output'],
        'optional_fields': ['acceptance_criteria'],
        'default_threshold': 0.5,
        'tags': ['knowledge', 'agent', 'single_turn'],
        'score_range': [0, 1]
    },

    'answer_correctness': {
        'key': 'answer_correctness',
        'name': 'Answer Correctness',
        'description': 'Evaluates answer correctness by classifying statements and '
                       'calculating F-score.',
        'required_fields': ['query', 'actual_output', 'expected_output'],
        'optional_fields': [],
        'default_threshold': 0.5,
        'tags': ['single_turn'],
        'score_range': [0, 1]
    },

    'answer_criteria': {
        'key': 'answer_criteria',
        'name': 'Answer Criteria',
        'description': 'Evaluates the responses based on user defined specified criteria.',
        'required_fields': ['query', 'actual_output'],
        'optional_fields': ['acceptance_criteria', 'additional_input'],
        'default_threshold': 0.5,
        'tags': ['knowledge', 'agent', 'single_turn'],
        'score_range': [0, 1]
    },

    'answer_relevancy': {
        'key': 'answer_relevancy',
        'name': 'Answer Relevancy',
        'description': 'Evaluates how relevant an answer is to the input query.',
        'required_fields': ['query', 'actual_output'],
        'optional_fields': [],
        'default_threshold': 0.5,
        'tags': ['knowledge', 'single_turn'],
        'score_range': [0, 1]
    },

    'citation_presence': {
        'key': 'citation_presence',
        'name': 'Citation Presence',
        'description': 'Evaluates whether the AI response includes properly formatted citations.',
        'required_fields': ['actual_output'],
        'optional_fields': [],
        'default_threshold': 0.5,
        'tags': ['agent', 'knowledge', 'heuristic'],
        'score_range': [0, 1]
    },

    'contains_match': {
        'key': 'contains_match',
        'name': 'Contains Match',
        'description': 'Returns 1.0 if the actual output contains the expected '
                       'output (after stripping).',
        'required_fields': ['actual_output', 'expected_output'],
        'optional_fields': [],
        'default_threshold': 0.5,
        'tags': ['heuristic', 'binary'],
        'score_range': [0, 1]
    },

    'contextual_ranking': {
        'key': 'contextual_ranking',
        'name': 'Contextual Ranking',
        'description': 'Evaluates whether relevant context chunks are ranked higher '
                       '(Mean Average Precision).',
        'required_fields': ['query', 'expected_output', 'retrieved_content'],
        'optional_fields': [],
        'default_threshold': 0.5,
        'tags': ['knowledge', 'single_turn'],
        'score_range': [0, 1]
    },

    'contextual_recall': {
        'key': 'contextual_recall',
        'name': 'ContextualRecall',
        'description': 'Evaluates if the retrieved context supports the expected answer.',
        'required_fields': ['expected_output', 'retrieved_content'],
        'optional_fields': [],
        'default_threshold': 0.5,
        'tags': ['knowledge', 'single_turn'],
        'score_range': [0, 1]
    },

    'contextual_relevancy': {
        'key': 'contextual_relevancy',
        'name': 'Contextual Relevancy',
        'description': "Evaluates if the retrieved context is relevant to the user's query.",
        'required_fields': ['query', 'retrieved_content'],
        'optional_fields': [],
        'default_threshold': 0.5,
        'tags': ['knowledge', 'single_turn'],
        'score_range': [0, 1]
    },

    'conversation_efficiency': {
        'key': 'conversation_efficiency',
        'name': 'Conversation Efficiency',
        'description': 'Evaluates whether the conversation achieved its goal through '
                       'the most efficient path',
        'required_fields': ['conversation'],
        'optional_fields': [],
        'default_threshold': 0.5,
        'tags': ['agent', 'multi_turn'],
        'score_range': [0, 1]
    },

    'conversation_flow': {
        'key': 'conversation_flow',
        'name': 'Conversation Flow',
        'description': 'Performs advanced, multi-layered analysis of conversation '
                       'structure, state, and user effort trajectory.',
        'required_fields': ['conversation'],
        'optional_fields': [],
        'default_threshold': 0.7,
        'tags': ['agent', 'multi_turn'],
        'score_range': [0, 1]
    },

    'exact_string_match': {
        'key': 'exact_string_match',
        'name': 'Exact String Match',
        'description': 'Checks whether the actual output matches the expected output '
                       'exactly (after stripping whitespace).',
        'required_fields': ['actual_output', 'expected_output'],
        'optional_fields': [],
        'default_threshold': 0.5,
        'tags': ['heuristic', 'binary'],
        'score_range': [0, 1]
    },

    'factual_accuracy_with_semantic_alignment': {
        'key': 'factual_accuracy_with_semantic_alignment',
        'name': 'Factual Accuracy with Semantic Alignment',
        'description': 'Evaluates answer quality by combining factual accuracy '
                       '(TP/FP/FN analysis) with semantic similarity to ground truth.',
        'required_fields': ['query', 'actual_output', 'expected_output'],
        'optional_fields': [],
        'default_threshold': 0.7,
        'tags': [],
        'score_range': [0, 1]
    },

    'faithfulness': {
        'key': 'faithfulness',
        'name': 'Faithfulness',
        'description': 'Evaluates if the answer is factually consistent with the '
                       'retrieved context.',
        'required_fields': ['query', 'actual_output', 'retrieved_content'],
        'optional_fields': [],
        'default_threshold': 0.7,
        'tags': ['knowledge', 'single_turn'],
        'score_range': [0, 1]
    },

    'goal_completion': {
        'key': 'goal_completion',
        'name': 'Goal Completion',
        'description': "Analyzes if the user's goal was achieved, tracking sub-goals "
                       "and goal evolution.",
        'required_fields': ['conversation'],
        'optional_fields': [],
        'default_threshold': 0.5,
        'tags': ['agent', 'multi_turn'],
        'score_range': [0, 1]
    },

    'latency': {
        'key': 'latency',
        'name': 'Latency',
        'description': 'Execution time for the task. Normalization options anchor '
                       'on threshold.',
        'required_fields': ['latency'],
        'optional_fields': ['query'],
        'default_threshold': 5.0,
        'tags': ['heuristic'],
        'score_range': [0, None]
    },

    'levenshtein_ratio': {
        'key': 'levenshtein_ratio',
        'name': 'Levenshtein Ratio',
        'description': 'Calculates the Levenshtein ratio (string similarity) between '
                       'actual and expected outputs. Returns a score between 0.0 and '
                       '1.0, where 1.0 means identical strings.',
        'required_fields': ['actual_output', 'expected_output'],
        'optional_fields': [],
        'default_threshold': 0.2,
        'tags': ['heuristic'],
        'score_range': [0, 1]
    },

    'persona_tone_adherence': {
        'key': 'persona_tone_adherence',
        'name': 'Persona & Tone Adherence',
        'description': 'Evaluates whether the agent maintains its intended persona '
                       'and tone consistently',
        'required_fields': ['conversation'],
        'optional_fields': ['additional_input'],
        'default_threshold': 0.5,
        'tags': ['agent', 'multi_turn'],
        'score_range': [0, 1]
    },

    'quality_session': {
        'key': 'quality_session',
        'name': 'Quality Session',
        'description': 'One shot evaluation to determine if a conversation '
                       'constitutes a Quality Session.',
        'required_fields': ['conversation'],
        'optional_fields': [],
        'default_threshold': 0.5,
        'tags': ['agent', 'multi_turn', 'binary'],
        'score_range': [0, 1]
    },

    'question_answer_relevance': {
        'key': 'question_answer_relevance',
        'name': 'Question Answer Relevance',
        'description': 'One shot evaluation to whether an actual answer is '
                       'relevant and factually correct compared to an expected answer.',
        'required_fields': ['actual_output', 'expected_output'],
        'optional_fields': ['query'],
        'default_threshold': 0.5,
        'tags': ['agent', 'knowledge', 'single_turn', 'binary'],
        'score_range': [0, 1]
    },

    'sentence_bleu': {
        'key': 'sentence_bleu',
        'name': 'Sentence BLEU',
        'description': 'Computes sentence-level BLEU score between a candidate and '
                       'reference sentence. Returns a score between 0.0 and 1.0. '
                       'Higher means more similar.',
        'required_fields': ['actual_output', 'expected_output'],
        'optional_fields': [],
        'default_threshold': 0.5,
        'tags': ['heuristic'],
        'score_range': [0, 1]
    },

    'tool_correctness': {
        'key': 'tool_correctness',
        'name': 'Tool Correctness',
        'description': 'Evaluates whether the expected tools were correctly called '
                       'by the agent.',
        'required_fields': ['tools_called', 'expected_tools'],
        'optional_fields': [],
        'default_threshold': 0.5,
        'tags': ['agent', 'tool'],
        'score_range': [0, 1]
    }
}

try:
    if Config.GATEWAY_AVAILABLE:
        import requests
        response = requests.get(f"{settings.ai_toolkit_url}/metrics", timeout=5)
        response.raise_for_status()
        METRIC_REGISTRY = response.json().get("metrics", FALLBACK_REGISTRY)
    else:
        METRIC_REGISTRY = FALLBACK_REGISTRY
except:
    METRIC_REGISTRY = FALLBACK_REGISTRY


def get_all_tags():
    """Get all unique tags from metrics"""
    tags = set()
    for metric in METRIC_REGISTRY.values():
        tags.update(metric["tags"])
    return sorted(tags)


def create_tag_button(tag):
    """Create a tag filter button"""
    tag_colors = {
        "knowledge": "primary",
        "agent": "success",
        "single_turn": "info",
        "multi_turn": "warning",
        "data": "secondary",
        "safety": "danger"
    }

    color_class = tag_colors.get(tag, "secondary")

    return dbc.Button(
        tag.replace("_", " ").title(),
        id={"type": "tag-filter", "tag": tag},
        className=f"tag-filter-btn tag-{color_class}",
        size="sm",
        outline=True,
        n_clicks=0
    )


def create_metric_card(key, metric, is_selected=False):
    """Create a single metric card"""
    return html.Div(
        className="metric-card" + (" selected" if is_selected else ""),
        id={"type": "metric-card", "key": key},
        children=[
            html.Div(className="metric-card-header", children=[
                html.Div(className="metric-name-row", children=[
                    dbc.Checkbox(
                        id={"type": "metric-checkbox", "key": key},
                        className="metric-checkbox",
                        value=is_selected,
                        persistence=True,
                        persistence_type='session'
                    ),
                    html.H4(metric["name"], className="metric-name"),
                ]),
                html.Div(className="metric-tags", children=[
                    html.Span(tag, className=f"metric-tag tag-{tag}")
                    for tag in metric["tags"]
                ])
            ]),

            html.Div(className="metric-card-body", children=[
                html.P(metric["description"], className="metric-description"),

                html.Div(className="metric-specs", children=[
                    html.Div(className="spec-item", children=[
                        html.I(className="fas fa-sliders-h spec-icon"),
                        html.Span(f"Threshold: {metric['default_threshold']}", className="spec-text")
                    ]),
                    html.Div(className="spec-item", children=[
                        html.I(className="fas fa-chart-line spec-icon"),
                        html.Span(f"Range: {metric['score_range'][0]}-{metric['score_range'][1]}",
                                  className="spec-text")
                    ])
                ])
            ]),

            html.Div(className="metric-card-footer", children=[
                html.Div(className="fields-section", children=[
                    html.Span("Required:", className="fields-label"),
                    html.Div(className="fields-list", children=[
                        html.Span(field, className="field-chip required")
                        for field in metric["required_fields"]
                    ])
                ]),

                html.Div(className="fields-section", children=[
                    html.Span("Optional:", className="fields-label"),
                    html.Div(className="fields-list", children=[
                        html.Span(field, className="field-chip optional")
                        for field in metric.get("optional_fields", [])
                    ] if metric.get("optional_fields") else [
                        html.Span("None", className="field-chip none")
                    ])
                ])
            ])
        ]
    )


def create_metric_registry_modal():
    """Create the metric registry modal component with tabs"""
    from components.custom_metric_creator import create_custom_metric_tab

    return html.Div([
        dcc.Store(id='selected-metrics-store', data=[]),
        dcc.Store(id='active-tags-store', data=[]),

        dbc.Modal([
            dbc.ModalHeader(className="metric-registry-header", children=[
                html.Div(className="registry-header-content", children=[
                    html.I(className="fas fa-cube registry-icon"),
                    html.H2("Metric Registry", className="registry-title"),
                    html.Span(f"{len(METRIC_REGISTRY)} Available", className="registry-count")
                ])
            ]),

            dbc.ModalBody(className="metric-registry-body", children=[
                # Tabs for Browse vs Create
                dbc.Tabs(id="metric-registry-tabs", active_tab="browse-tab", className="metric-tabs", children=[
                    # Browse Metrics Tab
                    dbc.Tab(
                        label="Browse Metrics",
                        tab_id="browse-tab",
                        className="metric-tab",
                        children=[
                            html.Div(className="tab-content-wrapper", children=[
                                html.Div(className="registry-controls", children=[
                                    html.Div(className="registry-search-wrapper", children=[
                                        html.I(className="fas fa-search search-icon"),
                                        dcc.Input(
                                            id="metric-search",
                                            type="text",
                                            placeholder="Search metrics by name or description...",
                                            className="metric-search-input",
                                            debounce=True,
                                            value=""
                                        )
                                    ]),

                                    html.Div(className="tag-filters", children=[
                                        html.Span("Filter by tags:", className="filter-label"),
                                        html.Div(id="tag-filter-buttons", className="tag-buttons", children=[
                                            create_tag_button(tag) for tag in get_all_tags()
                                        ])
                                    ]),

                                    html.Div(className="view-options", children=[
                                        dbc.ButtonGroup([
                                            dbc.Button(html.I(className="fas fa-th-large"), id="view-grid-metrics",
                                                       size="sm", className="view-btn active"),
                                            dbc.Button(html.I(className="fas fa-list"), id="view-list-metrics",
                                                       size="sm", className="view-btn")
                                        ])
                                    ])
                                ]),

                                html.Div(id="metrics-container", className="metrics-grid", children=[
                                    create_metric_card(key, metric) for key, metric in METRIC_REGISTRY.items()
                                ]),

                                html.Div(className="selected-metrics-summary", children=[
                                    html.Div(className="summary-header", children=[
                                        html.I(className="fas fa-check-circle me-2"),
                                        html.Span("Selected Metrics", className="summary-title"),
                                        html.Span("0", id="selected-count", className="selected-badge")
                                    ]),
                                    html.Div(id="selected-metrics-list", className="selected-list")
                                ])
                            ])
                        ]),

                    # Create Custom Metric Tab
                    dbc.Tab(
                        label="Create Custom",
                        tab_id="create-tab",
                        className="metric-tab",
                        children=[
                            html.Div(className="tab-content-wrapper", children=[
                                create_custom_metric_tab()
                            ])
                        ])
                ])
            ]),

            dbc.ModalFooter(className="metric-registry-footer", children=[
                html.Div(className="footer-actions", children=[
                    html.Div(className="action-info", children=[
                        html.I(className="fas fa-info-circle me-2"),
                        html.Span("Selected metrics will be added to your experiment configuration")
                    ]),
                    html.Div(className="action-buttons", children=[
                        dbc.Button("Cancel", id="cancel-metrics", className="btn-secondary-modal"),
                        dbc.Button([
                            html.I(className="fas fa-plus me-2"),
                            "Add Selected Metrics"
                        ], id="apply-metrics", className="btn-primary-modal")
                    ])
                ])
            ])
        ], id="metric-registry-modal", size="xl", is_open=False, className="metric-registry-modal",
            style={"maxWidth": "95%"})  # Make modal wider for custom metric creator
    ])


def create_metric_registry_button():
    """Create the button to open metric registry"""
    return html.Div(className="metric-registry-section", children=[
        dbc.Button([
            html.I(className="fas fa-cube-grid me-2"),
            "Browse Metric Registry",
            html.Span(f"{len(METRIC_REGISTRY)}", className="metric-count-badge")
        ], id="open-metric-registry",
            className="btn-metric-registry",
            size="sm")
    ])


def register_metric_registry_callbacks(app):
    """Register callbacks for metric registry"""
    from components.custom_metric_creator import register_custom_metric_callbacks

    register_custom_metric_callbacks(app)

    @app.callback(
        Output("metric-registry-modal", "is_open"),
        [Input("open-metric-registry", "n_clicks"),
         Input("cancel-metrics", "n_clicks"),
         Input("apply-metrics", "n_clicks")],
        State("metric-registry-modal", "is_open"),
        prevent_initial_call=True
    )
    def toggle_metric_modal(open_clicks, cancel_clicks, apply_clicks, is_open):
        """Toggle metric registry modal"""
        ctx = callback_context
        if not ctx.triggered:
            return False

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "open-metric-registry":
            return True
        elif trigger_id in ["cancel-metrics", "apply-metrics"]:
            return False

        return is_open

    @app.callback(
        [Output("view-grid-metrics", "className"),
         Output("view-list-metrics", "className"),
         Output("metrics-container", "className")],
        [Input("view-grid-metrics", "n_clicks"),
         Input("view-list-metrics", "n_clicks")],
        prevent_initial_call=False
    )
    def toggle_view_mode(grid_clicks, list_clicks):
        """Toggle between grid and list view"""
        ctx = callback_context

        if not ctx.triggered:
            return "view-btn active", "view-btn", "metrics-grid"

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "view-list-metrics":
            return "view-btn", "view-btn active", "metrics-list"
        else:
            return "view-btn active", "view-btn", "metrics-grid"

    @app.callback(
        Output('active-tags-store', 'data'),
        Input({"type": "tag-filter", "tag": ALL}, "n_clicks"),
        State({"type": "tag-filter", "tag": ALL}, "id"),
        prevent_initial_call=True
    )
    def update_active_tags(n_clicks_list, tag_ids):
        """Track which tag filters are active based on click count"""
        if not n_clicks_list or not tag_ids:
            return []

        active_tags = []
        for i, n_clicks in enumerate(n_clicks_list):
            if n_clicks and n_clicks % 2 == 1:
                active_tags.append(tag_ids[i]["tag"])

        return active_tags

    @app.callback(
        Output({"type": "tag-filter", "tag": ALL}, "className"),
        Input('active-tags-store', 'data'),
        State({"type": "tag-filter", "tag": ALL}, "id"),
        prevent_initial_call=False
    )
    def update_tag_button_styles(active_tags, tag_ids):
        """Update tag button classes based on active state"""
        if not tag_ids:
            return []

        active_tags = active_tags or []
        tag_colors = {
            "knowledge": "primary",
            "agent": "success",
            "single_turn": "info",
            "multi_turn": "warning",
            "data": "secondary",
            "safety": "danger"
        }

        classnames = []
        for tag_id in tag_ids:
            tag = tag_id["tag"]
            color_class = tag_colors.get(tag, "secondary")
            base_class = f"tag-filter-btn tag-{color_class}"

            if tag in active_tags:
                classnames.append(f"{base_class} active")
            else:
                classnames.append(base_class)

        return classnames

    @app.callback(
        Output('selected-metrics-store', 'data'),
        Input({"type": "metric-checkbox", "key": ALL}, "value"),
        State({"type": "metric-checkbox", "key": ALL}, "id"),
        prevent_initial_call=True
    )
    def update_selected_metrics(checkbox_values, checkbox_ids):
        """Update the store with selected metrics"""
        selected = []
        if checkbox_values and checkbox_ids:
            for i, value in enumerate(checkbox_values):
                if value and i < len(checkbox_ids):
                    selected.append(checkbox_ids[i]["key"])
        return selected

    # Main filtering callback
    @app.callback(
        [Output("metrics-container", "children"),
         Output("selected-count", "children"),
         Output("selected-metrics-list", "children")],
        [Input("metric-search", "value"),
         Input('active-tags-store', 'data'),
         Input('selected-metrics-store', 'data')],
        prevent_initial_call=False
    )
    def filter_and_display_metrics(search_term, active_tags, selected_metrics):
        """Filter metrics and update display"""
        active_tags = active_tags or []
        selected_metrics = selected_metrics or []

        # Filter metrics
        filtered_metrics = {}
        for key, metric in METRIC_REGISTRY.items():
            # Search filter
            if search_term:
                search_lower = search_term.lower()
                if (search_lower not in metric["name"].lower() and
                        search_lower not in metric["description"].lower() and
                        search_lower not in key.lower()):
                    continue

            # Tag filter
            if active_tags:
                if not any(tag in metric["tags"] for tag in active_tags):
                    continue

            filtered_metrics[key] = metric

        # Create metric cards with selection state
        cards = []
        for key, metric in filtered_metrics.items():
            is_selected = key in selected_metrics
            cards.append(create_metric_card(key, metric, is_selected))

        # Count selected
        selected_count = len(selected_metrics)

        # Create selected list
        selected_list = []
        for key in selected_metrics:
            if key in METRIC_REGISTRY:
                selected_list.append(
                    html.Div(className="selected-metric-item", children=[
                        html.I(className="fas fa-check me-2"),
                        html.Span(METRIC_REGISTRY[key]["name"])
                    ])
                )

        return cards, str(selected_count), selected_list

    @app.callback(
        Output("config-editor", "value", allow_duplicate=True),
        Input("apply-metrics", "n_clicks"),
        [State('selected-metrics-store', 'data'),
         State("config-editor", "value")],
        prevent_initial_call=True
    )
    def apply_selected_metrics(apply_clicks, selected_metrics, current_config):
        """Add selected metrics to configuration"""
        if not apply_clicks or not selected_metrics:
            return dash.no_update

        if not current_config:
            current_config = (
                "# Experiment Configuration\nmodel:\n  name: gpt-4\n  temperature: 0.7\n  max_tokens: 1000\n"
            )

        # Check if metrics section already exists
        if "metrics:" in current_config or "metric:" in current_config:
            lines = current_config.split('\n')
            metric_start = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('metric') and ':' in line:
                    metric_start = i
                    break

            if metric_start >= 0:
                metric_end = len(lines)
                for i in range(metric_start + 1, len(lines)):
                    if lines[i] and not lines[i].startswith(' ') and not lines[i].startswith('\t'):
                        metric_end = i
                        break

                lines = lines[:metric_start] + lines[metric_end:]
                current_config = '\n'.join(lines)

        # Add metrics section
        metrics_yaml = "\n# Selected Metrics from Registry\nmetrics:\n"

        for metric_key in selected_metrics:
            if metric_key in METRIC_REGISTRY:
                metric = METRIC_REGISTRY[metric_key]
                metric_name = metric_key

                metrics_yaml += f"  {metric_name}: \n"

                if "data" in metric.get("tags", []):
                    metrics_yaml += f"    threshold: {metric['default_threshold']}\n"
                else:
                    metrics_yaml += f"    model_name: gpt-4o\n"
                    metrics_yaml += f"    threshold: {metric['default_threshold']}\n"

                metrics_yaml += f"    # {metric['description']}\n"

                if metric.get('required_fields'):
                    metrics_yaml += f"    # Required fields: {', '.join(metric['required_fields'])}\n"
                if metric.get('optional_fields'):
                    metrics_yaml += f"    # Optional fields: {', '.join(metric['optional_fields'])}\n"

                metrics_yaml += "\n"

        return current_config.rstrip() + "\n" + metrics_yaml
````
```
from dash import html, dcc, Input, Output, State, callback_context, ALL, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import json
import requests
from config import settings

# Get metric registry
try:
    response = requests.get(f'{settings.ai_toolkit_url}/metrics')
    METRIC_REGISTRY = response.json()['metrics']
except:
    try:
        from components.metric_registry import METRIC_REGISTRY
    except:
        METRIC_REGISTRY = {}


def get_metrics_for_fields_and_goal(selected_fields, goal):
    """
    Intelligently recommend metrics based on available fields and goal.
    Returns a dict with categorized metrics.
    """
    recommended = {
        'perfect_match': [],
        'partial_match': [],
        'optional_enhancement': []
    }

    for key, metric in METRIC_REGISTRY.items():
        required_fields = metric.get('required_fields', [])
        optional_fields = metric.get('optional_fields', [])
        tags = metric.get('tags', [])

        has_all_required = all(field in selected_fields for field in required_fields)
        has_some_required = any(field in selected_fields for field in required_fields)
        has_optional = any(field in selected_fields for field in optional_fields)

        if not has_some_required and required_fields:
            continue

        goal_match = False
        if goal == 'quality':
            goal_match = 'single_turn' in tags or 'multi_turn' in tags or 'knowledge' in tags
        elif goal == 'accuracy':
            goal_match = 'expected_output' in selected_fields and key in [
                'answer_correctness', 'faithfulness', 'context_adherence'
            ]
        elif goal == 'rag_performance':
            goal_match = 'retrieved_content' in selected_fields or 'data' in tags
        elif goal == 'agent_behavior':
            goal_match = 'agent' in tags or 'multi_turn' in tags
        elif goal == 'comprehensive':
            goal_match = True

        if not goal_match:
            continue

        if has_all_required:
            recommended['perfect_match'].append(key)
        elif has_some_required:
            recommended['partial_match'].append(key)
        elif has_optional:
            recommended['optional_enhancement'].append(key)

    return recommended


def create_metric_selector_layout():
    """Create the interactive metric selector page"""
    return html.Div(className="metric-selector-page", children=[
        # Animated Background Elements
        html.Div(className="animated-bg", children=[
            html.Div(className="floating-shape shape-1"),
            html.Div(className="floating-shape shape-2"),
            html.Div(className="floating-shape shape-3"),
            html.Div(className="pulse-ring ring-1"),
            html.Div(className="pulse-ring ring-2"),
        ]),

        # Header with Level Indicator
        html.Div(className="experiment-header gamified-header", children=[
            html.Div(className="header-content", children=[
                html.Div(className="header-icon-exp animated-icon", children=[
                    html.I(className="fas fa-magic")
                ]),
                html.Div(children=[
                    html.H1("Metric Quest Builder", className="experiment-title glowing-text"),
                    html.P(
                        "Embark on your journey to discover the perfect evaluation metrics",
                        className="experiment-subtitle"
                    )
                ]),
                # Experience Level Badge
                html.Div(className="experience-badge", children=[
                    html.Div(className="xp-level", children=[
                        html.I(className="fas fa-star"),
                        html.Span("Level ", className="level-text"),
                        html.Span("1", id="user-level", className="level-number")
                    ]),
                    html.Div(className="xp-bar", children=[
                        html.Div(className="xp-progress", id="xp-progress", style={"width": "0%"})
                    ])
                ])
            ])
        ]),

        dbc.Container(fluid=True, className="metric-selector-container", children=[
            # Progress tracker with achievements
            html.Div(className="selector-progress-bar enhanced", children=[
                # Achievement Badges Row
                html.Div(className="achievement-row", children=[
                    html.Div(className="achievement-badge", id="achievement-data", children=[
                        html.I(className="fas fa-database"),
                        html.Span("Data Master")
                    ]),
                    html.Div(className="achievement-badge", id="achievement-goal", children=[
                        html.I(className="fas fa-bullseye"),
                        html.Span("Goal Setter")
                    ]),
                    html.Div(className="achievement-badge", id="achievement-metrics", children=[
                        html.I(className="fas fa-cubes"),
                        html.Span("Metric Expert")
                    ]),
                    html.Div(className="achievement-badge", id="achievement-complete", children=[
                        html.I(className="fas fa-trophy"),
                        html.Span("Quest Complete")
                    ])
                ]),

                # Progress Steps
                html.Div(className="progress-steps", children=[
                    html.Div(className="progress-step active", id="step-indicator-1", children=[
                        html.Div(className="step-number", children=[
                            html.I(className="fas fa-database step-icon")
                        ]),
                        html.Div(className="step-label", children="Data Fields")
                    ]),
                    html.Div(className="progress-connector"),
                    html.Div(className="progress-step", id="step-indicator-2", children=[
                        html.Div(className="step-number", children=[
                            html.I(className="fas fa-bullseye step-icon")
                        ]),
                        html.Div(className="step-label", children="Goal")
                    ]),
                    html.Div(className="progress-connector"),
                    html.Div(className="progress-step", id="step-indicator-3", children=[
                        html.Div(className="step-number", children=[
                            html.I(className="fas fa-sitemap step-icon")
                        ]),
                        html.Div(className="step-label", children="Metrics")
                    ]),
                    html.Div(className="progress-connector"),
                    html.Div(className="progress-step", id="step-indicator-4", children=[
                        html.Div(className="step-number", children=[
                            html.I(className="fas fa-trophy step-icon")
                        ]),
                        html.Div(className="step-label", children="Complete")
                    ])
                ])
            ]),

            # Hidden placeholder elements
            html.Div(style={'display': 'none'}, children=[
                dbc.Button("", id="step-1-next", n_clicks=0),
                dbc.Button("", id="step-2-next", n_clicks=0),
                dbc.Button("", id="step-3-next", n_clicks=0),
                dbc.Button("", id="step-2-back", n_clicks=0),
                dbc.Button("", id="step-3-back", n_clicks=0),
                dbc.Button("", id="step-4-back", n_clicks=0),
                html.Span("", id="field-selection-hint"),
                html.Div(id="metric-tree-container"),
                html.Div(id="selected-metrics-review"),
                html.Pre(id="yaml-preview"),
                dbc.Button("", id="download-config", n_clicks=0),
                dbc.Button("", id="copy-config", n_clicks=0),
            ]),

            # Data stores
            dcc.Store(id='selector-step', data=1),
            dcc.Store(id='selected-data-fields', data=[]),
            dcc.Store(id='selected-evaluation-goal', data=None),
            dcc.Store(id='recommended-metrics', data={}),
            dcc.Store(id='selector-metrics', data=[]),
            dcc.Store(id='user-xp', data=0),

            # Step content area
            html.Div(id="selector-step-content", className="step-content-animated",
                     children=create_step_1_content()),

            # Hidden download component
            dcc.Download(id="download-yaml")
        ])
    ])


def create_step_1_content():
    """Step 1: Select available data fields"""
    data_fields = [
        {
            "key": "query",
            "name": "Query / Input",
            "icon": "fa-comment-dots",
            "description": "The input question or prompt sent to your agent",
            "required": True,
            "color": "#3498DB",
            "points": 100
        },
        {
            "key": "actual_output",
            "name": "Actual Output",
            "icon": "fa-robot",
            "description": "The response generated by your agent",
            "required": True,
            "color": "#8B9F4F",
            "points": 100
        },
        {
            "key": "expected_output",
            "name": "Expected Output",
            "icon": "fa-check-circle",
            "description": "Ground truth or reference answer for comparison",
            "required": False,
            "color": "#27AE60",
            "points": 50
        },
        {
            "key": "retrieved_content",
            "name": "Retrieved Content",
            "icon": "fa-database",
            "description": "Context chunks retrieved from your knowledge base (RAG)",
            "required": False,
            "color": "#9B59B6",
            "points": 75
        },
        {
            "key": "context",
            "name": "Context",
            "icon": "fa-file-alt",
            "description": "Additional context or reference information",
            "required": False,
            "color": "#E74C3C",
            "points": 50
        },
        {
            "key": "conversation",
            "name": "Conversation History",
            "icon": "fa-comments",
            "description": "Multi-turn dialogue context",
            "required": False,
            "color": "#E67E22",
            "points": 75
        },
        {
            "key": "tools_called",
            "name": "Tools Called",
            "icon": "fa-tools",
            "description": "Functions or APIs invoked by your agent",
            "required": False,
            "color": "#34495E",
            "points": 60
        },
        {
            "key": "additional_input",
            "name": "Additional Context",
            "icon": "fa-puzzle-piece",
            "description": "Extra metadata or context for evaluation",
            "required": False,
            "color": "#16A085",
            "points": 40
        },
        {
            "key": "acceptance_criteria",
            "name": "Acceptance Criteria",
            "icon": "fa-clipboard-check",
            "description": "Specific rules or requirements to validate",
            "required": False,
            "color": "#D35400",
            "points": 60
        }
    ]

    return html.Div(className="step-content-wrapper", children=[
        # Quest Header
        html.Div(className="quest-header", children=[
            html.Div(className="quest-badge", children=[
                html.I(className="fas fa-scroll"),
                html.Span("Quest 1")
            ]),
            html.H2("Choose Your Data Arsenal", className="step-title"),
            html.P(
                "Select the data fields available in your evaluation dataset. Each field unlocks different metric possibilities!",
                className="step-description"
            ),
            # Points Counter
            html.Div(className="points-counter", children=[
                html.I(className="fas fa-coins"),
                html.Span("Points: "),
                html.Span("0", id="field-points", className="points-value")
            ])
        ]),

        # Data field grid with hover effects
        html.Div(className="data-field-grid gamified", children=[
            create_data_field_card(field) for field in data_fields
        ]),

        # Step actions with animated button
        html.Div(className="step-actions", children=[
            html.Div(className="selected-summary", children=[
                html.I(className="fas fa-info-circle me-2 pulse-icon"),
                html.Span("Select at least the required fields to continue", id="field-selection-hint")
            ]),
            dbc.Button([
                "Continue to Goal Selection",
                html.I(className="fas fa-arrow-right ms-2")
            ], id="step-1-next", className="btn-step-next glow-button", size="lg", disabled=True)
        ])
    ])


def create_data_field_card(field):
    """Create a selectable data field card with gamification"""
    return html.Div(
        className=f"data-field-card gamified {'required-field' if field['required'] else ''}",
        id={"type": "data-field-card", "key": field["key"]},
        **{"data-field": field["key"], "data-points": str(field.get("points", 0))},
        children=[
            # Card Glow Effect
            html.Div(className="card-glow"),

            html.Div(className="field-card-header", children=[
                html.Div(
                    className="field-icon hexagon",
                    style={"backgroundColor": field["color"]},
                    children=[html.I(className=f"fas {field['icon']}")]
                ),
                html.Div(className="field-info", children=[
                    html.H4(field["name"], className="field-name"),
                    html.Div(className="field-meta", children=[
                        html.Span(
                            "Required" if field["required"] else "Optional",
                            className=f"field-badge {'badge-required' if field['required'] else 'badge-optional'}"
                        ),
                        html.Span(f"+{field.get('points', 0)} XP", className="field-points")
                    ])
                ])
            ]),
            html.P(field["description"], className="field-description"),
            html.Div(className="field-check-indicator animated", children=[
                html.I(className="fas fa-check-circle")
            ])
        ]
    )


def create_step_2_content():
    """Step 2: Select evaluation goal"""
    evaluation_goals = [
        {
            "key": "quality",
            "name": "Response Quality",
            "icon": "fa-star",
            "description": "Evaluate how helpful, relevant, and complete responses are",
            "color": "#F39C12",
            "difficulty": "Beginner",
            "points": 100
        },
        {
            "key": "accuracy",
            "name": "Factual Accuracy",
            "icon": "fa-bullseye",
            "description": "Verify correctness against ground truth or retrieved context",
            "color": "#E74C3C",
            "difficulty": "Intermediate",
            "points": 150
        },
        {
            "key": "rag_performance",
            "name": "RAG Performance",
            "icon": "fa-search",
            "description": "Assess retrieval quality and context utilization",
            "color": "#9B59B6",
            "difficulty": "Advanced",
            "points": 200
        },
        {
            "key": "agent_behavior",
            "name": "Agent Behavior",
            "icon": "fa-robot",
            "description": "Evaluate tool usage, conversation flow, and task completion",
            "color": "#3498DB",
            "difficulty": "Advanced",
            "points": 200
        },
        {
            "key": "comprehensive",
            "name": "Comprehensive Evaluation",
            "icon": "fa-layer-group",
            "description": "Run multiple metrics for thorough assessment",
            "color": "#8B9F4F",
            "difficulty": "Expert",
            "points": 300
        }
    ]

    return html.Div(className="step-content-wrapper", children=[
        # Quest Header
        html.Div(className="quest-header", children=[
            html.Div(className="quest-badge", children=[
                html.I(className="fas fa-scroll"),
                html.Span("Quest 2")
            ]),
            html.H2("Define Your Mission", className="step-title"),
            html.P(
                "Choose your evaluation objective. Each goal unlocks unique metric combinations!",
                className="step-description"
            )
        ]),

        # Goal selection grid
        html.Div(className="goal-grid gamified", children=[
            create_goal_card(goal) for goal in evaluation_goals
        ]),

        # Step actions
        html.Div(className="step-actions", children=[
            dbc.Button([
                html.I(className="fas fa-arrow-left me-2"),
                "Back"
            ], id="step-2-back", className="btn-step-back", size="lg", outline=True),
            dbc.Button([
                "Discover Metrics",
                html.I(className="fas fa-sparkles ms-2")
            ], id="step-2-next", className="btn-step-next glow-button", size="lg", disabled=True)
        ])
    ])


def create_goal_card(goal):
    """Create a selectable evaluation goal card with gamification"""
    difficulty_colors = {
        "Beginner": "#27AE60",
        "Intermediate": "#F39C12",
        "Advanced": "#E74C3C",
        "Expert": "#8B59B6"
    }

    return html.Div(
        className="goal-card gamified",
        id={"type": "goal-card", "key": goal["key"]},
        **{"data-goal": goal["key"], "data-points": str(goal.get("points", 0))},
        children=[
            html.Div(className="goal-card-inner", children=[
                # Difficulty Badge
                html.Div(className="difficulty-badge",
                         style={"backgroundColor": difficulty_colors.get(goal["difficulty"], "#7F8C8D")},
                         children=goal["difficulty"]),

                html.Div(
                    className="goal-icon-large rotating",
                    style={"backgroundColor": goal["color"]},
                    children=[html.I(className=f"fas {goal['icon']}")]
                ),
                html.H3(goal["name"], className="goal-name"),
                html.P(goal["description"], className="goal-description"),

                # Points Display
                html.Div(className="goal-points", children=[
                    html.I(className="fas fa-coins"),
                    html.Span(f"+{goal.get('points', 0)} XP")
                ]),

                html.Div(className="goal-check animated", children=[
                    html.I(className="fas fa-check-circle")
                ])
            ])
        ]
    )


def create_step_3_content(available_fields, goal, recommended_metrics, selected_metrics=None):
    """Step 3: Interactive tree visualization for metric selection"""
    return html.Div(className="step-content-wrapper", children=[
        # Quest Header
        html.Div(className="quest-header", children=[
            html.Div(className="quest-badge", children=[
                html.I(className="fas fa-scroll"),
                html.Span("Quest 3")
            ]),
            html.H2("The Metric Constellation", className="step-title"),
            html.P([
                "Navigate the cosmic tree of metrics. Your ",
                html.Strong(f"{len(available_fields or [])} data fields"),
                " and ",
                html.Strong(f"{goal or 'chosen'} goal"),
                " have illuminated your path!"
            ], className="step-description")
        ]),

        # Recommendation summary cards with animations
        create_recommendation_summary(recommended_metrics),

        # Interactive tree container with enhanced visuals
        html.Div(id="metric-tree-container", className="metric-tree-visualization enhanced",
                 children=create_interactive_tree(recommended_metrics, selected_metrics)),

        # Step actions
        html.Div(className="step-actions", children=[
            dbc.Button([
                html.I(className="fas fa-arrow-left me-2"),
                "Back to Goals"
            ], id="step-3-back", className="btn-step-back", size="lg", outline=True),
            dbc.Button([
                "Forge Configuration",
                html.I(className="fas fa-hammer ms-2")
            ], id="step-3-next", className="btn-step-next glow-button", size="lg")
        ])
    ])


def create_recommendation_summary(recommended_metrics):
    """Create summary cards for metric recommendations with animations"""
    perfect = len(recommended_metrics.get('perfect_match', []))
    partial = len(recommended_metrics.get('partial_match', []))
    optional = len(recommended_metrics.get('optional_enhancement', []))

    return html.Div(className="recommendation-summary-cards animated", children=[
        html.Div(className="summary-card perfect-match-card floating", children=[
            html.Div(className="summary-card-icon pulse", children=[
                html.I(className="fas fa-crown")
            ]),
            html.Div(className="summary-card-content", children=[
                html.Div(className="summary-number counting", children=str(perfect)),
                html.Div(className="summary-label", children="Perfect Matches"),
                html.Div(className="summary-description",
                         children="All requirements satisfied")
            ])
        ]),
        html.Div(className="summary-card partial-match-card floating delay-1", children=[
            html.Div(className="summary-card-icon pulse", children=[
                html.I(className="fas fa-star-half-alt")
            ]),
            html.Div(className="summary-card-content", children=[
                html.Div(className="summary-number counting", children=str(partial)),
                html.Div(className="summary-label", children="Partial Matches"),
                html.Div(className="summary-description",
                         children="Some requirements met")
            ])
        ]),
        html.Div(className="summary-card optional-card floating delay-2", children=[
            html.Div(className="summary-card-icon pulse", children=[
                html.I(className="fas fa-plus-circle")
            ]),
            html.Div(className="summary-card-content", children=[
                html.Div(className="summary-number counting", children=str(optional)),
                html.Div(className="summary-label", children="Enhancements"),
                html.Div(className="summary-description",
                         children="Bonus capabilities")
            ])
        ])
    ])


def create_interactive_tree(recommended_metrics, selected_metrics=None):
    """Create the interactive tree visualization with enhanced visuals"""
    if not recommended_metrics or all(len(v) == 0 for v in recommended_metrics.values()):
        return html.Div(className="no-metrics-message animated", children=[
            html.I(className="fas fa-exclamation-circle rotating",
                   style={"fontSize": "3rem", "color": "#7F8C8D"}),
            html.H3("No metrics discovered yet", style={"marginTop": "1rem"}),
            html.P("Adjust your data fields or goal to unlock metrics.")
        ])

    selected_metrics = selected_metrics or []

    # Create the enhanced tree structure
    tree_content = html.Div(className="metric-tree-structure cosmic", children=[
        # Animated particles background
        html.Div(className="tree-particles", children=[
            html.Div(className="particle") for _ in range(20)
        ]),

        # Root node with pulsing effect
        html.Div(className="tree-root-node animated", children=[
            html.Div(className="root-node-content glowing", children=[
                html.I(className="fas fa-sitemap rotating"),
                html.Span("Metric Universe")
            ])
        ]),

        # Connection lines with gradient
        html.Svg(className="tree-connections", children=[
            html.Defs(children=[
                html.LinearGradient(id="gradient-perfect", children=[
                    html.Stop(offset="0%", style={"stopColor": "#27AE60", "stopOpacity": 1}),
                    html.Stop(offset="100%", style={"stopColor": "#8B9F4F", "stopOpacity": 0.3})
                ])
            ])
        ]),

        # Branches container with staggered animations
        html.Div(className="tree-branches animated", children=[
            create_tree_branch(
                "Perfect Matches",
                "fa-crown",
                "perfect-branch",
                recommended_metrics.get('perfect_match', []),
                "#27AE60",
                selected_metrics,
                0
            ),
            create_tree_branch(
                "Partial Matches",
                "fa-star-half-alt",
                "partial-branch",
                recommended_metrics.get('partial_match', []),
                "#F39C12",
                selected_metrics,
                1
            ),
            create_tree_branch(
                "Enhancements",
                "fa-plus-circle",
                "optional-branch",
                recommended_metrics.get('optional_enhancement', []),
                "#3498DB",
                selected_metrics,
                2
            )
        ])
    ])

    return tree_content


def create_tree_branch(branch_name, icon, branch_class, metric_keys, color, selected_metrics, delay_index):
    """Create an enhanced branch of the tree with animations"""
    if not metric_keys:
        return html.Div()

    # Group metrics by category
    categories = {
        'Quality & Relevance': [],
        'Accuracy & Correctness': [],
        'RAG & Retrieval': [],
        'Agent & Conversation': [],
        'Other': []
    }

    for metric_key in metric_keys:
        if metric_key not in METRIC_REGISTRY:
            continue

        metric = METRIC_REGISTRY[metric_key]
        tags = metric.get('tags', [])

        if 'single_turn' in tags or 'relevancy' in metric_key:
            categories['Quality & Relevance'].append(metric_key)
        elif 'correctness' in metric_key or 'accuracy' in metric_key:
            categories['Accuracy & Correctness'].append(metric_key)
        elif 'data' in tags or any(kw in metric_key for kw in ['chunk', 'retriev', 'context']):
            categories['RAG & Retrieval'].append(metric_key)
        elif 'agent' in tags or 'multi_turn' in tags:
            categories['Agent & Conversation'].append(metric_key)
        else:
            categories['Other'].append(metric_key)

    # Create enhanced branch structure
    return html.Div(className=f"tree-branch {branch_class} animated delay-{delay_index}", children=[
        html.Div(className="branch-connector glowing"),
        html.Div(className="branch-node floating", style={"backgroundColor": color}, children=[
            html.I(className=f"fas {icon} pulse"),
            html.Span(branch_name),
            html.Span(f"({len(metric_keys)})", className="branch-count badge")
        ]),
        html.Div(className="branch-categories", children=[
            create_category_group(cat_name, metrics, color, selected_metrics)
            for cat_name, metrics in categories.items()
            if metrics
        ])
    ])


def create_category_group(category_name, metric_keys, branch_color, selected_metrics):
    """Create a category group with enhanced styling"""
    return html.Div(className="category-group animated", children=[
        html.Div(className="category-header", children=[
            html.Div(className="category-connector animated"),
            html.Div(className="category-label glass", children=[
                html.I(className="fas fa-folder-open pulse"),
                html.Span(category_name)
            ])
        ]),
        html.Div(className="category-metrics", children=[
            create_tree_metric_node(metric_key, branch_color, metric_key in (selected_metrics or []))
            for metric_key in metric_keys
        ])
    ])


def create_tree_metric_node(metric_key, branch_color, is_selected=False):
    """Create an enhanced metric node in the tree"""
    if metric_key not in METRIC_REGISTRY:
        return html.Div()

    metric = METRIC_REGISTRY[metric_key]

    return html.Div(
        className=f"tree-metric-node animated {'selected' if is_selected else ''}",
        id={"type": "tree-metric-node", "key": metric_key},
        children=[
            html.Div(className="metric-node-connector glowing"),
            html.Div(className="metric-node-content glass", children=[
                html.Div(className="metric-node-header", children=[
                    dbc.Switch(
                        id={"type": "metric-toggle", "key": metric_key},
                        value=is_selected,
                        className="metric-node-toggle fancy"
                    ),
                    html.Div(className="metric-node-info", children=[
                        html.Div(className="metric-node-name", children=metric['name']),
                        html.Div(className="metric-node-key", children=f"({metric_key})")
                    ])
                ]),
                html.Div(className="metric-node-details", children=[
                    html.P(metric['description'], className="metric-node-description"),
                    html.Div(className="metric-node-tags", children=[
                        html.Span(tag.replace("_", " ").title(),
                                  className=f"metric-tag tag-{tag} glowing")
                        for tag in metric.get('tags', [])[:3]
                    ]),
                    html.Div(className="metric-node-fields", children=[
                        html.Div(className="field-requirement", children=[
                            html.I(className="fas fa-check-circle pulse",
                                   style={"color": branch_color, "marginRight": "0.5rem"}),
                            html.Span("Requires: "),
                            html.Span(", ".join(metric.get('required_fields', [])))
                        ])
                    ])
                ])
            ])
        ]
    )


def create_step_4_content(selected_metrics):
    """Step 4: Review and export configuration with celebration"""
    return html.Div(className="step-content-wrapper", children=[
        # Celebration Effects
        html.Div(className="celebration-container", children=[
            html.Div(className="confetti") for _ in range(30)
        ]),

        # Quest Complete Header
        html.Div(className="quest-header celebration", children=[
            html.Div(className="quest-badge gold", children=[
                html.I(className="fas fa-trophy"),
                html.Span("Quest Complete!")
            ]),
            html.H2("Configuration Forged!", className="step-title golden"),
            html.P([
                "Congratulations! You've selected ",
                html.Strong(f"{len(selected_metrics or [])} metrics", className="highlight"),
                ". Your evaluation arsenal is ready!"
            ], className="step-description")
        ]),

        # Success banner with animation
        html.Div(className="success-banner animated pulse-slow", children=[
            html.I(className="fas fa-check-circle rotating", style={"fontSize": "2rem", "color": "#27AE60"}),
            html.Div(style={"flex": 1}, children=[
                html.H3("Achievement Unlocked!", style={"margin": "0 0 0.5rem 0", "color": "#27AE60"}),
                html.P(
                    "Your metric configuration is ready for deployment. Export it to start evaluating!",
                    style={"margin": 0, "color": "#2C3E50"}
                )
            ])
        ], style={
            "display": "flex",
            "gap": "1.5rem",
            "padding": "2rem",
            "background": "linear-gradient(135deg, rgba(39, 174, 96, 0.1) 0%, rgba(139, 159, 79, 0.05) 100%)",
            "border": "2px solid rgba(39, 174, 96, 0.3)",
            "borderRadius": "16px",
            "marginBottom": "3rem",
            "alignItems": "center",
            "boxShadow": "0 0 30px rgba(39, 174, 96, 0.2)"
        }),

        dbc.Row([
            dbc.Col(lg=6, children=[
                html.Div(className="review-card glass animated fadeInLeft", children=[
                    html.H3([
                        html.I(className="fas fa-clipboard-list me-2"),
                        "Selected Metrics"
                    ], className="review-title"),
                    html.Div(id="selected-metrics-review", className="metrics-review-list")
                ])
            ]),
            dbc.Col(lg=6, children=[
                html.Div(className="review-card glass animated fadeInRight", children=[
                    html.H3([
                        html.I(className="fas fa-code me-2"),
                        "YAML Configuration"
                    ], className="review-title"),
                    html.Pre(id="yaml-preview", className="yaml-config-preview")
                ])
            ])
        ]),

        html.Div(className="step-actions final-actions", children=[
            dbc.Button([
                html.I(className="fas fa-arrow-left me-2"),
                "Back to Metrics"
            ], id="step-4-back", className="btn-step-back", size="lg", outline=True),
            html.Div(style={"display": "flex", "gap": "1rem"}, children=[
                dbc.Button([
                    html.I(className="fas fa-download me-2"),
                    "Download YAML"
                ], id="download-config", className="btn-download glow-button", size="lg"),
                dbc.Button([
                    html.I(className="fas fa-copy me-2"),
                    "Copy to Clipboard"
                ], id="copy-config", className="btn-copy glow-button", size="lg")
            ])
        ])
    ])


def register_metric_selector_callbacks(app):
    """Register all callbacks for the metric selector"""

    # Step navigation callback
    @app.callback(
        [Output('selector-step', 'data'),
         Output('selector-step-content', 'children'),
         Output('step-indicator-1', 'className'),
         Output('step-indicator-2', 'className'),
         Output('step-indicator-3', 'className'),
         Output('step-indicator-4', 'className'),
         Output('achievement-data', 'className'),
         Output('achievement-goal', 'className'),
         Output('achievement-metrics', 'className'),
         Output('achievement-complete', 'className')],
        [Input('step-1-next', 'n_clicks'),
         Input('step-2-next', 'n_clicks'),
         Input('step-3-next', 'n_clicks'),
         Input('step-2-back', 'n_clicks'),
         Input('step-3-back', 'n_clicks'),
         Input('step-4-back', 'n_clicks')],
        [State('selector-step', 'data'),
         State('selected-data-fields', 'data'),
         State('selected-evaluation-goal', 'data'),
         State('recommended-metrics', 'data'),
         State('selector-metrics', 'data')],
        prevent_initial_call=True
    )
    def navigate_steps(next1, next2, next3, back2, back3, back4,
                       current_step, fields, goal, recommended, metrics):
        from dash.exceptions import PreventUpdate
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Determine new step
        if 'next' in trigger_id:
            new_step = current_step + 1
        elif 'back' in trigger_id:
            new_step = current_step - 1
        else:
            new_step = current_step

        new_step = max(1, min(4, new_step))

        # Generate step content
        if new_step == 1:
            content = create_step_1_content()
        elif new_step == 2:
            content = create_step_2_content()
        elif new_step == 3:
            content = create_step_3_content(fields, goal, recommended, metrics)
        else:
            content = create_step_4_content(metrics)

        # Update progress indicators
        indicators = []
        for i in range(1, 5):
            if i < new_step:
                indicators.append("progress-step completed")
            elif i == new_step:
                indicators.append("progress-step active")
            else:
                indicators.append("progress-step")

        # Update achievement badges
        achievements = []
        achievements.append("achievement-badge unlocked" if new_step > 1 else "achievement-badge")
        achievements.append("achievement-badge unlocked" if new_step > 2 else "achievement-badge")
        achievements.append("achievement-badge unlocked" if new_step > 3 else "achievement-badge")
        achievements.append("achievement-badge unlocked gold" if new_step == 4 else "achievement-badge")

        return new_step, content, *indicators, *achievements

    # Data field selection callback
    @app.callback(
        [Output('selected-data-fields', 'data'),
         Output('step-1-next', 'disabled'),
         Output('field-selection-hint', 'children'),
         Output('field-points', 'children'),
         Output({"type": "data-field-card", "key": ALL}, "className")],
        Input({"type": "data-field-card", "key": ALL}, "n_clicks"),
        [State({"type": "data-field-card", "key": ALL}, "id"),
         State('selected-data-fields', 'data')],
        prevent_initial_call=True
    )
    def toggle_data_fields(n_clicks_list, card_ids, selected_fields):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update, no_update

        # Parse trigger
        trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        clicked_key = trigger['key']

        # Update selection
        selected_fields = selected_fields or []
        if clicked_key in selected_fields:
            selected_fields.remove(clicked_key)
        else:
            selected_fields.append(clicked_key)

        # Calculate points
        points_map = {
            'query': 100, 'actual_output': 100, 'expected_output': 50,
            'retrieved_content': 75, 'context': 50, 'conversation': 75,
            'tools_called': 60, 'additional_input': 40, 'acceptance_criteria': 60
        }
        total_points = sum(points_map.get(field, 0) for field in selected_fields)

        # Check if required fields are present
        required_fields = ['query', 'actual_output']
        has_required = all(rf in selected_fields for rf in required_fields)

        # Update button state
        button_disabled = not has_required

        # Update hint text
        if has_required:
            hint = f"✓ {len(selected_fields)} fields selected (+{total_points} XP)"
        else:
            missing = [rf for rf in required_fields if rf not in selected_fields]
            hint = f"Select required fields: {', '.join(missing)}"

        # Update card classes
        card_classes = []
        for card_id in card_ids:
            key = card_id['key']
            is_required = key in required_fields
            is_selected = key in selected_fields

            base_class = "data-field-card gamified"
            if is_required:
                base_class += " required-field"
            if is_selected:
                base_class += " selected"

            card_classes.append(base_class)

        return selected_fields, button_disabled, hint, str(total_points), card_classes

    # Goal selection callback
    @app.callback(
        [Output('selected-evaluation-goal', 'data'),
         Output('step-2-next', 'disabled'),
         Output({"type": "goal-card", "key": ALL}, "className")],
        Input({"type": "goal-card", "key": ALL}, "n_clicks"),
        [State({"type": "goal-card", "key": ALL}, "id"),
         State('selected-evaluation-goal', 'data')],
        prevent_initial_call=True
    )
    def select_goal(n_clicks_list, card_ids, current_goal):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update

        # Parse trigger
        trigger = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        clicked_key = trigger['key']

        # Update selection (only one goal at a time)
        new_goal = clicked_key

        # Update card classes
        card_classes = []
        for card_id in card_ids:
            key = card_id['key']
            base_class = "goal-card gamified"
            if key == new_goal:
                base_class += " selected"
            card_classes.append(base_class)

        return new_goal, False, card_classes

    # Generate recommended metrics for step 3
    @app.callback(
        [Output('recommended-metrics', 'data'),
         Output('selector-metrics', 'data', allow_duplicate=True)],
        Input('selector-step', 'data'),
        [State('selected-data-fields', 'data'),
         State('selected-evaluation-goal', 'data')],
        prevent_initial_call=True
    )
    def generate_recommendations(step, fields, goal):
        if step != 3 or not fields or not goal:
            return no_update, no_update

        recommended = get_metrics_for_fields_and_goal(fields, goal)
        # Start with perfect matches selected by default
        initial_selected = recommended.get('perfect_match', [])
        return recommended, initial_selected

    # Single callback to handle metric selection
    @app.callback(
        Output('selector-metrics', 'data'),
        Input({"type": "metric-toggle", "key": ALL}, "value"),
        [State({"type": "metric-toggle", "key": ALL}, "id"),
         State('selector-metrics', 'data')],
        prevent_initial_call=True
    )
    def update_selected_metrics(toggle_values, toggle_ids, current_selected):
        """Update selected metrics based on toggle switches"""
        if not toggle_ids:
            return current_selected or []

        selected = []
        for i, value in enumerate(toggle_values):
            if value and i < len(toggle_ids):
                selected.append(toggle_ids[i]['key'])
        return selected

    # Generate review content for step 4
    @app.callback(
        [Output('selected-metrics-review', 'children'),
         Output('yaml-preview', 'children')],
        Input('selector-step', 'data'),
        State('selector-metrics', 'data'),
        prevent_initial_call=True
    )
    def generate_review(step, selected_metrics):
        if step != 4:
            return no_update, no_update

        selected_metrics = selected_metrics or []

        # Create metric review list
        review_items = []
        for metric_key in selected_metrics:
            if metric_key not in METRIC_REGISTRY:
                continue

            metric = METRIC_REGISTRY[metric_key]
            review_items.append(
                html.Div(className="review-metric-item animated", children=[
                    html.Div(className="review-metric-icon pulse", children=[
                        html.I(className="fas fa-cube")
                    ]),
                    html.Div(className="review-metric-info", children=[
                        html.Div(metric['name'], className="review-metric-name"),
                        html.Div(metric['description'], className="review-metric-description")
                    ])
                ])
            )

        # Generate YAML configuration
        yaml_config = "# Metric Configuration\n"
        yaml_config += "# Generated by OMEGA Metric Quest Builder\n"
        yaml_config += f"# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n" if 'pd' in dir() else "\n"
        yaml_config += "metrics:\n"

        for metric_key in selected_metrics:
            if metric_key not in METRIC_REGISTRY:
                continue

            metric = METRIC_REGISTRY[metric_key]
            yaml_config += f"  {metric_key}:\n"

            if "data" not in metric.get("tags", []):
                yaml_config += f"    model_name: gpt-4o\n"

            yaml_config += f"    threshold: {metric['default_threshold']}\n"
            yaml_config += f"    # {metric['description']}\n"

            if metric.get('required_fields'):
                yaml_config += f"    # Required: {', '.join(metric['required_fields'])}\n"

            yaml_config += "\n"

        return review_items, yaml_config

    # Download YAML configuration
    @app.callback(
        Output('download-yaml', 'data'),
        Input('download-config', 'n_clicks'),
        State('yaml-preview', 'children'),
        prevent_initial_call=True
    )
    def download_yaml(n_clicks, yaml_content):
        if n_clicks:
            return dict(content=yaml_content, filename="metrics_config.yaml")
        return no_update

    # Copy to clipboard callback
    app.clientside_callback(
        """
        function(n_clicks, yaml_content) {
            if (n_clicks && yaml_content) {
                navigator.clipboard.writeText(yaml_content);
                // Return updated button with success state
                return [
                    {'props': {'className': 'fas fa-check me-2'}, 'type': 'I', 'namespace': 'dash_html_components'},
                    'Copied!'
                ];
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('copy-config', 'children'),
        Input('copy-config', 'n_clicks'),
        State('yaml-preview', 'children'),
        prevent_initial_call=True
    )
````
```
from typing import Any


def single_turn_metric_template(key: str, item: Any) -> str:
    """
    Generate a minimal working template for creating a custom metric.
    This provides the bare minimum code structure to get started.
    """
    class_name = 'SingleTurnJudge'

    return f'''from ai_toolkit.evaluation.metrics.base import BaseMetric, MetricEvaluationResult
from ai_toolkit.evaluation.dataset import DatasetItem

class {class_name}(BaseMetric):
    """TODO: Add description of what this metric evaluates."""

    instruction = (
        "TODO: Add your LLM-as-a-judge evaluation instruction here. "
        "Explain what the metric should evaluate and how. "
    )

    examples = [
        (
            DatasetItem(
                expected_output="TODO: Add expected output example",
                actual_output="TODO: Add actual output example",
                # add any more desired fields
            ),
            MetricEvaluationResult(
                score=1,  # Recommendation is to keep binary for prompt based judges
                explanation="TODO: Add step-by-step explanation of why this gets the score",
            ),
        ),
        # TODO: Add more examples as needed
    ]

metric = {class_name}()
# This expects the `DatasetItem` as an DatasetItem object
input_data = DatasetItem(actual_output='This is a test', expected_output='This is a test')
# Or as a dictionary
input_data = {{'actual_output': 'This is a test', 'expected_output': 'This is a test'}}
# Async run
await metric.execute(input_data)
'''


def multi_turn_metric_template(key: str, item: Any) -> str:
    """
    Generate a minimal working template for creating a multi-turn conversation metric.
    This provides the bare minimum code structure for multi-turn evaluations.
    """
    class_name = 'MultiTurnJudge'

    return f'''from ai_toolkit.evaluation.metrics.base import BaseMetric, MetricEvaluationResult
from ai_toolkit.evaluation.dataset import DatasetItem
from ai_toolkit.evaluation.dataset_schema import MultiTurnConversation, HumanMessage, AIMessage, ToolCall, ToolMessage

class {class_name}(BaseMetric):
    """TODO: Add description of what this metric evaluates for multi-turn conversations."""

    instruction = (
        "TODO: Add your LLM-as-a-judge evaluation instruction here for multi-turn conversations. "
        "Explain what the metric should evaluate and how. "
    )

    examples = [
        (
            DatasetItem(
                conversation=MultiTurnConversation(messages=[
                    HumanMessage(content="TODO: Add first human message"),
                    AIMessage(content="TODO: Add AI response"),
                    HumanMessage(content="TODO: Add follow-up human message"),
                    AIMessage(content="TODO: Add final AI response"),
                    # add any more desired messages, tool calls, etc.
                ])
            ),
            MetricEvaluationResult(
                score=1,  # Recommendation is to keep binary for prompt based judges
                explanation="TODO: Add step-by-step explanation of why this gets the score",
            ),
        ),
        # TODO: Add more examples as needed
    ]

metric = {class_name}()
# This expects the DatasetItem with a conversation field
input_data = DatasetItem(
    conversation=MultiTurnConversation(messages=[
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
    ])
)
# Async run
await metric.execute(input_data)
'''


def heuristic_metric_template(key: str, item: Any) -> str:
    """
    Generate a minimal working template for creating a heuristic/rule-based metric.
    This provides code structure for simple algorithmic evaluations without LLM judges.
    """
    class_name = 'HeuristicMetric'

    return f'''from ai_toolkit.evaluation.metrics.base import BaseMetric, MetricEvaluationResult
from ai_toolkit.evaluation.dataset import DatasetItem

class {class_name}(BaseMetric):
    """TODO: Add description of what this heuristic metric evaluates."""

    async def execute(self, item: DatasetItem) -> MetricEvaluationResult:
        """
        TODO: Implement your heuristic evaluation logic here.
        This method should contain rule-based logic, not LLM calls.

        Args:
            item: DatasetItem containing the data to evaluate

        Returns:
            MetricEvaluationResult with score and optional explanation
        """
        # TODO: Replace this with your actual heuristic logic
        # Example: Check if actual output matches expected output
        is_match = item.actual_output.strip() == item.expected_output.strip()

        score = 1.0 if is_match else 0.0
        explanation = f"Outputs {{'match' if is_match else 'do not match'}}"

        return MetricEvaluationResult(
            score=score,
            explanation=explanation
        )

# Usage example
metric = {class_name}()
input_data = DatasetItem(
    actual_output="Hello world",
    expected_output="Hello world"
)
# Async run
result = await metric.execute(input_data)
print(f"Score: {{result.score}}, Explanation: {{result.explanation}}")
'''


def yaml_metric_template(key: str, item: Any) -> str:
    """
    Generate a minimal working template for creating a YAML-based metric.
    This provides the bare minimum YAML structure for LLM-powered evaluation.
    """

    return """# Save this as: my_metric.yaml
name: 'MyMetric'
instruction: |
  TODO: Add your LLM-as-a-judge evaluation instruction here.
  Explain what the metric should evaluate and how.
  Provide a score of either 0 or 1 and explain your reasoning.

# Optional configuration
model_name: "gpt-4"
threshold: 0.7
required_fields:
  - "actual_output"
  - "expected_output"

examples:
  - input:
      actual_output: "TODO: Add example actual output"
      expected_output: "TODO: Add example expected output"
    output:
      score: 1
      explanation: "TODO: Add explanation for why this gets this score"

  - input:
      actual_output: "TODO: Add another example"
      expected_output: "TODO: Add another expected output"
    output:
      score: 0
      explanation: "TODO: Add explanation for why this gets this score"

##  Usage in Python:
# from ai_toolkit.evaluation.metrics.yaml_metrics import load_metric_from_yaml
# from ai_toolkit.evaluation.dataset import DatasetItem
#
# MetricClass = load_metric_from_yaml("my_metric.yaml")
# metric = MetricClass()
#
# input_data = DatasetItem(
#     actual_output="Test output",
#     expected_output="Expected output"
# )
#
# result = await metric.execute(input_data)
# print(result.pretty())
"""
````

```
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc


def create_custom_metric_tab():
    """Create the custom metric creation tab with template selection and code editor"""

    return html.Div(className="custom-metric-creator", children=[
        # Header Section
        html.Div(className="creator-header", children=[
            html.Div(className="creator-header-content", children=[
                html.I(className="fas fa-code creator-icon"),
                html.Div(children=[
                    html.H3("Create Your Own Metric", className="creator-title"),
                    html.P("Build custom evaluation metrics using our templates or write from scratch",
                           className="creator-subtitle")
                ])
            ])
        ]),

        # Template Selection Section
        html.Div(className="template-selection-section", children=[
            html.Label([
                html.I(className="fas fa-layer-group me-2"),
                "Choose a Template"
            ], className="section-label"),

            html.Div(className="template-cards-grid", children=[
                # Single Turn Template Card
                html.Div(
                    className="template-card",
                    id="template-single-turn",
                    n_clicks=0,
                    children=[
                        html.Div(className="template-card-header", children=[
                            html.I(className="fas fa-message template-icon"),
                            html.H4("Single Turn Judge", className="template-name")
                        ]),
                        html.P("LLM-as-a-judge metric for evaluating single turn interactions",
                               className="template-description"),
                        html.Div(className="template-tags", children=[
                            html.Span("LLM Judge", className="template-tag"),
                            html.Span("Single Turn", className="template-tag")
                        ])
                    ]
                ),

                # Multi Turn Template Card
                html.Div(
                    className="template-card",
                    id="template-multi-turn",
                    n_clicks=0,
                    children=[
                        html.Div(className="template-card-header", children=[
                            html.I(className="fas fa-comments template-icon"),
                            html.H4("Multi Turn Judge", className="template-name")
                        ]),
                        html.P("LLM-as-a-judge metric for multi-turn conversations",
                               className="template-description"),
                        html.Div(className="template-tags", children=[
                            html.Span("LLM Judge", className="template-tag"),
                            html.Span("Multi Turn", className="template-tag")
                        ])
                    ]
                ),

                # Heuristic Template Card
                html.Div(
                    className="template-card",
                    id="template-heuristic",
                    n_clicks=0,
                    children=[
                        html.Div(className="template-card-header", children=[
                            html.I(className="fas fa-calculator template-icon"),
                            html.H4("Heuristic Metric", className="template-name")
                        ]),
                        html.P("Rule-based algorithmic metric without LLM",
                               className="template-description"),
                        html.Div(className="template-tags", children=[
                            html.Span("Heuristic", className="template-tag"),
                            html.Span("No LLM", className="template-tag")
                        ])
                    ]
                ),

                # YAML Template Card
                html.Div(
                    className="template-card",
                    id="template-yaml",
                    n_clicks=0,
                    children=[
                        html.Div(className="template-card-header", children=[
                            html.I(className="fas fa-file-code template-icon"),
                            html.H4("YAML Config", className="template-name")
                        ]),
                        html.P("Configuration-based metric with YAML format",
                               className="template-description"),
                        html.Div(className="template-tags", children=[
                            html.Span("YAML", className="template-tag"),
                            html.Span("Config", className="template-tag")
                        ])
                    ]
                )
            ])
        ]),

        # Code Editor Section
        html.Div(className="code-editor-section", children=[
            html.Div(className="editor-header", children=[
                html.Label([
                    html.I(className="fas fa-code me-2"),
                    "Metric Code"
                ], className="section-label"),

                html.Div(className="editor-actions", children=[
                    dbc.Button([
                        html.I(className="fas fa-copy me-2"),
                        "Copy"
                    ], id="copy-metric-code", size="sm", className="btn-editor-action"),

                    dbc.Button([
                        html.I(className="fas fa-download me-2"),
                        "Download"
                    ], id="download-metric-code", size="sm", className="btn-editor-action"),

                    dbc.Button([
                        html.I(className="fas fa-sync-alt me-2"),
                        "Reset"
                    ], id="reset-metric-code", size="sm", className="btn-editor-action")
                ])
            ]),

            # Selected Template Indicator
            html.Div(id="selected-template-indicator", className="selected-template-indicator", children=[
                html.I(className="fas fa-info-circle me-2"),
                html.Span("Select a template to get started", id="template-indicator-text")
            ]),

            # Code Editor Textarea
            dcc.Textarea(
                id="custom-metric-code-editor",
                className="metric-code-editor",
                placeholder="Select a template above or write your custom metric code here...",
                value="# Select a template to get started",
                spellCheck=False,
                draggable=False
            ),

            # Help Text
            html.Div(className="editor-help", children=[
                html.Div(className="help-item", children=[
                    html.I(className="fas fa-lightbulb help-icon"),
                    html.Span("Tip: Modify the TODO sections in the template to create your custom metric")
                ]),
                html.Div(className="help-item", children=[
                    html.I(className="fas fa-book help-icon"),
                    html.A("View Documentation",
                           href="https://docs.example.com/custom-metrics",
                           target="_blank",
                           className="help-link")
                ])
            ])
        ]),

        # Save Section
        html.Div(className="save-section", children=[
            html.Div(className="save-form", children=[
                html.Label([
                    html.I(className="fas fa-tag me-2"),
                    "Metric Name"
                ], className="save-label"),
                dcc.Input(
                    id="custom-metric-name",
                    type="text",
                    placeholder="e.g., MyCustomMetric",
                    className="save-input"
                ),
            ]),

            html.Div(className="save-actions", children=[
                dbc.Button([
                    html.I(className="fas fa-save me-2"),
                    "Save Metric"
                ], id="save-custom-metric", className="btn-save-metric", size="lg"),

                html.Div(id="save-status", className="save-status")
            ])
        ]),

        # Store for selected template
        dcc.Store(id='selected-template-store', data=''),
        dcc.Download(id="download-metric-file")
    ])


def register_custom_metric_callbacks(app):
    """Register callbacks for custom metric creator"""

    # Template selection callbacks
    @app.callback(
        [Output('template-single-turn', 'className'),
         Output('template-multi-turn', 'className'),
         Output('template-heuristic', 'className'),
         Output('template-yaml', 'className'),
         Output('selected-template-store', 'data'),
         Output('custom-metric-code-editor', 'value'),
         Output('template-indicator-text', 'children')],
        [Input('template-single-turn', 'n_clicks'),
         Input('template-multi-turn', 'n_clicks'),
         Input('template-heuristic', 'n_clicks'),
         Input('template-yaml', 'n_clicks')],
        prevent_initial_call=True
    )
    def select_template(single_clicks, multi_clicks, heuristic_clicks, yaml_clicks):
        ctx = callback_context
        if not ctx.triggered:
            return ["template-card"] * 4, '', '# Select a template to get started', "Select a template to get started"

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Import template functions
        from components.metric_templates import (
            single_turn_metric_template,
            multi_turn_metric_template,
            heuristic_metric_template,
            yaml_metric_template
        )

        # Define template mapping
        templates = {
            'template-single-turn': {
                'func': single_turn_metric_template,
                'name': 'Single Turn Judge',
                'classes': ["template-card active", "template-card", "template-card", "template-card"]
            },
            'template-multi-turn': {
                'func': multi_turn_metric_template,
                'name': 'Multi Turn Judge',
                'classes': ["template-card", "template-card active", "template-card", "template-card"]
            },
            'template-heuristic': {
                'func': heuristic_metric_template,
                'name': 'Heuristic Metric',
                'classes': ["template-card", "template-card", "template-card active", "template-card"]
            },
            'template-yaml': {
                'func': yaml_metric_template,
                'name': 'YAML Config',
                'classes': ["template-card", "template-card", "template-card", "template-card active"]
            }
        }

        selected = templates.get(triggered_id)
        if selected:
            code = selected['func']('', None)
            indicator_text = f"Selected: {selected['name']}"
            return selected['classes'] + [triggered_id, code, indicator_text]

        return ["template-card"] * 4, '', '# Select a template to get started', "Select a template to get started"

    # Copy to clipboard
    @app.callback(
        Output('save-status', 'children', allow_duplicate=True),
        Input('copy-metric-code', 'n_clicks'),
        State('custom-metric-code-editor', 'value'),
        prevent_initial_call=True
    )
    def copy_code(n_clicks, code):
        if not n_clicks or not code:
            return ""

        # Note: Actual clipboard copy requires JavaScript, this shows a message
        return html.Div(className="status-message success", children=[
            html.I(className="fas fa-check-circle me-2"),
            "Code copied to clipboard! (Use Ctrl+C or Cmd+C)"
        ])

    # Download metric code
    @app.callback(
        Output('download-metric-file', 'data'),
        Input('download-metric-code', 'n_clicks'),
        [State('custom-metric-code-editor', 'value'),
         State('custom-metric-name', 'value'),
         State('selected-template-store', 'data')],
        prevent_initial_call=True
    )
    def download_code(n_clicks, code, metric_name, template_type):
        if not n_clicks or not code:
            return None

        # Determine file extension
        if 'yaml' in template_type.lower():
            filename = f"{metric_name or 'custom_metric'}.yaml"
        else:
            filename = f"{metric_name or 'custom_metric'}.py"

        return dict(content=code, filename=filename)

    # Reset code editor
    @app.callback(
        [Output('custom-metric-code-editor', 'value', allow_duplicate=True),
         Output('save-status', 'children', allow_duplicate=True)],
        Input('reset-metric-code', 'n_clicks'),
        State('selected-template-store', 'data'),
        prevent_initial_call=True
    )
    def reset_code(n_clicks, template_type):
        if not n_clicks:
            return dash.no_update, dash.no_update

        from components.metric_templates import (
            single_turn_metric_template,
            multi_turn_metric_template,
            heuristic_metric_template,
            yaml_metric_template
        )

        templates = {
            'template-single-turn': single_turn_metric_template,
            'template-multi-turn': multi_turn_metric_template,
            'template-heuristic': heuristic_metric_template,
            'template-yaml': yaml_metric_template
        }

        template_func = templates.get(template_type)
        if template_func:
            code = template_func('', None)
            status = html.Div(className="status-message info", children=[
                html.I(className="fas fa-info-circle me-2"),
                "Code reset to template"
            ])
            return code, status

        return '# Select a template to get started', ""

    # Save custom metric
    @app.callback(
        Output('save-status', 'children', allow_duplicate=True),
        Input('save-custom-metric', 'n_clicks'),
        [State('custom-metric-code-editor', 'value'),
         State('custom-metric-name', 'value')],
        prevent_initial_call=True
    )
    def save_metric(n_clicks, code, metric_name):
        if not n_clicks:
            return ""

        if not metric_name or not metric_name.strip():
            return html.Div(className="status-message error", children=[
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Please enter a metric name"
            ])

        if not code or code.strip() == '# Select a template to get started':
            return html.Div(className="status-message error", children=[
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Please write or select a metric template"
            ])

        return html.Div(className="status-message success", children=[
            html.I(className="fas fa-check-circle me-2"),
            f"Metric '{metric_name}' saved successfully!"
        ])
````

CSS Files. 
```
/* =====================================================
   Interactive Metric Selector - Gamified Experience
   Professional C-Suite Design with OMEGA Palette
   ===================================================== */

/* Main Container */
.metric-selector-page {
  min-height: 100vh;
  background: linear-gradient(135deg,
    rgba(20, 25, 30, 0.98) 0%,
    rgba(30, 35, 45, 0.95) 100%);
  position: relative;
  overflow: hidden;
}

/* Animated Background */
.animated-bg {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
  z-index: 1;
}

.floating-shape {
  position: absolute;
  border-radius: 50%;
  background: linear-gradient(135deg,
    rgba(139, 159, 79, 0.1) 0%,
    rgba(164, 184, 108, 0.05) 100%);
  animation: float 20s infinite ease-in-out;
}

.shape-1 {
  width: 300px;
  height: 300px;
  top: 10%;
  left: 10%;
  animation-delay: 0s;
}

.shape-2 {
  width: 200px;
  height: 200px;
  top: 60%;
  right: 15%;
  animation-delay: 5s;
}

.shape-3 {
  width: 150px;
  height: 150px;
  bottom: 20%;
  left: 30%;
  animation-delay: 10s;
}

@keyframes float {
  0%, 100% { transform: translate(0, 0) rotate(0deg); }
  25% { transform: translate(30px, -30px) rotate(90deg); }
  50% { transform: translate(-20px, 20px) rotate(180deg); }
  75% { transform: translate(-30px, -10px) rotate(270deg); }
}

.pulse-ring {
  position: absolute;
  border: 2px solid rgba(139, 159, 79, 0.3);
  border-radius: 50%;
  animation: pulse-expand 3s infinite;
}

.ring-1 {
  width: 400px;
  height: 400px;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}

.ring-2 {
  width: 600px;
  height: 600px;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  animation-delay: 1.5s;
}

@keyframes pulse-expand {
  0% {
    transform: translate(-50%, -50%) scale(1);
    opacity: 0.6;
  }
  100% {
    transform: translate(-50%, -50%) scale(1.5);
    opacity: 0;
  }
}

.metric-selector-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 2rem 1rem 4rem 1rem;
  position: relative;
  z-index: 2;
}

/* Gamified Header */
.gamified-header {
  background: linear-gradient(135deg,
    rgba(139, 159, 79, 0.15) 0%,
    rgba(164, 184, 108, 0.1) 100%);
  border: 1px solid rgba(139, 159, 79, 0.3);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3),
              inset 0 0 20px rgba(139, 159, 79, 0.1);
}

.animated-icon {
  animation: rotate-slow 10s linear infinite;
}

@keyframes rotate-slow {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.glowing-text {
  text-shadow: 0 0 20px rgba(139, 159, 79, 0.5);
}

/* Experience Badge */
.experience-badge {
  position: absolute;
  right: 2rem;
  top: 50%;
  transform: translateY(-50%);
  background: rgba(0, 0, 0, 0.5);
  border: 2px solid rgba(139, 159, 79, 0.4);
  border-radius: 12px;
  padding: 1rem 1.5rem;
  min-width: 150px;
}

.xp-level {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #8B9F4F;
  font-weight: 700;
  margin-bottom: 0.5rem;
}

.xp-level i {
  color: #F39C12;
  animation: pulse 2s infinite;
}

.level-number {
  font-size: 1.5rem;
  background: linear-gradient(135deg, #8B9F4F, #A4B86C);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.xp-bar {
  width: 100%;
  height: 8px;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 4px;
  overflow: hidden;
}

.xp-progress {
  height: 100%;
  background: linear-gradient(90deg, #8B9F4F, #A4B86C);
  transition: width 0.5s ease;
  box-shadow: 0 0 10px rgba(139, 159, 79, 0.5);
}

/* Enhanced Progress Bar */
.selector-progress-bar.enhanced {
  background: linear-gradient(135deg,
    rgba(0, 0, 0, 0.6) 0%,
    rgba(20, 25, 30, 0.4) 100%);
  border: 1px solid rgba(139, 159, 79, 0.2);
  box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4),
              inset 0 0 30px rgba(139, 159, 79, 0.05);
}

/* Achievement Row */
.achievement-row {
  display: flex;
  justify-content: space-around;
  margin-bottom: 2rem;
  padding-bottom: 2rem;
  border-bottom: 1px solid rgba(139, 159, 79, 0.1);
}

.achievement-badge {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(127, 140, 141, 0.3);
  border-radius: 20px;
  color: #7F8C8D;
  font-size: 0.9rem;
  transition: all 0.3s ease;
  opacity: 0.5;
}

.achievement-badge.unlocked {
  opacity: 1;
  background: linear-gradient(135deg,
    rgba(139, 159, 79, 0.2) 0%,
    rgba(164, 184, 108, 0.15) 100%);
  border-color: rgba(139, 159, 79, 0.5);
  color: #A4B86C;
  box-shadow: 0 0 20px rgba(139, 159, 79, 0.3);
  animation: achievement-pop 0.5s ease;
}

.achievement-badge.unlocked.gold {
  background: linear-gradient(135deg,
    rgba(241, 196, 15, 0.2) 0%,
    rgba(243, 156, 18, 0.15) 100%);
  border-color: rgba(241, 196, 15, 0.5);
  color: #F1C40F;
  box-shadow: 0 0 30px rgba(241, 196, 15, 0.4);
}

@keyframes achievement-pop {
  0% { transform: scale(0.8); }
  50% { transform: scale(1.2); }
  100% { transform: scale(1); }
}

/* Progress Steps Enhanced */
.progress-step .step-icon {
  font-size: 1.2rem;
}

.progress-step.active .step-number {
  animation: pulse-glow 2s ease-in-out infinite;
}

@keyframes pulse-glow {
  0%, 100% {
    box-shadow: 0 8px 24px rgba(139, 159, 79, 0.4);
    transform: scale(1.15);
  }
  50% {
    box-shadow: 0 8px 32px rgba(139, 159, 79, 0.6),
                0 0 40px rgba(139, 159, 79, 0.4);
    transform: scale(1.2);
  }
}

/* Quest Header */
.quest-header {
  text-align: center;
  margin-bottom: 3rem;
  padding: 2rem;
  background: linear-gradient(135deg,
    rgba(0, 0, 0, 0.5) 0%,
    rgba(139, 159, 79, 0.1) 100%);
  border-radius: 16px;
  border: 1px solid rgba(139, 159, 79, 0.2);
  position: relative;
  overflow: hidden;
}

.quest-header::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: linear-gradient(45deg,
    transparent,
    rgba(139, 159, 79, 0.1),
    transparent);
  animation: shimmer 3s infinite;
}

@keyframes shimmer {
  0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
  100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
}

.quest-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  background: linear-gradient(135deg, #8B9F4F, #A4B86C);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-weight: 700;
  margin-bottom: 1rem;
  box-shadow: 0 4px 15px rgba(139, 159, 79, 0.4);
}

.quest-badge.gold {
  background: linear-gradient(135deg, #F1C40F, #F39C12);
  box-shadow: 0 4px 20px rgba(241, 196, 15, 0.5);
  animation: golden-shine 2s ease-in-out infinite;
}

@keyframes golden-shine {
  0%, 100% { filter: brightness(1); }
  50% { filter: brightness(1.3); }
}

.points-counter {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  background: rgba(241, 196, 15, 0.1);
  border: 1px solid rgba(241, 196, 15, 0.3);
  border-radius: 20px;
  padding: 0.5rem 1rem;
  margin-top: 1rem;
  color: #F1C40F;
  font-weight: 600;
}

.points-value {
  font-size: 1.2rem;
  font-weight: 700;
  transition: all 0.3s ease;
}

/* Gamified Data Field Cards */
.data-field-grid.gamified {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1.5rem;
  margin-bottom: 3rem;
}

.data-field-card.gamified {
  background: linear-gradient(135deg,
    rgba(0, 0, 0, 0.6) 0%,
    rgba(20, 25, 30, 0.4) 100%);
  border: 2px solid rgba(139, 159, 79, 0.2);
  position: relative;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.card-glow {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 100%;
  height: 100%;
  background: radial-gradient(circle,
    rgba(139, 159, 79, 0.3) 0%,
    transparent 70%);
  transform: translate(-50%, -50%);
  opacity: 0;
  transition: opacity 0.3s ease;
  pointer-events: none;
}

.data-field-card.gamified:hover .card-glow {
  opacity: 1;
}

.data-field-card.gamified:hover {
  transform: translateY(-5px) scale(1.02);
  border-color: rgba(139, 159, 79, 0.5);
  box-shadow: 0 15px 40px rgba(139, 159, 79, 0.3);
}

.field-icon.hexagon {
  clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
  width: 54px;
  height: 54px;
}

.field-meta {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.field-points {
  color: #F1C40F;
  font-weight: 700;
  font-size: 0.8rem;
}

.data-field-card.gamified.selected {
  background: linear-gradient(135deg,
    rgba(139, 159, 79, 0.2) 0%,
    rgba(164, 184, 108, 0.15) 100%);
  border-color: #8B9F4F;
  box-shadow: 0 10px 30px rgba(139, 159, 79, 0.4),
              inset 0 0 20px rgba(139, 159, 79, 0.1);
}

.field-check-indicator.animated {
  animation: check-bounce 0.5s ease when selected;
}

@keyframes check-bounce {
  0% { transform: scale(0); }
  50% { transform: scale(1.3); }
  100% { transform: scale(1); }
}

/* Gamified Goal Cards */
.goal-grid.gamified {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 2rem;
  margin-bottom: 3rem;
}

.goal-card.gamified {
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  perspective: 1000px;
}

.goal-card.gamified .goal-card-inner {
  background: linear-gradient(135deg,
    rgba(0, 0, 0, 0.6) 0%,
    rgba(20, 25, 30, 0.4) 100%);
  border: 2px solid rgba(139, 159, 79, 0.2);
  position: relative;
  overflow: hidden;
}

.goal-card.gamified:hover .goal-card-inner {
  transform: translateY(-8px) rotateY(5deg);
  box-shadow: 0 20px 50px rgba(139, 159, 79, 0.3);
  border-color: rgba(139, 159, 79, 0.4);
}

.difficulty-badge {
  position: absolute;
  top: 1rem;
  right: 1rem;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: white;
}

.goal-icon-large.rotating {
  animation: rotate-3d 15s linear infinite;
}

@keyframes rotate-3d {
  from { transform: rotateY(0deg) rotateX(0deg); }
  to { transform: rotateY(360deg) rotateX(360deg); }
}

.goal-points {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 1rem;
  color: #F1C40F;
  font-weight: 700;
}

.goal-check.animated {
  animation: check-spin 0.6s ease when visible;
}

@keyframes check-spin {
  0% { transform: scale(0) rotate(-180deg); }
  100% { transform: scale(1) rotate(0deg); }
}

.goal-card.gamified.selected .goal-card-inner {
  background: linear-gradient(135deg,
    rgba(139, 159, 79, 0.25) 0%,
    rgba(164, 184, 108, 0.2) 100%);
  border-color: #8B9F4F;
  box-shadow: 0 15px 40px rgba(139, 159, 79, 0.4),
              inset 0 0 30px rgba(139, 159, 79, 0.1);
}

/* Enhanced Tree Visualization */
.metric-tree-visualization.enhanced {
  background: linear-gradient(135deg,
    rgba(0, 0, 0, 0.7) 0%,
    rgba(20, 25, 30, 0.5) 100%);
  border: 2px solid rgba(139, 159, 79, 0.3);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5),
              inset 0 0 40px rgba(139, 159, 79, 0.05);
  position: relative;
  overflow: hidden;
}

.metric-tree-structure.cosmic {
  position: relative;
  min-height: 600px;
  padding: 3rem;
}

/* Tree Particles Animation */
.tree-particles {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
}

.tree-particles .particle {
  position: absolute;
  width: 4px;
  height: 4px;
  background: rgba(139, 159, 79, 0.6);
  border-radius: 50%;
  animation: particle-float 20s infinite linear;
}

.tree-particles .particle:nth-child(even) {
  background: rgba(241, 196, 15, 0.6);
  animation-duration: 25s;
}

.tree-particles .particle:nth-child(3n) {
  background: rgba(52, 152, 219, 0.6);
  animation-duration: 30s;
}

@keyframes particle-float {
  0% {
    transform: translateY(100vh) translateX(0);
    opacity: 0;
  }
  10% {
    opacity: 1;
  }
  90% {
    opacity: 1;
  }
  100% {
    transform: translateY(-10vh) translateX(100px);
    opacity: 0;
  }
}

/* Enhanced Root Node */
.tree-root-node.animated {
  animation: fadeInDown 0.8s ease;
}

@keyframes fadeInDown {
  from {
    opacity: 0;
    transform: translateY(-30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.root-node-content.glowing {
  background: linear-gradient(135deg, #8B9F4F, #A4B86C);
  box-shadow: 0 10px 40px rgba(139, 159, 79, 0.5),
              0 0 60px rgba(139, 159, 79, 0.3);
  animation: glow-pulse 3s ease-in-out infinite;
}

@keyframes glow-pulse {
  0%, 100% {
    box-shadow: 0 10px 40px rgba(139, 159, 79, 0.5),
                0 0 60px rgba(139, 159, 79, 0.3);
  }
  50% {
    box-shadow: 0 10px 50px rgba(139, 159, 79, 0.7),
                0 0 80px rgba(139, 159, 79, 0.5);
  }
}

.root-node-content i.rotating {
  animation: rotate 8s linear infinite;
}

@keyframes rotate {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Tree Connections SVG */
.tree-connections {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

/* Enhanced Branches */
.tree-branches.animated {
  animation: fadeInUp 1s ease 0.5s backwards;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.tree-branch.animated {
  animation: slideInLeft 0.8s ease backwards;
}

.tree-branch.delay-0 {
  animation-delay: 0.6s;
}

.tree-branch.delay-1 {
  animation-delay: 0.8s;
}

.tree-branch.delay-2 {
  animation-delay: 1s;
}

@keyframes slideInLeft {
  from {
    opacity: 0;
    transform: translateX(-50px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

.branch-connector.glowing {
  background: linear-gradient(180deg,
    rgba(139, 159, 79, 0.8) 0%,
    rgba(139, 159, 79, 0.2) 100%);
  box-shadow: 0 0 20px rgba(139, 159, 79, 0.5);
  animation: pulse-line 2s ease-in-out infinite;
}

@keyframes pulse-line {
  0%, 100% {
    opacity: 0.6;
    transform: scaleY(1);
  }
  50% {
    opacity: 1;
    transform: scaleY(1.1);
  }
}

.branch-node.floating {
  animation: float-vertical 3s ease-in-out infinite;
}

@keyframes float-vertical {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-5px); }
}

.branch-node i.pulse {
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.1); }
}

.branch-count.badge {
  background: rgba(255, 255, 255, 0.2);
  backdrop-filter: blur(10px);
}

/* Enhanced Category Groups */
.category-group.animated {
  animation: fadeIn 0.6s ease backwards;
}

.category-connector.animated {
  animation: extend-line 0.8s ease backwards;
}

@keyframes extend-line {
  from {
    width: 0;
    opacity: 0;
  }
  to {
    width: 1rem;
    opacity: 1;
  }
}

.category-label.glass {
  background: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(139, 159, 79, 0.3);
}

/* Enhanced Metric Nodes */
.tree-metric-node.animated {
  animation: slideIn 0.5s ease backwards;
}

.metric-node-connector.glowing {
  position: relative;
}

.metric-node-connector.glowing::after {
  content: '';
  position: absolute;
  left: 0;
  top: -2px;
  width: 100%;
  height: 5px;
  background: linear-gradient(90deg,
    transparent,
    rgba(139, 159, 79, 0.8),
    transparent);
  animation: slide-glow 3s linear infinite;
}

@keyframes slide-glow {
  from { transform: translateX(-100%); }
  to { transform: translateX(100%); }
}

.metric-node-content.glass {
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(139, 159, 79, 0.3);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.metric-node-content:hover {
  background: rgba(139, 159, 79, 0.15);
  border-color: rgba(139, 159, 79, 0.6);
  transform: translateX(5px) scale(1.02);
  box-shadow: 0 10px 30px rgba(139, 159, 79, 0.3);
}

.metric-node-toggle.fancy .form-check-input {
  width: 3rem;
  height: 1.5rem;
  background: rgba(0, 0, 0, 0.5);
  border: 2px solid rgba(139, 159, 79, 0.3);
  position: relative;
}

.metric-node-toggle.fancy .form-check-input:checked {
  background: linear-gradient(90deg, #8B9F4F, #A4B86C);
  border-color: #8B9F4F;
  box-shadow: 0 0 15px rgba(139, 159, 79, 0.5);
}

.metric-tag.glowing {
  box-shadow: 0 0 10px rgba(139, 159, 79, 0.3);
  animation: tag-pulse 3s ease-in-out infinite;
}

@keyframes tag-pulse {
  0%, 100% { opacity: 0.8; }
  50% { opacity: 1; }
}

/* Summary Cards Enhanced */
.recommendation-summary-cards.animated {
  animation: fadeIn 0.8s ease 0.3s backwards;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.summary-card.floating {
  animation: float-subtle 4s ease-in-out infinite;
}

.summary-card.delay-1 {
  animation-delay: 1.3s;
}

.summary-card.delay-2 {
  animation-delay: 2.6s;
}

@keyframes float-subtle {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-3px); }
}

.summary-card-icon.pulse {
  animation: icon-pulse 2s ease-in-out infinite;
}

@keyframes icon-pulse {
  0%, 100% {
    transform: scale(1);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
  }
  50% {
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
  }
}

.summary-number.counting {
  animation: count-up 1s ease;
}

@keyframes count-up {
  from {
    opacity: 0;
    transform: scale(0.5);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

/* Step 4 Celebration */
.celebration-container {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
  overflow: hidden;
}

.confetti {
  position: absolute;
  width: 10px;
  height: 10px;
  background: linear-gradient(135deg, #8B9F4F, #A4B86C);
  animation: confetti-fall 3s linear;
}

.confetti:nth-child(even) {
  background: linear-gradient(135deg, #F39C12, #F1C40F);
  animation-duration: 2.5s;
}

.confetti:nth-child(3n) {
  background: linear-gradient(135deg, #3498DB, #2980B9);
  animation-duration: 3.5s;
}

@keyframes confetti-fall {
  0% {
    transform: translateY(-100vh) rotate(0deg);
    opacity: 1;
  }
  100% {
    transform: translateY(100vh) rotate(720deg);
    opacity: 0;
  }
}

.quest-header.celebration {
  animation: celebration-glow 2s ease-in-out;
}

@keyframes celebration-glow {
  0%, 100% {
    box-shadow: 0 0 30px rgba(241, 196, 15, 0.3);
  }
  50% {
    box-shadow: 0 0 60px rgba(241, 196, 15, 0.6);
  }
}

.step-title.golden {
  background: linear-gradient(135deg, #F1C40F, #F39C12);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: golden-shimmer 3s ease-in-out infinite;
}

@keyframes golden-shimmer {
  0%, 100% { filter: brightness(1); }
  50% { filter: brightness(1.5); }
}

/* Animated Buttons */
.glow-button {
  position: relative;
  overflow: hidden;
  background: linear-gradient(135deg, #8B9F4F, #A4B86C);
  box-shadow: 0 6px 25px rgba(139, 159, 79, 0.4);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.glow-button::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  width: 0;
  height: 0;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.3);
  transform: translate(-50%, -50%);
  transition: width 0.6s, height 0.6s;
}

.glow-button:hover {
  transform: translateY(-2px) scale(1.02);
  box-shadow: 0 8px 35px rgba(139, 159, 79, 0.5),
              0 0 25px rgba(139, 159, 79, 0.3);
}

.glow-button:hover::before {
  width: 300px;
  height: 300px;
}

.pulse-icon {
  animation: pulse 2s ease-in-out infinite;
}

/* Glass Effect */
.glass {
  background: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Review Cards Enhanced */
.review-card.glass {
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(15px);
  border: 2px solid rgba(139, 159, 79, 0.3);
}

.fadeInLeft {
  animation: fadeInLeft 0.8s ease;
}

.fadeInRight {
  animation: fadeInRight 0.8s ease;
}

@keyframes fadeInLeft {
  from {
    opacity: 0;
    transform: translateX(-30px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes fadeInRight {
  from {
    opacity: 0;
    transform: translateX(30px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

.review-metric-item.animated:hover {
  animation: highlight 0.3s ease;
}

@keyframes highlight {
  0%, 100% { background: rgba(139, 159, 79, 0.1); }
  50% { background: rgba(139, 159, 79, 0.2); }
}

.pulse-slow {
  animation: pulse 3s ease-in-out infinite;
}

/* Responsive */
@media (max-width: 768px) {
  .experience-badge {
    position: static;
    margin-top: 1rem;
    transform: none;
  }

  .achievement-row {
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .achievement-badge {
    font-size: 0.8rem;
    padding: 0.3rem 0.6rem;
  }

  .tree-branches {
    flex-direction: column;
  }
}
```
/* =====================================================
   Metric Registry Tabs - Professional Design
   ===================================================== */

/* Tab Navigation */
.metric-tabs {
    border-bottom: 2px solid rgba(139, 159, 79, 0.15);
    margin-bottom: 0;
}

.metric-tabs .nav-link {
    border: none;
    border-bottom: 3px solid transparent;
    color: #7F8C8D;
    font-weight: 600;
    font-size: 1rem;
    padding: 1rem 1.5rem;
    transition: all 0.3s ease;
    background: transparent;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Add icons via CSS pseudo-elements */
.metric-tabs .nav-link[aria-controls="browse-tab"]::before {
    content: "\f002";  /* FontAwesome search icon */
    font-family: "Font Awesome 5 Free", "Font Awesome 6 Free";
    font-weight: 900;
    margin-right: 0.5rem;
    transition: transform 0.3s ease;
}

.metric-tabs .nav-link[aria-controls="create-tab"]::before {
    content: "\f121";  /* FontAwesome code icon */
    font-family: "Font Awesome 5 Free", "Font Awesome 6 Free";
    font-weight: 900;
    margin-right: 0.5rem;
    transition: transform 0.3s ease;
}

.metric-tabs .nav-link:hover {
    color: #8B9F4F;
    background: rgba(139, 159, 79, 0.05);
    border-bottom-color: rgba(139, 159, 79, 0.3);
}

.metric-tabs .nav-link:hover::before {
    transform: scale(1.1);
}

.metric-tabs .nav-link.active {
    color: #8B9F4F;
    background: linear-gradient(135deg, rgba(139, 159, 79, 0.08), rgba(164, 184, 108, 0.05));
    border-bottom-color: #8B9F4F;
    position: relative;
}

.metric-tabs .nav-link.active::before {
    color: #8B9F4F;
}

.metric-tabs .nav-link.active::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #8B9F4F, #A4B86C);
    animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
    from {
        transform: scaleX(0);
    }
    to {
        transform: scaleX(1);
    }
}

/* Tab Content Wrapper */
.tab-content-wrapper {
    padding: 2rem 0;
    animation: fadeIn 0.4s ease-out;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Modal Width Adjustment */
.metric-registry-modal .modal-xl {
    max-width: 95%;
}

@media (min-width: 1400px) {
    .metric-registry-modal .modal-xl {
        max-width: 1320px;
    }
}

/* Tab Icons - Remove from here, already handled above */
.metric-tabs .nav-link.active i {
    color: #8B9F4F;
}

/* Responsive Tabs */
@media (max-width: 768px) {
    .metric-tabs .nav-link {
        padding: 0.75rem 1rem;
        font-size: 0.9rem;
    }

    .tab-content-wrapper {
        padding: 1rem 0;
    }
}
````
````
```
/* =====================================================
   Metric Registry - Professional C-Suite Design
   Matching OMEGA Design System
   ===================================================== */

/* Metric Registry Button */
.metric-registry-section {
  margin-top: 24px;
  padding-top: 24px;
  border-top: 1px solid rgba(139, 159, 79, 0.1);
}

.btn-metric-registry {
  background: linear-gradient(135deg, rgba(139, 159, 79, 0.1) 0%, rgba(164, 184, 108, 0.05) 100%);
  border: 2px solid #8B9F4F;
  color: #6B7A3A;
  padding: 12px 24px;
  border-radius: 10px;
  font-weight: 600;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  align-items: center;
  gap: 8px;
  position: relative;
  overflow: hidden;
}

.btn-metric-registry::before {
  content: '';
  position: absolute;
  top: 50%;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(139, 159, 79, 0.2), transparent);
  transform: translateY(-50%);
  transition: left 0.6s ease;
}

.btn-metric-registry:hover {
  background: linear-gradient(135deg, #8B9F4F 0%, #A4B86C 100%);
  color: white;
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(139, 159, 79, 0.3);
  border-color: transparent;
}

.btn-metric-registry:hover::before {
  left: 100%;
}

.metric-count-badge {
  background: white;
  color: #8B9F4F;
  padding: 2px 8px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 700;
  margin-left: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Metric Registry Modal */
.metric-registry-modal .modal-content {
  border: none;
  border-radius: 20px;
  background: linear-gradient(135deg,
    rgba(255, 255, 255, 0.98) 0%,
    rgba(248, 252, 248, 0.95) 100%);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
  max-height: 90vh;
}

/* Modal Header */
.metric-registry-header {
  background: linear-gradient(135deg, #8B9F4F 0%, #A4B86C 100%);
  border-radius: 20px 20px 0 0;
  border: none;
  padding: 28px 32px;
  position: relative;
  overflow: hidden;
}

.metric-registry-header::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: radial-gradient(circle at 20% 50%,
    rgba(255, 255, 255, 0.1) 0%,
    transparent 50%);
  pointer-events: none;
}

.registry-header-content {
  display: flex;
  align-items: center;
  gap: 16px;
  position: relative;
  z-index: 1;
}

.registry-icon {
  font-size: 2rem;
  color: white;
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.2));
  animation: iconRotate 20s linear infinite;
}

@keyframes iconRotate {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.registry-title {
  font-size: 1.5rem;
  font-weight: 700;
  color: white;
  margin: 0;
  letter-spacing: -0.02em;
  flex: 1;
}

.registry-count {
  background: rgba(255, 255, 255, 0.2);
  color: white;
  padding: 6px 16px;
  border-radius: 30px;
  font-size: 0.9rem;
  font-weight: 600;
  backdrop-filter: blur(10px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

/* Modal Body */
.metric-registry-body {
  padding: 32px;
  max-height: calc(90vh - 200px);
  overflow-y: auto;
}

/* Registry Controls */
.registry-controls {
  display: grid;
  gap: 20px;
  margin-bottom: 32px;
  padding: 24px;
  background: rgba(139, 159, 79, 0.03);
  border-radius: 12px;
  border: 1px solid rgba(139, 159, 79, 0.1);
}

.registry-search-wrapper {
  position: relative;
}

.registry-search-wrapper .search-icon {
  position: absolute;
  left: 16px;
  top: 50%;
  transform: translateY(-50%);
  color: #7F8C8D;
  font-size: 1rem;
  pointer-events: none;
}

.metric-search-input {
  width: 100%;
  padding: 12px 16px 12px 44px;
  border: 2px solid rgba(139, 159, 79, 0.15);
  border-radius: 10px;
  font-size: 0.95rem;
  transition: all 0.3s ease;
  background: white;
}

.metric-search-input:focus {
  outline: none;
  border-color: #8B9F4F;
  box-shadow: 0 0 0 3px rgba(139, 159, 79, 0.1);
  transform: translateY(-1px);
}

.metric-search-input::placeholder {
  color: #95A5A6;
  font-style: italic;
}

/* Tag Filters */
.tag-filters {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.filter-label {
  font-size: 0.9rem;
  font-weight: 600;
  color: #2C3E50;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.tag-buttons {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  flex: 1;
}

.tag-filter-btn {
  padding: 6px 14px;
  border-radius: 20px;
  font-size: 0.85rem;
  font-weight: 600;
  transition: all 0.3s ease;
  text-transform: capitalize;
}

.tag-filter-btn.tag-primary {
  border-color: #8B9F4F;
  color: #8B9F4F;
}

.tag-filter-btn.tag-primary:hover,
.tag-filter-btn.tag-primary.active {
  background: #8B9F4F;
  color: white;
  border-color: #8B9F4F;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(139, 159, 79, 0.25);
}

.tag-filter-btn.tag-success {
  border-color: #27AE60;
  color: #27AE60;
}

.tag-filter-btn.tag-success:hover,
.tag-filter-btn.tag-success.active {
  background: #27AE60;
  color: white;
  border-color: #27AE60;
}

.tag-filter-btn.tag-info {
  border-color: #3498DB;
  color: #3498DB;
}

.tag-filter-btn.tag-info:hover,
.tag-filter-btn.tag-info.active {
  background: #3498DB;
  color: white;
  border-color: #3498DB;
}

.tag-filter-btn.tag-warning {
  border-color: #F39C12;
  color: #F39C12;
}

.tag-filter-btn.tag-warning:hover,
.tag-filter-btn.tag-warning.active {
  background: #F39C12;
  color: white;
  border-color: #F39C12;
}

.tag-filter-btn.tag-danger {
  border-color: #E74C3C;
  color: #E74C3C;
}

.tag-filter-btn.tag-danger:hover,
.tag-filter-btn.tag-danger.active {
  background: #E74C3C;
  color: white;
  border-color: #E74C3C;
}

.tag-filter-btn.tag-secondary {
  border-color: #7F8C8D;
  color: #7F8C8D;
}

.tag-filter-btn.tag-secondary:hover,
.tag-filter-btn.tag-secondary.active {
  background: #7F8C8D;
  color: white;
  border-color: #7F8C8D;
}

/* View Options */
.view-options {
  display: flex;
  justify-content: flex-end;
}

.view-btn {
  background: white;
  border: 1px solid rgba(139, 159, 79, 0.2);
  color: #7F8C8D;
  padding: 8px 12px;
  transition: all 0.2s ease;
}

.view-btn:hover {
  background: rgba(139, 159, 79, 0.05);
  color: #8B9F4F;
}

.view-btn.active {
  background: #8B9F4F;
  color: white;
  border-color: #8B9F4F;
}

/* Metrics Grid */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 20px;
  margin-bottom: 32px;
  max-height: 500px;
  overflow-y: auto;
  padding-right: 8px;
}

.metrics-grid::-webkit-scrollbar {
  width: 8px;
}

.metrics-grid::-webkit-scrollbar-track {
  background: rgba(139, 159, 79, 0.05);
  border-radius: 4px;
}

.metrics-grid::-webkit-scrollbar-thumb {
  background: linear-gradient(#8B9F4F, #A4B86C);
  border-radius: 4px;
}

/* Metric Card */
.metric-card {
  background: white;
  border: 1px solid rgba(139, 159, 79, 0.12);
  border-radius: 12px;
  padding: 20px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
  cursor: pointer;
}

.metric-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 3px;
  background: linear-gradient(90deg, #8B9F4F, #A4B86C, #B8C78A);
  transform: scaleX(0);
  transition: transform 0.3s ease;
}

.metric-card:hover::before {
  transform: scaleX(1);
}

.metric-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 32px rgba(139, 159, 79, 0.15);
  border-color: rgba(139, 159, 79, 0.25);
}

.metric-card.selected {
  background: linear-gradient(135deg,
    rgba(139, 159, 79, 0.05) 0%,
    rgba(164, 184, 108, 0.03) 100%);
  border-color: #8B9F4F;
  box-shadow: 0 8px 24px rgba(139, 159, 79, 0.2);
}

/* Metric Card Header */
.metric-card-header {
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(139, 159, 79, 0.08);
}

.metric-name-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.metric-name {
  font-size: 1.1rem;
  font-weight: 600;
  color: #2C3E50;
  margin: 0;
  flex: 1;
}

.metric-checkbox {
  cursor: pointer;
  transform: scale(1.2);
}

.metric-checkbox .form-check-input {
  background-color: white;
  border: 2px solid #8B9F4F;
  cursor: pointer;
}

.metric-checkbox .form-check-input:checked {
  background-color: #8B9F4F;
  border-color: #6B7A3A;
  box-shadow: 0 0 8px rgba(139, 159, 79, 0.3);
}

.metric-tags {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.metric-tag {
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  opacity: 0.9;
}

.tag-knowledge {
  background: linear-gradient(135deg, #8B9F4F 0%, #A4B86C 100%);
  color: white;
}

.tag-agent {
  background: linear-gradient(135deg, #27AE60 0%, #229954 100%);
  color: white;
}

.tag-single_turn {
  background: linear-gradient(135deg, #3498DB 0%, #2E86AB 100%);
  color: white;
}

.tag-multi_turn {
  background: linear-gradient(135deg, #F39C12 0%, #D68910 100%);
  color: white;
}

.tag-data {
  background: linear-gradient(135deg, #7F8C8D 0%, #5D6D7E 100%);
  color: white;
}

.tag-safety {
  background: linear-gradient(135deg, #E74C3C 0%, #C0392B 100%);
  color: white;
}

/* Metric Card Body */
.metric-card-body {
  margin-bottom: 16px;
}

.metric-description {
  font-size: 0.9rem;
  color: #34495E;
  line-height: 1.5;
  margin-bottom: 12px;
}

.metric-specs {
  display: flex;
  gap: 20px;
  padding: 12px;
  background: rgba(139, 159, 79, 0.03);
  border-radius: 8px;
  border: 1px solid rgba(139, 159, 79, 0.08);
}

.spec-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.spec-icon {
  font-size: 0.9rem;
  color: #8B9F4F;
}

.spec-text {
  font-size: 0.85rem;
  color: #2C3E50;
  font-weight: 500;
}

/* Metric Card Footer */
.metric-card-footer {
  padding-top: 12px;
  border-top: 1px solid rgba(139, 159, 79, 0.08);
}

.fields-section {
  margin-bottom: 8px;
}

.fields-section:last-child {
  margin-bottom: 0;
}

.fields-label {
  font-size: 0.8rem;
  font-weight: 600;
  color: #7F8C8D;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-right: 8px;
}

.fields-list {
  display: inline-flex;
  gap: 6px;
  flex-wrap: wrap;
}

.field-chip {
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 0.75rem;
  font-weight: 500;
  font-family: 'Monaco', monospace;
}

.field-chip.required {
  background: rgba(231, 76, 60, 0.1);
  color: #E74C3C;
  border: 1px solid rgba(231, 76, 60, 0.2);
}

.field-chip.optional {
  background: rgba(52, 152, 219, 0.1);
  color: #3498DB;
  border: 1px solid rgba(52, 152, 219, 0.2);
}

.field-chip.none {
  background: rgba(127, 140, 141, 0.1);
  color: #7F8C8D;
  border: 1px solid rgba(127, 140, 141, 0.2);
  font-style: italic;
}

/* Selected Metrics Summary */
.selected-metrics-summary {
  background: linear-gradient(135deg,
    rgba(139, 159, 79, 0.05) 0%,
    rgba(164, 184, 108, 0.03) 100%);
  border: 1px solid rgba(139, 159, 79, 0.15);
  border-radius: 12px;
  padding: 20px;
  margin-top: 24px;
}

.summary-header {
  display: flex;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(139, 159, 79, 0.1);
}

.summary-header i {
  color: #27AE60;
  font-size: 1.1rem;
}

.summary-title {
  font-size: 1rem;
  font-weight: 600;
  color: #2C3E50;
  flex: 1;
}

.selected-badge {
  background: #8B9F4F;
  color: white;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 0.85rem;
  font-weight: 700;
  box-shadow: 0 2px 6px rgba(139, 159, 79, 0.25);
}

.selected-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 8px;
  max-height: 120px;
  overflow-y: auto;
  padding-right: 8px;
}

.selected-metric-item {
  padding: 8px 12px;
  background: white;
  border: 1px solid rgba(139, 159, 79, 0.1);
  border-radius: 8px;
  display: flex;
  align-items: center;
  font-size: 0.85rem;
  color: #2C3E50;
  transition: all 0.2s ease;
}

.selected-metric-item:hover {
  background: rgba(139, 159, 79, 0.05);
  transform: translateX(2px);
}

.selected-metric-item i {
  color: #27AE60;
  font-size: 0.8rem;
}

/* Modal Footer */
.metric-registry-footer {
  background: rgba(248, 252, 248, 0.5);
  border-top: 1px solid rgba(139, 159, 79, 0.1);
  padding: 20px 32px;
  border-radius: 0 0 20px 20px;
}

.footer-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.action-info {
  display: flex;
  align-items: center;
  color: #7F8C8D;
  font-size: 0.9rem;
  font-style: italic;
}

.action-info i {
  color: #3498DB;
  font-size: 1rem;
}

.action-buttons {
  display: flex;
  gap: 12px;
}

.btn-secondary-modal {
  background: white;
  border: 2px solid rgba(139, 159, 79, 0.2);
  color: #7F8C8D;
  padding: 10px 24px;
  border-radius: 8px;
  font-weight: 600;
  transition: all 0.3s ease;
}

.btn-secondary-modal:hover {
  background: rgba(139, 159, 79, 0.05);
  border-color: rgba(139, 159, 79, 0.3);
  color: #6B7A3A;
}

.btn-primary-modal {
  background: linear-gradient(135deg, #8B9F4F 0%, #A4B86C 100%);
  border: none;
  color: white;
  padding: 10px 28px;
  border-radius: 8px;
  font-weight: 600;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 4px 12px rgba(139, 159, 79, 0.25);
}

.btn-primary-modal:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(139, 159, 79, 0.35);
  background: linear-gradient(135deg, #6B7A3A 0%, #8B9F4F 100%);
}

/* List View Alternative */
.metrics-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  max-height: 500px;
  overflow-y: auto;
  padding-right: 8px;
}

.metrics-list .metric-card {
  display: flex;
  align-items: center;
  padding: 16px;
}

.metrics-list .metric-card-header {
  border: none;
  margin-bottom: 0;
  padding-bottom: 0;
  flex: 1;
  display: flex;
  align-items: center;
  gap: 20px;
}

.metrics-list .metric-name-row {
  margin-bottom: 0;
}

.metrics-list .metric-tags {
  margin-left: auto;
}

.metrics-list .metric-card-body,
.metrics-list .metric-card-footer {
  display: none;
}

/* Responsive Design */
@media (max-width: 1200px) {
  .metrics-grid {
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  }
}

@media (max-width: 768px) {
  .registry-controls {
    padding: 16px;
  }

  .metrics-grid {
    grid-template-columns: 1fr;
    max-height: 400px;
  }

  .selected-list {
    grid-template-columns: 1fr;
  }

  .footer-actions {
    flex-direction: column;
    gap: 16px;
    align-items: stretch;
  }

  .action-buttons {
    width: 100%;
    flex-direction: column;
  }

  .btn-secondary-modal,
  .btn-primary-modal {
    width: 100%;
    justify-content: center;
  }
}

/* Animation for metric selection */
@keyframes selectPulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.05); }
  100% { transform: scale(1); }
}

.metric-card.selecting {
  animation: selectPulse 0.3s ease;
}

/* Custom scrollbar for modal body */
.metric-registry-body::-webkit-scrollbar {
  width: 10px;
}

.metric-registry-body::-webkit-scrollbar-track {
  background: rgba(139, 159, 79, 0.05);
  border-radius: 5px;
}

.metric-registry-body::-webkit-scrollbar-thumb {
  background: linear-gradient(#8B9F4F, #A4B86C);
  border-radius: 5px;
}

.metric-registry-body::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(#6B7A3A, #8B9F4F);
}

/* Loading state */
.metrics-loading {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 300px;
  color: #8B9F4F;
  font-size: 2rem;
}

.metrics-loading i {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}














/* Metric Selection Section - Prominent Design */
.metric-selection-section {
  background: linear-gradient(135deg,
    rgba(139, 159, 79, 0.04) 0%,
    rgba(164, 184, 108, 0.02) 100%);
  border: 2px solid rgba(139, 159, 79, 0.2);
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 28px;
  position: relative;
  overflow: hidden;
}

.metric-selection-section::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 3px;
  background: linear-gradient(90deg, #8B9F4F, #A4B86C, #B8C78A);
  animation: shimmerMetric 4s ease-in-out infinite;
}

@keyframes shimmerMetric {
  0%, 100% { transform: translateX(-100%); }
  50% { transform: translateX(100%); }
}

.btn-browse-metrics {
  width: 100%;
  background: linear-gradient(135deg, #8B9F4F 0%, #A4B86C 100%);
  border: none;
  color: white;
  padding: 18px 32px;
  margin-top: 12px;
  border-radius: 12px;
  font-size: 1.1rem;
  font-weight: 600;
  letter-spacing: 0.3px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 6px 20px rgba(139, 159, 79, 0.3);
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

/* Button ripple effect on hover */
.btn-browse-metrics::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  width: 0;
  height: 0;
  background: rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  transform: translate(-50%, -50%);
  transition: width 0.6s, height 0.6s;
}

.btn-browse-metrics:hover {
  transform: translateY(-3px) scale(1.02);
  box-shadow: 0 10px 30px rgba(139, 159, 79, 0.4);
  background: linear-gradient(135deg, #6B7A3A 0%, #8B9F4F 100%);
}

.btn-browse-metrics:hover::before {
  width: 500px;
  height: 500px;
}

.btn-browse-metrics .btn-subtitle {
  font-size: 0.85rem;
  opacity: 0.9;
  font-weight: 400;
  font-style: italic;
}

/* Selected Metrics Preview */
.selected-metrics-preview {
  margin-top: 20px;
  padding: 16px;
  background: white;
  border: 1px solid rgba(139, 159, 79, 0.15);
  border-radius: 8px;
  min-height: 60px;
  display: none;
}

.selected-metrics-preview.has-metrics {
  display: block;
  animation: fadeInPreview 0.3s ease-out;
}

/* Preview animations */
@keyframes fadeInPreview {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Preview metric chips */
.preview-metric-chip {
  background: linear-gradient(135deg,
    rgba(139, 159, 79, 0.1) 0%,
    rgba(164, 184, 108, 0.08) 100%);
  border: 1px solid rgba(139, 159, 79, 0.2);
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 0.85rem;
  font-weight: 500;
  color: #6B7A3A;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: all 0.2s ease;
}

.preview-metric-chip:hover {
  background: rgba(139, 159, 79, 0.15);
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(139, 159, 79, 0.2);
}

/* Section Divider */
.section-divider {
  margin: 28px 0;
  border: none;
  border-top: 1px solid rgba(139, 159, 79, 0.1);
  position: relative;
}

.section-divider::after {
  content: 'Additional Configuration';
  position: absolute;
  top: -10px;
  left: 50%;
  transform: translateX(-50%);
  background: white;
  padding: 0 16px;
  color: #7F8C8D;
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

/* List View Alternative - Updated */
.metrics-list {
  display: flex;
  flex-direction: column;
  gap: 16px;  /* Increased spacing between items */
  max-height: 500px;
  overflow-y: auto;
  padding-right: 8px;
}

.metrics-list .metric-card {
  display: flex;
  align-items: center;
  padding: 18px 20px;  /* Increased padding */
}

.metrics-list .metric-card-header {
  border: none;
  margin-bottom: 0;
  padding-bottom: 0;
  flex: 1;
  display: flex;
  align-items: center;
  gap: 24px;  /* Increased gap between elements */
}

.metrics-list .metric-name-row {
  margin-bottom: 0;
  display: flex;
  align-items: center;
  gap: 16px;  /* Space between checkbox and name */
  flex: 1;
}

/* Checkbox on the left in list view */
.metrics-list .metric-checkbox {
  order: -1;  /* Ensures checkbox comes first */
  margin-right: 0;
}

/* Name takes up remaining space */
.metrics-list .metric-name {
  flex: 1;
  margin: 0;
}

.metrics-list .metric-tags {
  margin-left: auto;
  display: flex;
  gap: 8px;
}

.metrics-list .metric-card-body,
.metrics-list .metric-card-footer {
  display: none;
}
`
```
/* =====================================================
   Custom Metric Creator - Premium Design
   ===================================================== */

.custom-metric-creator {
    padding: 2rem;
    background: linear-gradient(135deg, #fdfeff 0%, #f5f8f5 100%);
    border-radius: 16px;
    min-height: 600px;
}

/* Creator Header */
.creator-header {
    background: linear-gradient(135deg, #8B9F4F 0%, #6B7A3A 100%);
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.creator-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 80%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    animation: floatHeader 20s infinite ease-in-out;
}

@keyframes floatHeader {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    50% { transform: translateY(-20px) rotate(180deg); }
}

.creator-header-content {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    position: relative;
    z-index: 1;
}

.creator-icon {
    width: 64px;
    height: 64px;
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    color: white;
    flex-shrink: 0;
}

.creator-title {
    color: white;
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}

.creator-subtitle {
    color: rgba(255, 255, 255, 0.9);
    font-size: 1rem;
    margin: 0.5rem 0 0 0;
}

/* Template Selection Section */
.template-selection-section {
    margin-bottom: 2rem;
}

.section-label {
    display: flex;
    align-items: center;
    font-size: 1.1rem;
    font-weight: 600;
    color: #2C3E50;
    margin-bottom: 1rem;
}

.template-cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.25rem;
}

.template-card {
    background: white;
    border: 2px solid rgba(139, 159, 79, 0.15);
    border-radius: 14px;
    padding: 1.5rem;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.template-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, #8B9F4F, #A4B86C);
    transform: scaleX(0);
    transition: transform 0.3s ease;
}

.template-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(139, 159, 79, 0.2);
    border-color: rgba(139, 159, 79, 0.4);
}

.template-card:hover::before {
    transform: scaleX(1);
}

.template-card.active {
    border-color: #8B9F4F;
    background: linear-gradient(135deg, rgba(139, 159, 79, 0.08), rgba(164, 184, 108, 0.05));
    box-shadow: 0 8px 24px rgba(139, 159, 79, 0.25);
}

.template-card.active::before {
    transform: scaleX(1);
}

.template-card-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.75rem;
}

.template-icon {
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, rgba(139, 159, 79, 0.1), rgba(107, 122, 58, 0.08));
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #8B9F4F;
    font-size: 1.25rem;
    flex-shrink: 0;
}

.template-card.active .template-icon {
    background: linear-gradient(135deg, #8B9F4F, #6B7A3A);
    color: white;
}

.template-name {
    font-size: 1.1rem;
    font-weight: 600;
    color: #2C3E50;
    margin: 0;
}

.template-description {
    font-size: 0.9rem;
    color: #7F8C8D;
    line-height: 1.5;
    margin-bottom: 0.75rem;
}

.template-tags {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.template-tag {
    background: rgba(139, 159, 79, 0.1);
    color: #6B7A3A;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 500;
    border: 1px solid rgba(139, 159, 79, 0.2);
}

.template-card.active .template-tag {
    background: rgba(139, 159, 79, 0.2);
    border-color: rgba(139, 159, 79, 0.4);
}

/* Code Editor Section */
.code-editor-section {
    background: white;
    border: 1px solid rgba(139, 159, 79, 0.15);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.editor-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid rgba(139, 159, 79, 0.1);
}

.editor-actions {
    display: flex;
    gap: 0.75rem;
}

.btn-editor-action {
    background: white;
    border: 1px solid rgba(139, 159, 79, 0.25);
    color: #6B7A3A;
    padding: 0.5rem 1rem;
    font-size: 0.875rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.btn-editor-action:hover {
    background: rgba(139, 159, 79, 0.1);
    border-color: #8B9F4F;
    color: #6B7A3A;
    transform: translateY(-2px);
}

.selected-template-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    background: linear-gradient(135deg, rgba(139, 159, 79, 0.08), rgba(164, 184, 108, 0.05));
    border-left: 3px solid #8B9F4F;
    border-radius: 8px;
    margin-bottom: 1rem;
    font-size: 0.9rem;
    color: #2C3E50;
    font-weight: 500;
}

.selected-template-indicator i {
    color: #8B9F4F;
}

/* Code Editor Textarea */
.metric-code-editor {
    width: 100%;
    min-height: 400px;
    padding: 1.25rem;
    border: 2px solid rgba(139, 159, 79, 0.15);
    border-radius: 10px;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Consolas', 'source-code-pro', monospace;
    font-size: 0.9rem;
    line-height: 1.6;
    background: #fafafa;
    color: #2C3E50;
    resize: vertical;
    transition: all 0.3s ease;
}

.metric-code-editor:focus {
    outline: none;
    border-color: #8B9F4F;
    background: white;
    box-shadow: 0 0 0 4px rgba(139, 159, 79, 0.08);
}

.metric-code-editor::placeholder {
    color: #95A5A6;
    font-style: italic;
}

/* Editor Help */
.editor-help {
    display: flex;
    gap: 2rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(139, 159, 79, 0.1);
}

.help-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    color: #7F8C8D;
}

.help-icon {
    color: #8B9F4F;
    font-size: 1rem;
}

.help-link {
    color: #8B9F4F;
    text-decoration: none;
    font-weight: 600;
    transition: all 0.2s ease;
}

.help-link:hover {
    color: #6B7A3A;
    text-decoration: underline;
}

/* Save Section */
.save-section {
    background: white;
    border: 1px solid rgba(139, 159, 79, 0.15);
    border-radius: 14px;
    padding: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 2rem;
}

.save-form {
    flex: 1;
    max-width: 400px;
}

.save-label {
    display: flex;
    align-items: center;
    font-size: 0.95rem;
    font-weight: 600;
    color: #2C3E50;
    margin-bottom: 0.5rem;
}

.save-input {
    width: 100%;
    padding: 0.875rem 1rem;
    border: 2px solid rgba(139, 159, 79, 0.15);
    border-radius: 10px;
    font-size: 0.95rem;
    transition: all 0.3s ease;
    background: #fafafa;
}

.save-input:focus {
    outline: none;
    border-color: #8B9F4F;
    background: white;
    box-shadow: 0 0 0 4px rgba(139, 159, 79, 0.08);
}

.save-actions {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.btn-save-metric {
    background: linear-gradient(135deg, #8B9F4F 0%, #6B7A3A 100%);
    border: none;
    color: white;
    padding: 0.875rem 2rem;
    border-radius: 12px;
    font-weight: 700;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 16px rgba(139, 159, 79, 0.3);
    position: relative;
    overflow: hidden;
}

.btn-save-metric::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.3);
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
}

.btn-save-metric:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(139, 159, 79, 0.4);
    background: linear-gradient(135deg, #6B7A3A 0%, #8B9F4F 100%);
}

.btn-save-metric:active::before {
    width: 300px;
    height: 300px;
}

/* Save Status Messages */
.save-status {
    min-height: 40px;
    display: flex;
    align-items: center;
}

.status-message {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.25rem;
    border-radius: 10px;
    font-size: 0.9rem;
    font-weight: 600;
    animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.status-message.success {
    background: rgba(39, 174, 96, 0.1);
    color: #27AE60;
    border: 1px solid rgba(39, 174, 96, 0.2);
}

.status-message.error {
    background: rgba(231, 76, 60, 0.1);
    color: #E74C3C;
    border: 1px solid rgba(231, 76, 60, 0.2);
}

.status-message.info {
    background: rgba(52, 152, 219, 0.1);
    color: #3498DB;
    border: 1px solid rgba(52, 152, 219, 0.2);
}

/* Responsive Design */
@media (max-width: 1024px) {
    .template-cards-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 768px) {
    .custom-metric-creator {
        padding: 1rem;
    }

    .creator-header {
        padding: 1.5rem;
    }

    .creator-title {
        font-size: 1.5rem;
    }

    .template-cards-grid {
        grid-template-columns: 1fr;
    }

    .save-section {
        flex-direction: column;
        align-items: stretch;
    }

    .save-form {
        max-width: 100%;
    }

    .save-actions {
        flex-direction: column;
        width: 100%;
    }

    .btn-save-metric {
        width: 100%;
        justify-content: center;
    }

    .editor-help {
        flex-direction: column;
        gap: 0.75rem;
    }

    .metric-code-editor {
        min-height: 300px;
        font-size: 0.85rem;
    }
}

/* Syntax Highlighting Hints */
.metric-code-editor {
    tab-size: 4;
    -moz-tab-size: 4;
}

/* Loading state for save button */
.btn-save-metric:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

/* Copy feedback animation */
@keyframes copyPulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.btn-editor-action.copied {
    animation: copyPulse 0.3s ease;
    background: rgba(39, 174, 96, 0.1);
    border-color: #27AE60;
    color: #27AE60;
}
`