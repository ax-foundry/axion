from typing import Any


def single_turn_metric_template(key: str, item: Any) -> str:
    """
    Generate a minimal working template for creating a custom metric.
    This provides the bare minimum code structure to get started.
    """
    class_name = 'SingleTurnJudge'

    return f'''from axion.metrics.base import BaseMetric, MetricEvaluationResult
from axion.dataset import DatasetItem

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

    return f'''from axion.metrics.base import BaseMetric, MetricEvaluationResult
from axion.dataset import DatasetItem
from axion.dataset_schema import MultiTurnConversation, HumanMessage, AIMessage, ToolCall, ToolMessage

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

    return f'''from axion.metrics.base import BaseMetric, MetricEvaluationResult
from axion.dataset import DatasetItem

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
# from axion.metrics.yaml_metrics import load_metric_from_yaml
# from axion.dataset import DatasetItem
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
