from typing import List, Literal

from pydantic import Field

from axion._core.logging import get_logger
from axion._core.schema import RichBaseModel
from axion.metrics.base import BaseMetric
from axion.metrics.internals.schema import (
    FaithfulnessVerdict,
    RelevancyVerdictModel,
)

logger = get_logger(__name__)


class StatementGeneratorInput(RichBaseModel):
    question: str = Field(description='The question to answer')
    answer: str = Field(description='The answer to the question')


class StatementGeneratorOutput(RichBaseModel):
    statements: List[str] = Field(description='The generated statements')


class StatementGenerator(BaseMetric[StatementGeneratorInput, StatementGeneratorOutput]):
    instruction = "Given a question, an answer, and sentences from the answer analyze the complexity of each sentence given under 'sentences' and break down each sentence into one or more fully understandable statements while also ensuring no pronouns are used in each statement."
    input_model = StatementGeneratorInput
    output_model = StatementGeneratorOutput
    description = 'Statement Generator Prompt'
    examples = [
        (
            StatementGeneratorInput(
                question='Who was Albert Einstein and what is he best known for?',
                answer='He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.',
            ),
            StatementGeneratorOutput(
                statements=[
                    'Albert Einstein was a German-born theoretical physicist.',
                    'Albert Einstein is recognized as one of the greatest and most influential physicists of all time.',
                    'Albert Einstein was best known for developing the theory of relativity.',
                    'Albert Einstein also made important contributions to the development of the theory of quantum mechanics.',
                ]
            ),
        )
    ]


class StatementExtractor(BaseMetric):
    """Extracts atomic statements from a body of text."""

    instruction = """### Instructions (Chain of Thought):
1. **Analyze the Input**: Read the provided text.
2. **Deconstruct Sentences**: Break down each sentence into its core factual statements.
3. **Ensure Atomicity**: Verify that each statement expresses only a single, verifiable fact.
4. **Format Output**: Return a JSON object with a 'statements' key containing the list of strings.

### Evaluation Criteria:
- Each statement must be self-contained and independently verifiable.
- Each statement must express a single fact or claim.
- Statements should be as concise as possible while remaining clear."""

    class Input(RichBaseModel):
        actual_output: str

    class Output(RichBaseModel):
        statements: List[str]

    input_model = Input
    output_model = Output

    examples = [
        (
            Input(
                actual_output='Our new laptop model features a high-resolution Retina display for crystal-clear visuals. It also includes a fast-charging battery, giving you up to 12 hours of usage on a single charge.'
            ),
            Output(
                statements=[
                    'The new laptop model has a high-resolution Retina display.',
                    'It includes a fast-charging battery with up to 12 hours of usage.',
                ]
            ),
        )
    ]

    async def execute(self, actual_output: str) -> 'Output':
        return await super().execute(self.input_model(actual_output=actual_output))


class RelevancyInput(RichBaseModel):
    input_query: str
    statements: List[str]


class RelevancyOutput(RichBaseModel):
    verdicts: List[RelevancyVerdictModel]


_STRICT_INSTRUCTION = """For the provided list of statements, determine whether each statement is relevant to address the input query.

A statement is relevant ONLY if it *directly* answers the user's query. Do not consider closely related topics as relevant.

For each statement, provide:
- verdict: 'yes' if the statement is relevant, 'no' if irrelevant, 'idk' if ambiguous
- reason: Only provide a reason if the verdict is 'no', explaining why the statement is irrelevant

The number of verdicts MUST equal the number of statements provided."""

_STRICT_EXAMPLES = [
    (
        RelevancyInput(
            input_query='What features does the new laptop have?',
            statements=[
                'The new laptop model has a high-resolution Retina display.',
                'Every purchase comes with a one-year warranty.',
                'Pineapples taste great on pizza.',
            ],
        ),
        RelevancyOutput(
            verdicts=[
                RelevancyVerdictModel(verdict='yes'),
                RelevancyVerdictModel(
                    verdict='no',
                    reason='A one-year warranty is a purchase benefit, not a feature of the laptop itself.',
                ),
                RelevancyVerdictModel(
                    verdict='no',
                    reason='The statement about pineapples on pizza is completely irrelevant to the input query.',
                ),
            ]
        ),
    )
]


_TASK_INSTRUCTION = """For the provided list of statements, determine whether each statement is relevant to the user's input query.

A statement is relevant if it:
1. Directly answers the user's query.
2. Provides essential, closely-related information that a user would likely need to complete the task implied by the query.

For each statement, provide:
- verdict: 'yes' if the statement is relevant (based on the rules above), 'no' if irrelevant, 'idk' if ambiguous
- reason: Only provide a reason if the verdict is 'no', explaining why the statement is irrelevant

The number of verdicts MUST equal the number of statements provided."""

_TASK_EXAMPLES = [
    (
        RelevancyInput(
            input_query='What features does the new laptop have?',
            statements=[
                'The new laptop model has a high-resolution Retina display.',
                'Every purchase comes with a one-year warranty.',
                'Pineapples taste great on pizza.',
            ],
        ),
        RelevancyOutput(
            verdicts=[
                RelevancyVerdictModel(verdict='yes'),
                RelevancyVerdictModel(verdict='yes'),
                RelevancyVerdictModel(
                    verdict='no',
                    reason='The statement about pineapples on pizza is completely irrelevant to the input query.',
                ),
            ]
        ),
    ),
    (
        RelevancyInput(
            input_query='How do I change my password?',
            statements=[
                "Go to Settings and click 'Change My Password'.",
                'You must also select a security question on that page.',
                'The weather today is sunny.',
            ],
        ),
        RelevancyOutput(
            verdicts=[
                RelevancyVerdictModel(verdict='yes'),
                RelevancyVerdictModel(verdict='yes'),
                RelevancyVerdictModel(
                    verdict='no',
                    reason='The weather is completely irrelevant to changing a password.',
                ),
            ]
        ),
    ),
]


class BatchRelevancyJudge(BaseMetric):
    """
    Judges the relevance of a list of statements to a query in a single batch call.
    Supports 'strict' (granular) and 'task' (lenient) relevancy modes.
    """

    as_structured_llm = True
    instruction: str = 'WILL BE OVERWRITTEN'
    examples: list = []

    input_model = RelevancyInput
    output_model = RelevancyOutput

    def __init__(
        self,
        relevancy_mode: Literal['strict', 'task'] = 'task',
        **kwargs,
    ):
        """
        Initialize the BatchRelevancyJudge.

        Args:
            relevancy_mode: The mode for judging relevancy.
                'strict': Only directly answering statements are relevant.
                'task': Closely related, helpful statements are also relevant (default).
            **kwargs: Additional arguments for BaseMetric.
        """
        # Call super() *first* to set up logging, etc.
        super().__init__(**kwargs)
        self.relevancy_mode = relevancy_mode
        if self.relevancy_mode == 'strict':
            self.instruction = _STRICT_INSTRUCTION
            self.examples = _STRICT_EXAMPLES
        else:
            self.instruction = _TASK_INSTRUCTION
            self.examples = _TASK_EXAMPLES

    async def execute(self, input_query: str, statements: List[str]) -> RelevancyOutput:
        return await super().execute(
            self.input_model(input_query=input_query, statements=statements)
        )


class RelevancyExplainer(BaseMetric):
    """Generates a human-readable explanation for a relevancy score."""

    instruction = """Given the answer relevancy score, the list of reasons for irrelevant statements, and the input query, provide a CONCISE reason for the score.

Explain why the score is not higher, but also why it is at its current level. If there are no irrelevant statements, provide positive feedback."""

    class Input(RichBaseModel):
        irrelevant_statements: List[str]
        input_query: str
        score: float

    class Output(RichBaseModel):
        explanation: str

    input_model = Input
    output_model = Output

    examples = [
        (
            Input(
                irrelevant_statements=[
                    'Customer support is a service, not a feature of the laptop.'
                ],
                input_query='What features does the new laptop have?',
                score=0.8,
            ),
            Output(
                explanation="The score is 0.8 because while most statements directly address laptop features, one response included a service rather than focusing solely on the device's technical specifications."
            ),
        )
    ]

    async def execute(
        self, irrelevant_statements: List[str], input_query: str, score: float
    ) -> 'Output':
        return await super().execute(
            self.input_model(
                irrelevant_statements=irrelevant_statements,
                input_query=input_query,
                score=score,
            )
        )


class FaithfulnessJudge(BaseMetric):
    """Judges if a claim is supported by the provided evidence."""

    instruction = """You are an expert fact-checker. Evaluate if the claim is logically entailed by the provided evidence.

### CRITICAL EVALUATION RULES:
1. **Modality Mismatch**: You MUST distinguish between "Recommended", "Required", and "Minimum".
   - If the claim says X is "required" but evidence says X is "recommended", this is **CONTRADICTORY**.
   - If the claim says X is the "minimum" but evidence lists a lower absolute minimum, this is **CONTRADICTORY**.
2. **Logical Entailment**: The claim must be completely supported by the text. Do not assume information that is not present.
3. **Numerical Precision**: Numbers must match exactly or be explicitly supported by ranges in the text.

### Verdict Categories:
- "Fully Supported": The evidence explicitly supports the claim, including the strength of the requirement (required vs. recommended).
- "Partially Supported": The evidence supports the core subject, but the claim exaggerates the certainty or gets minor details wrong.
- "No Evidence": The evidence does not contain the information needed to verify the claim.
- "Contradictory": The evidence directly contradicts the claim, OR the claim misrepresents a "recommendation" as a "requirement".

Return a JSON object with 'verdict' (one of the four categories) and 'reason' (brief explanation) keys."""

    class Input(RichBaseModel):
        claim: str
        evidence: str

    class Output(RichBaseModel):
        verdict: FaithfulnessVerdict
        reason: str

    input_model = Input
    output_model = Output

    examples = [
        (
            Input(
                claim='The minimum internet download speed required is 1.5 Mbps.',
                evidence='Recommended Minimum Bandwidth Speeds: Download 1.5 Mbps. Minimum absolute requirement: 1 Mbps.',
            ),
            Output(
                verdict=FaithfulnessVerdict.CONTRADICTORY,
                reason="The evidence lists 1.5 Mbps as 'recommended', not 'required'. The actual minimum requirement is 1 Mbps, which contradicts the claim.",
            ),
        ),
        (
            Input(
                claim='Paris has a population of 2.2 million',
                evidence='The city of Paris has 2,161,000 inhabitants.',
            ),
            Output(
                verdict=FaithfulnessVerdict.FULLY_SUPPORTED,
                reason="The evidence confirms the claim's population figure with high precision.",
            ),
        ),
        (
            Input(
                claim='You must have a webcam.',
                evidence='A webcam (built-in or external) is recommended for the best experience.',
            ),
            Output(
                verdict=FaithfulnessVerdict.CONTRADICTORY,
                reason='The claim states a webcam is a "must" (requirement), but the evidence only states it is "recommended".',
            ),
        ),
    ]

    async def execute(self, claim: str, evidence: str) -> 'Output':
        return await super().execute(self.input_model(claim=claim, evidence=evidence))


class BatchFaithfulnessJudge(BaseMetric):
    """Judges if a list of claims is supported by the provided evidence in a single batch call."""

    as_structured_llm = True
    instruction = """For each provided claim, evaluate if it is logically entailed by the provided evidence.

### CRITICAL EVALUATION RULES:
1. **Modality Mismatch**: You MUST distinguish between "Recommended", "Required", and "Minimum".
   - If a claim says X is "required" or "must" happen, but evidence says X is "recommended" or "preferred", this is **CONTRADICTORY**.
   - If a claim says X is the "minimum" but evidence lists a lower absolute minimum, this is **CONTRADICTORY**.
2. **Logical Entailment**: The claim must be completely supported by the text. Do not assume information that is not present.
3. **Numerical Precision**: Numbers must match exactly or be explicitly supported by ranges in the text.

For each claim, provide a verdict and a brief reason.
Verdict Categories: "Fully Supported", "Partially Supported", "No Evidence", "Contradictory".

The number of verdicts MUST equal the number of claims provided."""

    class Input(RichBaseModel):
        claims: List[str]
        evidence: str

    class Output(RichBaseModel):
        verdicts: List[FaithfulnessJudge.Output]

    input_model = Input
    output_model = Output

    examples = [
        (
            Input(
                claims=[
                    'The minimum internet download speed required is 1.5 Mbps.',
                    'A webcam is recommended for the exam.',
                    'You must use Windows 10.',
                ],
                evidence='Recommended Minimum Bandwidth Speeds Download: 1.5 MBPS. Absolute Minimum: 1 Mbps. A webcam is recommended. Supported OS: Windows 8 or Windows 10.',
            ),
            Output(
                verdicts=[
                    FaithfulnessJudge.Output(
                        verdict=FaithfulnessVerdict.CONTRADICTORY,
                        reason="The evidence lists 1.5 Mbps as 'recommended', while the claim states it is 'required'. The actual minimum is 1 Mbps.",
                    ),
                    FaithfulnessJudge.Output(
                        verdict=FaithfulnessVerdict.FULLY_SUPPORTED,
                        reason='The evidence explicitly matches the claim that a webcam is recommended.',
                    ),
                    FaithfulnessJudge.Output(
                        verdict=FaithfulnessVerdict.PARTIALLY_SUPPORTED,
                        reason="The evidence supports Windows 10, but the claim implies it is the only option ('must use'), ignoring Windows 8 support.",
                    ),
                ]
            ),
        )
    ]

    async def execute(self, claims: List[str], evidence: str) -> 'Output':
        return await super().execute(self.input_model(claims=claims, evidence=evidence))


class RecallJudge(BaseMetric):
    """Judges if a ground truth statement can be inferred from the context."""

    instruction = """Evaluate if the ground truth statement can be fully inferred or verified from the provided context.

A statement is SUPPORTED if all its key facts are present in or can be logically inferred from the context.
A statement is NOT SUPPORTED if any key fact is missing from the context.

Return a JSON object with a single boolean key: 'is_supported'."""

    class Input(RichBaseModel):
        ground_truth_statement: str
        context: str

    class Output(RichBaseModel):
        is_supported: bool

    input_model = Input
    output_model = Output

    examples = [
        (
            Input(
                ground_truth_statement='The Eiffel Tower was built in 1889.',
                context="The iconic Eiffel Tower, a landmark in Paris, was completed for the 1889 World's Fair.",
            ),
            Output(is_supported=True),
        )
    ]

    async def execute(self, ground_truth_statement: str, context: str) -> 'Output':
        return await super().execute(
            self.input_model(
                ground_truth_statement=ground_truth_statement, context=context
            )
        )


class BatchRecallJudge(BaseMetric):
    """Judges if a list of ground truth statements can be inferred from the context in a single batch call."""

    as_structured_llm = True
    instruction = """For each provided ground truth statement, evaluate if it can be fully inferred or verified from the provided context.

A statement is SUPPORTED if all its key facts are present in or can be logically inferred from the context.
A statement is NOT SUPPORTED if any key fact is missing from the context.

Return a list of JSON objects, each with a single boolean key: 'is_supported'.
The number of verdicts MUST equal the number of statements provided."""

    class Input(RichBaseModel):
        statements: List[str]
        context: str

    class Output(RichBaseModel):
        verdicts: List[RecallJudge.Output]

    input_model = Input
    output_model = Output

    examples = [
        (
            Input(
                statements=[
                    'Neil Armstrong was the first person on the moon.',
                    'Apollo 11 was the mission.',
                    'The landing occurred in 1969.',
                ],
                context='Apollo 11 was the spaceflight that first landed humans on the Moon in July 1969. Commander Neil Armstrong and lunar module pilot Buzz Aldrin were the first two people to land on the Moon.',
            ),
            Output(
                verdicts=[
                    RecallJudge.Output(is_supported=True),
                    RecallJudge.Output(is_supported=True),
                    RecallJudge.Output(is_supported=True),
                ]
            ),
        )
    ]

    async def execute(self, statements: List[str], context: str) -> 'Output':
        return await super().execute(
            self.input_model(statements=statements, context=context)
        )


class ChunkUsefulnessJudge(BaseMetric):
    """Judges if a context chunk is useful for generating an expected output."""

    instruction = """Evaluate if the context chunk contains useful information for generating the expected output.

A chunk is USEFUL if it contains facts or data directly relevant to generating the expected output.
A chunk is NOT USEFUL if it is tangential or too general.

Return a JSON object with a single boolean key: 'is_useful'."""

    class Input(RichBaseModel):
        context_chunk: str
        expected_output: str

    class Output(RichBaseModel):
        is_useful: bool

    input_model = Input
    output_model = Output

    examples = [
        (
            Input(
                expected_output='The first person on the moon was Neil Armstrong.',
                context_chunk='Apollo 11 was the spaceflight that first landed humans on the Moon. Commander Neil Armstrong was part of the crew.',
            ),
            Output(is_useful=True),
        )
    ]

    async def execute(self, context_chunk: str, expected_output: str) -> 'Output':
        return await super().execute(
            self.input_model(
                context_chunk=context_chunk, expected_output=expected_output
            )
        )


class UtilizationJudge(BaseMetric):
    """
    Judges whether a specific context chunk was utilized in generating the answer.
    Uses semantic similarity or LLM judgment to determine usage.
    """

    instruction = """### Instructions:
Determine whether the provided context chunk was actually used to generate the given answer.

A chunk is considered "utilized" if:
1. Information from the chunk appears in the answer (even if paraphrased)
2. The answer directly references or builds upon facts from the chunk
3. The chunk's content influenced the answer's substance

A chunk is NOT utilized if:
1. The information is not mentioned in the answer
2. The answer could have been generated without this chunk
3. The chunk is redundant with other utilized chunks

### Evaluation Criteria:
- Focus on whether the chunk contributed to the answer's content
- Look for semantic similarity, not just exact text matches
- Consider paraphrasing and synthesis of information"""

    class Input(RichBaseModel):
        context_chunk: str = Field(
            description='The context chunk to evaluate for utilization'
        )
        answer: str = Field(description='The generated answer to check against')

    class Output(RichBaseModel):
        is_utilized: bool = Field(
            description='Whether the context chunk was utilized in the answer'
        )
        reasoning: str = Field(
            default='',
            description='Explanation of why the chunk was or was not utilized',
        )

    input_model = Input
    output_model = Output

    examples = [
        (
            Input(
                context_chunk='The Eiffel Tower was completed in 1889 and stands 330 meters tall.',
                answer='The Eiffel Tower, finished in 1889, reaches a height of 330 meters and remains one of Paris most iconic landmarks.',
            ),
            Output(
                is_utilized=True,
                reasoning='The answer directly uses both facts from the chunk: the completion year (1889) and the height (330 meters).',
            ),
        ),
        (
            Input(
                context_chunk='The Louvre Museum houses over 380,000 objects and displays 35,000 works of art.',
                answer='The Eiffel Tower, finished in 1889, reaches a height of 330 meters.',
            ),
            Output(
                is_utilized=False,
                reasoning='The answer discusses the Eiffel Tower, while this chunk is about the Louvre Museum. None of this information appears in the answer.',
            ),
        ),
        (
            Input(
                context_chunk='Python was created by Guido van Rossum and first released in 1991.',
                answer='Python is a popular programming language known for its readability and extensive libraries.',
            ),
            Output(
                is_utilized=False,
                reasoning='While the answer discusses Python, it does not mention the creator or release year from this chunk. The answer could be generated without this specific information.',
            ),
        ),
    ]

    async def execute(self, context_chunk: str, answer: str) -> 'Output':
        return await super().execute(
            self.input_model(context_chunk=context_chunk, answer=answer)
        )


class UtilizationVerdict(RichBaseModel):
    """Verdict for a single chunk's utilization."""

    is_utilized: bool = Field(
        description='Whether this chunk was utilized in the answer'
    )
    reasoning: str = Field(
        default='', description='Explanation of the utilization decision'
    )


class BatchUtilizationJudge(BaseMetric):
    """
    Judges whether multiple context chunks were utilized in generating an answer.
    More efficient than individual judgments when evaluating many chunks.
    """

    instruction = """### Instructions:
For each provided context chunk, determine whether it was actually used to generate the given answer.

A chunk is considered "utilized" if:
1. Information from the chunk appears in the answer (even if paraphrased)
2. The answer directly references or builds upon facts from the chunk
3. The chunk's content influenced the answer's substance

A chunk is NOT utilized if:
1. The information is not mentioned in the answer
2. The answer could have been generated without this chunk
3. The chunk is redundant with other utilized chunks

For each chunk, provide:
- is_utilized: boolean indicating if the chunk was used
- reasoning: brief explanation of your decision

The number of verdicts MUST equal the number of chunks provided."""

    class Input(RichBaseModel):
        chunks: list[str] = Field(description='List of context chunks to evaluate')
        answer: str = Field(description='The generated answer to check against')

    class Output(RichBaseModel):
        verdicts: list[UtilizationVerdict] = Field(
            description='List of utilization verdicts, one per chunk'
        )

    input_model = Input
    output_model = Output

    examples = [
        (
            Input(
                chunks=[
                    'The Eiffel Tower was completed in 1889.',
                    'The tower stands 330 meters tall.',
                    'The Louvre Museum houses over 380,000 objects.',
                ],
                answer='The Eiffel Tower, finished in 1889, reaches a height of 330 meters.',
            ),
            Output(
                verdicts=[
                    UtilizationVerdict(
                        is_utilized=True,
                        reasoning='The completion year 1889 is mentioned in the answer.',
                    ),
                    UtilizationVerdict(
                        is_utilized=True,
                        reasoning='The height of 330 meters is stated in the answer.',
                    ),
                    UtilizationVerdict(
                        is_utilized=False,
                        reasoning='The Louvre Museum is not mentioned in the answer about the Eiffel Tower.',
                    ),
                ]
            ),
        )
    ]

    async def execute(self, chunks: list[str], answer: str) -> 'Output':
        return await super().execute(self.input_model(chunks=chunks, answer=answer))


class ContextSufficiencyJudge(BaseMetric):
    """
    Judges whether the provided context contains sufficient information to answer
    the query, independent of any generated answer.
    """

    instruction = """### Instructions:
Evaluate whether the provided context contains sufficient information to answer the user's query.

This is a DIAGNOSTIC evaluation - you are NOT evaluating the quality of any generated answer.
You are ONLY evaluating whether the context makes it POSSIBLE to answer the query.

Context is SUFFICIENT if:
1. All key information needed to answer the query is present in the context
2. The context provides enough detail for a complete answer
3. No critical information is missing

Context is INSUFFICIENT if:
1. Key information needed to answer the query is missing
2. The context is too vague or incomplete
3. The context discusses related topics but not the specific question asked

### Evaluation Criteria:
- Focus solely on the query and context relationship
- Do not consider what answer might be generated
- Be specific about what information is present or missing
- Consider whether the query could be fully answered with only this context"""

    class Input(RichBaseModel):
        query: str = Field(description='The user query that needs to be answered')
        context: str = Field(
            description='The retrieved context to evaluate for sufficiency'
        )

    class Output(RichBaseModel):
        is_sufficient: bool = Field(
            description='Whether the context is sufficient to answer the query'
        )
        reasoning: str = Field(
            description='Explanation of why the context is or is not sufficient'
        )

    input_model = Input
    output_model = Output

    examples = [
        (
            Input(
                query='What was our Q2 2024 revenue?',
                context='Our Q1 2024 revenue was $5M. Our Q3 2024 projections are strong. The company performed well in early 2024.',
            ),
            Output(
                is_sufficient=False,
                reasoning='The context provides Q1 2024 revenue and Q3 projections, but does not contain the specific Q2 2024 revenue figure that was requested.',
            ),
        ),
        (
            Input(
                query='What was our Q2 2024 revenue?',
                context='In Q2 2024, the company generated $7.2M in revenue, representing a 15% increase over Q1.',
            ),
            Output(
                is_sufficient=True,
                reasoning='The context explicitly states the Q2 2024 revenue as $7.2M, which directly answers the query.',
            ),
        ),
        (
            Input(
                query='How do I reset my password?',
                context='To update your account settings, go to the Settings page. Our security features include two-factor authentication and password policies.',
            ),
            Output(
                is_sufficient=False,
                reasoning='The context mentions settings and security features but does not provide the specific steps needed to reset a password.',
            ),
        ),
        (
            Input(
                query='What are the benefits of Python for data science?',
                context='Python offers extensive data science libraries like pandas, NumPy, and scikit-learn. It has a gentle learning curve and strong community support. Python integrates well with big data tools and provides excellent visualization capabilities through matplotlib and seaborn.',
            ),
            Output(
                is_sufficient=True,
                reasoning='The context provides comprehensive information about Python benefits for data science, including libraries, ease of learning, community support, integration capabilities, and visualization tools.',
            ),
        ),
    ]

    async def execute(self, query: str, context: str) -> 'Output':
        return await super().execute(self.input_model(query=query, context=context))
