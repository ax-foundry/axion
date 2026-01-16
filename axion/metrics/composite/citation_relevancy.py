import re
from typing import Dict, List, Optional, Tuple

from pydantic import Field

from axion._core.logging import get_logger
from axion._core.schema import AIMessage, HumanMessage, RichBaseModel
from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import (
    BaseMetric,
    MetricEvaluationResult,
    metric,
)
from axion.metrics.schema import SignalDescriptor

logger = get_logger(__name__)


class CitationRelevanceVerdict(RichBaseModel):
    citation: str = Field(description='The exact citation text or URL being judged.')
    is_relevant: bool = Field(
        description="True if the citation's content/source is relevant to the original query, False otherwise."
    )
    reason: str = Field(
        description='A concise reason explaining why the citation is RELEVANT or NOT RELEVANT to the original query.'
    )
    # Added field to track the turn number associated with the citation
    turn_index: Optional[int] = Field(
        default=None,
        description='The turn index where the citation was provided (0-indexed).',
    )
    original_query: str = Field(
        description='The user query that prompted this citation.'
    )


class CitationRelevanceJudgeInput(RichBaseModel):
    original_query: str = Field(description='The user’s original query.')
    citations: List[str] = Field(
        description='The list of extracted citations to judge.'
    )


class CitationRelevanceJudgeOutput(RichBaseModel):
    verdicts: List[CitationRelevanceVerdict] = Field(
        description='The relevance judgement for each citation provided in the input.'
    )


class CitationRelevanceJudge(
    BaseMetric[CitationRelevanceJudgeInput, CitationRelevanceJudgeOutput]
):
    as_structured_llm = True
    instruction = """For each citation provided, determine its relevance to the user's original query.

    The citation will be provided as either a raw URL or in the format '[Title](URL)' when available. Use the **Title** to infer the content of the source.

    A citation is RELEVANT if:
    1. It is highly likely to contain the information needed to answer the query.
    2. It is a definitive source for the domain or entity mentioned in the query.
    3. The source's title/URL/identifier clearly matches the topic of the query.

    A citation is NOT RELEVANT if:
    1. It is a general homepage or a link to a tangential topic.
    2. It requires significant searching or navigation to find the answer.
    3. It appears to be a broken or non-descriptive reference.

    Provide a concise reason for every verdict (Relevant or Not Relevant). The number of verdicts MUST equal the number of citations provided."""
    input_model = CitationRelevanceJudgeInput
    output_model = CitationRelevanceJudgeOutput
    description = 'Judges citation relevance against the original query.'
    examples = [
        (
            CitationRelevanceJudgeInput(
                original_query='What are the new features in Python 3.12?',
                citations=[
                    "[What's New in Python 3.12](https://docs.python.org/3/whatsnew/3.12.html)",
                    'https://www.python.org/',
                    'Smith et al., 2022',
                ],
            ),
            CitationRelevanceJudgeOutput(
                verdicts=[
                    CitationRelevanceVerdict(
                        citation="[What's New in Python 3.12](https://docs.python.org/3/whatsnew/3.12.html)",
                        is_relevant=True,
                        reason='The citation title directly discusses new features specifically in Python 3.12.',
                        turn_index=0,
                        original_query='What are the new features in Python 3.12?',
                    ),
                    CitationRelevanceVerdict(
                        citation='https://www.python.org/',
                        is_relevant=False,
                        reason='This is a general homepage for Python, not a specific article about Python 3.12 features.',
                        turn_index=0,
                        original_query='What are the new features in Python 3.12?',
                    ),
                    CitationRelevanceVerdict(
                        citation='Smith et al., 2022',
                        is_relevant=False,
                        reason='The citation is an academic reference from 2022, which is unlikely to contain details about Python 3.12 features released later.',
                        turn_index=0,
                        original_query='What are the new features in Python 3.12?',
                    ),
                ]
            ),
        ),
        (
            CitationRelevanceJudgeInput(
                original_query='What is the current population of Tokyo?',
                citations=[
                    'https://www.tokyo.gov.jp/population-stats/',
                    'Wang, J. (2019)',
                ],
            ),
            CitationRelevanceJudgeOutput(
                verdicts=[
                    CitationRelevanceVerdict(
                        citation='https://www.tokyo.gov.jp/population-stats/',
                        is_relevant=True,
                        reason="This URL clearly links to the official Tokyo government's population statistics page, making it a definitive and highly relevant source.",
                        turn_index=0,
                        original_query='What is the current population of Tokyo?',
                    ),
                    CitationRelevanceVerdict(
                        citation='Wang, J. (2019)',
                        is_relevant=False,
                        reason='This academic reference is outdated (2019) and therefore unlikely to contain the current population figure requested.',
                        turn_index=0,
                        original_query='What is the current population of Tokyo?',
                    ),
                ]
            ),
        ),
    ]

    async def execute(
        self, original_query: str, citations: List[str]
    ) -> 'CitationRelevanceJudgeOutput':
        return await super().execute(
            self.input_model(original_query=original_query, citations=citations)
        )


class CitationRelevanceResult(RichBaseModel):
    relevance_score: float = Field(
        description='The percentage of citations determined to be relevant (0.0 to 1.0).'
    )
    total_citations: int = Field(description='Total number of citations found.')
    relevant_citations_count: int = Field(
        description='Number of citations found to be relevant.'
    )
    breakdown: List[CitationRelevanceVerdict] = Field(
        description='Detailed relevance judgement for each citation.'
    )


@metric(
    name='Citation Relevancy',
    key='citation_relevancy',
    description='Measures the quality of citations by judging their relevance to the user query using heuristic extraction and LLM reasoning.',
    required_fields=['query', 'actual_output'],
    optional_fields=['conversation'],
    default_threshold=0.8,
    score_range=(0, 1),
    tags=['agent', 'knowledge', 'multi_turn', 'citation'],
)
class CitationRelevancy(BaseMetric):
    """
    Evaluates the quality of citations by assessing their relevance to the user's query
    using heuristic (regex) extraction and LLM judging, supporting both single and multi-turn evaluation.
    """

    as_structured_llm = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.relevance_judge = CitationRelevanceJudge(**kwargs)

    @trace(
        name='extract_citations_heuristically',
        capture_args=True,
        capture_response=False,
    )
    def _extract_citations_heuristically(self, text: str) -> List[str]:
        """
        Extract URLs, markdown links, and other citation formats.
        Prioritizes capturing the markdown format [Title](URL) as a single string.
        """
        citations = []
        seen_urls = set()

        # Markdown link pattern: [Title](URL)
        markdown_pattern = r'\[([^\]]+)\]\((https?://[^\s\)]+)\)'
        markdown_matches = list(re.finditer(markdown_pattern, text))

        for match in markdown_matches:
            url = match.group(2).strip()

            # Use the full markdown link as the citation string
            citation_string = match.group(0)

            # Check for uniqueness based on URL
            if url not in seen_urls:
                citations.append(citation_string)
                seen_urls.add(url)

        # Base patterns, excluding the full markdown pattern from search
        patterns = [
            r'https?://[^\s<>"\'\[\]{}|\\^`]+',  # HTTP/HTTPS URLs
            r'www\.[^\s<>"\'\[\]{}|\\^`]+',  # WWW URLs without protocol
            r'(?:doi:|DOI:)\s*[^\s\]\)]+',  # DOI patterns (full match)
            r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s+\d{4}\)',  # Academic (Author, Year)
        ]

        # Use finditer to get match objects for all occurrences
        combined_pattern = '|'.join(patterns)

        # Filter the text to remove the already found markdown links before running general patterns
        temp_text = text
        for match in reversed(markdown_matches):
            start, end = match.span()
            temp_text = temp_text[:start] + ' ' * (end - start) + temp_text[end:]

        general_matches = re.finditer(combined_pattern, temp_text, re.IGNORECASE)

        for match in general_matches:
            citation = match.group(0)

            # Clean up the extracted citation to remove trailing punctuation
            cleaned = re.sub(r'[.,;:!?\'")\]}]+$', '', citation)

            # A simplistic way to extract URL for uniqueness check
            url_check = cleaned.lower()
            if 'http' in url_check:
                url_check = re.search(r'(https?://[^\s]+)', url_check)
                url_check = url_check.group(0) if url_check else cleaned

            if url_check not in seen_urls and cleaned not in seen_urls:
                # For non-URL citations (DOI, Author/Year), use the cleaned string as the key
                seen_urls.add(url_check)
                seen_urls.add(cleaned)
                citations.append(cleaned)

        return citations

    @trace(name='get_citation_judgments', capture_args=True)
    async def _get_citation_judgments(
        self, item: DatasetItem
    ) -> Tuple[List[CitationRelevanceVerdict], int]:
        """
        Processes conversation turns or single-turn data to extract citations,
        associates them with the correct query, and judges relevance.
        Returns a list of all verdicts and the total number of citations found.
        """
        all_verdicts: List[CitationRelevanceVerdict] = []
        # {query: [(citation, turn_index)]}
        judgements_to_run: Dict[str, List[Tuple[str, int]]] = {}
        total_citations = 0

        if item.conversation and item.conversation.messages:
            # Multi-turn logic
            messages = item.conversation.messages
            current_user_query = None

            for i, message in enumerate(messages):
                if isinstance(message, HumanMessage):
                    current_user_query = message.content
                elif isinstance(message, AIMessage) and message.content:
                    # Assistant message index is i
                    if current_user_query is None:
                        # Skip if AI message isn't preceded by a user query
                        logger.warning(
                            f'Skipping AIMessage at index {i} without preceding HumanMessage.'
                        )
                        continue

                    citations = self._extract_citations_heuristically(message.content)
                    total_citations += len(citations)

                    if not citations:
                        continue

                    # Group citations by the user query that prompted them
                    if current_user_query not in judgements_to_run:
                        judgements_to_run[current_user_query] = []

                    for citation in citations:
                        # Store the citation and the index of the ASSISTANT turn it appeared in
                        judgements_to_run[current_user_query].append((citation, i))

                    # Reset the user query for the next turn
                    current_user_query = None

        elif item.actual_output and item.query:
            # Single-turn logic (uses actual_output and query)
            citations = self._extract_citations_heuristically(item.actual_output)
            total_citations = len(citations)

            if citations:
                judgements_to_run[item.query] = [(c, 0) for c in citations]

        else:
            return [], 0  # Neither conversation nor single-turn data available

        # 2. Batch Judge Relevance for each unique query context
        for query, citation_data in judgements_to_run.items():
            citations_for_query = [data[0] for data in citation_data]

            if not citations_for_query:
                continue

            try:
                judgement_output = await self.relevance_judge.execute(
                    original_query=query, citations=citations_for_query
                )

                # Pair the generic verdicts back with their original context (citation, turn_index)
                for i, verdict in enumerate(judgement_output.verdicts):
                    citation_text, turn_index = citation_data[i]

                    # Create the final, rich verdict object
                    all_verdicts.append(
                        CitationRelevanceVerdict(
                            citation=citation_text,
                            is_relevant=verdict.is_relevant,
                            reason=verdict.reason,
                            turn_index=turn_index,
                            original_query=query,
                        )
                    )
            except Exception as e:
                logger.error(
                    f'Relevance judging failed for query "{query[:50]}...": {e}'
                )
                # Log error and continue to the next query, or handle as needed
                continue

        return all_verdicts, total_citations

    @trace(name='CitationRelevancy', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """
        Executes the citation relevance workflow: extract, judge, score.
        """
        self._validate_required_metric_fields(item)

        # Extract and Judge Citations (handles single/multi-turn)
        all_verdicts, total_count = await self._get_citation_judgments(item)

        if total_count == 0:
            return MetricEvaluationResult(
                score=0.0, explanation='No citations were found in the response(s).'
            )

        # Calculate Score
        relevant_count = sum(1 for v in all_verdicts if v.is_relevant)
        relevance_score = relevant_count / total_count if total_count > 0 else 0.0

        # Compute cost estimate
        # Only the relevance judge is an LLM call now
        self.compute_cost_estimate([self.relevance_judge])

        # Build Result and Explanation
        result_data = CitationRelevanceResult(
            relevance_score=relevance_score,
            total_citations=total_count,
            relevant_citations_count=relevant_count,
            breakdown=all_verdicts,
        )

        explanation = (
            f'Citation Relevance Score: {relevance_score:.2f} '
            f'({relevant_count} out of {total_count} citations were relevant across all turns). '
        )
        if total_count > 0 and relevant_count < total_count:
            irrelevant_reasons = [
                f"Turn {v.turn_index}: '{v.citation[:30]}...': {v.reason}"
                for v in all_verdicts
                if not v.is_relevant
            ]
            # Use max(3) to keep explanation concise
            explanation += f'Irrelevant citations found (top 3): {"; ".join(irrelevant_reasons[:3])}'
            if len(irrelevant_reasons) > 3:
                explanation += f' and {len(irrelevant_reasons) - 3} more.'

        return MetricEvaluationResult(
            score=relevance_score, explanation=explanation, signals=result_data
        )

    def get_signals(
        self, result: CitationRelevanceResult
    ) -> List[SignalDescriptor[CitationRelevanceResult]]:
        """Generates detailed signals from the relevance evaluation."""
        signals = [
            # === HEADLINE SCORE ===
            SignalDescriptor(
                name='relevance_score',
                group='Overall',
                extractor=lambda r: r.relevance_score,
                headline_display=True,
                description='Percentage of citations relevant to the original query.',
            ),
            # === AGGREGATES ===
            SignalDescriptor(
                name='total_citations',
                group='Overall',
                extractor=lambda r: r.total_citations,
                description='Total number of citations extracted from the response.',
            ),
            SignalDescriptor(
                name='relevant_citations_count',
                group='Overall',
                extractor=lambda r: r.relevant_citations_count,
                description='Number of citations deemed relevant.',
            ),
            SignalDescriptor(
                name='irrelevant_citations_count',
                group='Overall',
                extractor=lambda r: r.total_citations - r.relevant_citations_count,
                description='Number of citations deemed irrelevant.',
            ),
        ]

        # === PER-CITATION BREAKDOWN ===
        for i, verdict in enumerate(result.breakdown):
            # Use the start of the citation as the group name
            citation_preview = verdict.citation[:50].replace('\n', ' ').strip()
            group_name = f'T{verdict.turn_index}-C{i + 1}: {citation_preview}{"..." if len(verdict.citation) > 50 else ""}'

            signals.extend(
                [
                    SignalDescriptor(
                        name='relevance_verdict',
                        group=group_name,
                        extractor=lambda r, idx=i: r.breakdown[idx].is_relevant,
                        headline_display=True,
                        description='Relevance status (True/False).',
                    ),
                    SignalDescriptor(
                        name='turn_index',
                        group=group_name,
                        extractor=lambda r, idx=i: r.breakdown[idx].turn_index,
                        description='The assistant turn index that provided the citation.',
                    ),
                    SignalDescriptor(
                        name='original_query',
                        group=group_name,
                        extractor=lambda r, idx=i: r.breakdown[idx].original_query,
                        description='The user query that preceded this turn.',
                    ),
                    SignalDescriptor(
                        name='citation_text',
                        group=group_name,
                        extractor=lambda r, idx=i: r.breakdown[idx].citation,
                        description='The full citation text/URL.',
                    ),
                    SignalDescriptor(
                        name='relevance_reason',
                        group=group_name,
                        extractor=lambda r, idx=i: r.breakdown[idx].reason,
                        description='Reasoning for the relevance judgement.',
                    ),
                ]
            )

        return signals
