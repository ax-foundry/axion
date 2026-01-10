class ExplanationTemplates:
    """
    Standard templates for metric explanations.
    Separated for easier maintenance and lookups.
    """

    @staticmethod
    def get_manual_explanation(metric_name, score):
        metric_explanations = {
            'CitationPresence': f'Citation tracking shows {score:.1%} of responses included source citations or links, {"meeting" if score >= 0.9 else "approaching" if score >= 0.7 else "missing"} attribution standards.',
            'ContextualSufficiency': f'Retrieved context was sufficient to answer the query in {score:.1%} of cases, {"indicating strong retrieval coverage" if score >= 0.8 else "suggesting retrieval improvements may be needed" if score < 0.7 else "showing adequate information retrieval"}.',
            'ContextualUtilization': f'Only {score:.1%} of relevant retrieved chunks were actually used in generating answers, {"indicating efficient context use" if score >= 0.6 else "suggesting the model may be overlooking available information" if score < 0.5 else "showing moderate utilization"}.',
            'HitRate': f'The system successfully found the correct answer within the top 5 results {score:.1%} of the time (Hits@5), measuring fundamental retrieval success without considering ranking order.',
            'MeanReciprocalRank': f'On average, the first correct answer appeared at position {f"{1 / score:.1f}" if score > 0 else "N/A"} in search results (MRR: {score:.2f}), {"rewarding top placements" if score >= 0.7 else "indicating ranking could be improved" if score < 0.5 else "showing moderate ranking quality"}.',
            'ContextualPrecision': f'{score:.1%} of retrieved document chunks were actually relevant to answering the query, measuring how much noise versus signal was included in the context window.',
            'ContextualRecall': f'{score:.1%} of the necessary information needed to answer queries was successfully retrieved, {"ensuring comprehensive coverage" if score >= 0.8 else "suggesting some information may be missed" if score < 0.7 else "providing adequate coverage"}.',
            'ContextualRelevancy': f'Retrieved context achieved {score:.1%} relevance to input queries, evaluating how well the retrieval system matches user information needs.',
            'AnswerRelevancy': f'{score:.1%} of generated answers directly addressed the specific question asked, measuring response focus and avoiding tangential information.',
            'Faithfulness': f'{score:.1%} of answer content was grounded in the retrieved context without hallucination, {"demonstrating excellent factual accuracy" if score >= 0.9 else "showing good grounding" if score >= 0.8 else "indicating potential hallucination issues"}.',
            'CriteriaCorrectness': f'{score:.1%} of user-defined correctness criteria (specific facts, required logic, constraints) were satisfied, using aspect-based evaluation to verify accuracy requirements.',
            'CriteriaRelevancy': f'{score:.1%} of user-defined relevancy criteria aspects were meaningfully addressed in responses, measuring alignment with high-level relevance expectations.',
        }
        return metric_explanations.get(metric_name)

    @staticmethod
    def get_category_explanation(metric_name, score):
        category_explanations = {
            'model': 'Overall model performance aggregated across all evaluation dimensions',
            'ANSWER_QUALITY': 'Aggregated score measuring the quality and correctness of model responses',
            'RAG_QUALITY': 'Aggregated score for Retrieval-Augmented Generation performance',
            'CONTEXTUAL': 'Composite score measuring how well the retrieved context supports answer generation',
            'GENERATION': 'Quality of text generation given the retrieved context',
            'RETRIEVAL': 'Composite score for document retrieval effectiveness',
        }
        if metric_name in category_explanations:
            base_desc = category_explanations[metric_name]
            qualifier = (
                'excellent'
                if score >= 0.8
                else 'good'
                if score >= 0.6
                else 'can be improved'
            )
            return (
                f'{base_desc}. Currently tracking as <b>{qualifier}</b> ({score:.1%}).'
            )
        return None
