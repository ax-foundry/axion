"""Tests for evaluation module."""

from axion.caliber.evaluation import CaliberMetric, EvaluationRunner
from axion.caliber.models import (
    AlignmentMetrics,
    Annotation,
    EvaluationConfig,
    EvaluationRecord,
    EvaluationResult,
    UploadedRecord,
)


class TestCaliberMetric:
    """Tests for CaliberMetric class."""

    def test_init_basic(self):
        """Test basic initialization."""
        metric = CaliberMetric(
            instruction='Score 1 if accurate, 0 otherwise.',
            model_name='gpt-4o',
            llm_provider='openai',
        )
        assert metric.instruction == 'Score 1 if accurate, 0 otherwise.'
        assert metric.examples == []

    def test_init_with_examples(self):
        """Test initialization with few-shot examples."""
        examples = [
            {
                'input': {
                    'id': 'ex1',
                    'query': 'What is 2+2?',
                    'actual_output': '4',
                },
                'output': {
                    'score': 1,
                    'explanation': 'Correct answer',
                },
            },
        ]
        metric = CaliberMetric(
            instruction='Score 1 if correct.',
            model_name='gpt-4o',
            examples=examples,
        )
        assert len(metric.examples) == 1

    def test_init_required_fields(self):
        """Test initialization with required fields."""
        metric = CaliberMetric(
            instruction='Test',
            required_fields=['query', 'actual_output'],
        )
        # required_fields is handled by BaseMetric
        assert metric.instruction == 'Test'


class TestEvaluationRunner:
    """Tests for EvaluationRunner class."""

    def test_init(self):
        """Test initialization."""
        config = EvaluationConfig(
            criteria='Score 1 if accurate.',
            model_name='gpt-4o',
        )
        runner = EvaluationRunner(config, max_concurrent=10)
        assert runner._config == config
        assert runner._max_concurrent == 10
        assert runner._metric is None

    def test_compute_metrics_empty(self):
        """Test computing metrics with no records."""
        config = EvaluationConfig(criteria='Test')
        runner = EvaluationRunner(config)

        metrics = runner._compute_metrics([])

        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.cohen_kappa == 0.0
        assert metrics.true_positives == 0

    def test_compute_metrics_perfect_alignment(self):
        """Test computing metrics with perfect alignment."""
        config = EvaluationConfig(criteria='Test')
        runner = EvaluationRunner(config)

        records = [
            EvaluationRecord(
                record_id='r1',
                human_score=1,
                llm_score=1,
                aligned=True,
                score_difference=0,
            ),
            EvaluationRecord(
                record_id='r2',
                human_score=0,
                llm_score=0,
                aligned=True,
                score_difference=0,
            ),
            EvaluationRecord(
                record_id='r3',
                human_score=1,
                llm_score=1,
                aligned=True,
                score_difference=0,
            ),
            EvaluationRecord(
                record_id='r4',
                human_score=0,
                llm_score=0,
                aligned=True,
                score_difference=0,
            ),
        ]

        metrics = runner._compute_metrics(records)

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.cohen_kappa == 1.0
        assert metrics.true_positives == 2
        assert metrics.true_negatives == 2
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0

    def test_compute_metrics_with_misalignment(self):
        """Test computing metrics with some misalignment."""
        config = EvaluationConfig(criteria='Test')
        runner = EvaluationRunner(config)

        records = [
            EvaluationRecord(
                record_id='r1',
                human_score=1,
                llm_score=1,
                aligned=True,
                score_difference=0,
            ),  # TP
            EvaluationRecord(
                record_id='r2',
                human_score=0,
                llm_score=0,
                aligned=True,
                score_difference=0,
            ),  # TN
            EvaluationRecord(
                record_id='r3',
                human_score=0,
                llm_score=1,
                aligned=False,
                score_difference=1,
            ),  # FP
            EvaluationRecord(
                record_id='r4',
                human_score=1,
                llm_score=0,
                aligned=False,
                score_difference=1,
            ),  # FN
        ]

        metrics = runner._compute_metrics(records)

        assert metrics.accuracy == 0.5  # 2 out of 4 aligned
        assert metrics.true_positives == 1
        assert metrics.true_negatives == 1
        assert metrics.false_positives == 1
        assert metrics.false_negatives == 1

    def test_build_confusion_matrix_empty(self):
        """Test building confusion matrix with no records."""
        config = EvaluationConfig(criteria='Test')
        runner = EvaluationRunner(config)

        matrix = runner._build_confusion_matrix([])

        assert matrix == {
            'LLM=0': {'Human=0': 0, 'Human=1': 0},
            'LLM=1': {'Human=0': 0, 'Human=1': 0},
        }

    def test_build_confusion_matrix(self):
        """Test building confusion matrix."""
        config = EvaluationConfig(criteria='Test')
        runner = EvaluationRunner(config)

        records = [
            EvaluationRecord(
                record_id='r1',
                human_score=1,
                llm_score=1,
                aligned=True,
                score_difference=0,
            ),  # TP
            EvaluationRecord(
                record_id='r2',
                human_score=0,
                llm_score=0,
                aligned=True,
                score_difference=0,
            ),  # TN
            EvaluationRecord(
                record_id='r3',
                human_score=0,
                llm_score=1,
                aligned=False,
                score_difference=1,
            ),  # FP
            EvaluationRecord(
                record_id='r4',
                human_score=1,
                llm_score=0,
                aligned=False,
                score_difference=1,
            ),  # FN
        ]

        matrix = runner._build_confusion_matrix(records)

        assert matrix['LLM=0']['Human=0'] == 1  # TN
        assert matrix['LLM=0']['Human=1'] == 1  # FN
        assert matrix['LLM=1']['Human=0'] == 1  # FP
        assert matrix['LLM=1']['Human=1'] == 1  # TP

    def test_coerce_binary_score_valid(self):
        """Test coercing valid binary scores."""
        config = EvaluationConfig(criteria='Test')
        runner = EvaluationRunner(config)

        assert runner._coerce_binary_score(0, 'r1') == (0, None)
        assert runner._coerce_binary_score(1, 'r1') == (1, None)
        assert runner._coerce_binary_score('0', 'r1') == (0, None)
        assert runner._coerce_binary_score('1', 'r1') == (1, None)
        assert runner._coerce_binary_score(' 1 ', 'r1') == (1, None)

    def test_coerce_binary_score_invalid(self):
        """Test coercing invalid scores."""
        config = EvaluationConfig(criteria='Test')
        runner = EvaluationRunner(config)

        score, note = runner._coerce_binary_score(2, 'r1')
        assert score == 0
        assert 'coerced to 0' in note

        score, note = runner._coerce_binary_score(None, 'r1')
        assert score == 0
        assert 'Missing score' in note

        score, note = runner._coerce_binary_score('invalid', 'r1')
        assert score == 0
        assert 'coerced to 0' in note

    def test_coerce_binary_score_nan(self):
        """Test coercing NaN scores."""

        config = EvaluationConfig(criteria='Test')
        runner = EvaluationRunner(config)

        score, note = runner._coerce_binary_score(float('nan'), 'r1')
        assert score == 0
        assert 'Missing score' in note

    def test_records_to_dataset_items(self):
        """Test converting records to dataset items."""
        config = EvaluationConfig(criteria='Test')
        runner = EvaluationRunner(config)

        records = [
            UploadedRecord(
                id='r1', query='Q1', actual_output='A1', expected_output='E1'
            ),
            UploadedRecord(id='r2', query='Q2', actual_output='A2'),
        ]
        annotations = {
            'r1': Annotation(record_id='r1', score=1),
            'r2': Annotation(record_id='r2', score=0),
        }

        items = runner._records_to_dataset_items(records, annotations)

        assert len(items) == 2
        assert items[0].id == 'r1'
        assert items[0].query == 'Q1'
        assert items[0].judgment == '1'
        assert items[1].judgment == '0'

    def test_records_to_dataset_items_missing_annotation(self):
        """Test converting records with missing annotations."""
        config = EvaluationConfig(criteria='Test')
        runner = EvaluationRunner(config)

        records = [
            UploadedRecord(id='r1', query='Q1', actual_output='A1'),
            UploadedRecord(id='r2', query='Q2', actual_output='A2'),
        ]
        annotations = {
            'r1': Annotation(record_id='r1', score=1),
            # r2 missing
        }

        items = runner._records_to_dataset_items(records, annotations)

        assert len(items) == 1
        assert items[0].id == 'r1'

    def test_build_evaluation_records(self):
        """Test building evaluation records from results."""
        config = EvaluationConfig(criteria='Test')
        runner = EvaluationRunner(config)

        records = [
            UploadedRecord(id='r1', query='Q1', actual_output='A1'),
            UploadedRecord(id='r2', query='Q2', actual_output='A2'),
        ]
        annotations = {
            'r1': Annotation(record_id='r1', score=1),
            'r2': Annotation(record_id='r2', score=0),
        }
        llm_evaluations = {
            'r1': {'llm_score': 1, 'llm_reasoning': 'Good'},
            'r2': {'llm_score': 1, 'llm_reasoning': 'Also good'},  # Misaligned
        }

        eval_records = runner._build_evaluation_records(
            records, annotations, llm_evaluations
        )

        assert len(eval_records) == 2
        assert eval_records[0].record_id == 'r1'
        assert eval_records[0].aligned is True
        assert eval_records[1].record_id == 'r2'
        assert eval_records[1].aligned is False
        assert eval_records[1].score_difference == 1


class TestEvaluationConfig:
    """Tests for EvaluationConfig model."""

    def test_create_minimal(self):
        """Test creating config with minimal fields."""
        config = EvaluationConfig(criteria='Score 1 if good.')
        assert config.criteria == 'Score 1 if good.'
        assert config.model_name == 'gpt-4o'  # default

    def test_create_full(self):
        """Test creating config with all fields."""
        config = EvaluationConfig(
            criteria='Score 1 if accurate.',
            model_name='gpt-4o-mini',
            llm_provider='openai',
            system_prompt='You are an evaluator.',
        )
        assert config.model_name == 'gpt-4o-mini'
        assert config.llm_provider == 'openai'
        assert config.system_prompt == 'You are an evaluator.'


class TestAlignmentMetrics:
    """Tests for AlignmentMetrics model."""

    def test_create_metrics(self):
        """Test creating alignment metrics."""
        metrics = AlignmentMetrics(
            accuracy=0.85,
            precision=0.9,
            recall=0.8,
            f1_score=0.85,
            cohen_kappa=0.7,
            specificity=0.9,
            true_positives=18,
            true_negatives=17,
            false_positives=2,
            false_negatives=3,
        )
        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.9
        assert metrics.true_positives == 18


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_create_result(self):
        """Test creating evaluation result."""
        metrics = AlignmentMetrics(
            accuracy=1.0,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            cohen_kappa=1.0,
            specificity=1.0,
            true_positives=5,
            true_negatives=5,
            false_positives=0,
            false_negatives=0,
        )
        config = EvaluationConfig(criteria='Test')
        records = [
            EvaluationRecord(
                record_id='r1',
                human_score=1,
                llm_score=1,
                aligned=True,
                score_difference=0,
            )
        ]

        result = EvaluationResult(
            records=records,
            metrics=metrics,
            confusion_matrix={
                'LLM=0': {'Human=0': 5, 'Human=1': 0},
                'LLM=1': {'Human=0': 0, 'Human=1': 5},
            },
            config=config,
        )

        assert len(result.records) == 1
        assert result.metrics.accuracy == 1.0
