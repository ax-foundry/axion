import pytest

from axion.caliber.pattern_discovery import (
    AnnotatedItem,
    AnnotationNote,
    ClusteringInput,
    ClusteringMethod,
    ClusteringOutput,
    DiscoveredPattern,
    EvidenceItem,
    LabelInput,
    LabelOutput,
    PatternCategory,
    PatternDiscovery,
    PatternDiscoveryResult,
)

# Check if BERTopic is available
try:
    import bertopic  # noqa: F401

    HAS_BERTOPIC = True
except ImportError:
    HAS_BERTOPIC = False

requires_bertopic = pytest.mark.skipif(
    not HAS_BERTOPIC, reason='BERTopic not installed'
)


class TestAnnotatedItem:
    """Tests for AnnotatedItem dataclass."""

    def test_create_minimal(self):
        """Test creating item with minimal fields."""
        item = AnnotatedItem(record_id='r1', score=1)
        assert item.record_id == 'r1'
        assert item.score == 1
        assert item.notes is None
        assert item.query is None
        assert item.actual_output is None

    def test_create_full(self):
        """Test creating item with all fields."""
        item = AnnotatedItem(
            record_id='r1',
            score=0,
            notes='Missing context',
            timestamp='2024-01-01T00:00:00Z',
            query='What is Python?',
            actual_output='A snake.',
        )
        assert item.record_id == 'r1'
        assert item.score == 0
        assert item.notes == 'Missing context'
        assert item.query == 'What is Python?'


class TestDiscoveredPattern:
    """Tests for DiscoveredPattern dataclass."""

    def test_create_pattern(self):
        """Test creating a discovered pattern."""
        pattern = DiscoveredPattern(
            category='Missing Context',
            description='Responses lacking background',
            count=5,
            record_ids=['r1', 'r2', 'r3', 'r4', 'r5'],
            examples=['Example 1', 'Example 2'],
            confidence=0.85,
        )
        assert pattern.category == 'Missing Context'
        assert pattern.count == 5
        assert len(pattern.record_ids) == 5
        assert pattern.confidence == 0.85


class TestPatternDiscoveryResult:
    """Tests for PatternDiscoveryResult dataclass."""

    def test_create_result(self):
        """Test creating a pattern discovery result."""
        patterns = [
            DiscoveredPattern(
                category='Pattern 1',
                description='Desc 1',
                count=3,
                record_ids=['r1', 'r2', 'r3'],
            ),
            DiscoveredPattern(
                category='Pattern 2',
                description='Desc 2',
                count=2,
                record_ids=['r4', 'r5'],
            ),
        ]
        result = PatternDiscoveryResult(
            patterns=patterns,
            uncategorized=['r6'],
            total_analyzed=6,
            method=ClusteringMethod.LLM,
            metadata={'model': 'gpt-4o'},
        )
        assert len(result.patterns) == 2
        assert result.uncategorized == ['r6']
        assert result.total_analyzed == 6
        assert result.method == ClusteringMethod.LLM


class TestClusteringMethod:
    """Tests for ClusteringMethod enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert ClusteringMethod.LLM.value == 'llm'
        assert ClusteringMethod.BERTOPIC.value == 'bertopic'
        assert ClusteringMethod.HYBRID.value == 'hybrid'


class TestPydanticModels:
    """Tests for Pydantic input/output models."""

    def test_annotation_note(self):
        """Test AnnotationNote model."""
        note = AnnotationNote(record_id='r1', notes='Good response')
        assert note.record_id == 'r1'
        assert note.notes == 'Good response'

    def test_clustering_input(self):
        """Test ClusteringInput model."""
        input_data = ClusteringInput(
            annotations=[
                AnnotationNote(record_id='r1', notes='Note 1'),
                AnnotationNote(record_id='r2', notes='Note 2'),
            ]
        )
        assert len(input_data.annotations) == 2

    def test_pattern_category(self):
        """Test PatternCategory model."""
        category = PatternCategory(
            category='Missing Info',
            record_ids=['r1', 'r2'],
            description='Responses missing information',
        )
        assert category.category == 'Missing Info'
        assert len(category.record_ids) == 2

    def test_clustering_output(self):
        """Test ClusteringOutput model."""
        output = ClusteringOutput(
            patterns=[
                PatternCategory(
                    category='Pattern 1',
                    record_ids=['r1'],
                    description='Desc',
                )
            ],
            uncategorized=['r2'],
        )
        assert len(output.patterns) == 1
        assert output.uncategorized == ['r2']


class TestPatternDiscovery:
    """Tests for PatternDiscovery class."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        discovery = PatternDiscovery()
        assert discovery._max_notes == 50
        assert discovery._min_category_size == 2

    def test_init_custom(self):
        """Test initialization with custom params."""
        discovery = PatternDiscovery(
            model_name='gpt-4o',
            llm_provider='openai',
            max_notes=100,
            min_category_size=3,
        )
        assert discovery._model_name == 'gpt-4o'
        assert discovery._llm_provider == 'openai'
        assert discovery._max_notes == 100
        assert discovery._min_category_size == 3

    def test_normalize_annotations_dict(self):
        """Test normalizing dict annotations."""
        discovery = PatternDiscovery()
        annotations = {
            'r1': {'record_id': 'r1', 'score': 1, 'notes': 'Good'},
            'r2': {'score': 0, 'notes': 'Bad'},
        }
        result = discovery._normalize_annotations(annotations)

        assert 'r1' in result
        assert 'r2' in result
        assert isinstance(result['r1'], AnnotatedItem)
        assert result['r1'].notes == 'Good'
        assert result['r2'].record_id == 'r2'  # Uses key as fallback

    def test_normalize_annotations_items(self):
        """Test normalizing AnnotatedItem annotations."""
        discovery = PatternDiscovery()
        annotations = {
            'r1': AnnotatedItem(record_id='r1', score=1, notes='Good'),
        }
        result = discovery._normalize_annotations(annotations)

        assert result['r1'].notes == 'Good'

    def test_normalize_annotations_invalid_type(self):
        """Test normalizing invalid annotation type."""
        discovery = PatternDiscovery()
        annotations = {'r1': 'invalid'}

        with pytest.raises(TypeError, match='Invalid annotation type'):
            discovery._normalize_annotations(annotations)

    def test_format_topic_name(self):
        """Test formatting BERTopic topic name."""
        discovery = PatternDiscovery()

        # Normal case
        topic_words = [('missing', 0.5), ('context', 0.4), ('info', 0.3), ('data', 0.2)]
        name = discovery._format_topic_name(topic_words)
        assert name == 'Missing / Context / Info'

        # Empty case
        assert discovery._format_topic_name([]) == 'Unknown Pattern'

    def test_constants(self):
        """Test class constants."""
        assert PatternDiscovery.BERTOPIC_MIN_DOCUMENTS == 5
        assert PatternDiscovery.MAX_EXAMPLES_PER_PATTERN == 3
        assert PatternDiscovery.EXAMPLE_PREVIEW_CHARS == 100


class TestPatternDiscoveryEmptyInput:
    """Tests for pattern discovery with empty/edge case inputs."""

    @pytest.mark.asyncio
    async def test_discover_no_notes(self):
        """Test discovery with no annotations having notes."""
        discovery = PatternDiscovery(model_name='gpt-4o')
        annotations = {
            'r1': AnnotatedItem(record_id='r1', score=1, notes=None),
            'r2': AnnotatedItem(record_id='r2', score=0, notes=None),
        }

        # annotations_to_evidence filters out items with no notes,
        # resulting in empty evidence → early return
        result = await discovery.discover(annotations)

        assert result.patterns == []
        assert result.metadata.get('error') == 'No items with text to cluster'


class TestLabelModels:
    """Tests for LabelInput and LabelOutput Pydantic models."""

    def test_label_input(self):
        """Test LabelInput model."""
        input_data = LabelInput(
            examples=['Missing context in response', 'Lacks background info'],
            keywords='missing, context, background, info',
        )
        assert len(input_data.examples) == 2
        assert 'missing' in input_data.keywords

    def test_label_output(self):
        """Test LabelOutput model."""
        output = LabelOutput(category_name='Missing Context')
        assert output.category_name == 'Missing Context'

    def test_label_output_alias_category(self):
        """Test LabelOutput accepts 'category' alias."""
        output = LabelOutput(category='Missing Context')
        assert output.category_name == 'Missing Context'

    def test_label_output_alias_name(self):
        """Test LabelOutput accepts 'name' alias."""
        output = LabelOutput(name='Missing Context')
        assert output.category_name == 'Missing Context'

    def test_label_output_alias_label(self):
        """Test LabelOutput accepts 'label' alias."""
        output = LabelOutput(label='Missing Context')
        assert output.category_name == 'Missing Context'


@requires_bertopic
class TestBERTopicClustering:
    """Tests for BERTopic clustering method."""

    @pytest.mark.asyncio
    async def test_bertopic_too_few_documents(self):
        """Test BERTopic with too few documents returns early."""
        discovery = PatternDiscovery()

        # Only 3 items - below BERTOPIC_MIN_DOCUMENTS (5)
        annotations = {
            'r1': AnnotatedItem(record_id='r1', score=0, notes='Missing context'),
            'r2': AnnotatedItem(record_id='r2', score=0, notes='Lacks detail'),
            'r3': AnnotatedItem(record_id='r3', score=1, notes='Good response'),
        }

        result = await discovery.discover(annotations, method=ClusteringMethod.BERTOPIC)

        assert result.patterns == []
        assert set(result.uncategorized) == {'r1', 'r2', 'r3'}
        assert result.method == ClusteringMethod.BERTOPIC
        assert 'Too few documents' in result.metadata.get('error', '')

    @pytest.mark.asyncio
    async def test_bertopic_no_notes(self):
        """Test BERTopic with no notes returns early."""
        discovery = PatternDiscovery()

        annotations = {
            'r1': AnnotatedItem(record_id='r1', score=0, notes=None),
            'r2': AnnotatedItem(record_id='r2', score=1, notes=None),
        }

        result = await discovery.discover(annotations, method=ClusteringMethod.BERTOPIC)

        assert result.patterns == []
        assert 'Too few documents' in result.metadata.get('error', '')

    def test_bertopic_embedding_model_config(self):
        """Test BERTopic embedding model can be configured."""
        discovery = PatternDiscovery(bertopic_embedding_model='paraphrase-MiniLM-L6-v2')
        assert discovery._bertopic_embedding_model == 'paraphrase-MiniLM-L6-v2'


@requires_bertopic
class TestHybridClustering:
    """Tests for Hybrid clustering method."""

    @pytest.mark.asyncio
    async def test_hybrid_too_few_documents(self):
        """Test Hybrid with too few documents returns early with correct method."""
        # small_corpus_llm_fallback defaults on (small corpora route to LLM); disable it
        # to assert the too-few-documents early-out contract.
        discovery = PatternDiscovery(
            model_name='gpt-4o', small_corpus_llm_fallback=False
        )

        # Only 3 items - below BERTOPIC_MIN_DOCUMENTS
        annotations = {
            'r1': AnnotatedItem(record_id='r1', score=0, notes='Missing context'),
            'r2': AnnotatedItem(record_id='r2', score=0, notes='Lacks detail'),
            'r3': AnnotatedItem(record_id='r3', score=1, notes='Good response'),
        }

        result = await discovery.discover(annotations, method=ClusteringMethod.HYBRID)

        # Should still return HYBRID method even though it fell back
        assert result.method == ClusteringMethod.HYBRID
        assert result.patterns == []

    def test_label_handler_lazy_init(self):
        """Test that label handler is lazily initialized."""
        discovery = PatternDiscovery(model_name='gpt-4o')
        assert discovery._label_handler is None
        # Handler created on first access (not testing actual call to avoid API)


class TestBertopicSmallCorpusResilience:
    """BERTopic's default UMAP crashes on small corpora; these pin the resilience fix."""

    def test_build_umap_clamps_for_small_n(self):
        discovery = PatternDiscovery()
        # At/below the boundary → a UMAP clamped to the corpus size.
        umap = discovery._build_umap(11)
        if umap is None:  # umap-learn not installed alongside bertopic
            pytest.skip('umap not installed')
        assert umap.n_neighbors == 10  # min(15, 11 - 1)
        assert umap.n_components == 5  # min(5, 11 - 2)

    def test_boundary_tracks_configured_n_neighbors(self):
        """The small-corpus boundary derives from n_neighbors, not a hardcoded 15."""
        # A caller that raises n_neighbors moves the boundary with it.
        discovery = PatternDiscovery(bertopic_n_neighbors=30)
        assert discovery._bertopic_n_neighbors == 30
        # 25 docs is safe under the default 15 but below a custom 30 → must clamp.
        umap = discovery._build_umap(25)
        if umap is None:
            pytest.skip('umap not installed')
        assert umap.n_neighbors == 24  # min(30, 25 - 1)

    def test_custom_n_neighbors_honored_on_large_corpus(self):
        """A non-default n_neighbors is applied even on large corpora (never a no-op)."""
        discovery = PatternDiscovery(bertopic_n_neighbors=8)
        umap = discovery._build_umap(500)
        if umap is None:
            pytest.skip('umap not installed')
        assert umap.n_neighbors == 8  # min(8, 500 - 1)

    def test_default_large_corpus_seeded(self):
        """Default n_neighbors + large corpus → a UMAP mirroring BERTopic's defaults but
        seeded. BERTopic's own default UMAP has random_state=None → stochastic clusters
        run-to-run; this pins the determinism fix."""
        discovery = PatternDiscovery()  # default n_neighbors=15, no seed
        umap = discovery._build_umap(500)
        if umap is None:
            pytest.skip('umap not installed')
        assert umap.n_neighbors == 15
        assert umap.n_components == 5
        assert umap.min_dist == 0.0
        assert umap.metric == 'cosine'
        assert umap.random_state == 42  # unseeded discovery still deterministic

    def test_umap_uses_configured_seed(self):
        discovery = PatternDiscovery(seed=7)
        umap = discovery._build_umap(500)
        if umap is None:
            pytest.skip('umap not installed')
        assert umap.random_state == 7

    def test_build_small_corpus_umap_alias(self):
        """Back-compat: the old method name still resolves to the new builder."""
        discovery = PatternDiscovery()
        assert PatternDiscovery._build_small_corpus_umap is PatternDiscovery._build_umap
        umap = discovery._build_small_corpus_umap(11)
        if umap is None:
            pytest.skip('umap not installed')
        assert umap.n_neighbors == 10

    @requires_bertopic
    @pytest.mark.asyncio
    async def test_bertopic_small_corpus_does_not_crash(self):
        """11 short docs used to raise 'Found array with 0 sample(s)'; now it clusters."""
        from axion.caliber.pattern_discovery import EvidenceItem

        texts = [
            'Landscaping risk with tree service exposure needs higher-hazard review.',
            'Home-based architecture consulting approved automatically for BPP coverage.',
            'Liquor store with 90% alcohol sales must not auto-decline.',
            'Apartment building frame construction over 1.5M exposure referred.',
            'Class code mismatch between declared DBA and observed operations.',
            'Retail NOC home-based risk needs actual operations verification.',
            'Chemical manufacturing exposure flagged for subcontractor work.',
            'Convenience store with fuel sales requires environmental review.',
            'Contractor landscape gardening with snow removal add-on service.',
            'Restaurant with liquor license and live entertainment exposure.',
            'Auto repair shop with paint booth fire hazard classification.',
        ]
        evidence = {
            f'e{i}': EvidenceItem(id=f'e{i}', text=t) for i, t in enumerate(texts)
        }

        discovery = PatternDiscovery(min_category_size=2)
        result = await discovery._cluster_evidence_with_bertopic(evidence)

        assert result.method == ClusteringMethod.BERTOPIC
        assert result.metadata.get('error') is None
        assert result.patterns  # small-but-valid corpus produced at least one topic

    @pytest.mark.asyncio
    async def test_hybrid_falls_back_to_llm_on_bertopic_failure(self, monkeypatch):
        """A BERTopic failure (not a too-few early-out) routes HYBRID to LLM clustering."""
        # Opt out of the small-corpus short-circuit to exercise the bertopic-failure path.
        discovery = PatternDiscovery(
            model_name='gpt-4o', small_corpus_llm_fallback=False
        )
        evidence = {'e1': EvidenceItem(id='e1', text='some note')}

        async def _failed_bertopic(_ev):
            return PatternDiscoveryResult(
                patterns=[],
                uncategorized=['e1'],
                total_analyzed=1,
                method=ClusteringMethod.BERTOPIC,
                metadata={'error': 'boom', 'reason': 'bertopic_failed'},
            )

        async def _llm(_ev):
            return PatternDiscoveryResult(
                patterns=[
                    DiscoveredPattern(
                        category='c', description='d', count=1, record_ids=['e1']
                    )
                ],
                uncategorized=[],
                total_analyzed=1,
                method=ClusteringMethod.LLM,
                metadata={},
            )

        monkeypatch.setattr(
            discovery, '_cluster_evidence_with_bertopic', _failed_bertopic
        )
        monkeypatch.setattr(discovery, '_cluster_evidence_with_llm', _llm)

        result = await discovery._cluster_evidence_hybrid(evidence)
        assert result.method == ClusteringMethod.HYBRID
        assert len(result.patterns) == 1  # served by the LLM fallback
        assert result.metadata['bertopic_fallback'] == 'boom'

    @pytest.mark.asyncio
    async def test_hybrid_too_few_documents_stays_empty(self, monkeypatch):
        """A deliberate too-few-documents early-out is NOT sent to the LLM fallback."""
        # With the small-corpus short-circuit off, the too-few contract still holds.
        discovery = PatternDiscovery(
            model_name='gpt-4o', small_corpus_llm_fallback=False
        )
        evidence = {'e1': EvidenceItem(id='e1', text='some note')}

        async def _too_few(_ev):
            return PatternDiscoveryResult(
                patterns=[],
                uncategorized=['e1'],
                total_analyzed=1,
                method=ClusteringMethod.BERTOPIC,
                metadata={'error': 'too few', 'reason': 'too_few_documents'},
            )

        async def _llm(_ev):  # pragma: no cover - must not be called
            raise AssertionError('LLM fallback must not run for too_few_documents')

        monkeypatch.setattr(discovery, '_cluster_evidence_with_bertopic', _too_few)
        monkeypatch.setattr(discovery, '_cluster_evidence_with_llm', _llm)

        result = await discovery._cluster_evidence_hybrid(evidence)
        assert result.method == ClusteringMethod.HYBRID
        assert result.patterns == []

    @pytest.mark.asyncio
    async def test_hybrid_small_corpus_routes_to_llm(self, monkeypatch):
        """With the fallback on (default), a corpus <= n_neighbors skips BERTopic."""
        discovery = PatternDiscovery(model_name='gpt-4o')  # fallback on, n_neighbors=15
        evidence = {
            f'e{i}': EvidenceItem(id=f'e{i}', text=f'note {i}') for i in range(10)
        }

        async def _bertopic(_ev):  # pragma: no cover - must not be called
            raise AssertionError('BERTopic must not run for a small HYBRID corpus')

        async def _llm(ev):
            return PatternDiscoveryResult(
                patterns=[
                    DiscoveredPattern(
                        category='c', description='d', count=10, record_ids=list(ev)
                    )
                ],
                uncategorized=[],
                total_analyzed=10,
                method=ClusteringMethod.LLM,
                metadata={},
            )

        monkeypatch.setattr(discovery, '_cluster_evidence_with_bertopic', _bertopic)
        monkeypatch.setattr(discovery, '_cluster_evidence_with_llm', _llm)

        result = await discovery._cluster_evidence_hybrid(evidence)
        assert result.method == ClusteringMethod.HYBRID
        assert result.metadata['small_corpus_llm_fallback'] == 10
        assert len(result.patterns) == 1

    @pytest.mark.asyncio
    async def test_hybrid_large_corpus_runs_bertopic(self, monkeypatch):
        """Above the boundary the short-circuit does not fire — BERTopic is attempted."""
        discovery = PatternDiscovery(model_name='gpt-4o')  # n_neighbors=15
        evidence = {
            f'e{i}': EvidenceItem(id=f'e{i}', text=f'note {i}') for i in range(20)
        }
        entered = {'bertopic': False}

        async def _bertopic(ev):
            entered['bertopic'] = True  # short-circuit was skipped
            return PatternDiscoveryResult(
                patterns=[],
                uncategorized=list(ev),
                total_analyzed=20,
                method=ClusteringMethod.BERTOPIC,
                metadata={'error': 'boom', 'reason': 'bertopic_failed'},
            )

        async def _llm(ev):
            return PatternDiscoveryResult(
                patterns=[
                    DiscoveredPattern(
                        category='c', description='d', count=20, record_ids=list(ev)
                    )
                ],
                uncategorized=[],
                total_analyzed=20,
                method=ClusteringMethod.LLM,
                metadata={},
            )

        monkeypatch.setattr(discovery, '_cluster_evidence_with_bertopic', _bertopic)
        monkeypatch.setattr(discovery, '_cluster_evidence_with_llm', _llm)

        result = await discovery._cluster_evidence_hybrid(evidence)
        assert entered['bertopic'] is True
        assert 'small_corpus_llm_fallback' not in result.metadata
        assert (
            result.metadata['bertopic_fallback'] == 'boom'
        )  # bertopic-failure fallback


class TestPatternDiscoveryMethodDispatch:
    """Tests for method dispatch in discover()."""

    @pytest.mark.asyncio
    async def test_discover_dispatches_to_llm(self):
        """Test discover() dispatches to LLM method."""
        discovery = PatternDiscovery(model_name='gpt-4o')
        annotations = {
            'r1': AnnotatedItem(record_id='r1', score=0, notes=None),
        }

        # With no notes, LLM method returns early
        result = await discovery.discover(annotations, method=ClusteringMethod.LLM)
        assert result.method == ClusteringMethod.LLM

    @requires_bertopic
    @pytest.mark.asyncio
    async def test_discover_dispatches_to_bertopic(self):
        """Test discover() dispatches to BERTopic method."""
        discovery = PatternDiscovery()
        annotations = {
            'r1': AnnotatedItem(record_id='r1', score=0, notes='Note 1'),
            'r2': AnnotatedItem(record_id='r2', score=0, notes='Note 2'),
        }

        # Too few docs, returns early
        result = await discovery.discover(annotations, method=ClusteringMethod.BERTOPIC)
        assert result.method == ClusteringMethod.BERTOPIC

    @requires_bertopic
    @pytest.mark.asyncio
    async def test_discover_dispatches_to_hybrid(self):
        """Test discover() dispatches to Hybrid method."""
        discovery = PatternDiscovery(model_name='gpt-4o')
        annotations = {
            'r1': AnnotatedItem(record_id='r1', score=0, notes='Note 1'),
            'r2': AnnotatedItem(record_id='r2', score=0, notes='Note 2'),
        }

        # Too few docs, returns early but with HYBRID method
        result = await discovery.discover(annotations, method=ClusteringMethod.HYBRID)
        assert result.method == ClusteringMethod.HYBRID

    @pytest.mark.asyncio
    async def test_discover_invalid_method(self):
        """Test discover() raises on invalid method."""
        discovery = PatternDiscovery()
        annotations = {'r1': AnnotatedItem(record_id='r1', score=0, notes='Note')}

        with pytest.raises(ValueError, match='Unknown clustering method'):
            await discovery.discover(annotations, method='invalid')
