"""Tests for ObservationNode and Trace tree building."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest

from axion._core.tracing.collection.models import ObservationsView
from axion._core.tracing.collection.observation_node import ObservationNode
from axion._core.tracing.collection.trace import Trace

# ---------------------------------------------------------------------------
# Fake trace data helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeObservation:
    id: str = 'obs-1'
    name: str = 'step1'
    type: str = 'SPAN'
    parent_observation_id: Optional[str] = None
    start_time: Any = None
    end_time: Any = None
    input: Any = ''
    output: Any = ''
    metadata: Optional[Dict] = None


@dataclass
class FakeRawTrace:
    id: str = 'trace-1'
    name: str = 'my-trace'
    input: Any = None
    output: Any = None
    observations: List[Any] = field(default_factory=list)
    latency: float = 0.5
    tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TestObservationNodeBasic
# ---------------------------------------------------------------------------


class TestObservationNodeBasic:
    def test_construction(self):
        obs = ObservationsView(id='o1', name='step1', type='SPAN')
        node = ObservationNode(obs)
        assert node.observation is obs
        assert node.parent is None
        assert node.children == []
        assert node.is_root is True
        assert node.is_leaf is True
        assert node.depth == 0

    def test_smart_access_delegation(self):
        obs = ObservationsView(id='o1', name='step1', type='GENERATION', input='hello')
        node = ObservationNode(obs)
        assert node.name == 'step1'
        assert node.type == 'GENERATION'
        assert node.input == 'hello'

    def test_parent_child_wiring(self):
        parent_obs = ObservationsView(id='p', name='parent', type='SPAN')
        child_obs = ObservationsView(id='c', name='child', type='GENERATION')
        parent = ObservationNode(parent_obs)
        child = ObservationNode(child_obs)

        parent._add_child(child)

        assert child.parent is parent
        assert child in parent.children
        assert parent.is_root is True
        assert parent.is_leaf is False
        assert child.is_root is False
        assert child.is_leaf is True

    def test_depth(self):
        nodes = []
        for i in range(4):
            obs = ObservationsView(id=f'n{i}', name=f'level{i}', type='SPAN')
            nodes.append(ObservationNode(obs))
        nodes[0]._add_child(nodes[1])
        nodes[1]._add_child(nodes[2])
        nodes[2]._add_child(nodes[3])

        assert nodes[0].depth == 0
        assert nodes[1].depth == 1
        assert nodes[2].depth == 2
        assert nodes[3].depth == 3

    def test_repr(self):
        obs = ObservationsView(id='o1', name='my-step', type='SPAN')
        node = ObservationNode(obs)
        r = repr(node)
        assert 'ObservationNode' in r
        assert 'my-step' in r
        assert 'SPAN' in r
        assert 'children=0' in r


# ---------------------------------------------------------------------------
# TestObservationNodeTime
# ---------------------------------------------------------------------------


class TestObservationNodeTime:
    def test_datetime_objects(self):
        t1 = datetime(2025, 1, 1, 12, 0, 0)
        t2 = datetime(2025, 1, 1, 12, 0, 5)
        obs = ObservationsView(
            id='o1', name='s', type='SPAN', start_time=t1, end_time=t2
        )
        node = ObservationNode(obs)
        assert node.start_time == t1
        assert node.end_time == t2
        assert node.duration == timedelta(seconds=5)

    def test_iso_string_timestamps(self):
        obs = ObservationsView(
            id='o1',
            name='s',
            type='SPAN',
            start_time='2025-01-01T12:00:00',
            end_time='2025-01-01T12:00:10',
        )
        node = ObservationNode(obs)
        assert node.start_time == datetime(2025, 1, 1, 12, 0, 0)
        assert node.end_time == datetime(2025, 1, 1, 12, 0, 10)
        assert node.duration == timedelta(seconds=10)

    def test_missing_timestamps(self):
        obs = ObservationsView(id='o1', name='s', type='SPAN')
        node = ObservationNode(obs)
        assert node.start_time is None
        assert node.end_time is None
        assert node.duration is None

    def test_partial_timestamps(self):
        t1 = datetime(2025, 1, 1, 12, 0, 0)
        obs = ObservationsView(id='o1', name='s', type='SPAN', start_time=t1)
        node = ObservationNode(obs)
        assert node.start_time == t1
        assert node.end_time is None
        assert node.duration is None


# ---------------------------------------------------------------------------
# TestObservationNodeTraversal
# ---------------------------------------------------------------------------


class TestObservationNodeTraversal:
    def test_walk_single_node(self):
        obs = ObservationsView(id='o1', name='root', type='SPAN')
        node = ObservationNode(obs)
        walked = list(node.walk())
        assert walked == [node]

    def test_walk_preorder(self):
        """
        Build:
            root
            ├── child1
            │   └── grandchild
            └── child2
        Expect pre-order: root, child1, grandchild, child2
        """
        root_obs = ObservationsView(id='r', name='root', type='SPAN')
        c1_obs = ObservationsView(id='c1', name='child1', type='SPAN')
        c2_obs = ObservationsView(id='c2', name='child2', type='SPAN')
        gc_obs = ObservationsView(id='gc', name='grandchild', type='GENERATION')

        root = ObservationNode(root_obs)
        c1 = ObservationNode(c1_obs)
        c2 = ObservationNode(c2_obs)
        gc = ObservationNode(gc_obs)

        root._add_child(c1)
        root._add_child(c2)
        c1._add_child(gc)

        walked = list(root.walk())
        assert [n.id for n in walked] == ['r', 'c1', 'gc', 'c2']


# ---------------------------------------------------------------------------
# TestTreeBuildFromTrace
# ---------------------------------------------------------------------------


class TestTreeBuildFromTrace:
    def _make_hierarchy_trace(self):
        """
        Simulates a trace like:
            root-span (id=r)
            ├── child-span (id=c1, parent=r, start=12:00:01)
            │   └── generation (id=g1, parent=c1, start=12:00:02)
            └── child-span2 (id=c2, parent=r, start=12:00:03)
        """
        raw = FakeRawTrace(
            id='trace-1',
            observations=[
                FakeObservation(
                    id='r',
                    name='root-span',
                    type='SPAN',
                    start_time=datetime(2025, 1, 1, 12, 0, 0),
                ),
                FakeObservation(
                    id='c1',
                    name='child-span',
                    type='SPAN',
                    parent_observation_id='r',
                    start_time=datetime(2025, 1, 1, 12, 0, 1),
                ),
                FakeObservation(
                    id='g1',
                    name='generation',
                    type='GENERATION',
                    parent_observation_id='c1',
                    start_time=datetime(2025, 1, 1, 12, 0, 2),
                ),
                FakeObservation(
                    id='c2',
                    name='child-span2',
                    type='SPAN',
                    parent_observation_id='r',
                    start_time=datetime(2025, 1, 1, 12, 0, 3),
                ),
            ],
        )
        return Trace(raw)

    def test_single_root_hierarchy(self):
        t = self._make_hierarchy_trace()
        roots = t.tree_roots
        assert len(roots) == 1
        root = roots[0]
        assert root.id == 'r'
        assert len(root.children) == 2
        assert root.children[0].id == 'c1'
        assert root.children[1].id == 'c2'
        assert len(root.children[0].children) == 1
        assert root.children[0].children[0].id == 'g1'

    def test_children_sorted_by_start_time(self):
        """Children with earlier start_time come first."""
        raw = FakeRawTrace(
            observations=[
                FakeObservation(
                    id='r',
                    name='root',
                    type='SPAN',
                    start_time=datetime(2025, 1, 1, 12, 0, 0),
                ),
                FakeObservation(
                    id='late',
                    name='late',
                    type='SPAN',
                    parent_observation_id='r',
                    start_time=datetime(2025, 1, 1, 12, 0, 10),
                ),
                FakeObservation(
                    id='early',
                    name='early',
                    type='SPAN',
                    parent_observation_id='r',
                    start_time=datetime(2025, 1, 1, 12, 0, 1),
                ),
            ],
        )
        t = Trace(raw)
        root = t.tree
        assert root.children[0].id == 'early'
        assert root.children[1].id == 'late'

    def test_caching(self):
        t = self._make_hierarchy_trace()
        roots1 = t.tree_roots
        roots2 = t.tree_roots
        assert roots1 is roots2

    def test_coexistence_with_steps(self):
        t = self._make_hierarchy_trace()
        # Steps still work (name-grouped)
        assert 'root-span' in t.step_names
        assert 'child-span' in t.step_names
        # Tree also works
        assert len(t.tree_roots) == 1

    def test_tree_returns_none_when_multi_root(self):
        raw = FakeRawTrace(
            observations=[
                FakeObservation(id='a', name='a', type='SPAN'),
                FakeObservation(id='b', name='b', type='SPAN'),
            ],
        )
        t = Trace(raw)
        assert len(t.tree_roots) == 2
        assert t.tree is None

    def test_tree_returns_root_when_single_root(self):
        t = self._make_hierarchy_trace()
        assert t.tree is not None
        assert t.tree.id == 'r'


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestTreeEdgeCases:
    def test_missing_parent_observation_id(self):
        """Observations without parent_observation_id are roots."""
        raw = FakeRawTrace(
            observations=[
                FakeObservation(
                    id='a', name='a', type='SPAN', parent_observation_id=None
                ),
                FakeObservation(
                    id='b', name='b', type='SPAN', parent_observation_id=None
                ),
            ],
        )
        t = Trace(raw)
        assert len(t.tree_roots) == 2

    def test_unknown_parent_observation_id(self):
        """Parent ID not in our set -> treated as root."""
        raw = FakeRawTrace(
            observations=[
                FakeObservation(
                    id='a', name='a', type='SPAN', parent_observation_id='nonexistent'
                ),
            ],
        )
        t = Trace(raw)
        assert len(t.tree_roots) == 1
        assert t.tree_roots[0].is_root is True

    def test_self_referencing_parent(self):
        """parent_observation_id == own id -> treated as root, no cycle."""
        raw = FakeRawTrace(
            observations=[
                FakeObservation(
                    id='a', name='a', type='SPAN', parent_observation_id='a'
                ),
            ],
        )
        t = Trace(raw)
        assert len(t.tree_roots) == 1
        assert t.tree_roots[0].is_root is True
        assert t.tree_roots[0].children == []

    def test_duplicate_observation_ids(self):
        """First occurrence wins, duplicate skipped."""
        raw = FakeRawTrace(
            observations=[
                FakeObservation(id='dup', name='first', type='SPAN'),
                FakeObservation(id='dup', name='second', type='GENERATION'),
            ],
        )
        t = Trace(raw)
        assert len(t.tree_roots) == 1
        assert t.tree_roots[0].name == 'first'

    def test_observations_with_no_id(self):
        """Observations without id get synthetic id and appear as roots."""
        obs1 = ObservationsView(name='no-id-1', type='SPAN')
        obs2 = ObservationsView(name='no-id-2', type='GENERATION')
        t = Trace([obs1, obs2])
        assert len(t.tree_roots) == 2

    def test_identical_start_times_stable_order(self):
        """When start_time is the same, original index determines order."""
        same_time = datetime(2025, 1, 1, 12, 0, 0)
        raw = FakeRawTrace(
            observations=[
                FakeObservation(id='r', name='root', type='SPAN', start_time=same_time),
                FakeObservation(
                    id='first',
                    name='first',
                    type='SPAN',
                    parent_observation_id='r',
                    start_time=same_time,
                ),
                FakeObservation(
                    id='second',
                    name='second',
                    type='SPAN',
                    parent_observation_id='r',
                    start_time=same_time,
                ),
            ],
        )
        t = Trace(raw)
        root = t.tree
        assert root.children[0].id == 'first'
        assert root.children[1].id == 'second'

    def test_empty_observations(self):
        """Empty observations -> tree_roots returns [], tree returns None."""
        raw = FakeRawTrace(observations=[])
        t = Trace(raw)
        assert t.tree_roots == []
        assert t.tree is None

    def test_dict_observations(self):
        """Dict-based observations work with tree building."""
        data = {
            'id': 'trace-dict',
            'observations': [
                {
                    'id': 'r',
                    'name': 'root',
                    'type': 'SPAN',
                    'parent_observation_id': None,
                    'start_time': '2025-01-01T12:00:00',
                },
                {
                    'id': 'c',
                    'name': 'child',
                    'type': 'GENERATION',
                    'parent_observation_id': 'r',
                    'start_time': '2025-01-01T12:00:01',
                },
            ],
        }
        t = Trace(data)
        assert len(t.tree_roots) == 1
        root = t.tree
        assert root.id == 'r'
        assert len(root.children) == 1
        assert root.children[0].id == 'c'


# ---------------------------------------------------------------------------
# TestObservationNodeSearch
# ---------------------------------------------------------------------------


class TestObservationNodeSearch:
    """Tests for find(), bracket access, iteration, len, and contains."""

    def _build_tree(self):
        """
        Build:
            root (SPAN)
            ├── child1 (SPAN)
            │   └── grandchild (GENERATION)
            └── child2 (GENERATION)
        """
        root = ObservationNode(ObservationsView(id='r', name='root', type='SPAN'))
        c1 = ObservationNode(ObservationsView(id='c1', name='child1', type='SPAN'))
        c2 = ObservationNode(
            ObservationsView(id='c2', name='child2', type='GENERATION')
        )
        gc = ObservationNode(
            ObservationsView(id='gc', name='grandchild', type='GENERATION')
        )
        root._add_child(c1)
        root._add_child(c2)
        c1._add_child(gc)
        return root, c1, c2, gc

    # -- find() -------------------------------------------------------------

    def test_find_by_name(self):
        root, c1, c2, gc = self._build_tree()
        assert root.find(name='grandchild') is gc

    def test_find_by_type(self):
        root, c1, c2, gc = self._build_tree()
        # First GENERATION in pre-order is grandchild (child1 -> grandchild before child2)
        assert root.find(type='GENERATION') is gc

    def test_find_by_name_and_type(self):
        root, c1, c2, gc = self._build_tree()
        assert root.find(name='child2', type='GENERATION') is c2

    def test_find_no_match(self):
        root, c1, c2, gc = self._build_tree()
        assert root.find(name='nonexistent') is None

    def test_find_no_match_wrong_type(self):
        root, c1, c2, gc = self._build_tree()
        assert root.find(name='child1', type='GENERATION') is None

    # -- __getitem__ (bracket access) ---------------------------------------

    def test_bracket_finds_descendant_by_name(self):
        root, c1, c2, gc = self._build_tree()
        assert root['grandchild'] is gc

    def test_bracket_falls_back_to_observation_field(self):
        root, c1, c2, gc = self._build_tree()
        assert root['id'] == 'r'

    def test_bracket_raises_key_error(self):
        root, c1, c2, gc = self._build_tree()
        with pytest.raises(KeyError):
            root['nonexistent']

    # -- __iter__ -----------------------------------------------------------

    def test_iter_direct_children(self):
        root, c1, c2, gc = self._build_tree()
        children = list(root)
        assert children == [c1, c2]

    def test_iter_leaf_node(self):
        root, c1, c2, gc = self._build_tree()
        assert list(gc) == []

    # -- __len__ ------------------------------------------------------------

    def test_len_returns_child_count(self):
        root, c1, c2, gc = self._build_tree()
        assert len(root) == 2
        assert len(c1) == 1
        assert len(gc) == 0

    # -- __contains__ -------------------------------------------------------

    def test_contains_finds_in_subtree(self):
        root, c1, c2, gc = self._build_tree()
        assert 'grandchild' in root
        assert 'child2' in root

    def test_contains_not_found(self):
        root, c1, c2, gc = self._build_tree()
        assert 'nonexistent' not in root

    def test_contains_does_not_match_self(self):
        root, c1, c2, gc = self._build_tree()
        assert 'root' not in root


# ---------------------------------------------------------------------------
# TraceCollection.trees
# ---------------------------------------------------------------------------


class TestTraceCollectionTrees:
    def test_trees_property(self):
        from axion._core.tracing.collection.trace_collection import TraceCollection

        data = [
            FakeRawTrace(
                observations=[
                    FakeObservation(id='r1', name='root', type='SPAN'),
                    FakeObservation(
                        id='c1', name='child', type='SPAN', parent_observation_id='r1'
                    ),
                ],
            ),
            FakeRawTrace(
                observations=[
                    FakeObservation(id='r2', name='root', type='SPAN'),
                ],
            ),
        ]
        coll = TraceCollection(data)
        trees = coll.trees
        assert len(trees) == 2
        assert len(trees[0]) == 1  # single root for first trace
        assert len(trees[1]) == 1  # single root for second trace
        assert trees[0][0].children[0].id == 'c1'
