"""Tests for PromptPatternsBase and create_extraction_pattern."""

import re

from axion._core.tracing.collection.prompt_patterns import (
    PromptPatternsBase,
    create_extraction_pattern,
)

# ---------------------------------------------------------------------------
# create_extraction_pattern
# ---------------------------------------------------------------------------


class TestCreateExtractionPattern:
    def test_basic_pattern(self):
        pattern = create_extraction_pattern('Assessment', r'(?:Recommendation|$)')
        text = 'Assessment: patient is stable Recommendation: continue'
        match = re.search(pattern, text, re.DOTALL)
        assert match is not None
        assert match.group(1).strip() == 'patient is stable'

    def test_special_chars_escaped(self):
        pattern = create_extraction_pattern('Risk (Score)', r'(?:End|$)')
        text = 'Risk (Score): 7/10 End'
        match = re.search(pattern, text, re.DOTALL)
        assert match is not None
        assert '7/10' in match.group(1)

    def test_no_match(self):
        pattern = create_extraction_pattern('Missing Label', r'(?:End|$)')
        text = 'Something else entirely'
        match = re.search(pattern, text, re.DOTALL)
        assert match is None


# ---------------------------------------------------------------------------
# PromptPatternsBase
# ---------------------------------------------------------------------------


class TestPromptPatternsBase:
    def test_base_returns_empty(self):
        result = PromptPatternsBase.get_for('anything')
        assert result == {}

    def test_subclass_discovery(self):
        class MyPatterns(PromptPatternsBase):
            @classmethod
            def _patterns_recommendation(cls):
                return {'key': 'pattern'}

        result = MyPatterns.get_for('recommendation')
        assert result == {'key': 'pattern'}

    def test_hyphenated_step_name(self):
        class MyPatterns(PromptPatternsBase):
            @classmethod
            def _patterns_location_extraction(cls):
                return {'loc': 'pat'}

        result = MyPatterns.get_for('location-extraction')
        assert result == {'loc': 'pat'}

    def test_special_chars_in_step_name(self):
        class MyPatterns(PromptPatternsBase):
            @classmethod
            def _patterns_step_v2(cls):
                return {'v': '2'}

        result = MyPatterns.get_for('step.v2')
        assert result == {'v': '2'}

    def test_no_match_returns_empty(self):
        class MyPatterns(PromptPatternsBase):
            @classmethod
            def _patterns_known(cls):
                return {'k': 'v'}

        result = MyPatterns.get_for('unknown')
        assert result == {}
