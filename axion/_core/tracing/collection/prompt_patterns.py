from __future__ import annotations

import re
from typing import Dict



def create_extraction_pattern(start_text: str, end_pattern: str) -> str:
    r"""
    Build a regex that captures text between a labelled start and a terminating pattern.

    Example::

        pattern = create_extraction_pattern('Assessment', r'(?:Recommendation|$)')
        # => r'Assessment:\s*(.*?)\s*(?:Recommendation|$)'
    """
    return rf'{re.escape(start_text)}:\s*(.*?)\s*(?:{end_pattern})'


class PromptPatternsBase:
    """
    Base registry for regex extraction patterns.

    Subclass and define ``_patterns_<step_name>()`` class methods that return
    ``Dict[str, str]`` mapping variable names to regex patterns.

    Example::

        class MyPatterns(PromptPatternsBase):
            @classmethod
            def _patterns_recommendation(cls) -> Dict[str, str]:
                return {
                    'assessment': create_extraction_pattern(
                        'Case Assessment', r'(?:Recommendation|$)'
                    ),
                }
    """

    @classmethod
    def get_for(cls, step_name: str) -> Dict[str, str]:
        """
        Look up extraction patterns for *step_name*.

        Normalizes the step name to find the matching ``_patterns_*`` method
        (handles hyphens, special characters, etc.).
        """
        raw = step_name.lower()
        candidates = [
            raw,
            raw.replace('-', '_'),
            re.sub(r'[^a-z0-9_]+', '_', raw).strip('_'),
            re.sub(r'[^a-z0-9]+', '', raw),
        ]
        for candidate in candidates:
            method_name = f'_patterns_{candidate}'
            if hasattr(cls, method_name):
                return getattr(cls, method_name)()
        return {}
