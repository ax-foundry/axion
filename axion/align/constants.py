# Model and provider options
from axion._handlers.llm.models import structured_outputs_models

MODEL_OPTIONS = {'llm_gateway': structured_outputs_models}


DEFAULT_INSTRUCTION = """You are an expert evaluator. Assess if the answer fully and directly addresses the user query.
Score 1 if it is both relevant and complete, otherwise 0.
Explain your reasoning.
"""

BEST_PRACTICES = """
# AlignEval Best Practices for LLM-as-a-Judge Calibration

## 1. Start with Data, Not Assumptions

**Work backward from real outputs, not forward from theory.**

❌ **Don't do this:**
- Write criteria based on what you think could go wrong
- Focus on generic metrics like "helpfulness" or "clarity"
- Start with complex rubrics before seeing your data

✅ **Do this instead:**
- Upload 50-100 real production samples
- Label pass/fail while studying actual model behaviors
- Look for patterns: *What actually breaks? How often?*

**Example:** You think grammar errors are a problem, but after labeling 100 samples, you find only 2 grammar issues but 15 cases of missing context. Focus your evaluator on context completeness, not grammar.

## 2. Calibrate Through Evaluation Loops

**Align human judgment with AI evaluation iteratively.**

**The process:**
1. **Label** → Human judgment on 50+ samples
2. **Evaluate** → Test LLM-as-judge against your labels
3. **Analyze** → Study false positives/negatives
4. **Refine** → Update criteria based on misalignments
5. **Optimize** → Use AlignEval's auto-improvement
6. **Repeat** → Continuous calibration

**Target metrics:** F1 > 0.8, but prioritize catching failures that impact business outcomes over perfect scores.

## 3. Scale with Domain Awareness

**Different tasks need different evaluators.**

**Domain-specific approach:**
- **Customer Support:** Focus on accuracy + tone
- **Content Generation:** Emphasize creativity + brand alignment
- **Code Review:** Prioritize functionality + security

**Scaling strategies:**
- Build separate evaluators per domain
- Ensemble 3 smaller models > 1 large model
- Monitor performance drift on new production data
- Maintain dedicated datasets for each use case

**Remember:** The goal isn't perfect evaluation—it's continuous improvement aligned with what matters to your business.
"""
