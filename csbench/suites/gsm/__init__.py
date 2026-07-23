"""GSM boundary suite: a DIAGNOSTIC, not a flagship result.

This suite exists to demonstrate two disclosed BOUNDARY cases for context
selection, NOT to headline a win:

  1. GSM8K -- clean, single-document math word problems. There is nothing to
     select away, so record-level selection is a NO-OP BY DESIGN: the arm
     correctly preserves the whole prompt.
  2. GSM-IC -- in-context distractors injected WITHIN a single record. These
     are OUTSIDE the selection regime, because record-level selection cannot
     remove a distractor sentence that lives inside the one record it must
     keep.

Flagship claims live with the RULER and the tool/document suites, not here.
"""
