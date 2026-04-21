from typing import List

def rank_candidates(candidates: List[str]) -> str:
    """
    Given a list of model-generated drafts, pick the best one.
    For v0.1, we just return the first one available.
    """
    if not candidates:
        return ""
    return candidates[0]
