def should_escalate(draft_attempt: str, error_count: int) -> bool:
    """
    Decide if we should escalate the task to a more expensive / larger model.
    Heuristics:
    - Multiple failures to generate valid text.
    - Context limits triggered.
    """
    if error_count >= 3:
        return True
    
    # If returned format is drastically bad, we might want to escalate.
    if len(draft_attempt) < 10 and error_count > 0:
        return True
        
    return False
