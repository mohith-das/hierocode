def slice_context(content: str, max_chars: int = 15000) -> str:
    """
    Very naive heuristic context slicer. 
    In future iterations, this should parse ASTs or use proper tokenizer.
    """
    if len(content) <= max_chars:
        return content
    
    # Just truncate with a marker for now
    half = max_chars // 2
    return content[:half] + "\n\n... [CONTENT TRUNCATED FOR CONTEXT SIZE] ...\n\n" + content[-half:]
