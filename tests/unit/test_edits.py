import pytest
from hierocode.broker.edits import EditBlock, EditApplyError, parse_edit_blocks, apply_edit_blocks

def test_parse_edit_blocks_no_markers():
    assert parse_edit_blocks("Hello world") == []

def test_parse_edit_blocks_happy_path():
    text = "```\n<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE\n```"
    blocks = parse_edit_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].search == "foo"
    assert blocks[0].replace == "bar"

def test_apply_edit_blocks_exact():
    original = "a\nfoo\nb"
    blocks = [EditBlock(search="foo", replace="bar")]
    assert apply_edit_blocks(original, blocks) == "a\nbar\nb"

def test_apply_edit_blocks_ambiguous():
    original = "foo\nfoo\n"
    blocks = [EditBlock(search="foo", replace="bar")]
    with pytest.raises(EditApplyError, match="not unique"):
        apply_edit_blocks(original, blocks)

def test_apply_edit_blocks_whitespace_fallback():
    original = "a\nfoo  \nbar\nb"
    # Drafter provides trailing whitespace differently or omits it
    blocks = [EditBlock(search="foo\nbar", replace="baz\nqux")]
    assert apply_edit_blocks(original, blocks) == "a\nbaz\nqux\nb"

def test_apply_edit_blocks_empty_search_appends():
    original = "foo\n"
    blocks = [EditBlock(search="", replace="bar")]
    assert apply_edit_blocks(original, blocks) == "foo\nbar\n"

def test_apply_edit_blocks_not_found():
    original = "foo\n"
    blocks = [EditBlock(search="baz", replace="bar")]
    with pytest.raises(EditApplyError, match="not found"):
        apply_edit_blocks(original, blocks)
