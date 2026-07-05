from hierocode.broker.edits import parse_edit_blocks, apply_edit_blocks

def test_debug():
    s = """
<<<<<<< SEARCH
def foo():
    pass
=======
def foo():
    return 1
>>>>>>> REPLACE
"""
    blocks = parse_edit_blocks(s)
    print("BLOCKS", blocks)
    original = "def foo():\n    pass\n"
    res = apply_edit_blocks(original, blocks)
    print("RES", repr(res))

if __name__ == "__main__":
    test_debug()
