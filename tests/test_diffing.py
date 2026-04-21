from hierocode.repo.diffing import generate_unified_diff

def test_generate_diff_same():
    orig = "a\nb\nc"
    diff = generate_unified_diff(orig, orig, "file.txt")
    assert not diff.strip()

def test_generate_diff_changed():
    orig = "a\nb\nc"
    mod = "a\nx\nc"
    diff = generate_unified_diff(orig, mod, "file.txt")
    assert "-b" in diff
    assert "+x" in diff
