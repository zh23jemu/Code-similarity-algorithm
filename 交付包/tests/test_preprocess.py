from similarity.preprocess import preprocess_code, remove_comments


def test_remove_comments_keeps_string_content():
    code = 'String url = "http://example.com"; // 注释内容\nint x = 1;'

    cleaned = remove_comments(code)

    assert "注释内容" not in cleaned
    assert "http://example.com" in cleaned


def test_preprocess_normalizes_identifiers_and_literals():
    code = "int answer = 123; String msg = \"hello\";"

    processed = preprocess_code(code)

    assert "IDENT" in processed.tokens
    assert "NUM_LITERAL" in processed.tokens
    assert "STR_LITERAL" in processed.tokens
    assert "answer" not in processed.tokens
