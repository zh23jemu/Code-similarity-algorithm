from similarity.features import extract_pair_features


def test_similar_code_has_high_base_similarity():
    code_a = "class A { public static void main(String[] args) { int x = 1; System.out.println(x); } }"
    code_b = "class B { public static void main(String[] args) { int y = 2; System.out.println(y); } }"

    features = extract_pair_features(code_a, code_b)

    assert features.base_similarity > 0.8


def test_different_code_has_lower_sequence_similarity():
    code_a = "class A { int add(int a, int b) { return a + b; } }"
    code_b = "class B { void loop() { for (int i = 0; i < 10; i++) { System.out.println(i); } } }"

    features = extract_pair_features(code_a, code_b)

    assert features.token_sequence_similarity < 0.8
