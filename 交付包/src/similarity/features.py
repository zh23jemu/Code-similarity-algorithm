from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import sqrt

from .preprocess import JAVA_KEYWORDS, ProcessedCode, preprocess_code


STRUCTURE_TOKENS = {
    "if",
    "else",
    "for",
    "while",
    "do",
    "switch",
    "case",
    "try",
    "catch",
    "return",
    "class",
    "new",
    "{",
    "}",
    "(",
    ")",
    "[",
    "]",
}

OPERATORS = {
    "+",
    "-",
    "*",
    "/",
    "%",
    "<",
    ">",
    "<=",
    ">=",
    "==",
    "!=",
    "=",
    "&&",
    "||",
    "!",
    "++",
    "--",
}


@dataclass(frozen=True)
class PairFeatures:
    """一对代码的数值化特征。

    这些特征既用于弱监督标签构造，也用于传统机器学习模型训练。
    """

    token_jaccard: float
    token_sequence_similarity: float
    keyword_cosine: float
    operator_cosine: float
    structure_cosine: float
    length_similarity: float
    line_count_similarity: float
    method_count_similarity: float
    loop_count_similarity: float
    branch_count_similarity: float
    base_similarity: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()

    def as_vector(self) -> list[float]:
        return [self.as_dict()[name] for name in FEATURE_NAMES]


FEATURE_NAMES = [
    "token_jaccard",
    "token_sequence_similarity",
    "keyword_cosine",
    "operator_cosine",
    "structure_cosine",
    "length_similarity",
    "line_count_similarity",
    "method_count_similarity",
    "loop_count_similarity",
    "branch_count_similarity",
    "base_similarity",
]


def _safe_ratio(a: int | float, b: int | float) -> float:
    """把两个数量差异转换成 0 到 1 的相似度。"""

    maximum = max(float(a), float(b), 1.0)
    return 1.0 - min(abs(float(a) - float(b)) / maximum, 1.0)


def _cosine(counter_a: Counter[str], counter_b: Counter[str]) -> float:
    """计算两个稀疏计数字典的余弦相似度。"""

    if not counter_a or not counter_b:
        return 0.0
    keys = set(counter_a) | set(counter_b)
    dot = sum(counter_a[key] * counter_b[key] for key in keys)
    norm_a = sqrt(sum(value * value for value in counter_a.values()))
    norm_b = sqrt(sum(value * value for value in counter_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _filtered_counter(processed: ProcessedCode, vocabulary: set[str]) -> Counter[str]:
    """统计指定词表内的 Token 频次，用于关键字、运算符和结构特征。"""

    return Counter(token for token in processed.tokens if token in vocabulary)


def _method_count(tokens: tuple[str, ...]) -> int:
    """粗略估计方法数量。

    Java 初学者提交通常结构较简单，检测“标识符 + 左括号 + 左大括号”
    的近似规则已经足够支撑本科毕设第一版实验。
    """

    count = 0
    for index in range(len(tokens) - 2):
        if tokens[index] in {"IDENT", "main"} and tokens[index + 1] == "(" and "{" in tokens[index + 2 : index + 8]:
            count += 1
    return count


def extract_pair_features(code_a: str, code_b: str) -> PairFeatures:
    """提取一对 Java 代码的相似度特征。"""

    processed_a = preprocess_code(code_a)
    processed_b = preprocess_code(code_b)
    tokens_a = processed_a.tokens
    tokens_b = processed_b.tokens
    set_a = set(tokens_a)
    set_b = set(tokens_b)

    union_size = len(set_a | set_b)
    token_jaccard = len(set_a & set_b) / union_size if union_size else 0.0
    token_sequence_similarity = SequenceMatcher(None, tokens_a, tokens_b).ratio()

    keyword_cosine = _cosine(_filtered_counter(processed_a, JAVA_KEYWORDS), _filtered_counter(processed_b, JAVA_KEYWORDS))
    operator_cosine = _cosine(_filtered_counter(processed_a, OPERATORS), _filtered_counter(processed_b, OPERATORS))
    structure_cosine = _cosine(
        _filtered_counter(processed_a, STRUCTURE_TOKENS),
        _filtered_counter(processed_b, STRUCTURE_TOKENS),
    )

    length_similarity = _safe_ratio(len(tokens_a), len(tokens_b))
    line_count_similarity = _safe_ratio(code_a.count("\n") + 1, code_b.count("\n") + 1)
    method_count_similarity = _safe_ratio(_method_count(tokens_a), _method_count(tokens_b))
    loop_count_similarity = _safe_ratio(
        sum(1 for token in tokens_a if token in {"for", "while", "do"}),
        sum(1 for token in tokens_b if token in {"for", "while", "do"}),
    )
    branch_count_similarity = _safe_ratio(
        sum(1 for token in tokens_a if token in {"if", "else", "switch", "case"}),
        sum(1 for token in tokens_b if token in {"if", "else", "switch", "case"}),
    )

    # 基础规则分数用于弱监督标签。权重偏向 Token 序列和结构相似度，
    # 这样变量名变化、常量变化不会过度影响“是否相似”的判断。
    base_similarity = (
        0.30 * token_jaccard
        + 0.30 * token_sequence_similarity
        + 0.15 * keyword_cosine
        + 0.10 * operator_cosine
        + 0.10 * structure_cosine
        + 0.05 * length_similarity
    )

    return PairFeatures(
        token_jaccard=token_jaccard,
        token_sequence_similarity=token_sequence_similarity,
        keyword_cosine=keyword_cosine,
        operator_cosine=operator_cosine,
        structure_cosine=structure_cosine,
        length_similarity=length_similarity,
        line_count_similarity=line_count_similarity,
        method_count_similarity=method_count_similarity,
        loop_count_similarity=loop_count_similarity,
        branch_count_similarity=branch_count_similarity,
        base_similarity=base_similarity,
    )
