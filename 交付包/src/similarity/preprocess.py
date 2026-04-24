from __future__ import annotations

import re
from dataclasses import dataclass


JAVA_KEYWORDS = {
    "abstract",
    "assert",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extends",
    "final",
    "finally",
    "float",
    "for",
    "goto",
    "if",
    "implements",
    "import",
    "instanceof",
    "int",
    "interface",
    "long",
    "native",
    "new",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "static",
    "strictfp",
    "super",
    "switch",
    "synchronized",
    "this",
    "throw",
    "throws",
    "transient",
    "try",
    "void",
    "volatile",
    "while",
}

TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|==|!=|<=|>=|\+\+|--|&&|\|\||[-+*/%<>=!&|^~?:;,.(){}\[\]]|\d+(?:\.\d+)?"
)


@dataclass(frozen=True)
class ProcessedCode:
    """Java 代码预处理结果。

    `normalized_code` 用于文本向量化，`tokens` 用于集合、序列和结构特征。
    """

    normalized_code: str
    tokens: tuple[str, ...]


def remove_comments(code: str) -> str:
    """移除 Java 单行与多行注释，同时尽量保留字符串内容。

    该实现使用轻量状态机而不是简单正则，避免把字符串里的 `//`
    或 `/*` 误判为注释起点。
    """

    result: list[str] = []
    i = 0
    state = "normal"
    while i < len(code):
        current = code[i]
        next_char = code[i + 1] if i + 1 < len(code) else ""

        if state == "normal":
            if current == "/" and next_char == "/":
                state = "line_comment"
                i += 2
                continue
            if current == "/" and next_char == "*":
                state = "block_comment"
                i += 2
                continue
            if current == '"':
                state = "string"
            elif current == "'":
                state = "char"
            result.append(current)
        elif state == "line_comment":
            if current == "\n":
                state = "normal"
                result.append(current)
        elif state == "block_comment":
            if current == "*" and next_char == "/":
                state = "normal"
                i += 2
                continue
        elif state == "string":
            result.append(current)
            if current == "\\" and next_char:
                result.append(next_char)
                i += 2
                continue
            if current == '"':
                state = "normal"
        elif state == "char":
            result.append(current)
            if current == "\\" and next_char:
                result.append(next_char)
                i += 2
                continue
            if current == "'":
                state = "normal"
        i += 1
    return "".join(result)


def normalize_literals(code: str) -> str:
    """归一化字符串、字符和数字字面量，降低常量细节对相似度的干扰。"""

    code = re.sub(r'"(?:\\.|[^"\\])*"', " STR_LITERAL ", code)
    code = re.sub(r"'(?:\\.|[^'\\])'", " CHAR_LITERAL ", code)
    return re.sub(r"\b\d+(?:\.\d+)?\b", " NUM_LITERAL ", code)


def tokenize_java(code: str, normalize_identifiers: bool = True) -> tuple[str, ...]:
    """将 Java 代码切分为轻量 Token 序列。

    非关键字标识符默认归一化为 `IDENT`，这样变量名替换不会显著降低
    相似度，更贴近代码查重场景。
    """

    raw_tokens = TOKEN_PATTERN.findall(code)
    tokens: list[str] = []
    for token in raw_tokens:
        if token in {"STR_LITERAL", "CHAR_LITERAL", "NUM_LITERAL"}:
            tokens.append(token)
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
            tokens.append(token if token in JAVA_KEYWORDS else ("IDENT" if normalize_identifiers else token))
        else:
            tokens.append(token)
    return tuple(tokens)


def preprocess_code(code: str, normalize_identifiers: bool = True) -> ProcessedCode:
    """执行完整预处理流程，返回规范化代码文本和 Token 序列。"""

    without_comments = remove_comments(code)
    normalized = normalize_literals(without_comments)
    tokens = tokenize_java(normalized, normalize_identifiers=normalize_identifiers)
    normalized_code = " ".join(tokens)
    return ProcessedCode(normalized_code=normalized_code, tokens=tokens)
