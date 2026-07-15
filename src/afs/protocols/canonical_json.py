"""Deterministic JSON encoding and hashing for versioned AFS protocols.

The encoder deliberately uses a small, language-neutral format: object keys are
sorted, arrays retain their order, strings use JSON escaping, and finite numbers
are rendered as plain base-10 values.  In particular, numerically equivalent
values such as ``500`` and ``500.0`` produce identical bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, InvalidOperation
from typing import Any


class CanonicalJSONError(ValueError):
    """Raised when a value cannot be represented by the canonical encoder."""


def ensure_utf8_text(value: Any, location: str = "(root)") -> None:
    """Reject lone surrogates and other text that has no strict UTF-8 encoding."""
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise CanonicalJSONError(
                f"{location}: text must contain only UTF-8-encodable Unicode"
            ) from exc
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJSONError(f"{location}: JSON object keys must be strings")
            ensure_utf8_text(key, f"{location}/<key>")
            ensure_utf8_text(item, f"{location}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            ensure_utf8_text(item, f"{location}/{index}")


def ensure_finite(value: Any, location: str = "(root)") -> None:
    """Reject non-finite and out-of-range numeric values recursively."""
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        try:
            binary64 = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise CanonicalJSONError(
                f"{location}: number is outside the supported finite range"
            ) from exc
        if not math.isfinite(binary64):
            raise CanonicalJSONError(f"{location}: non-finite numbers are not allowed")
    if isinstance(value, dict):
        for key, item in value.items():
            ensure_finite(item, f"{location}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            ensure_finite(item, f"{location}/{index}")


def ensure_interoperable_json(value: Any, location: str = "(root)") -> None:
    """Validate the numeric and Unicode domain shared by AFS v1 protocols."""
    ensure_finite(value, location)
    ensure_utf8_text(value, location)


def _reject_nonstandard_json_number(token: str) -> None:
    raise CanonicalJSONError(f"non-standard JSON number {token!r} is not allowed")


def _reject_duplicate_json_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJSONError(f"duplicate JSON object member {key!r}")
        result[key] = value
    return result


def strict_json_loads(data: str | bytes | bytearray) -> Any:
    """Parse interoperable JSON, rejecting duplicate members and extensions."""
    if isinstance(data, (bytes, bytearray)):
        data = bytes(data).decode("utf-8")
    parsed = json.loads(
        data,
        parse_constant=_reject_nonstandard_json_number,
        object_pairs_hook=_reject_duplicate_json_members,
    )
    ensure_interoperable_json(parsed)
    return parsed


def canonical_number_text(value: int | float | Decimal) -> str:
    """Render one finite number using the AFS v1 plain-decimal hash format."""
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CanonicalJSONError("canonical JSON requires a finite number") from exc
    if not number.is_finite():
        raise CanonicalJSONError("canonical JSON requires a finite number")
    if number == 0:
        return "0"

    sign, raw_digits, raw_exponent = number.as_tuple()
    if not isinstance(raw_exponent, int):
        raise CanonicalJSONError("canonical JSON requires a finite number")
    exponent = raw_exponent
    digits = list(raw_digits)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1
    digit_text = "".join(str(digit) for digit in digits)

    if exponent >= 0:
        body = digit_text + "0" * exponent
    else:
        point = len(digit_text) + exponent
        if point > 0:
            body = f"{digit_text[:point]}.{digit_text[point:]}"
        else:
            body = f"0.{('0' * -point)}{digit_text}"
    return ("-" if sign else "") + body


def encode_canonical_json(value: Any) -> str:
    """Encode a JSON value with stable object order and numeric tokens."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float, Decimal)):
        return canonical_number_text(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(encode_canonical_json(item) for item in value) + "]"
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise CanonicalJSONError("canonical JSON object keys must be strings")
        return (
            "{"
            + ",".join(
                json.dumps(key, ensure_ascii=False) + ":" + encode_canonical_json(item)
                for key, item in sorted(value.items())
            )
            + "}"
        )
    raise CanonicalJSONError(
        f"canonical JSON does not support values of type {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 bytes for a JSON-compatible value."""
    ensure_interoperable_json(value)
    return encode_canonical_json(value).encode("utf-8")


def canonical_json_text(value: Any, *, indent: int | None = None) -> str:
    """Return deterministic text, optionally formatted for human-readable output."""
    ensure_interoperable_json(value)
    if indent is None:
        return encode_canonical_json(value)
    if indent < 0:
        raise CanonicalJSONError("indent must be non-negative")
    # Formatting is intentionally a presentation concern; hashes always use the
    # compact encoder above.  Sorting remains deterministic for human output.
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=indent, sort_keys=True)


def sha256_canonical_json(value: Any) -> str:
    """Return the SHA-256 digest of :func:`canonical_json_bytes`."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


# Compatibility spelling for early adopters of this module.
canonical_json_sha256 = sha256_canonical_json
