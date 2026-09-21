"""Canonical JSON (RFC 8785 / JSON Canonicalization Scheme) and the request hash.

Number serialization follows ECMAScript number-to-string (RFC 8785's normative
rule) for the finite double space: shortest round-trip digits with ES6
notation thresholds. Object keys are sorted by UTF-16 code units; strings use
minimal JSON escaping with non-ASCII characters kept as UTF-8. Every number
is an IEEE-754 double (I-JSON), and strings containing unpaired surrogates are
rejected as invalid Unicode.
"""

import hashlib
import json
import math


def _es6_number(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        raise ValueError("NaN and Infinity are not JSON numbers")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    text = repr(abs(value))
    if "e" in text:
        mantissa, exponent = text.split("e")
        exp = int(exponent)
    else:
        mantissa, exp = text, 0
    point = mantissa.index(".") if "." in mantissa else len(mantissa)
    digits_all = mantissa.replace(".", "")
    n = exp + point
    digits = digits_all.lstrip("0")
    n -= len(digits_all) - len(digits)
    digits = digits.rstrip("0")
    if not digits:
        digits = "0"
    k = len(digits)
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    exponent_value = n - 1
    mantissa_part = digits[0] + ("." + digits[1:] if k > 1 else "")
    return sign + mantissa_part + "e" + ("+" if exponent_value >= 0 else "-") + str(abs(exponent_value))


def canonical_json(value) -> str:
    """Serialize one JSON value in RFC 8785 canonical form."""
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, int):
        # I-JSON/RFC 8785: every JSON number is an IEEE-754 double; integers
        # beyond 2**53 lose precision exactly as any conforming consumer sees.
        return _es6_number(float(value))
    if isinstance(value, float):
        return _es6_number(value)
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("string contains unpaired surrogates and is not valid Unicode") from exc
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise ValueError("object keys must be strings for canonical JSON")
            try:
                key.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("object key contains unpaired surrogates and is not valid Unicode") from exc
        # RFC 8785 sorts property names by UTF-16 code units.
        items = []
        for key in sorted(value, key=lambda k: k.encode("utf-16be", "surrogatepass")):
            items.append(json.dumps(key, ensure_ascii=False) + ":" + canonical_json(value[key]))
        return "{" + ",".join(items) + "}"
    raise ValueError(f"value is not JSON: {type(value)!r}")


def canonical_request_hash(value) -> str:
    """Lowercase hex SHA-256 of the UTF-8 canonical form (docs/CONTRACT.md 'Trace and Replay')."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
