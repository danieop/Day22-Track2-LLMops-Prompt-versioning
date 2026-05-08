import json
import re

from guardrails import Guard, OnFailAction, Validator, register_validator
from guardrails.validators import FailResult, PassResult


@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    PII_PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "PHONE": r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        redacted = value
        found = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            for match in re.findall(pattern, value):
                redacted = redacted.replace(match, f"[{pii_type}_REDACTED]")
                found.append(pii_type)
        if found:
            return FailResult(
                errorMessage=f"Detected PII types: {', '.join(found)}",
                fixValue=redacted,
            )
        return PassResult(value_override=value)


@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    @staticmethod
    def _repair(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        text = text.replace("'", '"')
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text

    def validate(self, value: str, metadata: dict):
        try:
            parsed = json.loads(value)
            return PassResult(value_override=json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            pass

        try:
            repaired_text = self._repair(value)
            parsed = json.loads(repaired_text)
            return FailResult(
                errorMessage="JSON was malformed but repairable",
                fixValue=json.dumps(parsed, indent=2),
            )
        except json.JSONDecodeError as exc:
            fallback = json.dumps({"error": "Invalid JSON after repair", "raw": value})
            return FailResult(errorMessage=str(exc), fixValue=fallback)


def _validated_output(guard: Guard, validator: Validator, text: str) -> str:
    outcome = guard.validate(text)
    if outcome.validated_output != text:
        return outcome.validated_output

    result = validator.validate(text, {})
    if isinstance(result, FailResult) and result.fix_value is not None:
        return result.fix_value
    if isinstance(result, PassResult) and result.value_override is not PassResult.ValueOverrideSentinel:
        return result.value_override
    return text


def demo_pii_guard() -> None:
    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)
    validator = PIIDetector(on_fail=OnFailAction.FIX)
    guard = Guard().use(validator)
    test_cases = [
        ("Email", "Contact John at john.doe@example.com for details."),
        ("Phone", "Call our support line at (555) 867-5309."),
        ("SSN", "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII", "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean", "No sensitive information in this text."),
    ]
    for label, text in test_cases:
        output = _validated_output(guard, validator, text)
        print(f"\n[{label}]")
        print(f"Input : {text}")
        print(f"Output: {output}")


def demo_json_guard() -> None:
    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)
    validator = JSONFormatter(on_fail=OnFailAction.FIX)
    guard = Guard().use(validator)
    test_cases = [
        ("Valid JSON", '{"name": "Alice", "age": 30}'),
        ("Markdown fences", '```json\n{"name": "Bob"}\n```'),
        ("Single quotes", "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma", '{"key": "value",}'),
        ("Truly invalid", "This is not JSON at all: ??? {]"),
    ]
    for label, text in test_cases:
        output = _validated_output(guard, validator, text)
        try:
            json.loads(text)
            status = "Pass"
        except json.JSONDecodeError:
            status = "Fixed"
        print(f"\n[{label}] {status}")
        print(f"Input : {text}")
        print(f"Output: {output}")


def main() -> None:
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)
    demo_pii_guard()
    demo_json_guard()
    print("\nStep 4 complete")


if __name__ == "__main__":
    main()
