# Native v1 Fixtures

Deterministic request, response, replay, and adapter-shape fixtures that pin
`docs/schemas/native-v1.schema.json`, which encodes `docs/CONTRACT.md`.

## What fixtures pin

Contract shape only: envelope closure, typed question and result shapes,
keyed maps, string-keyed Score level maps, nullability of unanswered fields,
error codes and classes, trace membership, the self-contained replay
envelope, and the Jev adapter result with its mandatory
`confidence_disclosure` string.

## What fixtures do not pin

Model usefulness, calibration, injection resistance, or any claim about a
real local model. UUIDs, hashes, timestamps, rendered messages, and runtime
versions in these files are illustrative placeholders, not recorded runs.
The acceptance-evidence corpus and gates live in `docs/CONTRACT.md`
(Acceptance Evidence); these fixtures are not that corpus.

## Validating

Each `manifest.json` row names a fixture file, a JSON Pointer target into the
schema (`$defs` member), and whether the fixture must validate (`valid`) or
must fail (`invalid`). Any Draft 2020-12 validator can check a row:

```python
import json
from jsonschema import Draft202012Validator, FormatChecker

schema = json.load(open("../schemas/native-v1.schema.json"))
row = {"file": "choice-valid.request.json", "target": "#/$defs/requestEnvelope"}
validator = Draft202012Validator(
    {"$ref": row["target"], "$defs": schema["$defs"]},
    format_checker=FormatChecker(),
)
errors = list(validator.iter_errors(json.load(open(row["file"]))))
assert not errors  # for expect: "valid"
```

Run every row to confirm the manifest holds. Fixtures are contract data, not
a test runner: no runtime code lives here.
