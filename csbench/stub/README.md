# Contract-conformance stub

A minimal local HTTP server that speaks the arm contract so you can run the
harness's hosted-method path (the `NeedlepathArm` client) with **no network and
no secrets**.

## What it is, and is not

- **It is** a conformance fixture: it accepts a `ContextRequest` at
  `POST /v1/context/select` and returns a schema-valid `ContextResponse` with
  every required field and a realistic timing shape.
- **It is NOT** a behavioral emulation of any method. It passes the input
  through unchanged: nothing is selected away, nothing is compressed, and it
  emits no gate or safety verdict. **Numbers it returns are not benchmark
  results.** Only their *shape* is real.

Use it to exercise wiring, schemas, and the row-writer in tests/CI. To reproduce
published numbers, point the client at the hosted endpoint instead.

## Run it

```bash
python -m csbench.stub --host 127.0.0.1 --port 8080
```

Then point an arm at it:

```python
from csbench.arms import NeedlepathArm
arm = NeedlepathArm(base_url="http://127.0.0.1:8080")
```

In tests, use the ephemeral-port context manager:

```python
from csbench.stub import StubServer
from csbench.arms import NeedlepathArm

with StubServer() as stub:
    arm = NeedlepathArm(base_url=stub.base_url)
    response = arm.select(request)
```
