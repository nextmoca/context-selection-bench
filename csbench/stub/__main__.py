"""Run the contract-conformance stub: ``python -m csbench.stub [--host H] [--port P]``."""

from __future__ import annotations

import argparse

from .server import serve_forever


def main() -> None:
    parser = argparse.ArgumentParser(description="Contract-conformance stub endpoint (not a behavioral emulation).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    serve_forever(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
