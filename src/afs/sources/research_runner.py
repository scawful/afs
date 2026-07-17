"""Private subprocess entry point for one selected research provider."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from ..config import load_config_model
from .models import ResearchRequest, ResearchSourceProvider
from .registry import load_source_provider_by_name
from .research import normalize_research_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    try:
        payload: Any = json.loads(
            Path(args.request).read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict):
            raise ValueError("research request must be a JSON object")
        request = ResearchRequest.from_dict(payload)
        if not request.network_allowed:
            raise ValueError("research runner requires explicit network consent")
        config = load_config_model(
            config_path=Path(args.config) if args.config else None,
            merge_user=True,
        )
        # Selected-extension import, factory, and invocation chatter all
        # belong on bounded stderr. Reserve stdout for exactly one JSON value.
        with contextlib.redirect_stdout(sys.stderr):
            provider = load_source_provider_by_name(args.provider, config=config)
            if not isinstance(provider, ResearchSourceProvider):
                raise TypeError(
                    f"source provider {args.provider!r} does not implement "
                    "research(request)"
                )
            records = provider.research(request)
        normalized = normalize_research_records(
            request,
            records,
            provider_name=args.provider,
        )
        print(
            json.dumps(
                [record.to_dict() for record in normalized],
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - private process reports bounded failure
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
