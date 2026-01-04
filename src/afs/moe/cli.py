"""CLI for MoE router testing."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Iterable

from .router import MoERouter, RouterConfig
from .classifier import QueryIntent


async def _test_routing(queries: list[str]) -> None:
    """Test routing decisions for queries."""
    router = MoERouter()

    print("=" * 60)
    print("MoE Router - Routing Test")
    print("=" * 60)

    for query in queries:
        decision = router.route(query)
        print(f"\nQuery: {query[:60]}...")
        print(f"  Intent: {decision.classification.intent.value}")
        print(f"  Confidence: {decision.classification.confidence:.2f}")
        print(f"  Expert: {decision.expert.name if decision.expert else 'fallback'}")
        if decision.classification.matched_patterns:
            print(f"  Patterns: {', '.join(decision.classification.matched_patterns[:3])}")


async def _generate(query: str, stream: bool = False, use_rag: bool = True) -> None:
    """Generate response for a query."""
    async with MoERouter() as router:
        decision = router.route(query)

        print(f"Routing to: {decision.expert.name if decision.expert else 'fallback'}")
        print(f"Intent: {decision.classification.intent.value}")
        print(f"Confidence: {decision.classification.confidence:.2f}")

        # Show retrieved context if RAG enabled
        if use_rag:
            retrieved = router.retrieve_context(query)
            if retrieved:
                print(f"Retrieved: {len(retrieved)} documents")
                for r in retrieved[:3]:
                    print(f"  - [{r.source}] {r.id} ({r.score:.2f})")

        print("-" * 40)

        if stream:
            async for chunk in await router.generate(query, stream=True, use_rag=use_rag):
                print(chunk, end="", flush=True)
            print()
        else:
            result = await router.generate(query, use_rag=use_rag)
            print(result.content)
            print(f"\n[Tokens: {result.tokens_generated}]")
            if result.retrieved_context:
                print(f"[Context: {len(result.retrieved_context)} docs]")


def _test_command(args: argparse.Namespace) -> int:
    """Run routing tests."""
    test_queries = [
        # Optimization queries -> din
        "Optimize this 65816 code:\nLDA #$00\nSTA $7EF340\nLDA #$00\nSTA $7EF341",
        "How can I make this loop faster and use fewer cycles?",
        "Can this be done with STZ instead?",
        "Use the hardware multiplier at $4202 to speed this up",

        # Generation queries -> nayru
        "Write a function to copy 256 bytes from ROM to RAM",
        "Generate a subroutine that checks if Link has the hookshot",
        "Implement a jump table for handling 8 different sprite states",
        "Create code that plays SFX $1B when the player collects a rupee",

        # Debug queries -> fallback (no farore yet)
        "Why does this code crash when A is zero?",
        "Debug this routine - it's not returning the right value",
        "What's wrong with this BNE branch?",

        # General queries -> fallback
        "What registers does the 65816 have?",
        "Explain how the stack works",
    ]

    asyncio.run(_test_routing(test_queries))
    return 0


def _generate_command(args: argparse.Namespace) -> int:
    """Generate a response."""
    asyncio.run(_generate(args.query, stream=args.stream, use_rag=not args.no_rag))
    return 0


def _list_command(args: argparse.Namespace) -> int:
    """List configured experts."""
    router = MoERouter()
    print("Configured Experts:")
    print("-" * 60)
    for expert in router.list_experts():
        print(f"  {expert.name}")
        print(f"    Model: {expert.model_id}")
        print(f"    Intent: {expert.intent.value}")
        print(f"    Host: {expert.host}")
        print()
    print(f"Fallback: {router.config.fallback_model}")
    return 0


def _eval_command(args: argparse.Namespace) -> int:
    """Run evaluations."""
    from .evals.runner import EvalRunner, run_routing_eval, run_full_eval

    if args.routing_only:
        metrics = run_routing_eval()
    else:
        metrics = asyncio.run(run_full_eval(use_rag=args.rag))

    # Save results if output specified
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


def _orchestrate_command(args: argparse.Namespace) -> int:
    """Run orchestrated query with Gemini planning."""
    from .orchestrator import orchestrate
    import json

    result = asyncio.run(orchestrate(args.query, verbose=args.verbose))

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Goal: {result['plan']['goal']}")
        print(f"Steps: {len(result['plan']['steps'])}")
        print()

        if args.verbose:
            print("Reasoning:")
            print(result['plan']['reasoning'][:500])
            print()

        print("=" * 60)
        print(result['final_response'])

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="afs-moe",
        description="Mixture of Experts router for 65816 assembly models"
    )
    sub = parser.add_subparsers(dest="command")

    # test command
    test_cmd = sub.add_parser("test", help="Test routing decisions")
    test_cmd.set_defaults(func=_test_command)

    # generate command
    gen_cmd = sub.add_parser("generate", help="Generate response")
    gen_cmd.add_argument("query", help="Query to process")
    gen_cmd.add_argument("--stream", "-s", action="store_true", help="Stream output")
    gen_cmd.add_argument("--no-rag", action="store_true", help="Disable RAG context retrieval")
    gen_cmd.set_defaults(func=_generate_command)

    # list command
    list_cmd = sub.add_parser("list", help="List configured experts")
    list_cmd.set_defaults(func=_list_command)

    # eval command
    eval_cmd = sub.add_parser("eval", help="Run evaluations")
    eval_cmd.add_argument(
        "--routing-only", "-r",
        action="store_true",
        help="Only evaluate routing (skip generation)"
    )
    eval_cmd.add_argument(
        "--rag",
        action="store_true",
        help="Enable RAG context retrieval during eval"
    )
    eval_cmd.add_argument(
        "--output", "-o",
        help="Save results to JSON file"
    )
    eval_cmd.set_defaults(func=_eval_command)

    # orchestrate command
    orch_cmd = sub.add_parser("orchestrate", help="Run orchestrated query with Gemini planning")
    orch_cmd.add_argument("query", help="Query to process")
    orch_cmd.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    orch_cmd.add_argument("--json", action="store_true", help="Output as JSON")
    orch_cmd.set_defaults(func=_orchestrate_command)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
