"""Entity CLI commands: extract, list, search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def entity_extract_command(args: argparse.Namespace) -> int:
    """Extract entities from training data."""
    from ..generators.base import TrainingSample
    from ..knowledge import EntityExtractor

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".entities.jsonl")

    extractor = EntityExtractor(include_hardware=not getattr(args, "no_hardware", False))

    # Load and process samples
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(TrainingSample.from_dict(json.loads(line)))

    print(f"Processing {len(samples)} samples...")

    total_entities = 0
    known_entities = 0

    for sample in samples:
        sample.populate_kg_entities(extractor, validate=getattr(args, "validate", False))
        result = extractor.extract(sample.output)
        total_entities += len(result.entities)
        known_entities += result.known_count

    # Write output
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict()) + "\n")

    print("\nEntity Extraction Complete")
    print(f"  Samples processed: {len(samples)}")
    print(f"  Total entities found: {total_entities}")
    print(f"  Known entities: {known_entities}")
    print(f"  Coverage: {100 * known_entities / total_entities:.1f}%" if total_entities else "  Coverage: N/A")
    print(f"  Output: {output_path}")

    return 0


def entity_list_command(args: argparse.Namespace) -> int:
    """List known entities."""
    from ..knowledge import (
        ALTTP_ADDRESSES,
        AddressCategory,
        get_addresses_by_category,
    )

    if args.category:
        try:
            category = AddressCategory(args.category.lower())
        except ValueError:
            print(f"Unknown category: {args.category}")
            print(f"Valid categories: {[c.value for c in AddressCategory]}")
            return 1

        addresses = get_addresses_by_category(category)
        print(f"\n{category.value.upper()} Addresses ({len(addresses)}):")
        print("-" * 60)
    else:
        addresses = ALTTP_ADDRESSES
        print(f"\nAll Known Addresses ({len(addresses)}):")
        print("-" * 60)

    for name, info in sorted(addresses.items()):
        print(f"  {name:30} {info.full_address:12} {info.description[:40]}")

    return 0


def entity_search_command(args: argparse.Namespace) -> int:
    """Search for entity by address."""
    from ..knowledge import lookup_by_address

    query = args.address.upper().lstrip("$")

    # Parse address
    if len(query) == 6:
        bank = query[:2]
        offset = int(query[2:], 16)
    elif len(query) == 4:
        bank = "7E"
        offset = int(query, 16)
    elif len(query) == 2:
        bank = "7E"
        offset = int(query, 16)
    else:
        print(f"Invalid address format: {args.address}")
        print("Expected: $XX, $XXXX, or $XXXXXX")
        return 1

    matches = lookup_by_address(offset, bank)

    if not matches:
        print(f"No matches found for ${bank}{offset:04X}")
        return 0

    print(f"\nMatches for ${bank}{offset:04X}:")
    print("-" * 60)
    for name, info in matches:
        print(f"  Name: {name}")
        print(f"  Address: {info.full_address}")
        print(f"  Category: {info.category.value}")
        print(f"  Description: {info.description}")
        if info.notes:
            print(f"  Notes: {info.notes}")
        print()

    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register entity command parsers."""
    entity_parser = subparsers.add_parser(
        "entity", help="Entity extraction and knowledge base utilities."
    )
    entity_sub = entity_parser.add_subparsers(dest="entity_command")

    # entity extract
    ent_extract = entity_sub.add_parser(
        "extract", help="Extract entities from training data."
    )
    ent_extract.add_argument(
        "--input", required=True, help="Input JSONL or ASM file."
    )
    ent_extract.add_argument(
        "--output", help="Output JSONL (default: input.entities.jsonl)."
    )
    ent_extract.add_argument(
        "--no-hardware",
        action="store_true",
        help="Exclude hardware register entities.",
    )
    ent_extract.add_argument(
        "--validate",
        action="store_true",
        help="Validate entity usage in context.",
    )
    ent_extract.set_defaults(func=entity_extract_command)

    # entity list
    ent_list = entity_sub.add_parser("list", help="List known entities.")
    ent_list.add_argument(
        "--category",
        help="Filter by category (e.g., link_state, sprite, hardware).",
    )
    ent_list.set_defaults(func=entity_list_command)

    # entity search
    ent_search = entity_sub.add_parser(
        "search", help="Search for entity by address."
    )
    ent_search.add_argument(
        "address", help="Address to search (e.g., $7EF36C, $0D00, $00)."
    )
    ent_search.set_defaults(func=entity_search_command)
