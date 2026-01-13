import json

from afs.embeddings import (
    build_embedding_index,
    evaluate_embedding_index,
    load_embedding_eval_cases,
)


def test_embeddings_eval(tmp_path) -> None:
    src_dir = tmp_path / "docs"
    src_dir.mkdir()
    doc1 = src_dir / "doc1.txt"
    doc2 = src_dir / "doc2.txt"
    doc1.write_text("alpha bravo", encoding="utf-8")
    doc2.write_text("charlie delta", encoding="utf-8")

    index_root = tmp_path / "index"
    build_embedding_index([src_dir], index_root, embed_fn=None)

    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"query": "alpha", "expected_path": "doc1.txt"}) + "\n",
        encoding="utf-8",
    )

    cases = load_embedding_eval_cases(eval_path)
    result = evaluate_embedding_index(
        index_root,
        cases,
        embed_fn=None,
        top_k=2,
        min_score=0.0,
        match_mode="path",
    )

    assert result.total == 1
    assert result.hits == 1
    assert result.hit_rate == 1.0
