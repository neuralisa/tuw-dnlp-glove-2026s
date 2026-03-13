from __future__ import annotations

import argparse
from pathlib import Path

from .core import GloveIndex, analogy_vector, cosine_similarity, save_plot
from .hf_data import DEFAULT_HF_FILENAME, DEFAULT_HF_REPO_ID, resolve_glove_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone GloVe visualizer for TU Wien DNLP teaching"
    )
    parser.add_argument("--glove-path", help="optional local path to glove.6B.100d.txt")
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--hf-filename", default=DEFAULT_HF_FILENAME)
    parser.add_argument("--top-k", type=int, default=10)

    subparsers = parser.add_subparsers(dest="command", required=True)

    neighbors = subparsers.add_parser("neighbors", help="show nearest neighbors for one word")
    neighbors.add_argument("word")
    neighbors.add_argument("--plot", type=Path)

    similarity = subparsers.add_parser("similarity", help="show cosine similarity between two words")
    similarity.add_argument("left")
    similarity.add_argument("right")

    analogy = subparsers.add_parser("analogy", help="run a - b + c")
    analogy.add_argument("a")
    analogy.add_argument("b")
    analogy.add_argument("c")
    analogy.add_argument("--plot", type=Path)

    return parser.parse_args()


def print_neighbors(index: GloveIndex, word: str, top_k: int):
    query = index.vector(word)
    neighbors = index.neighbors(query, top_k=top_k, exclude={word})
    print(f"\nNearest neighbors for {word!r}\n")
    for rank, item in enumerate(neighbors, start=1):
        print(f"{rank:>2}. {item.word:<20} cosine={item.score:.4f}")
    return query, neighbors


def main() -> None:
    args = parse_args()
    glove_path = resolve_glove_path(
        args.glove_path,
        repo_id=args.hf_repo_id,
        filename=args.hf_filename,
    )
    print(f"\nUsing GloVe file: {glove_path}\n")
    index = GloveIndex.load(glove_path)
    print(f"Loaded {len(index.words):,} words with {index.dims} dimensions.\n")

    if args.command == "neighbors":
        query, neighbors = print_neighbors(index, args.word, args.top_k)
        if args.plot:
            save_plot(args.word, query, neighbors, index, args.plot)
            print(f"\nSaved plot to {args.plot}\n")
        return

    if args.command == "similarity":
        score = cosine_similarity(index.vector(args.left), index.vector(args.right))
        print(f"\ncos({args.left}, {args.right}) = {score:.4f}\n")
        return

    if args.command == "analogy":
        query = analogy_vector(index, positive=[args.a, args.c], negative=[args.b])
        neighbors = index.neighbors(query, top_k=args.top_k, exclude={args.a, args.b, args.c})
        print(f"\nAnalogy: {args.a} - {args.b} + {args.c}\n")
        for rank, item in enumerate(neighbors, start=1):
            print(f"{rank:>2}. {item.word:<20} cosine={item.score:.4f}")
        if args.plot:
            save_plot(f"{args.a}-{args.b}+{args.c}", query, neighbors, index, args.plot)
            print(f"\nSaved plot to {args.plot}\n")
        return

    raise RuntimeError("unknown command")
