import argparse
import os
from typing import Optional

from src.retrieval.indexer import build_index, derive_index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an embedding index for a JSONL corpus using E5/Contriever.\n\n"
            "Each line of the corpus should be a JSON object with at least "
            "'title' and either 'text' or 'contents' fields."
        )
    )

    parser.add_argument(
        "--corpus-path",
        type=str,
        required=True,
        help="Path to the input corpus JSONL file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="intfloat/e5-large-v2",
        help="Encoder model name or path (default: intfloat/e5-large-v2).",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help=(
            "Directory to store the index. "
            "If not provided, it is derived from corpus path and model name."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for encoding (default: 1024).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100000,
        help="Number of documents per embedding shard (default: 100000).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device to run the encoder on (e.g., 'cuda', 'cpu'). "
            "If not set, defaults to CUDA when available, otherwise CPU."
        ),
    )

    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    corpus_path = os.path.abspath(args.corpus_path)
    model_name = args.model_name

    if args.index_dir is None:
        index_dir = derive_index_path(corpus_path, model_name)
    else:
        index_dir = os.path.abspath(args.index_dir)

    print(f"Corpus path: {corpus_path}")
    print(f"Model name: {model_name}")
    print(f"Index directory: {index_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Shard size: {args.shard_size}")

    # Let indexer choose default device when args.device is None
    device = args.device if args.device is not None else (
        "cuda"  # indexer.py will still check torch.cuda.is_available()
    )

    build_index(
        corpus_path=corpus_path,
        index_dir=index_dir,
        model_name=model_name,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        device=device,
    )


if __name__ == "__main__":
    main()

