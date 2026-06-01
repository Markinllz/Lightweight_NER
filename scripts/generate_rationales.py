#!/usr/bin/env python3
"""CLI: генерация JSONL с рационалами. Логика в rationale_student.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rationale_student import (  # noqa: E402
    run_rationale_generation,
    run_rationale_generation_all_conll_txt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="CoNLL-2003 + rationales (gpt-oss API)")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "train_with_rationale.jsonl")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data",
        help="С --conll-dir: сюда пишутся train/valid/test JSONL",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conll-dir",
        type=Path,
        default=None,
        help="Каталог с train.txt, valid.txt, test.txt — сгенерировать все три JSONL в --output-dir",
    )
    parser.add_argument(
        "--conll-txt",
        type=Path,
        default=None,
        help="Один CoNLL .txt + --split (без Hugging Face)",
    )
    args = parser.parse_args()

    common = dict(
        limit=args.limit,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        sleep_s=args.sleep,
        seed=args.seed,
        dotenv_path=ROOT / ".env",
    )

    if args.conll_dir is not None:
        run_rationale_generation_all_conll_txt(conll_dir=args.conll_dir, output_dir=args.output_dir, **common)
    elif args.conll_txt is not None:
        run_rationale_generation(
            split=args.split,
            output_path=args.output,
            conll_txt_path=args.conll_txt,
            **common,
        )
    else:
        run_rationale_generation(split=args.split, output_path=args.output, **common)


if __name__ == "__main__":
    main()
