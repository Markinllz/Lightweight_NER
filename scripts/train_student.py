#!/usr/bin/env python3
"""CLI: обучение студента. Логика в rationale_student.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rationale_student import run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "checkpoints" / "student_distilbert")
    parser.add_argument("--encoder", type=str, default="distilbert-base-cased")
    parser.add_argument("--decoder", type=str, default="distilgpt2")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--max_enc_len", type=int, default=256)
    parser.add_argument("--max_dec_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--eval_hf", action="store_true")
    parser.add_argument(
        "--eval-conll-dir",
        type=Path,
        default=None,
        help="Каталог с valid.txt|train.txt (CoNLL) для val NER F1; игнорируется, если задан --val",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=None,
        help="JSONL валидации (tokens/tags/rationale) — приоритетнее --eval-conll-dir и --eval_hf",
    )
    parser.add_argument(
        "--batch-log-every",
        type=int,
        default=10,
        help="Логировать батч в history каждые N шагов",
    )
    parser.add_argument(
        "--no-eval-conll-test",
        action="store_true",
        help="Не оценивать test.txt из --eval-conll-dir",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=None,
        help="JSONL с этапа генерации (test) — после обучения NER span F1 на тесте",
    )
    args = parser.parse_args()

    res = run_training(
        train_jsonl=args.train,
        output_dir=args.output_dir,
        encoder=args.encoder,
        decoder=args.decoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        early_stopping_patience=args.early_stopping_patience,
        eval_conll_test=not args.no_eval_conll_test,
        max_enc_len=args.max_enc_len,
        max_dec_len=args.max_dec_len,
        seed=args.seed,
        fp16=args.fp16,
        eval_hf=args.eval_hf,
        eval_conll_dir=args.eval_conll_dir,
        val_jsonl=args.val,
        batch_log_every=args.batch_log_every,
        test_jsonl=args.test,
    )
    print(res["output_dir"])
    fm = res.get("final_metrics") or {}
    if fm.get("test_jsonl"):
        print("test JSONL F1:", fm["test_jsonl"]["f1"])
    if fm.get("test_conll_txt"):
        print("test CoNLL F1:", fm["test_conll_txt"]["f1"])


if __name__ == "__main__":
    main()
