"""
Общий код для генерации рационалов (API) и обучения студента (DistilBERT + NER-голова + DistilGPT2).
Используется скриптами в scripts/ и ноутбуками notebooks/*_kaggle.ipynb.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, EncoderDecoderModel

PACKAGE_ROOT = Path(__file__).resolve().parent

# Как в классическом HF ``conll2003`` (порядок id для NER).
CONLL2003_NER_LABEL_NAMES: tuple[str, ...] = (
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
)

CONLL2003_RAW_URLS: dict[str, str] = {
    "validation": "https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.testa",
    "test": "https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.testb",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- generation ---


def build_prompt(tokens: list[str], tags: list[str]) -> str:
    words_line = " ".join(tokens)
    tags_line = " ".join(tags)
    return (
        "You assist with named entity recognition (NER) on English CoNLL-style BIO tags "
        "(O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC).\n\n"
        "Given the sentence and the gold BIO tags (one tag per word, same order), write "
        "ONE short paragraph in English explaining why these spans and entity types are "
        "reasonable. Do not change or restate the tags as a new sequence. Do not output JSON.\n\n"
        f"Sentence:\n{words_line}\n\n"
        f"Gold BIO tags:\n{tags_line}\n\n"
        "Explanation:"
    )


def make_client():
    import httpx
    from openai import OpenAI

    base_url = os.environ.get("GPT_OSS_BASE_URL", "").strip().rstrip("/")
    api_key = os.environ.get("GPT_OSS_API_KEY", "").strip()
    model = os.environ.get("GPT_OSS_MODEL", "").strip()
    http_user = os.environ.get("GPT_OSS_HTTP_USER", "").strip()
    http_password = os.environ.get("GPT_OSS_HTTP_PASSWORD", "").strip()

    if not base_url or not api_key or not model:
        raise ValueError(
            "Задайте переменные окружения GPT_OSS_BASE_URL, GPT_OSS_API_KEY и GPT_OSS_MODEL "
            "(Kaggle: Secrets или Add-ons → Environment)."
        )

    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    auth = None
    if http_user or http_password:
        auth = httpx.BasicAuth(http_user or "", http_password or "")

    http_client = httpx.Client(auth=auth, timeout=httpx.Timeout(120.0, connect=30.0))
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client), model


def extract_rationale_from_message(msg: Any) -> str:
    """
    Текст ответа модели. У gpt-oss ответ может быть в ``reasoning``,
    а ``content`` остаётся null, пока не выделено достаточно max_tokens.
    """
    text = (getattr(msg, "content", None) or "").strip()
    if text:
        return text
    reason = getattr(msg, "reasoning", None)
    if reason is None and hasattr(msg, "model_dump"):
        try:
            reason = msg.model_dump().get("reasoning")
        except Exception:  # noqa: BLE001
            reason = None
    if reason is not None:
        return str(reason).strip()
    return ""


def ner_tags_to_labels(example: dict[str, Any], label_names: list[str]) -> list[str]:
    return [label_names[i] for i in example["ner_tags"]]


def load_conll2003_txt(path: Path) -> list[dict[str, Any]]:
    """
    CoNLL-2003 .txt: предложения разделены пустой строкой; в строке ≥2 полей —
    токен = первый столбец, NER-тег = последний. Строки -DOCSTART- пропускаются.
    """
    path = Path(path)
    cur: list[tuple[str, str]] = []
    out: list[dict[str, Any]] = []

    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            if not line.strip():
                if cur:
                    out.append({"tokens": [a for a, _ in cur], "tags": [b for _, b in cur]})
                    cur = []
                continue
            line_st = line.strip()
            if line_st.startswith("-DOCSTART-"):
                continue
            parts = line_st.split()
            if len(parts) < 2:
                continue
            token, tag = parts[0], parts[-1]
            cur.append((token, tag))
    if cur:
        out.append({"tokens": [a for a, _ in cur], "tags": [b for _, b in cur]})
    return out


def resolve_conll_txt_for_split(conll_dir: Path, split: str) -> Path | None:
    """train.txt / valid.txt|validation.txt|dev.txt / test.txt в каталоге датасета."""
    conll_dir = Path(conll_dir)
    candidates: dict[str, tuple[str, ...]] = {
        "train": ("train.txt", "eng.train", "train.conll"),
        "validation": ("valid.txt", "validation.txt", "dev.txt", "eng.testa"),
        "test": ("test.txt", "eng.testb"),
    }
    for name in candidates.get(split, ()):
        p = conll_dir / name
        if p.is_file():
            return p
    return None


def run_rationale_generation(
    *,
    split: str,
    output_path: Path,
    limit: int = 0,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    sleep_s: float = 0.0,
    seed: int = 42,
    dotenv_path: Path | None = None,
    conll_txt_path: Path | None = None,
) -> Path:
    """Пишет JSONL: id, split, tokens, tags, rationale.

    Если задан conll_txt_path — читает предложения из этого CoNLL-файла.
    Иначе — сплит из Hugging Face ``conll2003``.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path or PACKAGE_ROOT / ".env")
    except ImportError:
        pass

    random.seed(seed)

    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train|validation|test")

    sents: list[dict[str, Any]] | None = None
    part = None
    label_names: list[str] | None = None

    if conll_txt_path is not None:
        sents = load_conll2003_txt(Path(conll_txt_path))
        n_all = len(sents)
    else:
        from datasets import load_dataset

        ds = load_dataset("conll2003")
        part = ds[split]
        label_names = list(part.features["ner_tags"].feature.names)
        n_all = len(part)

    client, model_name = make_client()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if limit <= 0:
        indices = list(range(n_all))
        random.shuffle(indices)
    else:
        indices = random.sample(range(n_all), min(limit, n_all))
    n_total = len(indices)

    with tmp_path.open("w", encoding="utf-8") as fout:
        for idx in tqdm(indices, desc="Генерация rationales", unit="ex"):
            if sents is not None:
                ex = sents[idx]
                tokens = ex["tokens"]
                tags = ex["tags"]
            else:
                assert part is not None and label_names is not None
                row = part[idx]
                tokens = row["tokens"]
                tags = ner_tags_to_labels(row, label_names)
            prompt = build_prompt(tokens, tags)

            rationale = ""
            last_err: BaseException | None = None
            for attempt in range(5):
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    choice = resp.choices[0].message
                    rationale = extract_rationale_from_message(choice)
                    if not rationale:
                        ch0 = resp.choices[0]
                        fr = getattr(ch0, "finish_reason", None)
                        raise ValueError(f"Пустой ответ модели (finish_reason={fr!r})")
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    wait = min(60.0, 2.0**attempt)
                    print(f"[warn] idx={idx} attempt {attempt + 1}/5: {e!r}; sleep {wait:.1f}s")
                    time.sleep(wait)

            if not rationale:
                raise RuntimeError(f"Не удалось получить rationale для idx={idx}: {last_err!r}")

            row = {
                "id": int(idx),
                "split": split,
                "tokens": tokens,
                "tags": tags,
                "rationale": rationale,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            if sleep_s > 0:
                time.sleep(sleep_s)

    tmp_path.replace(output_path)
    print(f"Готово: {n_total} строк -> {output_path}")
    return output_path


def run_rationale_generation_all_conll_txt(
    *,
    conll_dir: Path,
    output_dir: Path,
    limit: int = 0,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    sleep_s: float = 0.0,
    seed: int = 42,
    dotenv_path: Path | None = None,
) -> dict[str, Path]:
    """
    Для train / validation / test ищет .txt в ``conll_dir`` и пишет JSONL в ``output_dir``:
    train_with_rationale.jsonl, valid_with_rationale.jsonl, test_with_rationale.jsonl.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs: list[tuple[str, str]] = [
        ("train", "train_with_rationale.jsonl"),
        ("validation", "valid_with_rationale.jsonl"),
        ("test", "test_with_rationale.jsonl"),
    ]
    done: dict[str, Path] = {}
    for split, out_name in specs:
        txt = resolve_conll_txt_for_split(conll_dir, split)
        if txt is None:
            print(f"[skip] нет .txt для split={split} в {conll_dir}")
            continue
        outp = output_dir / out_name
        run_rationale_generation(
            split=split,
            output_path=outp,
            limit=limit,
            temperature=temperature,
            max_tokens=max_tokens,
            sleep_s=sleep_s,
            seed=seed,
            dotenv_path=dotenv_path,
            conll_txt_path=txt,
        )
        done[split] = outp
    if not done:
        raise FileNotFoundError(f"Не найдено ни одного train/valid/test .txt в {conll_dir}")
    return done


# --- training ---


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def eval_rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    """Только tokens/tags для NER-оценки (как load_hf_eval_rows)."""
    rows = read_jsonl(path)
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({"tokens": r["tokens"], "tags": r["tags"]})
    return out


def conll_label_vocab() -> tuple[dict[str, int], list[str]]:
    names = list(CONLL2003_NER_LABEL_NAMES)
    return {n: i for i, n in enumerate(names)}, names


def load_conll_eval_rows_from_dir(conll_dir: Path, split: str, max_n: int = 0) -> list[dict[str, Any]]:
    """Локальные ``train.txt`` / ``valid.txt`` / ``test.txt`` (как на Kaggle-датасетах)."""
    p = resolve_conll_txt_for_split(Path(conll_dir), split)
    if p is None:
        raise FileNotFoundError(
            f"В {conll_dir!r} нет файла для split={split!r} "
            "(ожидаются train.txt, valid.txt|validation.txt|dev.txt, test.txt)."
        )
    rows = load_conll2003_txt(p)
    if max_n and len(rows) > max_n:
        rows = rows[:max_n]
    return rows


class JsonlNerRationaleDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        enc_tok: Any,
        dec_tok: Any,
        label2id: dict[str, int],
        max_enc_len: int,
        max_dec_len: int,
    ) -> None:
        self.rows = rows
        self.enc_tok = enc_tok
        self.dec_tok = dec_tok
        self.label2id = label2id
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.rows[idx]
        tokens = ex["tokens"]
        tags = ex["tags"]
        rationale = ex.get("rationale") or ""

        enc = self.enc_tok(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_enc_len,
            padding=False,
        )
        word_ids = enc.word_ids()
        ner_labels = [-100] * len(enc["input_ids"])
        for word_idx, tag in enumerate(tags):
            if word_idx >= len(tokens):
                break
            lab = self.label2id[tag]
            positions = [i for i, w in enumerate(word_ids) if w == word_idx]
            if not positions:
                continue
            ner_labels[positions[0]] = lab

        dec = self.dec_tok(
            rationale,
            truncation=True,
            max_length=self.max_dec_len,
            padding=False,
            add_special_tokens=True,
        )
        decoder_labels = list(dec["input_ids"])

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "ner_labels": ner_labels,
            "decoder_labels": decoder_labels,
        }


def collate_batch(batch: list[dict[str, Any]], enc_pad: int, dec_pad: int) -> dict[str, torch.Tensor]:
    max_e = max(len(x["input_ids"]) for x in batch)
    max_d = max(len(x["decoder_labels"]) for x in batch)

    bs = len(batch)
    input_ids = torch.full((bs, max_e), enc_pad, dtype=torch.long)
    attention_mask = torch.zeros((bs, max_e), dtype=torch.long)
    ner_labels = torch.full((bs, max_e), -100, dtype=torch.long)
    decoder_labels = torch.full((bs, max_d), -100, dtype=torch.long)

    for i, x in enumerate(batch):
        n = len(x["input_ids"])
        input_ids[i, :n] = torch.tensor(x["input_ids"], dtype=torch.long)
        attention_mask[i, :n] = torch.tensor(x["attention_mask"], dtype=torch.long)
        ner_labels[i, :n] = torch.tensor(x["ner_labels"], dtype=torch.long)
        m = len(x["decoder_labels"])
        decoder_labels[i, :m] = torch.tensor(x["decoder_labels"], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ner_labels": ner_labels,
        "decoder_labels": decoder_labels,
    }


class DualHeadStudent(nn.Module):
    """
    Энкодер DistilBERT (через EncoderDecoderModel.get_encoder) + линейная NER-голова;
    полный EncoderDecoder для teacher-forcing по рационалу (decoder = DistilGPT2).
    Итоговый лосс: alpha * CE(NER) + (1 - alpha) * LM(rationale).
    """

    def __init__(self, enc_dec: EncoderDecoderModel, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.enc_dec = enc_dec
        h = enc_dec.config.encoder.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.ner_head = nn.Linear(h, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ner_labels: torch.Tensor,
        decoder_labels: torch.Tensor,
        alpha: float,
    ) -> dict[str, torch.Tensor]:
        enc_out = self.enc_dec.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        h = self.dropout(enc_out.last_hidden_state)
        ner_logits = self.ner_head(h)
        loss_ner = F.cross_entropy(
            ner_logits.view(-1, self.num_labels),
            ner_labels.view(-1),
            ignore_index=-100,
        )

        out = self.enc_dec(
            encoder_outputs=enc_out,
            attention_mask=attention_mask,
            labels=decoder_labels,
        )
        loss_gen = out.loss
        if loss_gen is None or torch.isnan(loss_gen):
            loss_gen = torch.zeros((), device=input_ids.device, dtype=loss_ner.dtype)

        loss = alpha * loss_ner + (1.0 - alpha) * loss_gen
        return {
            "loss": loss,
            "loss_ner": loss_ner.detach(),
            "loss_gen": loss_gen.detach(),
            "ner_logits": ner_logits,
        }


@torch.no_grad()
def eval_ner_only(
    model: DualHeadStudent,
    rows: list[dict[str, Any]],
    enc_tok: Any,
    id2label: list[str],
    device: torch.device,
    max_enc_len: int,
) -> dict[str, Any]:
    model.eval()
    from seqeval.metrics import classification_report, f1_score

    all_preds: list[list[str]] = []
    all_refs: list[list[str]] = []

    for ex in rows:
        tokens = ex["tokens"]
        tags = ex["tags"]
        enc = enc_tok(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_enc_len,
            padding=False,
        )
        word_ids = enc.word_ids()
        input_ids = torch.tensor([enc["input_ids"]], device=device)
        attention_mask = torch.tensor([enc["attention_mask"]], device=device)
        enc_out = model.enc_dec.get_encoder()(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = model.ner_head(model.dropout(enc_out.last_hidden_state))
        pred_ids = logits.argmax(-1)[0].tolist()

        preds_words: list[str] = []
        refs_words = list(tags[: len(tokens)])
        for widx in range(len(tokens)):
            pos = next((j for j, wid in enumerate(word_ids) if wid == widx), None)
            if pos is None:
                preds_words.append("O")
                continue
            pid = pred_ids[pos]
            preds_words.append(id2label[pid] if 0 <= pid < len(id2label) else "O")
        all_preds.append(preds_words)
        all_refs.append(refs_words)

    f1 = f1_score(all_refs, all_preds)
    return {
        "f1": float(f1),
        "report": classification_report(all_refs, all_preds),
        "n_sentences": len(all_refs),
    }


def _save_training_checkpoint(
    path: Path,
    model: DualHeadStudent,
    enc_tok: Any,
    dec_tok: Any,
    label2id: dict[str, int],
    id2label: list[str],
    alpha: float,
    encoder: str,
    decoder: str,
    epoch: int,
    val_f1: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_f1": val_f1,
            "model_state": model.state_dict(),
            "ner_head": model.ner_head.state_dict(),
            "label2id": label2id,
            "id2label": id2label,
            "alpha": alpha,
            "encoder_name": encoder,
            "decoder_name": decoder,
        },
        path,
    )
    model.enc_dec.save_pretrained(path.parent / "encoder_decoder")
    enc_tok.save_pretrained(path.parent / "tokenizer_encoder")
    dec_tok.save_pretrained(path.parent / "tokenizer_decoder")


def _load_training_checkpoint(path: Path, model: DualHeadStudent) -> dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return ckpt


def load_hf_eval_rows(split: str, max_n: int = 0) -> list[dict[str, Any]]:
    """
    Dev/test CoNLL-2003 без ``load_dataset('conll2003')`` (в datasets 3.x скрипты Hub отключены).
    """
    if split not in CONLL2003_RAW_URLS:
        raise ValueError(f"split must be one of {tuple(CONLL2003_RAW_URLS)}, got {split!r}")
    url = CONLL2003_RAW_URLS[split]
    req = urllib.request.Request(url, headers={"User-Agent": "fp_project-rationale-student/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 — фиксированные URL
            raw = resp.read()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Не удалось скачать CoNLL для оценки ({url}): {e}") from e
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    try:
        rows = load_conll2003_txt(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    if max_n and len(rows) > max_n:
        rows = rows[:max_n]
    return rows


def run_training(
    *,
    train_jsonl: Path,
    output_dir: Path,
    encoder: str = "distilbert-base-cased",
    decoder: str = "distilgpt2",
    epochs: int = 25,
    batch_size: int = 8,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    alpha: float = 0.7,
    max_enc_len: int = 256,
    max_dec_len: int = 256,
    seed: int = 42,
    fp16: bool = False,
    eval_hf: bool = False,
    eval_conll_dir: Path | None = None,
    val_jsonl: Path | None = None,
    batch_log_every: int = 10,
    test_jsonl: Path | None = None,
    early_stopping_patience: int = 6,
    eval_conll_test: bool = True,
) -> dict[str, Any]:
    """
    Обучение студента. По val F1 сохраняется лучший чекпойнт; после обучения — оценка на test.

    Val: ``val_jsonl`` → иначе ``eval_conll_dir``/valid.txt → иначе ``eval_hf``.
  Test: ``test_jsonl`` и при ``eval_conll_test`` — test.txt из ``eval_conll_dir``.
    Итог: ``final_metrics.json``, ``test_ner_metrics.json``, ``best_checkpoint.pt``.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, list] = {
        "epoch": [],
        "train_loss_mean": [],
        "train_loss_ner_mean": [],
        "train_loss_gen_mean": [],
        "val_f1": [],
        "global_step": [],
        "batch_loss": [],
        "batch_loss_ner": [],
        "batch_loss_gen": [],
    }

    train_rows = read_jsonl(train_jsonl)
    if not train_rows:
        raise ValueError("Пустой train JSONL")

    label2id, id2label = conll_label_vocab()
    for tags in (ex["tags"] for ex in train_rows):
        for t in tags:
            if t not in label2id:
                raise ValueError(f"Неизвестный тег {t!r} (ожидаются теги CoNLL-2003).")

    test_rows_eval: list[dict[str, Any]] = []
    if test_jsonl is not None:
        test_path = Path(test_jsonl)
        if not test_path.is_file():
            raise FileNotFoundError(f"test_jsonl не найден: {test_path}")
        test_rows_eval = eval_rows_from_jsonl(test_path)
        if not test_rows_eval:
            raise ValueError("Пустой test JSONL")
        for tags in (ex["tags"] for ex in test_rows_eval):
            for t in tags:
                if t not in label2id:
                    raise ValueError(f"Неизвестный тег в test: {t!r}")

    enc_tok = AutoTokenizer.from_pretrained(encoder, use_fast=True)
    dec_tok = AutoTokenizer.from_pretrained(decoder, use_fast=True)
    if dec_tok.pad_token is None:
        dec_tok.pad_token = dec_tok.eos_token

    enc_dec = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder, decoder)
    enc_dec.config.decoder_start_token_id = dec_tok.bos_token_id or dec_tok.eos_token_id
    enc_dec.config.pad_token_id = dec_tok.pad_token_id
    enc_dec.config.eos_token_id = dec_tok.eos_token_id
    enc_dec.config.vocab_size = enc_dec.config.decoder.vocab_size

    model = DualHeadStudent(enc_dec, num_labels=len(label2id)).to(device)
    ds = JsonlNerRationaleDataset(train_rows, enc_tok, dec_tok, label2id, max_enc_len, max_dec_len)

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return collate_batch(batch, enc_pad=enc_tok.pad_token_id or 0, dec_pad=dec_tok.pad_token_id or 0)

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_amp = bool(fp16 and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    eval_rows: list[dict[str, Any]] = []
    if val_jsonl is not None:
        vp = Path(val_jsonl)
        if not vp.is_file():
            raise FileNotFoundError(f"val_jsonl не найден: {vp}")
        eval_rows = eval_rows_from_jsonl(vp)
        if not eval_rows:
            raise ValueError("Пустой val JSONL")
        for tags in (ex["tags"] for ex in eval_rows):
            for t in tags:
                if t not in label2id:
                    raise ValueError(f"Неизвестный тег в val: {t!r}")
    elif eval_conll_dir is not None:
        eval_rows = load_conll_eval_rows_from_dir(Path(eval_conll_dir), "validation")
        for tags in (ex["tags"] for ex in eval_rows):
            for t in tags:
                if t not in label2id:
                    raise ValueError(f"Неизвестный тег в conll val: {t!r}")
    elif eval_hf:
        eval_rows = load_hf_eval_rows("validation")

    global_step = 0
    batch_log_every = max(1, int(batch_log_every))
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improve = 0
    best_ckpt_path = output_dir / "best_checkpoint.pt"
    early_stopping_patience = max(0, int(early_stopping_patience))

    epoch_pbar = tqdm(range(1, epochs + 1), desc="Эпохи", unit="epoch", position=0, leave=True)
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0.0
        total_ner = 0.0
        total_gen = 0.0
        n_batches = 0

        batch_pbar = tqdm(
            dl,
            desc=f"Батчи ep {epoch}/{epochs}",
            leave=False,
            unit="batch",
            position=1,
        )
        for batch in batch_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        ner_labels=batch["ner_labels"],
                        decoder_labels=batch["decoder_labels"],
                        alpha=alpha,
                    )
                    loss = out["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
            else:
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    ner_labels=batch["ner_labels"],
                    decoder_labels=batch["decoder_labels"],
                    alpha=alpha,
                )
                loss = out["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            lv = float(loss.detach())
            ln = float(out["loss_ner"])
            lg = float(out["loss_gen"])
            total_loss += lv
            total_ner += ln
            total_gen += lg
            n_batches += 1
            global_step += 1

            if (n_batches % batch_log_every) == 0:
                history["global_step"].append(global_step)
                history["batch_loss"].append(lv)
                history["batch_loss_ner"].append(ln)
                history["batch_loss_gen"].append(lg)

            batch_pbar.set_postfix(loss=lv, ner=ln, gen=lg, step=global_step)

        mean_loss = total_loss / max(n_batches, 1)
        mean_ner = total_ner / max(n_batches, 1)
        mean_gen = total_gen / max(n_batches, 1)
        history["epoch"].append(epoch)
        history["train_loss_mean"].append(mean_loss)
        history["train_loss_ner_mean"].append(mean_ner)
        history["train_loss_gen_mean"].append(mean_gen)

        val_f1: float | None = None
        if eval_rows:
            m = eval_ner_only(model, eval_rows, enc_tok, id2label, device, max_enc_len)
            val_f1 = float(m["f1"])
            history["val_f1"].append(val_f1)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                epochs_without_improve = 0
                _save_training_checkpoint(
                    best_ckpt_path,
                    model,
                    enc_tok,
                    dec_tok,
                    label2id,
                    id2label,
                    alpha,
                    encoder,
                    decoder,
                    epoch,
                    val_f1,
                )
            else:
                epochs_without_improve += 1
            print(f"epoch {epoch} mean_loss={mean_loss:.4f}  val_F1={val_f1:.4f}  best={best_val_f1:.4f}@{best_epoch}")
        else:
            history["val_f1"].append(float("nan"))
            print(f"epoch {epoch} mean_loss={mean_loss:.4f}")

        epoch_pbar.set_postfix(loss=f"{mean_loss:.4f}", f1=f"{val_f1:.4f}" if val_f1 is not None else "n/a")

        if (
            early_stopping_patience > 0
            and eval_rows
            and epochs_without_improve >= early_stopping_patience
        ):
            print(f"Early stopping: val F1 не улучшался {early_stopping_patience} эпох (лучшая ep {best_epoch})")
            break

    if best_ckpt_path.is_file():
        meta = _load_training_checkpoint(best_ckpt_path, model)
        print(f"Загружена лучшая модель: эпоха {meta['epoch']}, val F1={meta['val_f1']:.4f}")
    model.eval()

    final_metrics: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1 if best_epoch > 0 else None,
        "epochs_trained": history["epoch"][-1] if history["epoch"] else 0,
        "hyperparameters": {
            "encoder": encoder,
            "decoder": decoder,
            "epochs_max": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "alpha": alpha,
            "early_stopping_patience": early_stopping_patience,
        },
        "validation": None,
        "test_jsonl": None,
        "test_conll_txt": None,
    }

    if eval_rows:
        val_final = eval_ner_only(model, eval_rows, enc_tok, id2label, device, max_enc_len)
        final_metrics["validation"] = {
            "f1": val_final["f1"],
            "n_sentences": val_final["n_sentences"],
            "source": str(val_jsonl) if val_jsonl else ("conll_dir" if eval_conll_dir else "hf_download"),
        }

    test_ner: dict[str, Any] | None = None
    if test_rows_eval:
        test_ner = eval_ner_only(model, test_rows_eval, enc_tok, id2label, device, max_enc_len)
        final_metrics["test_jsonl"] = {
            "f1": test_ner["f1"],
            "n_sentences": test_ner["n_sentences"],
            "path": str(test_jsonl),
            "report": test_ner["report"],
        }
        with (output_dir / "test_ner_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"f1": test_ner["f1"], "report": test_ner["report"], "path": str(test_jsonl)},
                f,
                indent=2,
                ensure_ascii=False,
            )

    if eval_conll_test and eval_conll_dir is not None:
        conll_test_path = resolve_conll_txt_for_split(Path(eval_conll_dir), "test")
        if conll_test_path is not None:
            conll_test_rows = load_conll2003_txt(conll_test_path)
            conll_m = eval_ner_only(model, conll_test_rows, enc_tok, id2label, device, max_enc_len)
            final_metrics["test_conll_txt"] = {
                "f1": conll_m["f1"],
                "n_sentences": conll_m["n_sentences"],
                "path": str(conll_test_path),
                "report": conll_m["report"],
            }

    with (output_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    torch.save(
        {
            "ner_head": model.ner_head.state_dict(),
            "label2id": label2id,
            "id2label": id2label,
            "alpha": alpha,
            "encoder_name": encoder,
            "decoder_name": decoder,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
        },
        output_dir / "student_head.pt",
    )

    print("\n" + "=" * 60)
    print("ИТОГОВЫЕ МЕТРИКИ NER (seqeval span F1, лучший чекпойнт по val)")
    print("=" * 60)
    if final_metrics.get("validation"):
        print(f"  Validation F1: {final_metrics['validation']['f1']:.4f}")
    if final_metrics.get("test_jsonl"):
        print(f"  Test JSONL F1:  {final_metrics['test_jsonl']['f1']:.4f}  ({final_metrics['test_jsonl']['path']})")
    if final_metrics.get("test_conll_txt"):
        print(
            f"  Test CoNLL F1:  {final_metrics['test_conll_txt']['f1']:.4f}  "
            f"({final_metrics['test_conll_txt']['path']})"
        )
    if not final_metrics.get("test_jsonl") and not final_metrics.get("test_conll_txt"):
        print("  Test: не задан (укажите test_jsonl или eval_conll_dir с test.txt)")
    print(f"  Файлы: {output_dir / 'final_metrics.json'}")
    print("=" * 60 + "\n")

    if test_ner is not None:
        print("Classification report (test JSONL):\n")
        print(test_ner["report"])

    out: dict[str, Any] = {
        "output_dir": output_dir,
        "history": history,
        "final_metrics": final_metrics,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
    }
    if test_ner is not None:
        out["test_ner"] = test_ner
    return out
