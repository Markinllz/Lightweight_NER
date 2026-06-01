# Lightweight NER with Rationale-Augmented Training

This repository contains the code and report for a course research project on named entity recognition (NER) with rationale-augmented training. The project studies a compact student model for CoNLL-2003: a DistilBERT encoder with a token-classification head, trained together with a DistilGPT2 rationale decoder used only during optimization.

At inference time the decoder is not used; tagging is performed by the encoder and NER head only. The main metric is span-level F1 computed with `seqeval`.

## Repository structure

- `report/final_report_lms.pdf` — final report with the official title page.
- `notebooks/01_kaggle_generate_rationales.ipynb` — generation of rationale-augmented training and validation files.
- `notebooks/02_kaggle_train_student.ipynb` — student training and evaluation notebook.
- `rationale_student.py` — reusable training and evaluation code.
- `scripts/` — command-line entry points for rationale generation and student training.
- `graphics/` — figures used in the report.
- `requirements-ml.txt` — Python dependencies for the ML pipeline.

## Results

The reported experiment trains on rationale-augmented CoNLL-2003 data and evaluates on the official test split. The best run reaches 94.88% validation span F1 and 93.35% test span F1. The result is intended as an implementation and evaluation of rationale-supervised NER training; a controlled no-rationale ablation is left for future work.

## Running the notebooks

The notebooks are prepared for Kaggle. Add the CoNLL-2003 dataset as an input and configure the required API variables for rationale generation in Kaggle Secrets.

```bash
pip install -r requirements-ml.txt
```

For local script-based runs, see `scripts/generate_rationales.py` and `scripts/train_student.py`.

## Report

The final PDF report is available at `report/final_report_lms.pdf`.
