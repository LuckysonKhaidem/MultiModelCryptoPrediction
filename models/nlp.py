# models/nlp.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import zipfile
import os

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm  # progress bar

FINBERT_MODEL_NAME = "ProsusAI/finbert"


@dataclass
class FinBertEncoder:
    device: Optional[str] = None
    max_length: int = 128

    def __post_init__(self):
        if self.device is None:
            # cpu / cuda / mps (Apple Silicon) if available
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        self.model = AutoModel.from_pretrained(FINBERT_MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Encode a list of texts into CLS embeddings.
        Returns a tensor of shape [len(texts), hidden_size] on CPU.
        """
        all_embeddings = []

        num_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(texts))
            batch = texts[start:end]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            # CLS embedding at position 0
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, H]
            all_embeddings.append(cls_embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)  # [N, H]


def _read_csv_maybe_zipped(path: str) -> pd.DataFrame:
    """
    Read a CSV file, supporting both plain .csv and .zip containing a CSV.
    If it's a zip, we pick the first .csv file inside (or the first file if none end with .csv).
    """
    if not path.lower().endswith(".zip"):
        return pd.read_csv(path)

    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        if not names:
            raise ValueError(f"Zip file {path} is empty")

        csv_names = [n for n in names if n.lower().endswith(".csv")]
        target_name = csv_names[0] if csv_names else names[0]

        print(f"Detected zip input. Reading inner file: {target_name}")
        with zf.open(target_name) as f:
            return pd.read_csv(f)


def _count_existing_rows(path: str) -> int:
    """
    Count how many data rows already exist in output_csv (for resume/checkpoint).
    Assumes there's one header row. Returns 0 if file does not exist.
    """
    if not os.path.exists(path):
        return 0

    with open(path, "r", encoding="utf-8") as f:
        # subtract 1 for header
        lines = sum(1 for _ in f)
    return max(lines - 1, 0)


def add_finbert_embeddings_to_csv(
    input_csv: str,
    output_csv: str,
    text_col: str = "title",
    embedding_col: str = "finbert_embedding",
    row_chunk_size: int = 20000,   # rows per chunk
    batch_size: int = 32,          # FinBERT batch size
):
    """
    Load a CSV (or ZIP containing a CSV), compute FinBERT embeddings for `text_col`
    in row chunks, store them in `embedding_col` as list-of-floats, and write to `output_csv`
    incrementally to avoid blowing up memory.

    If output_csv already exists, we resume from the row after the last written row
    (checkpointing).
    """
    df = _read_csv_maybe_zipped(input_csv)
    n_rows = len(df)
    print(f"Loaded {n_rows} rows from {input_csv}")

    # ---- checkpoint: see how many rows we've already written ----
    already_done = _count_existing_rows(output_csv)
    if already_done > 0:
        print(f"Resuming from checkpoint: {already_done} rows already in {output_csv}")
    else:
        print("No existing checkpoint found. Starting from scratch.")

    if already_done >= n_rows:
        print("All rows already processed. Nothing to do.")
        return

    encoder = FinBertEncoder()

    first_chunk = already_done == 0

    # progress over rows
    for start in tqdm(
        range(0, n_rows, row_chunk_size),
        desc="Encoding rows with FinBERT",
    ):
        end = min(start + row_chunk_size, n_rows)

        # skip chunks that are fully completed
        if end <= already_done:
            continue

        # if we resume mid-chunk, adjust start so we don't re-encode rows
        chunk_start = max(start, already_done)
        sub = df.iloc[chunk_start:end].copy()

        texts = sub[text_col].fillna("").astype(str).tolist()
        if not texts:
            continue

        embeddings = encoder.encode(texts, batch_size=batch_size)

        sub[embedding_col] = [emb.tolist() for emb in embeddings]

        mode = "w" if first_chunk else "a"
        header = first_chunk
        sub.to_csv(output_csv, mode=mode, header=header, index=False)
        first_chunk = False

    print(f"Finished. Saved {n_rows} rows with embeddings to {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to input news CSV or ZIP containing CSV")
    parser.add_argument(
        "--output_csv",
        help="Path to output CSV",
        default="news_with_finbert_embeddings.csv",
    )
    parser.add_argument(
        "--text_col",
        help="Text column to encode (default: title)",
        default="title",
    )
    parser.add_argument(
        "--row_chunk_size",
        type=int,
        default=100,
        help="Number of rows to process per chunk (default: 20000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="FinBERT batch size (default: 32)",
    )
    args = parser.parse_args()

    add_finbert_embeddings_to_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        text_col=args.text_col,
        row_chunk_size=args.row_chunk_size,
        batch_size=args.batch_size,
    )
