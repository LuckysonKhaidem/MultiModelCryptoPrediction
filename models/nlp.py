# models/nlp.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import io
import zipfile

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
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        self.model = AutoModel.from_pretrained(FINBERT_MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Encode a list of texts into CLS embeddings.

        Returns a tensor of shape [len(texts), hidden_size].
        """
        all_embeddings = []

        indices = range(0, len(texts), batch_size)
        if show_progress:
            num_batches = (len(texts) + batch_size - 1) // batch_size
            indices = tqdm(
                indices,
                total=num_batches,
                desc="Encoding with FinBERT",
            )

        for i in indices:
            batch = texts[i : i + batch_size]
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
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)


def _read_csv_maybe_zipped(path: str) -> pd.DataFrame:
    """
    Read a CSV file, supporting both plain .csv and .zip containing a CSV.
    If it's a zip, we pick the first .csv file inside (or the first file if none end with .csv).
    """
    if not path.lower().endswith(".zip"):
        return pd.read_csv(path)

    # Zip case: open in memory and read inner CSV
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        if not names:
            raise ValueError(f"Zip file {path} is empty")

        # Prefer a .csv file if present
        csv_names = [n for n in names if n.lower().endswith(".csv")]
        target_name = csv_names[0] if csv_names else names[0]

        print(f"Detected zip input. Reading inner file: {target_name}")
        with zf.open(target_name) as f:
            # f is a file-like object with bytes
            return pd.read_csv(f)


def add_finbert_embeddings_to_csv(
    input_csv: str,
    output_csv: str,
    text_col: str = "title",
    embedding_col: str = "finbert_embedding",
):
    """
    Load a CSV (or ZIP containing a CSV), compute FinBERT embeddings for `text_col`,
    store them in `embedding_col` as list-of-floats, and write to `output_csv`.
    """
    df = _read_csv_maybe_zipped(input_csv)

    texts = df[text_col].fillna("").astype(str).tolist()

    print("Running text embedding now...")
    encoder = FinBertEncoder()
    embeddings = encoder.encode(texts, batch_size=32, show_progress=True)

    df[embedding_col] = [emb.tolist() for emb in embeddings]

    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows with embeddings to {output_csv}")


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
    args = parser.parse_args()

    add_finbert_embeddings_to_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        text_col=args.text_col,
    )
