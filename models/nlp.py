# models/nlp.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

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
    def encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
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


def add_finbert_embeddings_to_csv(
    input_csv: str,
    output_csv: str,
    text_col: str = "title",
    embedding_col: str = "finbert_embedding",
):
    df = pd.read_csv(input_csv)

    texts = df[text_col].fillna("").astype(str).tolist()

    print("Running text embedding now")
    encoder = FinBertEncoder()
    embeddings = encoder.encode(texts)  

    df[embedding_col] = [emb.tolist() for emb in embeddings]

    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows with embeddings to {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to input news CSV")
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
