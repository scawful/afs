"""Encoder pre-training for domain-specific embeddings.

Trains BERT-style encoders on domain corpora using masked language modeling.
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator

from .corpus_builder import CorpusBuilder, CorpusChunk

logger = logging.getLogger(__name__)


@dataclass
class PretrainConfig:
    """Configuration for encoder pre-training."""
    model_name: str = "bert-base-uncased"
    output_dir: Path = Path("pretrained_encoder")
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 1000
    mask_probability: float = 0.15
    max_seq_length: int = 512
    save_steps: int = 1000
    eval_steps: int = 500


@dataclass
class MaskedSample:
    """A masked language modeling sample."""
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]  # -100 for non-masked tokens


class EncoderPretrainer:
    """Pre-trains encoder models on domain corpora."""

    def __init__(
        self,
        corpus_builder: CorpusBuilder,
        config: Optional[PretrainConfig] = None,
    ):
        self.corpus = corpus_builder
        self.config = config or PretrainConfig()
        self._tokenizer = None
        self._model = None

    def _load_tokenizer(self):
        """Load or create tokenizer."""
        if self._tokenizer is not None:
            return self._tokenizer

        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        except ImportError:
            logger.warning("transformers not installed, using simple tokenizer")
            self._tokenizer = SimpleTokenizer()

        return self._tokenizer

    def _load_model(self):
        """Load or create model."""
        if self._model is not None:
            return self._model

        try:
            from transformers import AutoModelForMaskedLM
            self._model = AutoModelForMaskedLM.from_pretrained(self.config.model_name)
        except ImportError:
            raise ImportError("transformers required for pre-training")

        return self._model

    def create_mlm_samples(
        self,
        chunks: Iterator[CorpusChunk],
        max_samples: Optional[int] = None,
    ) -> Iterator[MaskedSample]:
        """Create masked language modeling samples from chunks."""
        tokenizer = self._load_tokenizer()
        count = 0

        for chunk in chunks:
            if max_samples and count >= max_samples:
                break

            # Tokenize
            encoding = tokenizer(
                chunk.text,
                max_length=self.config.max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            # Create labels (copy of input_ids)
            labels = input_ids.copy()

            # Mask random tokens
            for i in range(len(input_ids)):
                if attention_mask[i] == 0:
                    labels[i] = -100  # Ignore padding
                    continue

                if random.random() < self.config.mask_probability:
                    # 80% mask, 10% random, 10% keep
                    r = random.random()
                    if r < 0.8:
                        input_ids[i] = tokenizer.mask_token_id
                    elif r < 0.9:
                        input_ids[i] = random.randint(0, tokenizer.vocab_size - 1)
                    # else keep original
                else:
                    labels[i] = -100  # Don't compute loss on non-masked

            yield MaskedSample(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            count += 1

    def train(self, resume_from: Optional[Path] = None) -> dict:
        """Run pre-training loop."""
        try:
            from transformers import Trainer, TrainingArguments
            import torch
        except ImportError:
            raise ImportError("transformers and torch required for pre-training")

        model = self._load_model()
        tokenizer = self._load_tokenizer()

        # Collect training data
        logger.info("Collecting corpus chunks...")
        chunks = list(self.corpus.build_chunks(tokenizer=tokenizer.tokenize))
        logger.info(f"Collected {len(chunks)} chunks")

        # Create MLM samples
        logger.info("Creating MLM samples...")
        samples = list(self.create_mlm_samples(iter(chunks)))
        logger.info(f"Created {len(samples)} samples")

        # Create dataset
        class MLMDataset(torch.utils.data.Dataset):
            def __init__(self, samples):
                self.samples = samples

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                s = self.samples[idx]
                return {
                    "input_ids": torch.tensor(s.input_ids),
                    "attention_mask": torch.tensor(s.attention_mask),
                    "labels": torch.tensor(s.labels),
                }

        dataset = MLMDataset(samples)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            save_steps=self.config.save_steps,
            logging_steps=100,
            save_total_limit=2,
        )

        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        if resume_from:
            trainer.train(resume_from_checkpoint=str(resume_from))
        else:
            trainer.train()

        # Save final model
        trainer.save_model(str(self.config.output_dir / "final"))
        tokenizer.save_pretrained(str(self.config.output_dir / "final"))

        return {
            "samples_trained": len(samples),
            "output_dir": str(self.config.output_dir),
        }

    def export_embeddings(
        self,
        texts: list[str],
        output_path: Path,
    ) -> int:
        """Export embeddings for texts to FAISS index."""
        try:
            import torch
            import numpy as np
        except ImportError:
            raise ImportError("torch and numpy required for embeddings")

        model = self._load_model()
        tokenizer = self._load_tokenizer()

        model.eval()
        embeddings = []

        with torch.no_grad():
            for text in texts:
                encoding = tokenizer(
                    text,
                    max_length=self.config.max_seq_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                outputs = model.base_model(**encoding)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding[0])

        embeddings = np.array(embeddings)

        # Save embeddings
        np.save(output_path, embeddings)

        return len(embeddings)


class SimpleTokenizer:
    """Simple fallback tokenizer when transformers not available."""

    def __init__(self):
        self.vocab = {}
        self.vocab_size = 30000
        self.mask_token_id = 103
        self.pad_token_id = 0

    def __call__(self, text, **kwargs):
        tokens = text.split()
        max_length = kwargs.get("max_length", 512)

        input_ids = [hash(t) % self.vocab_size for t in tokens[:max_length]]
        attention_mask = [1] * len(input_ids)

        # Pad
        while len(input_ids) < max_length:
            input_ids.append(self.pad_token_id)
            attention_mask.append(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def tokenize(self, text):
        return text.split()
