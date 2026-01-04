"""ASM-ELECTRA: Discriminator for 65816 assembly quality.

Fine-tunes ELECTRA to distinguish real assembly from generated errors.
Can be used for:
- Pre-training data filtering
- Reward model for RLHF
- Inference-time rejection sampling
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class ElectraConfig:
    """Configuration for ASM-ELECTRA training."""

    # Model
    base_model: str = "google/electra-base-discriminator"
    max_length: int = 512

    # Training
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./asm-electra"))
    save_steps: int = 500

    # Device
    device: str = "auto"  # auto, cpu, cuda, mps

    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "max_length": self.max_length,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "output_dir": str(self.output_dir),
            "save_steps": self.save_steps,
        }


class ElectraDatasetWrapper(Dataset):
    """PyTorch Dataset wrapper for ELECTRA training."""

    def __init__(
        self,
        samples: list[dict],
        tokenizer: Any,
        max_length: int = 512,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(sample["label"], dtype=torch.long),
        }


class ASMElectra:
    """ASM-ELECTRA discriminator for assembly quality scoring."""

    def __init__(
        self,
        config: ElectraConfig | None = None,
        model_path: Path | str | None = None,
    ):
        """Initialize ASM-ELECTRA.

        Args:
            config: Training configuration (for new models)
            model_path: Path to load existing model from
        """
        self.config = config or ElectraConfig()
        self.model = None
        self.tokenizer = None
        self._device = None

        if model_path:
            self.load(model_path)

    @property
    def device(self) -> torch.device:
        if self._device is None:
            if self.config.device == "auto":
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                elif torch.backends.mps.is_available():
                    self._device = torch.device("mps")
                else:
                    self._device = torch.device("cpu")
            else:
                self._device = torch.device(self.config.device)
        return self._device

    def _load_base_model(self) -> None:
        """Load base ELECTRA model and tokenizer."""
        from transformers import (
            ElectraForSequenceClassification,
            ElectraTokenizerFast,
        )

        self.tokenizer = ElectraTokenizerFast.from_pretrained(
            self.config.base_model
        )
        self.model = ElectraForSequenceClassification.from_pretrained(
            self.config.base_model,
            num_labels=2,  # real vs fake
        )
        self.model.to(self.device)

    def train(
        self,
        train_data: list[dict],
        val_data: list[dict] | None = None,
        callbacks: list | None = None,
    ) -> dict:
        """Train the discriminator.

        Args:
            train_data: List of {"text": str, "label": int} dicts
            val_data: Optional validation data
            callbacks: Optional training callbacks

        Returns:
            Training metrics
        """
        from transformers import (
            Trainer,
            TrainingArguments,
            EarlyStoppingCallback,
        )
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        if self.model is None:
            self._load_base_model()

        # Create datasets
        train_dataset = ElectraDatasetWrapper(
            train_data, self.tokenizer, self.config.max_length
        )
        val_dataset = None
        if val_data:
            val_dataset = ElectraDatasetWrapper(
                val_data, self.tokenizer, self.config.max_length
            )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            save_steps=self.config.save_steps,
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=self.config.save_steps if val_dataset else None,
            logging_steps=100,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_f1" if val_dataset else None,
            save_total_limit=2,
        )

        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="binary"
            )
            acc = accuracy_score(labels, preds)
            return {
                "accuracy": acc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }

        # Setup callbacks
        trainer_callbacks = callbacks or []
        if val_dataset:
            trainer_callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

        # Train
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=trainer_callbacks,
        )

        result = trainer.train()

        # Save final model
        self.save(self.config.output_dir / "final")

        return {
            "train_loss": result.training_loss,
            "epochs": self.config.epochs,
            "steps": result.global_step,
        }

    def score(self, text: str) -> float:
        """Score a single text sample.

        Returns probability that the text is REAL (not fake).
        Higher score = more likely to be real/correct assembly.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train() first.")

        self.model.eval()
        with torch.no_grad():
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            outputs = self.model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1)

            # Return probability of being real (label 0)
            return probs[0, 0].item()

    def score_batch(self, texts: list[str], batch_size: int = 32) -> list[float]:
        """Score multiple texts efficiently."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() or train() first.")

        self.model.eval()
        scores = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            with torch.no_grad():
                encoding = self.tokenizer(
                    batch,
                    truncation=True,
                    max_length=self.config.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                encoding = {k: v.to(self.device) for k, v in encoding.items()}

                outputs = self.model(**encoding)
                probs = torch.softmax(outputs.logits, dim=-1)

                # Probability of being real
                batch_scores = probs[:, 0].cpu().tolist()
                scores.extend(batch_scores)

        return scores

    def predict(self, text: str, threshold: float = 0.5) -> tuple[int, float]:
        """Predict if text is real or fake.

        Returns:
            (prediction, confidence) where prediction is 0=real, 1=fake
        """
        score = self.score(text)
        prediction = 0 if score >= threshold else 1
        confidence = score if prediction == 0 else (1 - score)
        return prediction, confidence

    def save(self, path: Path | str) -> None:
        """Save model and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save config
        with open(path / "asm_electra_config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    def load(self, path: Path | str) -> None:
        """Load model from path."""
        from transformers import (
            ElectraForSequenceClassification,
            ElectraTokenizerFast,
        )

        path = Path(path)

        # Load config if exists
        config_path = path / "asm_electra_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
                # Filter to only known fields
                known_fields = {f.name for f in self.config.__dataclass_fields__.values()}
                filtered = {k: v for k, v in config_dict.items() if k in known_fields}
                self.config = ElectraConfig(**filtered)

        self.tokenizer = ElectraTokenizerFast.from_pretrained(path)
        self.model = ElectraForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()


def train_asm_electra(
    train_path: Path,
    output_dir: Path,
    val_path: Path | None = None,
    config: ElectraConfig | None = None,
) -> ASMElectra:
    """Convenience function to train ASM-ELECTRA from JSONL files.

    Args:
        train_path: Path to training JSONL (text, label fields)
        output_dir: Where to save the model
        val_path: Optional validation JSONL
        config: Training config

    Returns:
        Trained ASMElectra instance
    """
    from .data import ElectraDataset

    config = config or ElectraConfig()
    config.output_dir = output_dir

    # Load data
    train_dataset = ElectraDataset.from_jsonl(train_path)
    train_data = train_dataset.to_hf_format()

    val_data = None
    if val_path:
        val_dataset = ElectraDataset.from_jsonl(val_path)
        val_data = val_dataset.to_hf_format()

    print(f"Training data: {len(train_data)} samples")
    if val_data:
        print(f"Validation data: {len(val_data)} samples")

    # Train
    electra = ASMElectra(config=config)
    metrics = electra.train(train_data, val_data)

    print(f"Training complete: {metrics}")
    return electra
