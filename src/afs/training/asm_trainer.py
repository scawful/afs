"""ASM Encoder Trainer for AFS domain capabilities.

Part of the AFS (Agentic File System) framework, this module provides
a training pipeline for encoder-based models (BERT, ELECTRA, etc.)
using the custom 65816 assembly tokenizer.

These encoder models can be used by AFS agents for:
- Assembly code understanding and embedding
- Semantic similarity between code snippets
- Pre-training for downstream assembly tasks

Example:
    ```python
    from afs.training import ASMTrainer, ASMTrainerConfig
    from afs.tokenizer import ASMTokenizer

    # Load tokenizer
    tokenizer = ASMTokenizer.load("models/asm-tokenizer")

    # Configure training
    config = ASMTrainerConfig(
        output_dir="./asm-encoder",
        num_epochs=10,
        batch_size=16,
    )

    # Train
    trainer = ASMTrainer(tokenizer=tokenizer, config=config)
    trainer.train(train_data, val_data)
    ```
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

# Lazy import torch to allow module loading without it
if TYPE_CHECKING:
    import torch
    from torch.utils.data import Dataset, DataLoader


@dataclass
class ASMTrainerConfig:
    """Configuration for ASM model training."""

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./asm-model"))

    # Model architecture
    model_type: str = "bert"  # bert, electra, roberta
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    intermediate_size: int = 1024
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 512

    # Training
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 5e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1

    # MLM settings
    mlm_probability: float = 0.15

    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100

    # Device
    device: str = "auto"  # auto, cpu, cuda, mps

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_dir": str(self.output_dir),
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_dropout": self.hidden_dropout,
            "attention_dropout": self.attention_dropout,
            "max_position_embeddings": self.max_position_embeddings,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "mlm_probability": self.mlm_probability,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ASMTrainerConfig":
        """Create from dictionary."""
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])
        return cls(**data)


def _get_dataset_base():
    """Get Dataset base class, importing torch lazily."""
    from torch.utils.data import Dataset
    return Dataset


class ASMDataset:
    """PyTorch Dataset for assembly code.

    Note: Inherits from torch.utils.data.Dataset at runtime.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: Any,
        max_length: int = 512,
        mlm_probability: float = 0.15,
    ):
        """Initialize dataset.

        Args:
            texts: List of assembly code strings.
            tokenizer: ASMTokenizer instance.
            max_length: Maximum sequence length.
            mlm_probability: Probability of masking tokens for MLM.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch

        text = self.texts[idx]

        # Encode
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )

        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)

        # Create MLM labels
        labels = input_ids.clone()

        # Create mask for MLM (don't mask special tokens)
        special_ids = set(self.tokenizer.all_special_ids)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Don't mask special tokens
        for i, token_id in enumerate(input_ids.tolist()):
            if token_id in special_ids:
                probability_matrix[i] = 0.0

        # Also don't mask padding
        probability_matrix = probability_matrix * attention_mask.float()

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set non-masked tokens to -100 (ignored by loss)
        labels[~masked_indices] = -100

        # Replace masked tokens with [MASK]
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ASMTrainer:
    """Trainer for encoder models with ASM tokenizer."""

    def __init__(
        self,
        tokenizer: Any,
        config: ASMTrainerConfig | None = None,
        model: Any | None = None,
    ):
        """Initialize trainer.

        Args:
            tokenizer: ASMTokenizer instance.
            config: Training configuration.
            model: Optional pre-initialized model.
        """
        self.tokenizer = tokenizer
        self.config = config or ASMTrainerConfig()
        self.model = model
        self._device = None

    @property
    def device(self) -> Any:
        """Get training device."""
        import torch

        if self._device is None:
            if self.config.device == "auto":
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = torch.device("mps")
                else:
                    self._device = torch.device("cpu")
            else:
                self._device = torch.device(self.config.device)
        return self._device

    def _create_model(self) -> Any:
        """Create a BERT-style model for MLM."""
        from transformers import BertConfig, BertForMaskedLM

        config = BertConfig(
            vocab_size=len(self.tokenizer),
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_layers,
            num_attention_heads=self.config.num_heads,
            intermediate_size=self.config.intermediate_size,
            hidden_dropout_prob=self.config.hidden_dropout,
            attention_probs_dropout_prob=self.config.attention_dropout,
            max_position_embeddings=self.config.max_position_embeddings,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return BertForMaskedLM(config)

    def train(
        self,
        train_texts: list[str],
        val_texts: list[str] | None = None,
        callbacks: list[Callable] | None = None,
    ) -> dict[str, Any]:
        """Train the model.

        Args:
            train_texts: Training assembly code samples.
            val_texts: Optional validation samples.
            callbacks: Optional training callbacks.

        Returns:
            Training metrics.
        """
        # Create model if not provided
        if self.model is None:
            self.model = self._create_model()

        self.model.to(self.device)

        # Create datasets
        train_dataset = ASMDataset(
            train_texts,
            self.tokenizer,
            max_length=self.config.max_position_embeddings,
            mlm_probability=self.config.mlm_probability,
        )

        val_dataset = None
        if val_texts:
            val_dataset = ASMDataset(
                val_texts,
                self.tokenizer,
                max_length=self.config.max_position_embeddings,
                mlm_probability=self.config.mlm_probability,
            )

        # Create data loaders
        from torch.utils.data import DataLoader

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
            )

        # Setup optimizer
        import torch

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        num_training_steps = len(train_loader) * self.config.num_epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_steps,
        )

        # Training loop
        global_step = 0
        best_val_loss = float("inf")
        train_losses = []

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss += loss.item() * self.config.gradient_accumulation_steps

                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    print(f"Epoch {epoch+1}, Step {global_step}, Loss: {avg_loss:.4f}")

                # Evaluation
                if val_loader and global_step % self.config.eval_steps == 0:
                    val_loss = self._evaluate(val_loader)
                    print(f"  Validation Loss: {val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save(self.config.output_dir / "best")

                # Checkpointing
                if global_step % self.config.save_steps == 0:
                    self.save(self.config.output_dir / f"checkpoint-{global_step}")

            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1} complete. Average Loss: {avg_epoch_loss:.4f}")

        # Save final model
        self.save(self.config.output_dir / "final")

        return {
            "final_loss": train_losses[-1] if train_losses else None,
            "best_val_loss": best_val_loss if val_loader else None,
            "epochs": self.config.num_epochs,
            "global_steps": global_step,
            "train_losses": train_losses,
        }

    def _evaluate(self, val_loader: Any) -> float:
        """Evaluate model on validation set."""
        import torch

        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()

        self.model.train()
        return total_loss / len(val_loader)

    def save(self, path: Path | str) -> None:
        """Save model, tokenizer, and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(path)

        # Save tokenizer
        self.tokenizer.save(path / "tokenizer")

        # Save training config
        with open(path / "training_config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "ASMTrainer":
        """Load trainer from path."""
        from transformers import BertForMaskedLM
        from afs.tokenizer import ASMTokenizer

        path = Path(path)

        # Load config
        config_path = path / "training_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = ASMTrainerConfig.from_dict(json.load(f))
        else:
            config = ASMTrainerConfig()

        # Load tokenizer
        tokenizer = ASMTokenizer.load(path / "tokenizer")

        # Load model
        model = BertForMaskedLM.from_pretrained(path)

        return cls(tokenizer=tokenizer, config=config, model=model)


def train_asm_encoder(
    train_path: Path | str,
    tokenizer_path: Path | str,
    output_dir: Path | str,
    val_path: Path | str | None = None,
    config: ASMTrainerConfig | None = None,
) -> dict[str, Any]:
    """Convenience function to train ASM encoder from files.

    Args:
        train_path: Path to training data (one sample per line, or JSONL).
        tokenizer_path: Path to saved ASMTokenizer.
        output_dir: Output directory for model.
        val_path: Optional validation data path.
        config: Training configuration.

    Returns:
        Training metrics.
    """
    from afs.tokenizer import ASMTokenizer

    # Load tokenizer
    tokenizer = ASMTokenizer.load(tokenizer_path)
    print(f"Loaded tokenizer with {len(tokenizer)} tokens")

    # Load training data
    train_path = Path(train_path)
    train_texts = _load_texts(train_path)
    print(f"Loaded {len(train_texts)} training samples")

    val_texts = None
    if val_path:
        val_texts = _load_texts(Path(val_path))
        print(f"Loaded {len(val_texts)} validation samples")

    # Setup config
    config = config or ASMTrainerConfig()
    config.output_dir = Path(output_dir)

    # Train
    trainer = ASMTrainer(tokenizer=tokenizer, config=config)
    metrics = trainer.train(train_texts, val_texts)

    print(f"Training complete: {metrics}")
    return metrics


def _load_texts(path: Path) -> list[str]:
    """Load texts from file (plain text or JSONL)."""
    texts = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Try JSON first
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    # Look for text field
                    if "text" in data:
                        texts.append(data["text"])
                    elif "code" in data:
                        texts.append(data["code"])
                    elif "input" in data:
                        texts.append(data["input"])
                    continue
                except json.JSONDecodeError:
                    pass

            # Plain text
            texts.append(line)

    return texts
