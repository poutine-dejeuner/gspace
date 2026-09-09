"""
Fine-tune a small LM on modular addition (a + b mod p).

Generates a dataset of (a,b) pairs, formats as "a + b =" → answer,
and trains with next-token prediction on the answer tokens.

Supports:
- LoRA (via peft) for efficient fine-tuning
- Full fine-tuning for very small models
- Curriculum: start with mod 7, increase to mod 59
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset as HFDataset
import numpy as np


@dataclass
class FinetuneConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B"
    modulus: int = 59
    op: str = "+"  # only + for now
    n_train_pairs: int = 3481  # all 59×59 pairs
    val_split: float = 0.1
    output_dir: str = "models/mod59"
    seed: int = 42

    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_seq_length: int = 64

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    # Precision
    dtype: str = "bfloat16"

    # Generation
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100


def generate_dataset(modulus: int, op: str, seed: int):
    """Generate all (a,b) pairs for mod p addition."""
    rng = np.random.RandomState(seed)
    pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
    rng.shuffle(pairs)

    result_fn = {"+": lambda a, b: (a + b) % modulus}[op]

    data = []
    for a, b in pairs:
        result = result_fn(a, b)
        # Format: "a + b =" → answer. Model learns to predict answer after "="
        prompt = f"{a} + {b} ="
        answer = str(result)
        full_text = f"{prompt}{answer}"
        data.append({
            "a": a, "b": b, "result": result,
            "prompt": prompt, "answer": answer, "text": full_text,
        })
    return data


def tokenize_dataset(data: list[dict], tokenizer, max_length: int = 64):
    """Tokenize with labels only on answer tokens."""
    tokenized = {"input_ids": [], "attention_mask": [], "labels": []}

    for item in data:
        # Tokenize full text with labels only on answer
        prompt_ids = tokenizer.encode(item["prompt"], add_special_tokens=False)
        answer_ids = tokenizer.encode(item["answer"], add_special_tokens=False)
        full_ids = prompt_ids + answer_ids

        # Pad/truncate
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]

        # Labels: -100 for prompt, answer ids for answer
        labels = [-100] * len(prompt_ids) + answer_ids
        if len(labels) > max_length:
            labels = labels[:max_length]

        # Pad
        pad_len = max_length - len(full_ids)
        input_ids = full_ids + [tokenizer.pad_token_id or tokenizer.eos_token_id] * pad_len
        attention_mask = [1] * len(full_ids) + [0] * pad_len
        labels = labels + [-100] * pad_len

        tokenized["input_ids"].append(input_ids)
        tokenized["attention_mask"].append(attention_mask)
        tokenized["labels"].append(labels)

    return HFDataset.from_dict(tokenized)


def compute_metrics(eval_pred, tokenizer):
    """Compute accuracy on answer tokens only."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Only compute on answer tokens (labels != -100)
    mask = labels != -100
    correct = (predictions == labels) & mask
    total = mask.sum()

    if total == 0:
        return {"accuracy": 0.0}

    # Per-token accuracy
    token_acc = correct.sum() / total

    # Full-sequence accuracy: all answer tokens correct
    seq_correct = 0
    seq_total = 0
    for i in range(len(labels)):
        seq_mask = mask[i]
        if seq_mask.sum() > 0:
            if correct[i][seq_mask].all():
                seq_correct += 1
            seq_total += 1

    return {
        "token_accuracy": float(token_acc),
        "sequence_accuracy": float(seq_correct / seq_total) if seq_total > 0 else 0.0,
    }


def main():
    config = FinetuneConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Fine-tuning {config.model_id} on {config.op} mod {config.modulus}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate dataset
    data = generate_dataset(config.modulus, config.op, config.seed)
    print(f"Generated {len(data)} problems")

    # Split
    n_val = int(len(data) * config.val_split)
    train_data = data[n_val:]
    val_data = data[:n_val]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Tokenize
    train_ds = tokenize_dataset(train_data, tokenizer, config.max_seq_length)
    val_ds = tokenize_dataset(val_data, tokenizer, config.max_seq_length)

    # Load model
    print("Loading model...")
    dtype = getattr(torch, config.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if config.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="sequence_accuracy",
        greater_is_better=True,
        bf16=(config.dtype == "bfloat16"),
        fp16=(config.dtype == "float16"),
        report_to="none",
        seed=config.seed,
        dataloader_num_workers=0,
        save_total_limit=3,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Custom compute_metrics closure
    def make_compute_metrics(tok):
        def _compute(eval_pred):
            return compute_metrics(eval_pred, tok)
        return _compute

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    final_path = os.path.join(config.output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")

    # Quick evaluation
    print("\n=== Final Evaluation ===")
    eval_results = trainer.evaluate()
    print(json.dumps(eval_results, indent=2))

    # Test on a few examples
    print("\n=== Sample predictions ===")
    model.eval()
    model_to_use = model.module if hasattr(model, 'module') else model
    test_pairs = [(0, 0), (1, 1), (58, 1), (30, 29), (25, 30), (53, 3), (10, 20)]
    for a, b in test_pairs:
        result = (a + b) % config.modulus
        prompt = f"{a} + {b} ="
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device if hasattr(model, 'device') else device)
        with torch.no_grad():
            out = model_to_use.generate(**inputs, max_new_tokens=5, do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id)
            full = tokenizer.decode(out[0], skip_special_tokens=True)
            pred = full[len(prompt):].strip()
        print(f"  {prompt} {pred} (expected {result}) {'✓' if pred == str(result) else '✗'}")

    # Clean up
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
