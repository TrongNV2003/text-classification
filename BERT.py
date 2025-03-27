import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from training.dataloader import Dataset, LlmDataCollator
from training.evaluate import Tester
from training.trainer import LlmTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument("--dataloader_workers", type=int, default=2)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--pad_mask_id", type=int, default=-100)
parser.add_argument("--model", type=str, default="vinai/phobert-base-v2")
parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--valid_batch_size", type=int, default=8)
parser.add_argument("--test_batch_size", type=int, default=8)
parser.add_argument("--warmup_steps", type=int, default=200)
parser.add_argument("--train_file", type=str, default="dataset/train.json")
parser.add_argument("--valid_file", type=str, default="dataset/valid.json")
parser.add_argument("--test_file", type=str, default="dataset/test.json")
parser.add_argument(
    "--output_dir", type=str, default="./models/bert-classification"
)
parser.add_argument("--evaluate_on_accuracy", type=bool, default=True)
parser.add_argument(
    "--pin_memory", dest="pin_memory", action="store_true", default=False
)
parser.add_argument(
    "--early_stopping_patience",
    type=int,
    default=3,
    help="Number of epochs to wait for improvement",
    required=True,
)
parser.add_argument(
    "--early_stopping_threshold",
    type=float,
    default=0.001,
    help="Minimum improvement to reset early stopping counter",
    required=True,
)
args = parser.parse_args()


def get_tokenizer(checkpoint: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer


def get_model(
    checkpoint: str, device: str, tokenizer: AutoTokenizer
) -> AutoModelForSequenceClassification:
    config = AutoConfig.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, config=config
    )
    model = model.to(device)
    return model


if __name__ == "__main__":
    set_seed(args.seed)

    tokenizer = get_tokenizer(args.model)

    train_set = Dataset(args.train_file)
    valid_set = Dataset(args.valid_file)
    test_set = Dataset(args.test_file)

    collator = LlmDataCollator(tokenizer=tokenizer, max_length=args.max_length)

    model = get_model(args.model, args.device, tokenizer)
    trainer = LlmTrainer(
        dataloader_workers=args.dataloader_workers,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        model=model,
        pin_memory=args.pin_memory,
        save_dir=args.output_dir,
        tokenizer=tokenizer,
        train_set=train_set,
        valid_set=valid_set,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        collator_fn=collator,
        evaluate_on_accuracy=args.evaluate_on_accuracy,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
    )
    trainer.train()

    # test model
    MODEL = "models/bert-classification"
    tuned_model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(
        args.device
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    tester = Tester(model=tuned_model, test_loader=test_loader)

    tester.test_llm()
