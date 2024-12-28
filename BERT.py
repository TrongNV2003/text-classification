import torch
import random
import argparse
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from training.evaluate import Tester
from training.trainer import LlmTrainer
from training.dataloader import Dataset, LlmDataCollator

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()

parser.add_argument("--dataloader_workers", type=int, default=2)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--pad_mask_id", type=int, default=-100)
parser.add_argument("--model", type=str, default="tabularisai/multilingual-sentiment-analysis")
parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
parser.add_argument("--save_dir", type=str, default="./bert-classification")
parser.add_argument("--train_batch_size", type=int, default=16)
parser.add_argument("--valid_batch_size", type=int, default=8)
parser.add_argument("--train_file", type=str, default="dataset/train.json")
parser.add_argument("--valid_file", type=str, default="dataset/valid.json")
parser.add_argument("--test_file", type=str, default="dataset/test.json")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

def get_tokenizer(checkpoint: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # tokenizer.add_special_tokens(
    #     {'additional_special_tokens': ['<sep>']}
    # )
    return tokenizer

def get_model(checkpoint: str, device: str, tokenizer: AutoTokenizer) -> AutoModelForSequenceClassification:
    config = AutoConfig.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    return model

if __name__ == "__main__":
    set_seed(args.seed)
    
    tokenizer = get_tokenizer(args.model)
    train_set = Dataset(
        json_file=args.train_file
    )

    valid_set = Dataset(
        json_file=args.valid_file
    )
    collator = LlmDataCollator(tokenizer=tokenizer, max_length=args.max_length)

    model = get_model(args.model, args.device, tokenizer)
    trainer = LlmTrainer(
        dataloader_workers=args.dataloader_workers,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        model=model,
        pin_memory=args.pin_memory,
        save_dir=args.save_dir,
        tokenizer=tokenizer,
        train_set=train_set,
        valid_set=valid_set,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        collator_fn=collator
    )
    trainer.train()

# test model
    MODEL = "bert-classification"
    tuned_model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    test_set = Dataset(args.test_file)
    collator = LlmDataCollator(tokenizer=tokenizer, max_length=args.max_length)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, collate_fn=collator)
    tester = Tester(model=tuned_model, test_loader=test_loader)

    tester.test_llm()