import argparse
import random

import numpy as np
import torch
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader

from training.dataloader import Dataset, DnnDataCollator
from training.evaluate import Tester
from training.models import CNN
from training.trainer import DNNTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--train_file", type=str, default="dataset/train.json")
parser.add_argument("--val_file", type=str, default="dataset/valid.json")
parser.add_argument("--test_file", type=str, default="dataset/test.json")
parser.add_argument(
    "--dict_path", type=str, default="models/wiki.vi.model.bin"
)
parser.add_argument("--save_dir", type=str, default="models/CNN_model/")

args = parser.parse_args()


if __name__ == "__main__":
    set_seed(args.seed)
    pretrain_embed = KeyedVectors.load_word2vec_format(
        args.dict_path, binary=True
    )

    pretrained_dict = {"<pad>": 0, "<unk>": 1}
    for idx, word in enumerate(pretrain_embed.key_to_index, start=2):
        pretrained_dict[word] = idx

    train = Dataset(args.train_file)
    valid = Dataset(args.val_file)
    test = Dataset(args.test_file)

    collator = DnnDataCollator(pretrained_dict, max_length=args.max_length)
    train_loader = DataLoader(
        train, shuffle=True, batch_size=args.batch_size, collate_fn=collator
    )
    valid_loader = DataLoader(
        valid, shuffle=False, batch_size=args.batch_size, collate_fn=collator
    )
    test_loader = DataLoader(
        test, shuffle=False, batch_size=args.batch_size, collate_fn=collator
    )

    if torch.cuda.is_available():
        print("Training on GPU.")
    else:
        print("No GPU available, training on CPU.")

    vocab_size = len(pretrained_dict)
    output_size = 1  # binary (1 or 0)
    embedding_dim = pretrain_embed.vector_size  # 400
    num_filters = 100
    kernel_sizes = [3, 4, 5]

    cnn_model = CNN(
        pretrain_embed,
        vocab_size,
        output_size,
        embedding_dim,
        num_filters,
        kernel_sizes,
    )

    print(cnn_model)
    dtype = next(cnn_model.parameters()).dtype
    print(f"dtype {dtype}")

    trainer = DNNTrainer(
        model=cnn_model,
        epochs=args.epochs,
        learning_rate=args.lr,
        train_loader=train_loader,
        valid_loader=valid_loader,
        save_dir=args.save_dir,
    )
    trainer.train_cnn()

    tester = Tester(model=cnn_model, test_loader=test_loader)
    path = f"CNN_model/model_checkpoint_{args.epochs}.pth"
    trainer.load_model(path)
    tester.test_cnn()
