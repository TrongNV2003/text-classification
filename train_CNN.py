import torch
import random
import argparse
import numpy as np
from training.models import CNN
from training.trainer import Trainer
from training.evaluate import Tester
from gensim.models import KeyedVectors
from training.dataloader import Dataset
from torch.utils.data import DataLoader

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="dataset/train.json")
    parser.add_argument("--test_file", type=str, default="dataset/test.json")
    parser.add_argument("--eval_file", type=str, default="dataset/evaluate.json")
    parser.add_argument("--dict_path", type=str, default="models/wiki.vi.model.bin")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)

    return parser.parse_args()


if __name__ == "__main__":
    set_seed(42)
    args = parse_args()

    pretrain_embed = KeyedVectors.load_word2vec_format(args.dict_path, binary=True)

    pretrained_dict = {'<pad>': 0, '<unk>': 1}
    for idx, word in enumerate(pretrain_embed.key_to_index, start=2):
        pretrained_dict[word] = idx
        
    train_data = Dataset(args.train_file, pretrained_dict)._return_tensor()
    valid_data = Dataset(args.test_file, pretrained_dict)._return_tensor()
    test_data = Dataset(args.eval_file, pretrained_dict)._return_tensor()

    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)

    if(torch.cuda.is_available()):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')
        
    vocab_size = len(pretrained_dict)
    output_size = 1 # binary (1 or 0)
    embedding_dim = pretrain_embed.vector_size
    num_filters = 100
    kernel_sizes = [3, 4, 5]

    cnn_model = CNN(pretrain_embed, vocab_size, output_size, embedding_dim, num_filters, kernel_sizes)

    print(cnn_model)

    trainer = Trainer(
        model=cnn_model,
        epochs=args.epochs,
        learning_rate=args.lr,
        train_loader=train_loader,
        valid_loader=valid_loader
    )
    trainer.train()

    tester = Tester(
        model=cnn_model,
        test_loader=test_loader
    )
    tester.test()