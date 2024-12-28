import torch
import random
import argparse
import numpy as np
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader

from training.models import RNN
from training.trainer import Trainer
from training.evaluate import Tester
from training.dataloader import Dataset, DatasetCollator

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()

parser.add_argument("--train_file", type=str, default="dataset/train.json")
parser.add_argument("--test_file", type=str, default="dataset/test.json")
parser.add_argument("--eval_file", type=str, default="dataset/evaluate.json")
parser.add_argument("--dict_path", type=str, default="models/wiki.vi.model.bin")
parser.add_argument("--save_dir", type=str, default="RNN_model/")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

if __name__ == "__main__":
    set_seed(args.seed)

    pretrain_embed = KeyedVectors.load_word2vec_format(args.dict_path, binary=True)

    pretrained_dict = {'<pad>': 0, '<unk>': 1}
    for idx, word in enumerate(pretrain_embed.key_to_index, start=2):
        pretrained_dict[word] = idx

    train = Dataset(args.train_file)
    valid = Dataset(args.test_file)
    test = Dataset(args.test_file)

    collator = DatasetCollator(pretrained_dict)

    train_loader = DataLoader(train, shuffle=True, batch_size=args.batch_size, collate_fn=collator, drop_last=True)
    valid_loader = DataLoader(valid, shuffle=False, batch_size=args.batch_size, collate_fn=collator, drop_last=True)
    test_loader = DataLoader(test, shuffle=False, batch_size=args.batch_size, collate_fn=collator, drop_last=True)

    if(torch.cuda.is_available()):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')
        
    vocab_size = len(pretrained_dict) + 1
    output_size = 1 # binary (1 or 0)
    embedding_dim = pretrain_embed.vector_size # 400
    hidden_dim = 256
    n_layers = 2

    rnn_model = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

    print(rnn_model)
    dtype = next(rnn_model.parameters()).dtype
    print(f"dtype {dtype}")

    trainer = Trainer(
        model=rnn_model,
        epochs=args.epochs,
        learning_rate=args.lr,
        train_loader=train_loader,
        valid_loader=valid_loader,
        save_dir=args.save_dir
    )
    trainer.train_rnn()

    tester = Tester(
        model=rnn_model,
        test_loader=test_loader
    )
    path = "RNN_model/model_checkpoint_3.pth"
    trainer.load_model(path)
    tester.test_rnn()