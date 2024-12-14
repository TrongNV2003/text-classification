import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from training.evaluate import Tester
from gensim.models import KeyedVectors
from training.dataloader import Dataset
from torch.utils.data import TensorDataset, DataLoader
from training.trainer import CNN, Tokenizer, Trainer

tokenizer = Tokenizer()
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
    parser.add_argument("--model_path", type=str, default="models/wiki.vi.model.bin")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_length", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    set_seed(42)
    args = parse_args()

    train_set = Dataset(args.train_file)
    test_set = Dataset(args.test_file)
    eval_set = Dataset(args.eval_file)

    train_text = [train_set[i][0] for i in range(len(train_set))]
    train_label = [train_set[i][1] for i in range(len(train_set))]

    eval_text = [eval_set[i][0] for i in range(len(eval_set))]
    eval_label = [eval_set[i][1] for i in range(len(eval_set))]

    test_text = [test_set[i][0] for i in range(len(test_set))]
    test_label = [test_set[i][1] for i in range(len(test_set))]


    # Tokenizer
    pretrain_embed = KeyedVectors.load_word2vec_format(args.model_path, binary=True)

    pretrained_words = {'<pad>': 0, '<unk>': 1}
    for idx, word in enumerate(pretrain_embed.key_to_index, start=2):
        pretrained_words[word] = idx

    train_tokenized = tokenizer.tokenize(pretrained_words, train_text)
    eval_tokenized = tokenizer.tokenize(pretrained_words, eval_text)
    test_tokenized = tokenizer.tokenize(pretrained_words, test_text)

    features_train = tokenizer.padding(train_tokenized, args.max_length)
    features_eval = tokenizer.padding(eval_tokenized, args.max_length)
    features_test = tokenizer.padding(test_tokenized, args.max_length)

    train_data = TensorDataset(torch.from_numpy(features_train), torch.from_numpy(np.array(train_label)))
    valid_data = TensorDataset(torch.from_numpy(features_eval), torch.from_numpy(np.array(eval_label)))
    test_data = TensorDataset(torch.from_numpy(features_test), torch.from_numpy(np.array(test_label)))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=args.batch_size)

    train_on_gpu=torch.cuda.is_available()

    if(train_on_gpu):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')
        
    vocab_size = len(pretrained_words)
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