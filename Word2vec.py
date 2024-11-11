import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from tqdm import tqdm  # For progress bars
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


Corpus = """
apple juice
orange juice
cherry juice
apricot juice
facebook company
google company
microsoft company
mcdonalds company
yahoo company
"""

sentences = [sentence for sentence in Corpus.split("\n") if sentence != ""]

dataset = [sentence.split() for sentence in sentences]

dataset = pd.DataFrame(dataset, columns=["input", "output"])

words = list(dataset.input) + list(dataset.output)

unique_words = list(set(words))

id2tok = dict(enumerate(set(Corpus.split())))
tok2id = {tok: i for i, tok in id2tok.items()}


# defining the Dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        X = data[["input", "output"]].copy()

        X.input = X.input.map(tok2id)
        X.output = X.output.map(tok2id)

        # Change X.input and X.output
        self.X = X.output.values
        self.y = X.input.values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.expand = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, input):
        # Encode input to lower-dimensional representation
        hidden = self.embed(input)
        # Expand hidden layer to predictions
        logits = self.expand(hidden)
        return logits


dataset = CustomDataset(dataset)

# Instantiate 2 model
EMBED_SIZE = 2  # Quite small, just for the tutorial
size = len(unique_words)
model = Word2Vec(size, EMBED_SIZE)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
# Define training parameters
LR = 3e-4
EPOCHS = 1000
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

progress_bar = tqdm(range(EPOCHS * len(dataloader)))
running_loss = []
for epoch in range(EPOCHS):
    epoch_loss = 0
    for i, (batch_X, batch_y) in enumerate(dataloader):
        center, context = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(input=context)
        loss = loss_fn(logits, center)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        progress_bar.update(1)
    epoch_loss /= len(dataloader)
    running_loss.append(epoch_loss)
