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

import pandas as pd

dataset = [sentence.split() for sentence in sentences]

dataset = pd.DataFrame(dataset, columns=["input", "output"])

words = list(dataset.input) + list(dataset.output)
unique_words = list(set(words))

# id2tok = dict(enumerate(set(Corpus.split())))
# tok2id = {tok: i for i, tok in id2tok.items()}

# X = Corpus[["input", "output"]].copy()
# X.input = X.input.map(tok2id)
# X.output = X.output.map(tok2id)
# X