# DL24.py CS5173/6073 2020 cheng
# making centers, contexts, and negatives for PennTreebank data
# building vocabulary, performing subsampling and negative sampling
# Skip-gram word embedding as a translation from MXNet to Pytorch of d2l chapter 14
# Usage: python DL24.py
import zipfile
import collections
import random
import math
import torch
import numpy as np

f = zipfile.ZipFile('data/ptb.zip', 'r')
raw_text = f.read('ptb/ptb.train.txt').decode("utf-8")
sentences = [line.split() for line in raw_text.split('\n')]
tokens = [tk for line in sentences for tk in line]
counter = collections.Counter(tokens)
uniq_tokens = [token for token, freq in list(counter.items()) if counter[token] >= 10]
idx_to_token, token_to_idx = [], dict()
for token in uniq_tokens:
    idx_to_token.append(token)
    token_to_idx[token] = len(idx_to_token) - 1
s = [[idx_to_token[token_to_idx.get(tk, 0)] for tk in line] for line in sentences]
tokens = [tk for line in s for tk in line]
counter = collections.Counter(tokens)
num_tokens = sum(counter.values())
subsampled = [[tk for tk in line if random.uniform(0, 1) < math.sqrt(1e-4 / counter[tk] * num_tokens)] for line in s]
corpus = [[token_to_idx.get(tk) for tk in line] for line in subsampled]
tokens = [tk for line in corpus for tk in line]
counter = collections.Counter(tokens)
sampling_weights = [counter[i]**0.75 for i in range(len(counter))]
population = list(range(len(sampling_weights)))
candidates = random.choices(population, sampling_weights, k=10000)
max_window_size = 5
K = 5
j = 0
data = []
maxLen = 0
for line in corpus:
    if len(line) < 2:
        continue
    for i in range(len(line)):
        window_size = random.randint(1, max_window_size)
        indices = list(range(max(0, i - window_size), min(len(line), i + 1 + window_size)))
        indices.remove(i)
        for idx in indices:
            context = [line[idx] for idx in indices]
        neg = []
        while len(neg) < len(context) * K:
            ne = candidates[j]
            j += 1
            if j >= 10000:
                j = 0
            if ne not in context:
                neg.append(ne)
        data.append([line[i], context, neg])

max_len = max(len(c) + len(n) for _, c, n in data)
centers, contexts_negatives, labels = [], [], []
for center, context, negative in data:
    cur_len = len(context) + len(negative)
    centers += [center]
    contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
    labels += [[1] * len(context) + [0] * (max_len - len(context))]

class PTBdataset(torch.utils.data.Dataset):
    def __init__(self):
        super(PTBdataset).__init__()
        self.centers = np.array(centers).reshape(-1, 1)
        self.contexts_negatives = np.array(contexts_negatives)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts_negatives[idx], self.labels[idx]
pdata = PTBdataset()
data_iter = torch.utils.data.DataLoader(pdata, batch_size=512, shuffle=True)

vocab_size = len(idx_to_token)
embed_size = 100
import torch.nn as nn
import torch.optim as optim
net = nn.Sequential(
    nn.Embedding(vocab_size, embed_size),
    nn.Embedding(vocab_size, embed_size))
loss = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), 0.01)
m = nn.Sigmoid()

for epoch in range(5):
    for i, batch in enumerate(data_iter):
        center, context_negative, label = batch
        v = net[0](center.to(torch.int64))
        u = net[1](context_negative.to(torch.int64))
        pred = torch.tensordot(v, torch.transpose(u, 1, 2))
        l = loss(m(pred), label.to(torch.float32))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        if (i + 1) % 50 == 0:
            print(epoch, i, float(l))




#change this
def get_similar_tokens(query_token, k, embed):
        W = embed.weight.data()
        x = W[vocab[query_token]]
        cos = np.dot(W,x)/np.sqrt(np.sum(W*W, axis1) * np.sum(x * x) + le-9)
        topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
        for i in topk[1:]:
            print('cosine sim=%.3f: %s' % (cos[i],(vocab.idx_to_token[i])))
get_similar_tokens('chip', 3, net[0])
