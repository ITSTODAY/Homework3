from gensim.models import KeyedVectors
import torch
import numpy as np

f = open("/Users/matianyi/Desktop/vocab",encoding="utf-8")
vocab = f.readline()
vocab = vocab.strip("\n")
vocab = vocab.split(" ")
vocab = set(vocab)
#print(vocab)
word_to_idx = {word: i+1 for i,word in enumerate(vocab)}
word_to_idx["<unk>"] = 0
idx_to_word = {i+1: word for i, word in enumerate(vocab)}
idx_to_word[0] = '<unk>'
vocab_size = len(vocab) + 1
embed_size = 300

print(vocab_size)


wvmodel = KeyedVectors.load_word2vec_format("/Users/matianyi/Desktop/word2vec预训练/sgns.sogou.word",binary=False,encoding="utf-8")


weight = torch.zeros(vocab_size, embed_size)

for i in range(len(wvmodel.index2word)):
    try:
        index = word_to_idx[wvmodel.index2word[i]]
    except:
        continue
    weight[index, :] = torch.from_numpy(wvmodel.get_vector(
        idx_to_word[word_to_idx[wvmodel.index2word[i]]]))

weight = weight.numpy()
np.save("/Users/matianyi/Desktop/weight.npy",weight)
word_to_idx = np.array(word_to_idx)
np.save("/Users/matianyi/Desktop/index.npy",word_to_idx)