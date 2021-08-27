# Reference: https://github.com/toastynews/hong-kong-fastText

import torch
import torchtext.vocab

glove = torchtext.vocab.Vectors('toastynews.vec', cache='./vectors/')

print(f'There are {len(glove.itos)} words in the vocabulary')

def get_vector(embeddings, word):
    assert word in embeddings.stoi, f'*{word}* is not in the vocab!'
    return embeddings.vectors[embeddings.stoi[word]]

def closest_words(embeddings, vector, n=10):
    distances = [(w, torch.dist(vector, get_vector(embeddings, w)).item()) for w in embeddings.itos]
    return sorted(distances, key = lambda w: w[1])[:n]

def print_tuples(tuples):
    for w, d in tuples:
        print(f'({d:02.04f}) {w}')

#
# Similar Contexts
#
# Now to start looking at the context of different words.
# If we want to find the words similar to a certain input word, we first find the vector of this input word, 
# then we scan through our vocabulary finding any vectors similar to this input word vector.
print_tuples(closest_words(glove, get_vector(glove, '香港')))
print()
print_tuples(closest_words(glove, get_vector(glove, '宅男')))
print()
print_tuples(closest_words(glove, get_vector(glove, '股票')))
print()

# 
# Analogies
#
def analogy(embeddings, word1, word2, word3, n=10):
    candidate_words = closest_words(embeddings, get_vector(embeddings, word2) - get_vector(embeddings, word1) + get_vector(embeddings, word3), n+3)
    candidate_words = [x for x in candidate_words if x[0] not in [word1, word2, word3]][:n]
    print(f'{word1} is to {word2} as {word3} is to...')
    return candidate_words

print_tuples(analogy(glove, '台灣', '台語', '香港'))
print()
print_tuples(analogy(glove, '香港', '的士', '台灣'))
print()
print_tuples(analogy(glove, '匯控', '匯豐控股', '長實'))
print()
