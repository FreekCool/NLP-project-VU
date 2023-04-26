import math
from collections import Counter
import nltk
from scipy.sparse import dok_matrix
from tqdm import tqdm
import random

# Download and load the Brown corpus
nltk.download('brown')

# Use 10% of the brown corpus due to computational limit of laptop
# Load the corpus and select 10% of its words randomly
corpus = nltk.corpus.brown.words()
corpus_size = len(corpus)
subset_size = int(corpus_size * 1)
random.seed(42)  # for reproducibility
subset_indices = random.sample(range(corpus_size), subset_size)
subset = [corpus[i] for i in subset_indices]

# Calculate the frequency of each word in the subset
word_freq = Counter(subset)

# Set the threshold for word frequency
freq_threshold = 10

# Create a generator expression for all successive pairs of words in the subset
pairs = ((subset[i], subset[i+1]) for i in range(len(subset)-1))

# Use a sparse matrix to store the frequency counts for each word and word pair
freq_matrix = dok_matrix((len(word_freq), len(word_freq)), dtype=int)
word_indices = {word: i for i, word in enumerate(word_freq.keys())}

pair_count = 0
for word1, word2 in tqdm(pairs, total=len(corpus)-1):
    if word_freq[word1] >= freq_threshold and word_freq[word2] >= freq_threshold:
        freq_matrix[word_indices[word1], word_indices[word2]] += 1
        pair_count += 1

# Calculate the PMI for each word pair
pmi_scores = {}
for word1, word2 in tqdm(freq_matrix.keys(), total=len(freq_matrix.keys())):
    pair_freq = freq_matrix[word1, word2]
    word1_prob = word_freq[list(word_freq.keys())[word1]] / len(corpus)
    word2_prob = word_freq[list(word_freq.keys())[word2]] / len(corpus)
    pair_prob = pair_freq / pair_count
    pmi = math.log2(pair_prob / (word1_prob * word2_prob))
    pmi_scores[(list(word_freq.keys())[word1], list(word_freq.keys())[word2])] = pmi

# Sort the PMI scores in descending order
sorted_pmi = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)

# Print the top 20 and bottom 20 word pairs by PMI score
print("Top 20 word pairs by PMI:")
for pair, score in sorted_pmi[:20]:
    print(pair[0], pair[1], score)

print("Bottom 20 word pairs by PMI:")
for pair, score in sorted_pmi[-20:]:
    print(pair[0], pair[1], score)
