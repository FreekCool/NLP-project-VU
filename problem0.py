import nltk
from nltk.corpus import brown
import matplotlib.pyplot as plt

def remove_punctuation(tokens):
    return [word for word in tokens if word.isalpha()]

# nltk.download('brown')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

'''0. i, unique words of the whole corpus by frequency'''
tokens = brown.words()

freq_dist = nltk.FreqDist(tokens)

sorted_words = sorted(freq_dist, key = freq_dist.get, reverse = True)

'''0. ii, unique words of two chosen categories'''
# Category romance
romance_words = brown.words(categories = 'romance')

romance_freq_dist = nltk.FreqDist(romance_words)

romance_sorted_words = sorted(romance_freq_dist, key = romance_freq_dist.get, reverse = True)

# Category learned
learned_words = brown.words(categories = 'learned')

learned_freq_dist = nltk.FreqDist(learned_words)

learned_sorted_words = sorted(learned_freq_dist, key = learned_freq_dist.get, reverse = True)

# Print number of tokens
print(f"Number of tokens {len(tokens)}")

# print number of types
print(f"Number of types: {len(set(tokens))}")

# print number of words
words = remove_punctuation(tokens)
print(f"Number of words: {len(set(words))}")

# print average number of words per sentence
sentences = brown.sents()

new_sentences = []

for sentence in sentences:
    new_sentences.append(remove_punctuation(sentence))

total_words = 0

for sentence in new_sentences:
    total_words += len(sentence)

avg_num_words = total_words/len(new_sentences)
print(f"Average number of words per sentence is: {avg_num_words}")

# print the average word length
total_length = 0

for sentence in new_sentences:
    for word in sentence:
        total_length += len(word)

average_word_length = total_length/total_words
print(f"Average word length is: {average_word_length}")

# POS tagging
pos_tags = nltk.pos_tag(tokens)

pos_freq_dist = nltk.FreqDist(tag for (word, tag) in pos_tags)

# Create frequency distribution of POS tags
tag_freq_dist = nltk.FreqDist(tag for (word, tag) in pos_tags)

# Get list of 10 most frequent POS tags
top_tags = tag_freq_dist.most_common(10)

# print the top 10 most used POS tags
print(f"The top 10 most used POS tags are: {top_tags}")

# plot frequency curves
# Plot frequency curve for corpus
plt.figure(figsize=(12,6))
freq_dist.plot(5000, cumulative=False)
plt.title('Frequency distribution of words in the Brown corpus')
plt.xlabel('Word rank')
plt.ylabel('Frequency')
plt.gca().set_xticklabels([])
plt.show()

# Plot frequency curve for romance category
plt.figure(figsize=(12,6))
romance_freq_dist.plot(5000, cumulative=False)
plt.title('Frequency distribution of words in the romance category')
plt.xlabel('Word rank')
plt.ylabel('Frequency')
plt.gca().set_xticklabels([])
plt.show()

# Plot frequency curve for learned category
plt.figure(figsize=(12,6))
learned_freq_dist.plot(5000, cumulative=False)
plt.title('Frequency distribution of words in the learned category')
plt.xlabel('Word rank')
plt.ylabel('Frequency')
plt.gca().set_xticklabels([])
plt.show()

# Plot frequency curves on log-log scales
plt.figure(figsize=(12,6))
freq_dist.plot(5000, cumulative=False)
romance_freq_dist.plot(5000, cumulative=False)
learned_freq_dist.plot(5000, cumulative=False)
plt.xscale('log')
plt.yscale('log')
plt.title('Log-log frequency distribution of words in the Brown corpus and two genres')
plt.xlabel('Log word rank')
plt.ylabel('Log frequency')
plt.legend(['Corpus', 'Romance', 'Learned'])
plt.gca().set_xticklabels([])
plt.show()

