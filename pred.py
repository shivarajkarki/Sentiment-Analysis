import os
import io
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

rows = []
for line in open('C:/Users/Soumya Ranjan Panda/Downloads/Sarcasm_Headlines_Dataset_v2.json', 'r'):
    rows.append(json.loads(line))

stopwords=[ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

#Function to remove stopwords
def remove_stopwords(sentence):
    for word in stopwords:
        token =' '+word+' '
        sentence = sentence.replace(token,' ')
        sentence = sentence.replace('  ',' ')
    return sentence


# Function to separate the articles, headlines and labels from the data
def extract(array):
    # articles=[]
    headlines = []
    labels = []
    for i in range(len(array)):
        labels.append(array[i]['is_sarcastic'])
        headlines.append(remove_stopwords(array[i]['headline']))
        # articles.append(get_article(array[i]['article_link']))
    # return articles,headlines,labels
    labels = np.array(labels)
    headlines = np.array(headlines)
    return headlines, labels


train_headlines, train_labels=extract(rows)

#Preprocessing the training and test headlines for the word embedding
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_headlines)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(train_headlines)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

model_conv = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(16,3,activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_conv.load_weights('D:\\checkpoints\\sarcasm_13.hdf5')

txt = ["This is quite possibly one of the best products i have ever s"]
seq = tokenizer.texts_to_sequences(txt)
padd = pad_sequences(seq, maxlen=max_length)
pred = model_conv.predict(padd)
print(pred)


