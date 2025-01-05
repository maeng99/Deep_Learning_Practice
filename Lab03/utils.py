import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
import numpy as np

def plot_learning_curves(history, title):
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

def decode_imdb(sentence):
  word_to_idx = imdb.get_word_index()
  idx_to_word = dict([(value, key) for (key, value) in word_to_idx.items()])
  
  decoded_review = ' '.join([idx_to_word.get(i) for i in sentence])
  return decoded_review

def sentence_to_vector(test_sentence):
  
  word_to_idx = imdb.get_word_index()
  encode_test_sentence = []
  for word in test_sentence.split(' '):
    idx = int(word_to_idx.get(word))
    encode_test_sentence.append(idx)
  print(encode_test_sentence)
  encode_test_sentence = np.array(encode_test_sentence)

  encode_test_sentence = np.reshape(encode_test_sentence, (1, -1, 1))
  return encode_test_sentence