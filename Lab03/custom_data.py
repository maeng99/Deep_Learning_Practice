import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd


# 시계열 데이터 생성
def generate_time_series(n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, 1000, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # 주기 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # 주기 2
    series += 0.1 * (np.random.rand(1000, n_steps) - 0.5)         # 노이즈
    return series[..., np.newaxis]

def signal_one(n_steps = 50):
    n_steps = n_steps
    series = generate_time_series(n_steps + 1)
    X_train, y_train = series[:700, :n_steps], series[:700, -1]
    X_valid, y_valid = series[700:900, :n_steps], series[700:900, -1]
    X_test, y_test = series[900:, :n_steps], series[900:, -1]
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def signal_two():
  X_data = []
  y_data = []

  for i in range(2500):
      lst = np.random.rand(100)
      idx = np.random.choice(100, 2, replace = False)
      zeros = np.zeros(100)
      zeros[idx] = 1
      X_data.append(np.array(list(zip(zeros, lst))))
      y_data.append(np.prod(lst[idx]))
  X_data = np.array(X_data)
  y_data = np.array(y_data)

  return X_data, y_data

def ko_review():
    # 데이터셋 로드
  path_to_train_file = tf.keras.utils.get_file('train_txt', 'https://raw.githubusercontent.com/hmkim312/datas/main/navermoviereview/ratings_train.txt')
  path_to_test_file = tf.keras.utils.get_file('test_txt', 'https://raw.githubusercontent.com/hmkim312/datas/main/navermoviereview/ratings_test.txt')

  train_data = pd.read_csv(path_to_train_file, sep='\t')
  test_data = pd.read_csv(path_to_test_file, sep='\t')
  return train_data,test_data
