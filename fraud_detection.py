import pandas as pd
import math
import random
import numpy as np
from keras import models 
from keras import layers
from matplotlib import pyplot as plt
import sys

data = pd.read_csv("C:\\Users\\Tadeas\\Downloads\\fraudic\\fraud.csv")

def drop_data(x):
  ones = x[x["isFraud"] == 1]
  zeros = x[x["isFraud"] == 0][:8200]
  
  d = pd.concat([ones, zeros])

  return d

data = drop_data(data)

y = data["isFraud"].to_list()

data.drop(columns=["isFraud", "isFlaggedFraud", "nameDest", "nameOrig", "step"], inplace=True)

data = pd.get_dummies(data, columns=["type"])

x = data.values.tolist()



def rozptyl(y):
  all_nums = []
  k = sum(y) / len(y)
  for i in(y):
    rozdil = i - k 
    num = rozdil * rozdil
    all_nums.append(num)

  rozptyls = sum(all_nums) / len(all_nums)
  
  return rozptyls


def smerodatna_odchylka(y):
  
  smerodatna_odchylka = math.sqrt(rozptyl(y))

  return smerodatna_odchylka


def Z_score(data):
  z = []
  mean = sum(data) / len(data)
  std  = smerodatna_odchylka(data)
  for x in data:
    new_x = (x-mean) / std
    z.append(new_x)

  return z

for i in range(5):
  z =  Z_score([g[i] for g in x])
  for k in range(len(x)):
    x[k][i] = z[i]

splitenzi = math.ceil(len(x) * 0.8)







def split_data(x, y, t_split):
  random.seed(7)
  x = random.sample(x, len(x))
  random.seed(7)
  y = random.sample(y, len(y))

  x_train = x[:t_split]
  y_train = y[:t_split]
  x_test = x[t_split:]
  y_test = y[t_split:]

  return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = split_data(x, y, splitenzi)


def my_model(x, y, test):
  x = np.asarray(x)
  y = np.asarray(y)
  test = np.asarray(test)
  name = "fraud_model2.h5"

  try:
    m = models.load_model(name)
  
  except:
    m = models.Sequential()
    m.add(layers.Dense(32, activation="relu", input_shape=(x.shape[1],)))
    m.add(layers.Dense(64, activation="relu"))
    m.add(layers.Dense(32, activation="relu"))
    m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["acc"])

    m.fit(x, y, epochs=50, batch_size=32, validation_split=0.2)

    m.save(name)

  preds = m.predict(test)

  return preds

preds = my_model(x_train, y_train, x_test)

class classification_evaluate:
  
  def __init__(self, predicted, real):
    self.predicted = predicted
    self.real = real
  
  def accuracy(self):
    a =  0
    for i in range(len(self.predicted)):
      if self.predicted[i] == self.real[i]:
        a += 1
    
      else:
        pass

    return a/len(self.real)

  def recall(self):
    only_ones = []
    pred = []
    for i in range(len(self.real)):
      if self.real[i] == 1:
        only_ones.append(1)
        pred.append(self.predicted[i])

      else:
        pass

    a = 0
    for i in pred:
      if i == 1:
        a += 1

      else:
        pass

    return a / len(only_ones)

  def specificity(self):
    only_zeros = []
    pred = []
    for i in range(len(self.real)):
      if self.real[i] == 0:
        only_zeros.append(0)
        pred.append(self.predicted[i])

      else:
        pass

    a = 0
    for i in pred:
      if i == 0:
        a += 1

      else:
        pass

    return a / len(only_zeros)
  
  def Precision(self):
    pred_ones = []
    realitka = []
    for i in range(len(self.predicted)):
      if self.predicted[i] == 1:
        pred_ones.append(1)
        realitka.append(self.real[i])

      else:
        pass

    a = 0

    for i in realitka:
      if i == 1:
        a += 1

      else:
        pass

    return a/len(pred_ones)

  def confusion_matrix(self):
    conf_matrix = [[0, 0], [0, 0]]
    for i in range(len(self.real)):
      if self.real[i] == self.predicted[i]:
        if self.real[i] == 1:
          conf_matrix[0][0] += 1

        else:
          conf_matrix[1][1] += 1

      elif self.real[i] != self.predicted[i]:
        if self.real[i] == 1:
          conf_matrix[0][1] += 1

        else:
          conf_matrix[1][0] += 1


      else:
        sys.exit("Something is wrong")

    return conf_matrix


def ROC(real, probs):
  x = [1]
  y = [1]
  thresholds = []
  for i in range(10):
    threshold = i/10
    thresholds.append(threshold)
    preds = []
    for b in probs:
      if b > threshold:
        preds.append(1)

      else:
        preds.append(0)

    x_i = 1 - classification_evaluate(preds, real).specificity()
    y_i = classification_evaluate(preds, real).recall()

    x.append(x_i)
    y.append(y_i)
    

  best_threshhold = []
  for i in range(len(x)):
    if x[i] == 0 and y[i] == 1:
      best_threshhold.append(thresholds[i])

    else:
      pass

  plt.style.use("fivethirtyeight")

  plt.plot(x, y, "-o")

  plt.show()

  if len(best_threshhold) != 0:
    return best_threshhold

  else:
    return x, y


ROC(y_test, preds)

real_preds = []


for i in preds:
  if i > 0.5:
    real_preds.append(1)

  else:
    real_preds.append(0)

print(classification_evaluate(real_preds, y_test).accuracy())