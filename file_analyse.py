import pandas
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import sys
from keras import layers
from keras import models


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
  
  smerodatna_odchylk = math.sqrt(rozptyl(y))

  return smerodatna_odchylk



def T_test(x, y):
  mean = sum(y) / len(x)
  ss_mean = []

  for i in y:
    ss_mean.append((i-mean)**2)

  lolajs = []
  for k in x:
    if k not in lolajs:
      lolajs.append(k)

  unique = []

  for i in range(len(lolajs)):
    unique.append([])

  for i in range(len(y)):
    unique[x[i]].append(y[i])

  ss_fit = []
  
  for i in range(len(unique)):
    a = sum(unique[i]) / len(unique[i])
    for u in range(100):
      der = []
      for t in range(len(unique[i])):
        b = 2*(a-unique[i][t])
        der.append(b)

      a = a - 0.001*(sum(der))

    for n in unique[i]:
      ss_fit.append((a-n)**2)

  p = sum(ss_mean) - sum(ss_fit) / (len(unique) - 1)
  fin = p / (sum(ss_fit) / (len(x) - len(unique)))

  return fin


data = pandas.read_csv("C:\\Users\\42072\Downloads\\archive (1)\heart_failure_clinical_records_dataset.csv")

data = data.fillna(0)

num_of_all = len(data)

fill_man = (data["sex"] == 1) 
fill_women = (data["sex"] == 0) 

fill_death_man = fill_man & (data["DEATH_EVENT"] == 1)
fill_women_death = fill_women & (data["DEATH_EVENT"] == 1)

num_man_death = len(data[fill_death_man])
num_women_death = len(data[fill_women_death])

_all = [len(data[fill_women]), len(data[fill_man])]

death_all = [num_women_death, num_man_death]

def percent_of_death(all, death):
  percents = []
  for i in range(len(all)):
    percents.append(death[i] / all[i])

  for i in range(len(percents)):
    percents[i] = percents[i] * 100


  
  plt.style.use("fivethirtyeight")
  plt.title("Percent of death")
  plt.xlabel("genders")
  plt.ylabel("percents")
  plt.ylim(0, 100)
  plt.bar(["women", "man"], percents)
  plt.tight_layout()
  plt.show()
  
 
percent_of_death(_all, death_all)


patelets = data["platelets"]

x_gender = data["sex"]


predictors = ["serum_creatinine", "ejection_fraction"]


fig, (a1, a2) = plt.subplots(nrows=len(predictors), ncols=1)

plt.style.use("seaborn")

a1.set_title("Predictors histograms")
a1.set_ylabel("Total of it in that group")
a1.set_xlabel("Serum_creatinine")

a1.hist(data[predictors[0]], bins=10, edgecolor="black")
a2.hist(data[predictors[1]], bins=10, edgecolor="black")
a2.set_xlabel("Ejection_fraction")  


plt.tight_layout()

plt.show()


print(T_test(x_gender, patelets))

y = data["DEATH_EVENT"]

def Z_score(x):
  mean = sum(x) / len(x)
  std = smerodatna_odchylka(x)

  for i in range(len(x)):
    x[i] = (x[i]-mean) / std

  return x




class Guassian_naive_bayes:
  def __init__(self, x, y, test):
    self.x = x
    self.y = y
    self.test = test

  def normal_distribution_likelyhood(stdandard_dev, mean, x):
    m = math.pi * 2 * (stdandard_dev**2)
    m = math.sqrt(m)
    n = 1/m
    s = (x - mean)**2
    s = -s
    p = 2 * (stdandard_dev**2)
    h = s/p
    h = math.e ** h
    fin = h * n
    return fin

  def main(self):
    mean = []
    std = []
    unique = []
    for i in self.y:
      if i not in unique:
        unique.append(i)

      else:
        pass
    
    for i in range(len(unique)):
      mean.append([])
      std.append([])
      m = []
      for c in range(len(self.x)):
        if self.y[c] == unique[i]:
          m.append(self.x[c])

        else:
          pass

      for d in range(len(m[0])):
        v = []
        for t in m:
          v.append(t[d])

        un_mean = sum(v) / len(v)
        un_std = smerodatna_odchylka(v)
        mean[i].append(un_mean)
        std[i].append(un_std)
    
    def predict(means, stds, uniques):
      preds = []
      for i in self.test:
        log_likely = []
        for g in range(len(uniques)):
          likelyhoods = []
          for d in range(len(i)):
            likelyhood = Guassian_naive_bayes.normal_distribution_likelyhood(stds[g][d], means[g][d], i[d])
            likelyhoods.append(likelyhood)
          a = 0
          for u in likelyhoods:
            a += math.log(u)
          
          log_likely.append(a)

        fa = sorted(log_likely)[-1]
        preds.append(uniques[log_likely.index(fa)])

      return preds

    return predict(mean, std, unique)
  



class prediction_models:
  def __init__(self, x, y, normalization, train_batch):
    self.x = x
    self.y = y
    self.normalization = normalization
    self.train_batch = train_batch
    self.models_types = []



  def preparing_data(self):
    new_data = []
    supa_new_data = []
    for n ,i in enumerate(self.x):
      b = []
      if self.normalization[n] == 1: 
        datas = Z_score(data[i])
        for c in datas:
          b.append(c)

      else:
        for c in data[i]:
          b.append(c)

      new_data.append(b)

    for h in range(len(new_data[0])):
      sample = []
      for g in new_data:
        sample.append(g[h])

      supa_new_data.append(sample)

    supa_y = []
    for s in self.y:
      supa_y.append(s)

    def shuffles(x, y, seeds=1):
      random.seed(seeds)
      xa = random.sample(x, len(x))
      random.seed(seeds)
      ya = random.sample(y, len(y))

      return xa, ya

    supa_new_data, supa_y = shuffles(supa_new_data, supa_y)


    train_x = supa_new_data[:self.train_batch]
    train_y = supa_y[:self.train_batch]
    test_x = supa_new_data[self.train_batch:]
    test_y = supa_y[self.train_batch:]

    return train_x, train_y, test_x, test_y

  def Guassian_naive_bayes_classifier(self):
    train_x, train_y, test_x, test_y = prediction_models.preparing_data(self)
    predicitons = Guassian_naive_bayes(train_x, train_y, test_x).main()
    return predicitons, test_y

  def Neural_Network(self):
    train_x, train_y, test_x, test_y = prediction_models.preparing_data(self)
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    test_x = np.asarray(test_x)

    
    m = models.Sequential()
    m.add(layers.Dense(32, activation="relu",  input_shape=(train_x.shape[1],)))
    m.add(layers.Dense(64, activation="relu"))
    m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])

    m.fit(train_x, train_y, epochs=30, batch_size=8)
    

    predictions = m.predict(test_x)
   

    return predictions, test_y

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

modelsa = [prediction_models(predictors, y, [0, 0], 230).Neural_Network(), prediction_models(predictors, y, [0, 0], 230).Guassian_naive_bayes_classifier()]
names = ["Neural Network", "Naive_Gaussian_bayes"]

def ROC(real, probs):
  x = [1]
  y = [1]
  thresholds = []
  acc = []
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
    accuracy = classification_evaluate(preds, real).accuracy()
    acc.append(accuracy)

    x.append(x_i)
    y.append(y_i)
    

  plt.title("ROC curve")
  plt.style.use("fivethirtyeight")
  
  plt.xlabel("1 - Specifity")
  plt.ylabel("Recall")

  plt.plot(x, y, "-o")

  plt.show()

  a = sorted(acc)[-1]
  b = acc.index(a)
  return thresholds[b]




for n ,i in enumerate(modelsa):
  if names[n] == "Neural Network":
    preds = []
    predicted, real = i
    threshold = ROC(real, predicted)
    
    for g in predicted:
      if g[0] > threshold:
        preds.append(1)

      else:
        preds.append(0)

    predicted = preds
  
  else:
    predicted, real = i
 
  metrices_names = ["accuracy", "precision", "specifity", "recall"]

  accuracy = classification_evaluate(predicted, real).accuracy()
  precision = classification_evaluate(predicted, real).Precision()
  specificity = classification_evaluate(predicted, real).specificity()
  recall = classification_evaluate(predicted, real).recall()
  metricis = [accuracy, precision, specificity, recall]
  
  for t, r in enumerate(metricis):
    print(f"{names[n]}'s {metrices_names[t]} is {r}" )

  print("""------------------------------
------------------------------
------------------------------"""
)

  


    


    
  
  








  
