import os
import pandas as pd
from matplotlib import pyplot as plt
import math
from keras import layers
from keras import models
import numpy as np
from keras.models import load_model


def Z_score(x):
  mean = sum(x) / len(x)
  std = smerodatna_odchylka(x)

  for i in range(len(x)):
    x[i] = (x[i]-mean) / std

  return x

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




start_dir = "C:\\Users\\42072\Downloads\cars"

b = os.listdir(start_dir)

#data = pd.get_dummies(data, columns= ["transmission"], drop_first=False)

names = []
avgs = []



for i in b:
  data = pd.read_csv(start_dir + "\\" + i)
  names.append(i.split(".")[0])
  avg = sum(data["price"]) / len(data)
  avgs.append(avg)

plt.style.use("seaborn")

plt.barh(names, avgs)

plt.title("Prices of cars")

plt.ylabel("Car brands")

plt.xlabel("Price")

plt.tight_layout()

plt.show()


#Because I am from Czech republic, I will take a better look at Škoda

skoda_data = pd.read_csv(start_dir+ "\\skoda.csv")

skoda_data = pd.get_dummies(skoda_data, columns= ["transmission", "fuelType"], drop_first=False)

u = skoda_data["model"].unique()

pricese = []

for i in u:
  model_data = skoda_data[skoda_data["model"] == i]["price"]
  pricese.append(sum(model_data) / len(model_data))


plt.style.use("seaborn")

plt.barh(u, pricese)

plt.title("Prices of models")

plt.ylabel("Car models")

plt.xlabel("Price")

plt.tight_layout()

plt.show()


skoda_data = pd.get_dummies(skoda_data, columns= ["model"], drop_first=False)

y = skoda_data["price"]

skoda_data = skoda_data.drop(["price", "year"], axis=1)

predictors = skoda_data.columns

scale = 10000


x = []

for i in range(len(skoda_data)):
  dah = []
  for t in range(len(skoda_data.columns)):
    dah.append(skoda_data[predictors].iloc[i][t])

  x.append(dah)

train_split = 4000


def preparing_data(x, y, train_split):
  for h in range(3):
    o = []
    for g in x:
      o.append(g[h])

    z = Z_score(o)
    for g in range(len(x)):
      x[g][h] = z[g]      

  
  train_x = x[:train_split]
  train_y = y[:train_split]
  test_x = x[train_split:]
  test_y = y[train_split:]

  return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = preparing_data(x, y, train_split)


class modelsaj:

  def Neural_Network(train_x, train_y, test_x, type_model):
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    test_x = np.asarray(test_x)
    try:
      m = load_model(f"cars{type_model}.h5")

    except:
      m = models.Sequential()
      m.add(layers.Dense(32, activation="relu",  input_shape=(train_x.shape[1],)))
      m.add(layers.Dense(64, activation="relu"))
      m.add(layers.Dense(1))
      m.compile(optimizer="Adam", loss="mse", metrics=["mae"])

      #m.fit(train_x, train_y, epochs=50, batch_size=8)
    

    predictions = m.predict(test_x)


    return predictions


class prediction:
  def __init__(self, train_x, train_y, test_x, type_model):
    self.train_y = train_y
    self.train_x = train_x
    self.test_x = test_x
    self.type_model = type_model

  def Neural_Network_predictions(self):
    predictions = modelsaj.Neural_Network(self.train_x, self.train_y, self.test_x, self.type_model)
    return predictions

type_models = ["_normal", "_scaled", "_normalized"]
names = ["normal", "scaled", "normalized"]

predictions = []

for i in type_models:
  preds = prediction(train_x, train_y, test_x, i).Neural_Network_predictions()
  predictions.append(preds)



class evaluate:
  def __init__(self, real, predictions):
    self.real = real
    self.predictions = predictions

  def RMSE(self):
    p = []
    for i in range(len(self.real)):
      b = (self.real[i] - self.predictions[i])**2
      p.append(b)

    return math.sqrt(sum(p) / len(p))

  def R_squared(self):
    f = rozptyl(self.real)
    fucku = []
    for t in range(len(self.real)):
      g = self.real[t] - self.predictions[t]
      fucku.append(g**2)

    ahojda = sum(fucku) / len(fucku)

    finalejesmegusta = (f - ahojda) / f
    return finalejesmegusta

  def F(self):
    mean = sum(self.real) / len(self.real)
    meansquared = []
    for i in self.real:
      meansquared.append((i-mean)**2)

    fucku = []
    for t in range(len(self.real)):
      g = self.real[t] - self.predictions[t]
      fucku.append(g**2)

    fin_eq = (sum(meansquared) - sum(fucku)) / (sum(fucku) /(len(x)-len(train_x[0])))

    return fin_eq

  def MAE(self):
    absolutes = []
    for i in range(len(self.predictions)):
      absolute = abs(self.real[i] - self.predictions[i])
      absolutes.append(absolute)

    fin = sum(absolutes) / len(absolutes)

    return fin


for i in range(len(predictions)):
  y_new = []
  if names[i] == "scaled":
    for o in test_y:
      y_new.append(o/scale)
  
  elif names[i] == "normalized":
    y_j = Z_score(y)
    for s in y_j[train_split:]:
      y_new.append(s)
      
  else:
    for k in test_y:
      y_new.append(k)

 
  metrics = [evaluate(y_new, predictions[i]).RMSE(), 
             evaluate(y_new, predictions[i]).R_squared(), 
             evaluate(y_new, predictions[i]).F(), 
             evaluate(y_new, predictions[i]).MAE()]
 
  metric_names = ["RMSE", "R**2", "F", "MAE"]

  for c in range(len(metrics)):
    print(f"{names[i]}'s {metric_names[c]} is {metrics[c]}")

  print("""[<<<>>>-----------------<<<>>>-------------
---------<<<>>>-----------<<<>>>----------
---------------<<<>>>---------------<<<>>>]"""
)

  

  

  
    
    
  
      
    


    

      

    
  

