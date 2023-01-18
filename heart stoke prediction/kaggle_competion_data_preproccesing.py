import pandas as pd
import matplotlib.pyplot as plt
import math
import pickle
from collections import Counter


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


training_data1 = pd.read_csv("train.csv")
training_data2 = pd.read_csv("train1.csv")

training_data1 = training_data1.dropna()


print(len(training_data1), len(training_data2))


discrete_features = ['gender', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type',
       'smoking_status']

continuous_features = ['age', 'avg_glucose_level', 'bmi']

y_feature = ["stroke"]

discrete_mapping = [{}, {}, {}, {}, {}, {}, {}]

x_data = training_data1.append(training_data2)
x_data_continous = x_data[continuous_features].values.tolist()

del x_data_continous[14272]


x_data_discrete = x_data[discrete_features].values.tolist()
del x_data_discrete[14272]



y = x_data["stroke"].tolist()

del y[14272]

for i in range(len(discrete_features)):
  for j in x_data_discrete:
    if j[i] not in discrete_mapping[i]:
      discrete_mapping[i][j[i]] = len(discrete_mapping[i])



def euclidian_distance(vector_1, vector_2, indexes_to_pass):
  distance = 0
  
  if len(vector_1) == len(vector_2):
    for n in range(len(vector_1)):
      if n not in indexes_to_pass:
        distance += (vector_1[n]-vector_2[n])**2

  return math.sqrt(distance)



class replace_missing_data:
  def __init__(self, neighbours):
    self.data_storage = {}
    self.neighbours = neighbours
    #none_features = {30: 168, 28: 78, 29: 78}

  def train(self, x, y):
    for n, i in enumerate(x):
      if (3 == i[-2]) or (2 == i[-1]):
        continue
      if y[n] not in self.data_storage:
        if 3 in i:
          print("fuck")
        self.data_storage[y[n]] = [i]

      else:
        if 3 in i:
          print("fuck")
        self.data_storage[y[n]].append(i)

  def apply(self, x, y):
    if self.data_storage == {}:
      return "not trained"

    new_data = []
    for i in list(self.data_storage.keys()):
      for j in self.data_storage[i]:
        if 3 in j:
          print("sus")

    for n, i in enumerate(x):
      if 3 != i[-2] and 2 != i[-1]:
        if 3 in i:
          print(i[-2])
        
        new_data.append(i)
        continue






      
      none_indexes = [len(x[0])-2, len(x[0])-1]

      distances = []
      new_values = []




      

      for vector in self.data_storage[y[n]]:
        dist = euclidian_distance(vector, i, none_indexes)
        vals = [vector[q] for q in none_indexes]

        distances.append(dist)
        new_values.append(vals)

      distances, new_values = (list(t) for t in zip(*sorted(zip(distances, new_values))))

      for j in range(len(none_indexes)):
        best_values = []
        w = 0
        for k in range(self.neighbours):
        
          
          for _ in range(self.neighbours-w):
            best_values.append(new_values[k][j])
        
          w =+ 1

        
        occurence_count = Counter(best_values)
        new_val_fin = occurence_count.most_common(1)[0][0]

        i[none_indexes[j]] = new_val_fin

      new_data.append(i)





    return new_data, y

  def save(self, file_name):
    
    to_save = {"ds": self.data_storage, "ne": self.neighbours}
    
    with open(file_name, "wb") as f:
      pickle.dump(to_save, f)

  def load(self, file_name):
    try:
      with open(file_name, "rb") as f:
        repla = pickle.load(f)

      self.data_storage = repla["ds"]
      self.neighbours= repla["ne"]

    except:
      print("File doesnt exist")





class normalize:
  def __init__(self):
    self.means = []
    self.stds = []

  def calculate(self, x):
    columns = [[]]*len(x[0])
    for i in x:
      for n, j in enumerate(i):
        if j!= None:
          columns[n].append(j)

    for i in columns:
      self.means.append(sum(i) / len(i))
      self.stds.append(smerodatna_odchylka(i))

  def apply(self, x):
    if self.means == []:
      return

    new_data = []

    for i in x:
      x_i = []
      for n, j in enumerate(i):
        if j != None:
          z_value = (j-self.means[n]) / self.stds[n]
          x_i.append(z_value)

        else:
          x_i.append(None)

      new_data.append(x_i)

    return new_data

  def save(self, file_name):
    if self.means == []:
      return

    to_save = {"means": self.means, "stds": self.stds}

    with open(file_name, "wb") as f:
      pickle.dump(to_save, f)

  def load(self, file_name):
    try:
      with open(file_name, "rb") as f:
        normalizer = pickle.load(f)

      self.means = normalizer["means"]
      self.stds = normalizer["stds"]

    except:
      print("File doesnt exist")

continous_variables_normalizer = normalize()

continous_variables_normalizer.calculate(x_data_continous)

normalized_variables = continous_variables_normalizer.apply(x_data_continous)

continous_variables_normalizer.save("kaggle_continous_variables_normalize.pickle")





for n, i in enumerate(x_data_discrete):
  normalized_variables[n].append(discrete_mapping[-1][i[-1]])
  normalized_variables[n].append(discrete_mapping[0][i[0]])



replacer = replace_missing_data(5)


replacer.train(normalized_variables, y)



normalized_variables, y = replacer.apply(normalized_variables, y)

replacer.save("kaggle_replacer.pickle")

discrete_mapping.insert(0, discrete_mapping[-1])
del discrete_mapping[-1]





for n, i in enumerate(x_data_discrete):
  new_f = []
  ind = 2
  for j in i[1:len(i)-1]:
    new_f.append(discrete_mapping[ind][j])
    ind += 1

  normalized_variables[n] += new_f



  


with open("kaggle_normalized_variables.pickle", "wb") as f:
  pickle.dump(normalized_variables, f)

with open("kaggle_y.pickle", "wb") as f:
  pickle.dump(y, f)

with open("kaggle_discrete_mapping.pickle", "wb") as f:
  pickle.dump(discrete_mapping, f)


























