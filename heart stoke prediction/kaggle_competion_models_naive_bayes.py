import pickle
import math
import numpy as np
from sklearn.metrics import roc_auc_score
import random
import pandas as pd
from collections import Counter



with open("kaggle_normalized_variables.pickle", "rb") as f:
  x = pickle.load(f)

with open("kaggle_y.pickle", "rb") as f:
  y = pickle.load(f)

with open("kaggle_discrete_mapping.pickle", "rb") as f:
  discrete_mapping = pickle.load(f)


def splitting(x, y):
  splitoss = math.ceil(0.74*len(x))
  seed = 18

  random.seed(seed)
  x = random.sample(x, len(x))
  random.seed(seed)
  y = random.sample(y, len(y))

  x_train = x[:splitoss]
  y_train = y[:splitoss]
  x_test = x[splitoss:]
  y_test = y[splitoss:]

  return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = splitting(x, y)


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



def Naive_bayes_nowords(x, y, test):
  unique = []
  for i in y:
    if i not in unique:
      unique.append(i)

  print(unique)

  standard_pr = []

  for k in unique:
    a = 0
    for u in y:
      if u == k:
        a +=1

      else:
        pass
    
    standard_pr.append(a/len(y))
  
  print(standard_pr)

  v = []
  for t in range(len(unique)):
    v.append([])

  for a, i in enumerate(unique):
    for b in range(len(x)):
      if y[b] == i:
        v[a].append(x[b])

      else:
        pass

  supa = []

  for i in v:
    w = []
    for q in range(len(i[0])):
      w.append([])
    for r in range(len(i[0])):
      for g in i:
        w[r].append(g[r])

    supa.append(w)

  probs = []

  for k in range(len(supa)):
    probs.append([])

  for b , j in enumerate(supa):
    for t in j:
      uni = []
      for d in t:
        if d not in uni:
          uni.append(d)
        
      count = []
      
      for h in uni:
        a = 0
        for e in t:
          if e == h:
            a += 1

          else:
            pass

        count.append(a)

      ajaja = {}

      for v in range(len(count)):
        pr = count[v] / sum(count)
        ajaja[uni[v]] = pr

      probs[b].append(ajaja)
  
  estimated_prob = []

  preds = []

  for w in test:
    estimated_prob = []
    for i in range(len(probs)):
      prob = 1
      for n ,q in enumerate(w):
        try:
          prob = probs[i][n][q] * prob

        except:
          prob = prob * 0.001

      estimated_prob.append(prob*standard_pr[i])

    
    preds.append(estimated_prob)
  
  def save_model(name):
    with open(f"{name}.pickle", "wb") as f:
      pickle.dump(probs, f)
    
    print("Model saved")

  save_model("kaggle_competion_bayes_no_words")
  
  return preds, unique


class Guassian_naive_bayes:
  def __init__(self, x, y, test):
    self.x = x
    self.y = y
    self.test = test
    self.predictors = [None, None]

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
    
    self.predictors[0] = mean
    self.predictors[1] = std
    print(unique)
    
    def predict(means, stds, uniques):
      
      preds = []
      for i in self.test:
        log_likely = []
        for g in range(len(uniques)):
          likelyhoods = []
          for d in range(len(i)):
            likelyhood = Guassian_naive_bayes.normal_distribution_likelyhood(stds[g][d], means[g][d], i[d])
            likelyhoods.append(likelyhood)
          a = 1
          for u in likelyhoods:
            if u == 0:
              u = 0.0001
            a *= u
          
          log_likely.append(a)

        
        preds.append(log_likely)

      return preds
    
    return predict(mean, std, unique), unique

  def save_model(self, name):
    print(self.predictors)
    with open(f"{name}.pickle", "wb") as f:
      pickle.dump(self.predictors, f)
    
    print("Model saved")






def Main_Predict(x_train, y_train, x_test, y_test):
  normal_bayes_train = []
  gaussian_bayes_train = []
  normal_bayes_test = []
  gaussian_bayes_test = []
  for i in x_train:
    gaussian_bayes_train.append(i[:3])
    normal_bayes_train.append(i[3:])

  for i in x_test:
    gaussian_bayes_test.append(i[:3])
    normal_bayes_test.append(i[3:])

  preds_naive_bayes, unq = Naive_bayes_nowords(normal_bayes_train, y_train, normal_bayes_test)
  m = Guassian_naive_bayes(gaussian_bayes_train, y_train, gaussian_bayes_test)
  preds_gaussian_bayes, unique = m.main()
  m.save_model("kaggle_gaussian_bayes")

  for i in range(len(preds_gaussian_bayes)):
    u = 1 / sum(preds_gaussian_bayes[i])
    for j in range(len(preds_gaussian_bayes[i])):
      preds_gaussian_bayes[i][j] = preds_gaussian_bayes[i][j] * u

  for i in range(len(preds_naive_bayes)):
    u = 1 / sum(preds_naive_bayes[i])
    for j in range(len(preds_naive_bayes[i])):
      preds_naive_bayes[i][j] = preds_naive_bayes[i][j] * u


  final_preds = []
  probs = []
  for i in range(len(preds_naive_bayes)):
    fin_pred = []
    for j in range(len(preds_naive_bayes[i])):
      fin_pred.append((preds_naive_bayes[i][j] + preds_gaussian_bayes[i][j])/2)

    probs.append(fin_pred[unq.index(1)])

    

    final_preds.append(unique[fin_pred.index(max(fin_pred))])

  evaluation = roc_auc_score(y_test, probs)


  return evaluation

print(Main_Predict(x_train, y_train, x_test, y_test))




def load_models(name1 , name2):
  with open(name1, "rb") as f:
    m1 = pickle.load(f)

  with open(name2, "rb") as f:
    m2 = pickle.load(f)


  return m1, m2

class funkce:
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

  def normalization(x):
    pr = [] 
    u =  1 / sum(x)
    for i in x:
      pr.append(i*u)

    return pr

  def std(y):
    all_nums = []
    k = sum(y) / len(y)
    for i in(y):
      rozdil = i - k 
      num = rozdil * rozdil
      all_nums.append(num)

    rozptyls = sum(all_nums) / len(all_nums)
  
    return math.sqrt(rozptyls)


def euclidian_distance(vector_1, vector_2, indexes_to_pass):
  distance = 0
  
  if len(vector_1) == len(vector_2):
    for n in range(len(vector_1)):
      if n not in indexes_to_pass:
        distance += (vector_1[n]-vector_2[n])**2

  return math.sqrt(distance)


class model:
  def __init__(self):
    self.naive_bayes, self.gaussian_naive_bayes = load_models("kaggle_competion_bayes_no_words.pickle", "kaggle_gaussian_bayes.pickle")
    self.teams_count = 2
    self.standard_pr = [0.9585521001615509, 0.04144789983844911]

  def predict_naive_bayes(self, x):
    probs = []
    for i in range(self.teams_count):
      prob = 1
      for n ,q in enumerate(x):
        try:
          prob = self.naive_bayes[i][n][q] * prob

        except:
          pass

      probs.append(prob*self.standard_pr[i])

    return probs

  def predict_gaussian_naive_bayes(self, x):
    probs = []
    
    for g in range(self.teams_count):
      a = 1
      for d in range(len(x)):
        try:
     
          a *= funkce.normal_distribution_likelyhood(self.gaussian_naive_bayes[1][g][d], self.gaussian_naive_bayes[0][g][d], x[d])

        except:
          pass

    
    
      probs.append(a)

    return probs

def main_predict(naive_bayes_pred, gaussian_naive_bayes_pred):
  naive_bayes_pred = funkce.normalization(naive_bayes_pred)
  gaussian_naive_bayes_pred = funkce.normalization(gaussian_naive_bayes_pred) 
  fin_pred = []
  for j in range(len(naive_bayes_pred)):
    fin_pred.append((naive_bayes_pred[j] + gaussian_naive_bayes_pred[j])/2)

  return funkce.normalization(fin_pred)

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




      

      for vector in self.data_storage[0] + self.data_storage[1]:
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









test_data = pd.read_csv("test.csv")


discrete_features = ['gender', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type',
       'smoking_status']

continuous_features = ['age', 'avg_glucose_level', 'bmi']

x_data_continous = test_data[continuous_features].values.tolist()
x_data_discrete = test_data[discrete_features].values.tolist()

continous_variables_normalizer = normalize()

continous_variables_normalizer.load("kaggle_continous_variables_normalize.pickle")

normalized_variables = continous_variables_normalizer.apply(x_data_continous)


for n, i in enumerate(x_data_discrete):
  normalized_variables[n].append(discrete_mapping[0][i[-1]])
  normalized_variables[n].append(discrete_mapping[1][i[0]])


replacer = replace_missing_data(5)

replacer.load("kaggle_replacer.pickle")

normalized_variables, [] = replacer.apply(normalized_variables, [])







for n, i in enumerate(x_data_discrete):
  new_f = []
  ind = 2
  for j in i[1:len(i)-1]:
    new_f.append(discrete_mapping[ind][j])
    ind += 1

  normalized_variables[n] += new_f


print(normalized_variables[0])
  

final_test_probs = []

my_model = model()


submit_data = {"id": test_data["id"].values.tolist(), }

for i in normalized_variables:
  gauss_preds = my_model.predict_gaussian_naive_bayes(i[:3])
  discrete_preds = my_model.predict_naive_bayes(i[3:])
  fin_pred = main_predict(discrete_preds, gauss_preds)
  final_test_probs.append(fin_pred[1])


print(final_test_probs[0])
submit_data = {"id": test_data["id"].values.tolist(), "stroke": final_test_probs}

df = pd.DataFrame(submit_data)

df.to_csv("submision3.csv", index=False)




