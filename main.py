#acquisizione dei dati dalle API di google
import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage_v1beta1

#necessario importare un file con le credenziali per poter accedere alle API di google
import credentials

credentials, your_project_id = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

bqclient = bigquery.Client(
    credentials=credentials,
    project=your_project_id,
)
bqstorageclient = bigquery_storage_v1beta1.BigQueryStorageClient(
    credentials=credentials
)

#codice in SQL per ricevere i dati
query_string = """
SELECT *
FROM `bigquery-public-data.covid19_italy.national_trends`
ORDER BY date
LIMIT 1000
"""
dataframe = (
    bqclient.query(query_string)
    .result()
    .to_dataframe(bqstorage_client=bqstorageclient)
)

from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

#scelta dei dati dal dataframe
data = dataframe.filter(['recovered', 'deaths'])
array = data.to_numpy()

#i dati vengono riadattati secondo una scala che mantiene i valori tra 0 e 1
data_scaler = RobustScaler()
array = data_scaler.fit_transform(array)

#serve molto per capire quali dati usare in base alla omogeneità visiva dei dati
'''from matplotlib import pyplot
dataframe.hist()
pyplot.show()'''

#suddivido i dati in training set e test set
X_train1 = np.transpose(array)[0][0:60].reshape(60, 1)
X_valid = np.transpose(array)[0][60:91].reshape(31, 1)
Y_train1 = np.transpose(array)[1][1:61].reshape(60, 1)
Y_valid = np.transpose(array)[1][61:92].reshape(31, 1)

#creo un modello di apprendimento knn regressor
knnr = KNeighborsRegressor(1)
knnr.fit(X_train1, Y_train1)
Y_predic = knnr.predict(X_valid)

#cerco il k che genera un errore minimo
min_MAE = mean_absolute_error(Y_valid, Y_predic)
MAE = []
for k in range(2,60):
  knnr = KNeighborsRegressor(k)
  knnr.fit(X_train1, Y_train1)
  Y_predic = knnr.predict(X_valid)
  now_MAE = mean_absolute_error(Y_valid, Y_predic)
  MAE.append(now_MAE)
  if now_MAE < min_MAE:
    min_MAE = now_MAE

#cerca l'errore minimo all'interno della lista di errori per eliminare il k
k_optimal = 0
for x in range(len(MAE)):
  if MAE[x] == min_MAE:
    k_optimal = x + 2
if k_optimal == 0:
  k_optimal = 1

#genera un modella knn regressor con un k ottimale
knnr = KNeighborsRegressor(k_optimal)
knnr.fit(X_train1, Y_train1)
Y_predic = knnr.predict(X_valid)
print("MAE =",mean_absolute_error(Y_valid, Y_predic))
print("MQE =",mean_squared_error(Y_valid, Y_predic))
print("R2 =",r2_score(Y_valid, Y_predic))
print(Y_predic)


