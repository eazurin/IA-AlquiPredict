import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle

#Leemos los datos con pandas
data=pd.read_excel('bd_alquiler.xlsx')
#Mostramos 5 registros como ejemplo
#print(data.head())


#Mirando informacion como el numero de filas y columnas de nuestra base de datos
#print(data.shape)
#Informacion sobre los features, ademas el tipo de dato y el uso de memoria
#print(data.info())

#Extrayendo frecuencias de ocurrencia de cada feature en todos los registros
#for column in data.columns:
    #print(data[column].value_counts())
    #print("*"*12)

#Verificando la cantidad de registros que tienen algun campo faltante
#print(data.isna().sum())

data=data.drop(columns=['Unnamed: 0', 'Piso de ubicación', 'Vista al exterior', 'Tipo de cambio',
                        'Alquiler mensual en dólares corrientes', 'Alquiler mensual en soles constantes de 2009', 'Tipo de cambio', 'IPC'])

# Rellenar valores faltantes en 'Años de antigüedad', 'Número de baños' y 'Número de garages' con la mediana
data['Años de antigüedad'].fillna(data['Años de antigüedad'].median(), inplace=True)
data['Número de baños'].fillna(data['Número de baños'].median(), inplace=True)
data['Número de garages'].fillna(data['Número de garages'].median(), inplace=True)

#print(data.describe())
data.to_csv('dataset_final.csv')

# Convertir variables categóricas en variables dummy
data = pd.get_dummies(data, columns=['Distrito'], drop_first=True)
# Identificar filas con valores NaN en cualquier columna
# rows_with_nan = data[data.isnull().any(axis=1)]
# print(rows_with_nan)
# Eliminar filas con valores NaN    
data = data.dropna()

# Separar características (X) y la variable objetivo (y)
X = data.drop(columns=['Alquiler mensual en soles corrientes'])
y = data['Alquiler mensual en soles corrientes']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir el modelo de Random Forest con un número específico de estimadores
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
score = model.score(X_test, y_test)
print(f'R^2 score: {score}')
print(f'Mean Squared Error: {mse}')

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Guardar las columnas que se utilizaron para entrenar el modelo
with open('columns.pkl', 'wb') as columns_file:
    pickle.dump(X.columns.tolist(), columns_file)
