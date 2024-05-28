import pandas as pd
import numpy as np

#Leemos los datos con pandas
data=pd.read_excel('bd_alquiler.xlsx')
#Mostramos 5 registros como ejemplo
print(data.head())

#Mirando informacion como el numero de filas y columnas de nuestra base de datos
print(data.shape)
#Informacion sobre los features, ademas el tipo de dato y el uso de memoria
print(data.info())

#Extrayendo frecuencias de ocurrencia de cada feature en todos los registros
for column in data.columns:
    print(data[column].value_counts())
    print("*"*12)

#Verificando la cantidad de registros que tienen algun campo faltante
print(data.isna().sum())

data=data.drop(columns=['Unnamed: 0', 'Piso de ubicación', 'Vista al exterior', 'Tipo de cambio',
                        'Alquiler mensual en dólares corrientes', 'Alquiler mensual en soles constantes de 2009', 'Tipo de cambio', 'IPC'])

print(data.describe())