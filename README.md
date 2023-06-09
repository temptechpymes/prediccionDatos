# prediccionDatos
Algoritmo de Python de ejemplo para predicción de ventas

Es un ejemplo sencillo donde se puede trabajar la predicción de datos, es una estructura de datos de series temporales.

El archivo que contiene el código es "prediccion.ipynb", esta extensión debe trabajarse en Jupyter Notebook, para trabajar aquí se debe descargar Anaconda Navigator.

El algoritmo trabaja con librerías de Python como pandas, numpy y matplotlib que se utilizan para el analisis de datos y graficación.



En esta primera parte del código se importan las librerías con las que se van a trabajar y se define la función para que lea el archivo en sus columnas "fecha" y "unidades"
"import pandas as pd
import numpy as np
import matplotlib.pylab as plt

plt.rcParams['figure.figsize'] = (16,9)
plt.style.use('fast')

from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('Desktop/predicción/prediccion.csv', parse_dates=[0], header=None, index_col=0
, append.squeeze==True, names=['fecha', 'unidades'])
df.head()"


Imprime la fecha minima y máxima.
print(df.index.min())
print(df.index.max())

Cantidad de datos en 2017 y 2018
print(len(df['2017']))
print(len(df['2018']))


df.describe()
#Son un total de 604 registros, la media de venta de unidades es de 215 y un desvío de 75, es decir que por lo general estaremos entre 140 y 290 unidades.

La media en cada mes
meses =df.resample('M').mean()
meses


Imprime la gráfica de 2017 y 2018 con sus respectivos valores
plt.plot(meses['2017'].values)
plt.plot(meses['2018'].values)


Es un ejemplo sobre ventas en una heladería y se compara las ventas en verano del 2017 y 2018
verano2017 = df['2017-06-01':'2017-09-01']
plt.plot(verano2017.values)
verano2018 = df['2018-06-01':'2018-09-01']
plt.plot(verano2018.values)

##gráfica de ventas diarias (en unidades) en junio y julio



#Pronóstico de Ventas Diarias con Redes Neuronal

#7 días previos para “obtener” el octavo.

PASOS = 7
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
# load dataset
values = df.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, PASOS, 1)
reframed.head()



# split into train and test sets
values = reframed.values
n_train_days = 315+289 - (30+PASOS)
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
x_train, y_train = train[:, :-1], train[:, -1]
x_val, y_val = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)


Se crea el modelo
def crear_modeloFF():
    model = Sequential() 
    model.add(Dense(PASOS, input_shape=(1,PASOS),activation='tanh'))
    model.add(Flatten())
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mean_absolute_error',optimizer='Adam',metrics=["mse"])
    model.summary()
    return model
    
    
#En pocos segundos vemos una reducción del valor de pérdida tanto del set de entrenamiento como del de validación.

EPOCHS=40
 
model = crear_modeloFF()
 
history=model.fit(x_train,y_train,epochs=EPOCHS,validation_data=(x_val,y_val),batch_size=PASOS)




results=model.predict(x_val)
plt.scatter(range(len(y_val)),y_val,c='g')
plt.scatter(range(len(results)),results,c='r')
plt.title('validate')
plt.show()

#En la gráfica vemos que los puntitos verdes intentan aproximarse a los rojos. Cuanto más cerca ó superpuestos mejor. TIP: Si aumentamos la cantidad de EPOCHS mejora cada vez más.



Aquí se seleccionan los últimos días para tenerlos como fuente histórica para crear la predicción
ultimosDias = df['2018-11-16':'2018-11-30']
ultimosDias


values = ultimosDias.values
values = values.astype('float32')
# normalize features
values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, PASOS, 1)
reframed.drop(reframed.columns[[7]], axis=1, inplace=True)
reframed.head(7)

values = reframed.values
x_test = values[6:, :]
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
x_test

def agregarNuevoValor(x_test,nuevoValor):
    for i in range(x_test.shape[2]-1):
        x_test[0][0][i] = x_test[0][0][i+1]
    x_test[0][0][x_test.shape[2]-1]=nuevoValor
    return x_test

results=[]
for i in range(7):
    parcial=model.predict(x_test)
    results.append(parcial[0])
    print(x_test)
    x_test=agregarNuevoValor(x_test,parcial[0])
    

adimen = [x for x in results]    
inverted = scaler.inverse_transform(adimen)
inverted


Con esta última se crea la predicción de manera gráfica que muestra las próximas ventas en 7 días

prediccion1SemanaDiciembre = pd.DataFrame(inverted)
prediccion1SemanaDiciembre.columns = ['pronostico']
prediccion1SemanaDiciembre.plot()
prediccion1SemanaDiciembre.to_csv('pronostico.csv')








