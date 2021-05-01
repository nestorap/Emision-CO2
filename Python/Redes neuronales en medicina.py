# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Librerías

import os
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#%%
os.chdir('/home/nestor/Proyectos/Udemy Data Science/Tarea 5')

# Guardo la ruta donde están las carpetas con las distintas imágenes
directorio = '/home/nestor/Proyectos/Udemy Data Science/Tarea 5/4. Operations Department/Dataset'

# Compruebo las carpetas que hay en el directorio que he guardado
print(os.listdir(directorio))
#%%

# Creo el objeto generator. Es el train_test_split de las imágenes
img_generator = ImageDataGenerator(rescale=1./255, # Normalizo. 255 es el máximos en la escala RGB
                                   validation_split=0.20)

# Genero la muestra de train y val

train_gen = img_generator.flow_from_directory(batch_size=4,
                                              directory=directorio,
                                              shuffle=True,
                                              target_size=(256,256),
                                              class_mode='categorical',
                                              subset='training')

val_gen = img_generator.flow_from_directory(batch_size=4,
                                            directory=directorio,
                                            shuffle=True,
                                            target_size=(256,256),
                                            class_mode='categorical',
                                            subset='validation')
#%%
# voy a imprimir un lote de 40

train_img, train_labels = next(train_gen) # Esto me genera las imágenes con los lotes

#%%    

print(train_img.shape) # Elemento cuatridimensional. Primero es el número de imagenes, los siguientes 2 los pixeles y el ultimo escala RGB
print(train_labels.shape)# 40 imágenes por 4 categorias. Está como dummy

#%%
# Guardo las etiquetas para entender mejor
etiquetas = {0 : 'covid',
             1 : 'sano',
             2 : 'vírica',
             3 : 'bacteriana'}
#%%

# Visualizar algunas imágenes que voy a meter a la red

L = 6 # Largo
W = 6 # Ancho

fig, axes = plt.subplots(L, W, figsize=(12,12))
axes = axes.ravel() # Apalana la matriz de las imágenes

for i in np.arange(0, L*W):
    axes[i].imshow(train_img[i])
    axes[i].set_title(etiquetas[np.argmax(train_labels[i])])
    axes[i].axis('off')
    
    
plt.subplots_adjust(wspace = 0.5)

#%%

# Cargamos el modelo con los pesos preentrenados para imágenes
basemodel = ResNet50(weights='imagenet',
                     include_top = False,
                     input_tensor=Input(shape=(256,256,3))) #El input lo elijo yo

# Printo la red que he elegido.
#print(basemodel.summary())

#%%
# Congelamos las 10 últimas capas

for capa in basemodel.layers[:-10]: # Todas las capas excepto las últimas 10
    capa.trainable = False

#%%
    
headmodel = basemodel.output
headmodel = AveragePooling2D(pool_size=(4,4))(headmodel)
headmodel = Flatten(name = 'flatten')(headmodel)
headmodel = Dense(256, activation = 'relu')(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(128, activation = 'relu')(headmodel)
headmodel = Dropout(0.25)(headmodel)
headmodel = Dense(4, activation = 'softmax')(headmodel)


modelo = Model(inputs = basemodel.input, outputs = headmodel)

modelo.compile(loss = 'categorical_crossentropy',
               optimizer = optimizers.RMSprop(learning_rate= 1e-4,
                                              decay=1e-6),
               metrics = ['accuracy'])

earlystop = EarlyStopping(monitor='val_loss',
                          mode = min,
                          patience = 20)

checkpointer = ModelCheckpoint(filepath = 'pesos.hdf5',
                               verbose = 1,
                               save_best_only=True)

#%%

history = modelo.fit_generator(train_gen,
                              steps_per_epoch = train_gen.n//4,
                              epochs = 20,
                              validation_data = val_gen,
                              validation_steps = val_gen.n//4,
                              callbacks = [checkpointer,  earlystop])

#%%
# Este código es para cargar los pesos del modelo entrenado por si se ha cerrado

modelo.load_weights('pesos.hdf5')
modelo.compile(loss = 'categorical_crossentropy',
              optimizer = optimizers.RMSprop(learning_rate= 1e-4,
                                              decay=1e-6),
              metrics = ['accuracy'])

#%%
print(history.history.keys())

#%%
# Evalucación del modelo
# Juntar todos los gráficos en la misma celda
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.legend(['precisión en entrenamiento', 'pérdida en entrenamiento'])
#%%

plt.plot(history.history['val_loss'])

#%%

plt.plot(history.history['vall_accuracy'])

#%%

test_directorio = '/home/nestor/Proyectos/Udemy Data Science/Tarea 5/4. Operations Department/Test'

#%%

test = ImageDataGenerator(rescale = 1./255)
test_gen = test.flow_from_directory(batch_size = 40,
                                    directory = test_directorio,
                                    shuffle = True,
                                    target_size = (256,256),
                                    class_mode='categorical'
                                    )
evaluar = modelo.evaluate_generator(test_gen,
                                   steps = test_gen.n//4,
                                   verbose = 1)

print('precisión: ', evaluar[1])

#Obtenemos un accuracy de 0.64, algo flojo

#%%

''' Voy a hacer 3 listas donde voy a guardar la imagen con la pred
y la clasificación real'''


image = list()
prediccion = list()
origin = list()

for i in range(len(os.listdir(test_directorio))): # Recorro los números de las carpetas clasificadoras
    for item in os.listdir(os.path.join(test_directorio, str(i))): # Recorro los nómbres de las imágenes por carpeta
        img = cv2.imread(os.path.join(test_directorio, str(i), item))
        img = cv2.resize(img, (256,256))
        image.append(img)
        img = img/255 # Para normalizar
        img = img.reshape(-1, 256,256,3)
        predic = modelo.predict(img)
        predic = np.argmax(predic)
        prediccion.append(predic)
        origin.append(i)
    
    
#%%
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

L = 8
W = 5

fig, axes = plt.subplots(nrows = L,
                         ncols = W,
                         figsize = (12,12))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(image[i])
    axes[i].set_title('pred ={}\norigin ={}'.format(str(etiquetas[prediccion[i]]),
                                                    str(etiquetas[origin[i]])))
    axes[i].axis('off')
  
plt.subplots_adjust(wspace = 1.2, hspace=1)
# Ahora pinto las imágenes con su predicción y la realidad


#%%    
    
# Printeamos el report

print(classification_report(np.asaray(origin).
                            np.asaray(prediccion)))    

#%%

# Matriz de confusión

confusion = confusion_matrix(origin, prediccion)
ax = plt.subplot()

print(sns.heatmap(confusion, annot = True, ax = ax))
ax.set_xlabel('prediccion')
ax.set_ylabel('original')
ax.set_title('Matriz de confusion')

''' En la matriz de confusión observamos que el podelo clasifica correctamente
los casos sanos y los que tienen covid, en cambio. los problemas víricos y
bacterianos no los identifica bien. De aquí podemos concluir que:
    - Convendría recopilar más muestra y aumentar las epochs de entrenamiento
    para mejorar el modelo.
    - Es un buen inicio para cambiar los parámetros de las capas densas.
    - Las enfermedades víricas(2) se confunde con los que tienen covid y 
    los problemas de pulmón bacteriano son difíciles de separar. Esto puede
    orientar los análisis teniendo en cuenta que pueda existir alguna similitud.'''   
    
    
    
    
    