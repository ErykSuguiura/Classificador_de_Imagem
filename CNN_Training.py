#!/usr/bin/python3

#======================================================================================#
#                                      IMPORTS
#======================================================================================#
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn import metrics
#======================================================================================#
#                                      FUNCOES
#======================================================================================#

#--------------------------------------------------------------------------------------#
def plot_samples(input_array, rows=4, cols=5, title=''):
    '''
    Funcao para imprimir as imagens de referencia para treinamento e teste da rede neural
    - input_array: matriz com as imagens a serem usadas
    - rows: numero de linhas de imagens a serem impressas
    - cols: numero de colunas de imagens a serem impressas
    - title: titulo da da impressao
    '''
    fig, ax = plt.subplots(figsize=(cols,rows))
    ax.axis('off')
    plt.title(title)

    for i in list(range(0, min(len(input_array),(rows*cols)) )):      
        a = fig.add_subplot(rows,cols,i+1)
        imgplot = plt.imshow(input_array[i,:784].reshape((28,28)), cmap='gray_r', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
    plt.show()
#--------------------------------------------------------------------------------------#

def cnn_model():
    '''
    Funcao que realiza a construcao da rede neural convolucional(CNN) com o Keras
    '''
    
    model = Sequential()
    model.add(Conv2D(30, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
#--------------------------------------------------------------------------------------#

def matriz_confusao(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """
    Gera a matrix de confusao a partir dos dados recebidos como parametro.

    - confusion_matrix: matriz retornada da funcao sklearn.metrics.confusion_matrix()
    
    - class_names: uma lista para determinar a ordem em que as classes serao representadas
                   na matrix confusao

    - figsize: um tuple de dois valores para definir o tamanho horizontal e vertical da 
               imagem de saida.

    - fontsize: altera o tamanho da fonte na imagem.
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
#--------------------------------------------------------------------------------------#

#======================================================================================#
#                                      CODIGO
#======================================================================================#

K.image_data_format()


# load the data
banana = np.load('data/banana.npy')
abacaxi = np.load('data/pineapple.npy')
pizza = np.load('data/pizza.npy')
hamburger = np.load('data/hamburger.npy')
uva = np.load('data/grapes.npy')
bolo = np.load('data/cake.npy')

print(banana.shape)
print(abacaxi.shape)
print(pizza.shape)
print(hamburger.shape)
print(uva.shape)
print(bolo.shape)
print("====================")

banana = np.c_[banana, np.zeros(len(banana))]
abacaxi = np.c_[abacaxi, np.ones(len(abacaxi))]
pizza = np.c_[pizza, 2*np.ones(len(pizza))]
hamburger = np.c_[hamburger, 3*np.ones(len(hamburger))]
uva = np.c_[uva, 4*np.ones(len(uva))]
bolo = np.c_[bolo, 5*np.ones(len(bolo))]



plot_samples(abacaxi, title='Sample bolo drawings\n')

#l = 1/0

# Merging arrays and splitting the features and labels
# 10000 pode ser aumentado
X = np.concatenate((banana[:10000,:-1], abacaxi[:10000,:-1], pizza[:10000,:-1], hamburger[:10000,:-1], uva[:10000,:-1], bolo[:10000, :-1]), axis=0).astype('float32') # all columns but the last
y = np.concatenate((banana[:10000,-1], abacaxi[:10000,-1], pizza[:10000,-1], hamburger[:10000,-1], uva[:10000,-1],  bolo[:10000,-1]), axis=0).astype('float32') # the last column

# We than split data between train and test (80 - 20 usual ratio). Normalizing the value between 0 and 1
X_train, X_test, y_train, y_test = train_test_split(X/255.,y,test_size=0.2,random_state=0)

# one hot encode outputs
y_train_cnn = np_utils.to_categorical(y_train)
y_test_cnn = np_utils.to_categorical(y_test)
num_classes = y_test_cnn.shape[1]

# reshape to be [samples][width][height][pixels]
X_train_cnn = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test_cnn = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')





np.random.seed(0)
# build the model
model_cnn = cnn_model()
# Fit the model
model_cnn.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=15, batch_size=200)
# Final evaluation of the model
scores = model_cnn.evaluate(X_test_cnn, y_test_cnn, verbose=0)
print('Final CNN accuracy: ', scores[1])



y_pred_cnn = model_cnn.predict(X_test_cnn, verbose=0)
classes_y=np.argmax(y_pred_cnn,axis=1)

c_matrix = metrics.confusion_matrix(y_test, classes_y)



class_names = ['banana', 'abacaxi', 'pizza', 'hamburger', 'uva', 'bolo']
matriz_confusao(c_matrix, class_names, figsize = (10,7), fontsize=14)


with open('model_cnn.pkl', 'wb') as file:
      pickle.dump(model_cnn, file)
