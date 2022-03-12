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


with open(f'model_cnn.pkl', 'rb') as f:
                model_cnn = pickle.load(f)

# Carregando 'x_test' e ' y_test'
X_test = np.load('./static/x_test.npy')
y_test = np.load('./static/y_test.npy')



X_test_cnn = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
y_test_cnn = np_utils.to_categorical(y_test)


scores = model_cnn.evaluate(X_test_cnn, y_test_cnn, verbose=0)
print('Precisao CNN: ', scores[1])

y_pred_cnn = model_cnn.predict(X_test_cnn, verbose=0)
classes_y=np.argmax(y_pred_cnn,axis=1)

# matriz confusao
c_matrix = metrics.confusion_matrix(y_test, classes_y)

class_names = ['banana', 'abacaxi', 'pizza', 'hamburger', 'uva', 'bolo']
matriz_confusao(c_matrix, class_names, figsize = (10,7), fontsize=14) # gerar a matriz de confusao[em outro codigo]



                