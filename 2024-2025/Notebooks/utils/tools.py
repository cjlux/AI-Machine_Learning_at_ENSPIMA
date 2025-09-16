# version 2.2 du 23 septembre 2022
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
    
def plot_loss_accuracy(history):
    '''Plot training & validation loss & accuracy values, giving an argument
       'history' of type 'tensorflow.python.keras.callbacks.History'. '''
    
    plt.figure(figsize=(15,5))
    ax1 = plt.subplot(1,2,1)
    if history.history.get('accuracy'):
        ax1.plot(np.array(history.epoch)+1, history.history['accuracy'], 'o-',label='Train')
    if history.history.get('val_accuracy'):
        ax1.plot(np.array(history.epoch)+1, history.history['val_accuracy'], 'o-', label='Test')
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch') 
    ax1.grid()
    ax1.legend(loc='best')
    
    # Plot training & validation loss values
    ax2 = plt.subplot(1,2,2)
    if history.history.get('loss'):
        ax2.plot(np.array(history.epoch)+1, history.history['loss'], 'o-', label='Train')
    if history.history.get('val_loss'):
        ax2.plot(np.array(history.epoch)+1, history.history['val_loss'], 'o-',  label='Test')
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='best')
    ax2.grid()
    plt.show()

def plot_images(image_array:np.ndarray, R:int, C:int, r:int=0, 
                figsize:tuple=None, reverse:bool=False):
    '''
    Plot the images from image_array on a R x C grid, starting at image rank r.
    Arguments:
       image_array: an array of images
       R:int: the number of rows
       C:int: the number of columns
       r:int: the starting rank in the array image_array (default: 0)
       figsize:tuple: the sise of the display (default: (C//2+1, R//2+1))
       reverse:bool: wether to reverse video the image or not (default: False)
    '''
    if figsize is None: figsize=(C//2+1, R//2+1)
    plt.figure(figsize=figsize)
    for i in range(R*C):
        plt.subplot(R, C, i+1)
        im = image_array[r+i]
        if reverse: im = 255 - im
        plt.imshow(im, cmap='gray')
        plt.axis('off');


def show_cm(true, results, classes):
    ''' true  : the actual labels 
        results : the labels computed by the trained network (one-hot format)
        classes : list of possible label values'''
    predicted = np.argmax(results, axis=-1) # tableau d'entiers entre 0 et 9 
    cm = confusion_matrix(true, predicted)
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(11,9))
    heatmap(df_cm, annot=True, cbar=False, fmt="3d")
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()
    
def scan_dir(path):
    tree = ''
    data = [item for item in os.walk(path)]
    for item in data:
        if item[2]:
            for file in item[2]:
                tree += f'{item[0]}/{file}\n'
        else:
            tree += f'{item[0]}/\n'
    return tree
        
