# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:20:49 2017

@author: Madhu
"""
from loadPackages import *

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Function to load the data
def loadData():
    try:
             # Lets load our data. We will limit the number of words to 5,000 as that is how the data is setup.
             (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

             print("_"*100)
             print("train_data ", x_train.shape)
             print("train_labels ", y_train.shape)
             print("_"*100)
             print("test_data ", x_test.shape)
             print("test_labels ", y_test.shape)
             print("_"*100)
             print("Maximum value of a word index ")
             print(max([max(sequence) for sequence in x_train]))
             print("Maximum length num words of review in train ")
             print(max([len(sequence) for sequence in x_train]))
             print("_"*100)
             # See an actual review in words
             # Reverse from integers to words using the DICTIONARY (given by keras...need to do nothing to create it)

             word_index = imdb.get_word_index()

             reverse_word_index = dict(
                 [(value, key) for (key, value) in word_index.items()])

             decoded_review = ' '.join(
                 [reverse_word_index.get(i - 0, '?') for i in x_train[1]])

             #print(x_train[1])
             print(decoded_review)
             print("_"*100)
             return x_train, y_train, x_test, y_test
            
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)

# We pad the data because not all sentences in our data are the same length. We want to use a number that is larger than our largest data. Here I will choose 400.           
def padInput(x_train,x_test):
    try:        
        x_train = sequence.pad_sequences(x_train, maxlen=400)
        x_test = sequence.pad_sequences(x_test, maxlen=400)
        return x_train, x_test
    
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)        
           
def cnn1D(x_train,x_test, y_train, y_test):
    try:        
        # Lets start with a very simple 1D CNN model. We will use this as our baseline for everything else in this lab.
        model = Sequential()

        # This embedding is a trainable parameter. We aren't using GloVE for this model.
        model.add(Embedding(5000,50,input_length=400))
        model.add(Dropout(0.2))

        # There isn't much of a difference with how 1D and 2D CNNs work. They still use filters and scan the data.
        # we will use a similar model as our 2D CNN with the adition of an embedding layer at the beginning.
        model.add(Conv1D(64,3,padding='valid',activation='relu',strides=1))
        model.add(BatchNormalization())
        model.add(Conv1D(64,3,padding='valid',activation='relu',strides=1))
        model.add(BatchNormalization())
        model.add(GlobalMaxPooling1D())

        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))

        # We will use a sigmoid and a 1 neuron dense output since our data is binary.
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                          optimizer='Nadam',
                          metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                            batch_size=128,
                            epochs=2,
                            validation_data=(x_test, y_test))

        plt.clf()
        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, (len(history_dict['loss']) + 1))
        plt.plot(epochs, loss_values, 'bo', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)   
           
def cnn11D(x_train,x_test, y_train, y_test):
    try:        
        # Lets start with a very simple 1D CNN model. We will use this as our baseline for everything else in this lab.
        model = Sequential()

        # This embedding is a trainable parameter. We aren't using GloVE for this model.
        model.add(keras.layers.Embedding(1000,50,input_length=400))
        model.add(keras.layers.Dropout(0.2))

        # There isn't much of a difference with how 1D and 2D CNNs work. They still use filters and scan the data.
        # we will use a similar model as our 2D CNN with the adition of an embedding layer at the beginning.
        model.add(keras.layers.Conv1D(64,3,padding='valid',activation='relu',strides=1))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv1D(64,3,padding='valid',activation='relu',strides=1))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.GlobalMaxPooling1D())

        model.add(keras.layers.Dense(512))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Activation('relu'))

        # We will use a sigmoid and a 1 neuron dense output since our data is binary.
        model.add(keras.layers.Dense(1))
        model.add(keras.layers.Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                          optimizer='Nadam',
                          metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                            batch_size=128,
                            epochs=2,
                            validation_data=(x_test, y_test))

        plt.clf()
        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, (len(history_dict['loss']) + 1))
        plt.plot(epochs, loss_values, 'bo', label='Training loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    except Exception as ex:
           print ("-----------------------------------------------------------------------")
           template = "An exception of type {0} occurred. Arguments:\n{1!r}"
           message = template.format(type(ex).__name__, ex.args)
           print (message)            