import tensorflow.keras as tfk

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import math

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Model():
    def __init__(self):
        self.model = tfk.Sequential()
    
    def load_model(self, filename):
        self.model = tfk.models.load_model(filename)
    
    def create_model(self, input_shape):
        #Add layers
        #LSTM Layer
        self.model.add(tfk.layers.LSTM(100, input_shape=input_shape, return_sequences=True))
        #Dropout Layer
        self.model.add(tfk.layers.Dropout(0.2)) #dropout_rate = 0.2
        #LSTM Layer
        self.model.add(tfk.layers.LSTM(100, return_sequences=True))
        #LSTM Layer
        self.model.add(tfk.layers.LSTM(100, return_sequences=False))
        #Dropout Layer
        self.model.add(tfk.layers.Dropout(0.2)) #dropout_rate = 0.2
        #Dense Layer
        self.model.add(tfk.layers.Dense(1, activation='linear'))

        #compile the model
        self.model.compile(loss='mse', optimizer='adam')
        print('<Model> Compiled')

    
    def train(self, train_gen, batch_size, epochs, steps_per_epoch, save_dir):
        print('<Model> Training Started')
        print('<Model> %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
        save_file = os.path.join(save_dir, '%s-e%s.h5' % (datetime.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        
        #callbacks
        callbacks = [
            tfk.callbacks.ModelCheckpoint(filepath=save_file, monitor='loss', save_best_only=True)
        ]
        
        #fit model (fit now supports generators)
        self.model.fit(
            x=train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1
        )

        print('<Model> Training Completed. Model saved as %s' % save_file)

    def predict_sequence_full(self, data, window_size):
        #Shift the window by 1 new prediction each time, re-run predictions on new window
        print('<Model> Predicting Full Sequence...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            #use newaxis to add another dimension: (49,2) -> (1,49,2) this tells it that it is a "single" data point
            predicted.append(self.model.predict(curr_frame[np.newaxis,:,:])[0,0])
            #remove first val in frame
            curr_frame = curr_frame[1:]
            #insert the most recently predicted point to the end of the current frame
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        return predicted
    
    def predict_sequences_multiple(self, data, window_size, prediction_len):
        #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)+1):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[np.newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs



class Data():

    def __init__(self, filename, train_test_split, columns):
        #get data from stock file
        df = pd.read_csv(filename)
        split_idx = int(len(df)*train_test_split) #split index for train/test split

        self.data_train = df[:split_idx][columns].values
        self.data_test = df[split_idx:][columns].values
        self.train_len = len(self.data_train)

    #get generator of training data
    def get_train_batch_generator(self, seq_size, batch_size):
        i = 0 #seq_size
        #len(self.data_train)
        while i < (self.train_len - seq_size):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                # stop-condition for final batch if batch_size doesn't divide evenly into length of data_train
                #len(self.data_train)
                if i >= (self.train_len - seq_size):
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                #next window
                window = self.data_train[i:i+seq_size] #self.data_train[i-seq_size:i]
                #normalize
                norm_window = []
                for col_idx in range(window.shape[1]):
                    mn = np.mean(window[:,col_idx], axis=0)
                    std = np.std(window[:,col_idx], axis=0)
                    norm_col = (window[:,col_idx] - mn)/std
                    norm_window.append(norm_col)
                #Transpose normalized window back to correct shape
                norm_window = np.array(norm_window).T
                
                x = norm_window[:-1]
                y = norm_window[-1, [0]] #Assuming index 0 is the closing price
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    #get test x and y
    def get_test_data(self, seq_size):
        test_x = []
        test_y = []
        #create a window for each price from seq_size to length of data_train
        for i in range(seq_size, len(self.data_test)):
            #next window
            window = self.data_test[i-seq_size:i]
            #normalize
            norm_window = []
            for col_idx in range(window.shape[1]):
                mn = np.mean(window[:,col_idx], axis=0)
                std = np.std(window[:,col_idx], axis=0)
                norm_col = (window[:,col_idx] - mn)/std
                norm_window.append(norm_col)
            #Transpose normalized window back to correct shape
            norm_window = np.array(norm_window).T
            x = norm_window[:-1]
            y = norm_window[-1, [0]] #Assuming index 0 is the closing price
            test_x.append(x)
            test_y.append(y)
        #convert to numpy arrays and return x,y
        return np.array(test_x), np.array(test_y)




def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        # plt.legend()
    plt.show()

def main():

    #Variables for neural net
    sequence_size = 50
    batch_size = 32
    epochs = 2
    cols = ['Close', 'Volume']
    train_test_split = 0.85

    input_shape = (sequence_size-1, len(cols))
    save_dir = 'saved_models'

    data = Data("SPY.csv", train_test_split, cols)

    model = Model()
    load_filename = ''
    if load_filename:
        model.load_model(load_filename)
    else:
        #Create model
        model.create_model(input_shape)

        steps_per_epoch = math.ceil((len(data.data_train) - sequence_size) / batch_size)

        #Train model
        model.train(
            train_gen=data.get_train_batch_generator(
                seq_size=sequence_size,
                batch_size=batch_size
            ),
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            save_dir=save_dir
        )


    #Test data
    x_test, y_test = data.get_test_data(sequence_size)

    # predicted = model.predict_sequence_full(x_test, sequence_size)
    # plot_results(predicted, y_test)

    predicted = model.predict_sequences_multiple(x_test, sequence_size, sequence_size)
    plot_results_multiple(predicted, y_test, sequence_size)


if __name__ == "__main__":
    main()