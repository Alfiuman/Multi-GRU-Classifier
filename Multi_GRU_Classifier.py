

import numpy as np
np.random.seed(0)

import pandas as pd
from keras.layers import Dense, Dropout, GRU
from keras.models import Sequential
from keras import regularizers
from matplotlib import pyplot


#Loading the data
dataset_x_train = pd.read_csv('xValTrain.csv', header=None)
dataset_x_valid = pd.read_csv('xValValid.csv', header=None)
dataset_x_test = pd.read_csv('xValTest.csv', header=None)
dataset_y_train = pd.read_csv('yValTrain.csv', header=None)
dataset_y_valid = pd.read_csv('yValValid.csv', header=None)
dataset_y_test  = pd.read_csv('yValTest.csv', header=None)

#Preparing the data
values_x_train = dataset_x_train.values
train_x = values_x_train.astype('float32')

values_x_valid = dataset_x_valid.values
valid_x = values_x_valid.astype('float32')

values_x_test = dataset_x_test.values
test_x = values_x_test.astype('float32')

values_y_train = dataset_y_train.values
train_y = values_y_train.astype('float32')

values_y_valid = dataset_y_valid.values
valid_y = values_y_valid.astype('float32')

values_y_test = dataset_y_test.values
test_y = values_y_test.astype('float32')

train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
valid_x = valid_x.reshape((valid_x.shape[0], 1, valid_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))


#Sequential GRU model BEGIN
dropout_perc = 0.2
l2_reg = 0.01

model = Sequential()
model.add(Dropout(dropout_perc, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(GRU(100, return_sequences=True))
model.add(Dropout(dropout_perc))
model.add(GRU(100, return_sequences=True))
model.add(Dropout(dropout_perc))
model.add(GRU(100))
model.add(Dropout(dropout_perc))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(dropout_perc))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(l2_reg)))
model.compile(loss='mse', optimizer='adam')
#Sequential GRU model END

#Training the model
trained_model = model.fit(train_x, train_y, epochs=100, batch_size=72, validation_data=(valid_x, valid_y), verbose=2, shuffle=False)

#Plotting the training and the validation
pyplot.plot(trained_model.history['loss'], label='train')
pyplot.plot(trained_model.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

#Making prediction on test data
y_pred = model.predict(test_x)
y_predicted = [1.0 if y > 0.5 else 0.0 for y in y_pred]

ones_correct = 0
ones_wrong = 0
zeros_correct = 0
zeros_wrong = 0

for k in range(len(y_predicted)):
    if y_predicted[k] == test_y[k] and y_predicted[k] == 1:
        ones_correct += 1
    elif y_predicted[k] == test_y[k] and y_predicted[k] == 0:
        zeros_correct += 1
    elif y_predicted[k] != test_y[k] and test_y[k] == 1:
        ones_wrong += 1
    elif y_predicted[k] != test_y[k] and test_y[k] == 0:
        zeros_wrong += 1

#Showing the results of the test
print('Correct Ones are ', ones_correct, "   ", np.round(ones_correct / (ones_correct + ones_wrong), 5), " perc.")
print('Correct Zeros are ', zeros_correct, "   ", np.round(zeros_correct / (zeros_correct + zeros_wrong), 5), " perc.")
print('Wrong Ones are ', ones_wrong)
print('Wrong Zeros are ', zeros_wrong)




















