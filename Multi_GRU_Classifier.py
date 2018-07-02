

import numpy as np
np.random.seed(0)

import pandas as pd
from keras.layers import Dense, Dropout, GRU
from keras.models import Sequential
from keras import regularizers
from matplotlib import pyplot


#Loading the data
dataset_X_Train = pd.read_csv('xValTrain.csv', header=None)
dataset_X_Valid = pd.read_csv('xValValid.csv', header=None)
dataset_X_Test = pd.read_csv('xValTest.csv', header=None)
dataset_Y_Train = pd.read_csv('yValTrain.csv', header=None)
dataset_Y_Valid = pd.read_csv('yValValid.csv', header=None)
dataset_Y_Test  = pd.read_csv('yValTest.csv', header=None)

#Preparing the data
values_X_Train = dataset_X_Train.values
train_X = values_X_Train.astype('float32')

values_X_Valid = dataset_X_Valid.values
valid_X = values_X_Valid.astype('float32')

values_X_Test = dataset_X_Test.values
test_X = values_X_Test.astype('float32')

values_Y_Train = dataset_Y_Train.values
train_Y = values_Y_Train.astype('float32')

values_Y_Valid = dataset_Y_Valid.values
valid_Y = values_Y_Valid.astype('float32')

values_YTest = dataset_Y_Test.values
test_Y = values_YTest.astype('float32')

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
valid_X = valid_X.reshape((valid_X.shape[0], 1, valid_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


#Sequential GRU model BEGIN
dropout_Perc = 0.2
l2_Reg = 0.01

model = Sequential()
model.add(Dropout(dropout_Perc, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(GRU(100, return_sequences=True))
model.add(Dropout(dropout_Perc))
model.add(GRU(100, return_sequences=True))
model.add(Dropout(dropout_Perc))
model.add(GRU(100))
model.add(Dropout(dropout_Perc))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(dropout_Perc))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(l2_Reg)))
model.compile(loss='mse', optimizer='adam')
#Sequential GRU model END

#Training the model
trained_Model = model.fit(train_X, train_Y, epochs=100, batch_size=72, validation_data=(valid_X, valid_Y), verbose=2, shuffle=False)

#Plotting the training and the validation
pyplot.plot(trained_Model.history['loss'], label='train')
pyplot.plot(trained_Model.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

#Making prediction on test data
y_Pred = model.predict(test_X)
y_Predicted = [1.0 if y > 0.5 else 0.0 for y in y_Pred]

ones_Correct = 0
ones_Wrong = 0
zeros_Correct = 0
zeros_Wrong = 0

for k in range(len(y_Predicted)):
    if y_Predicted[k] == test_Y[k] and y_Predicted[k] == 1:
        ones_Correct += 1
    elif y_Predicted[k] == test_Y[k] and y_Predicted[k] == 0:
        zeros_Correct += 1
    elif y_Predicted[k] != test_Y[k] and test_Y[k] == 1:
        ones_Wrong += 1
    elif y_Predicted[k] != test_Y[k] and test_Y[k] == 0:
        zeros_Wrong += 1

#Showing the results of the test
print('Correct Ones are ', ones_Correct, "   ", np.round(ones_Correct / (ones_Correct + ones_Wrong), 5), " perc.")
print('Correct Zeros are ', zeros_Correct, "   ", np.round(zeros_Correct / (zeros_Correct + zeros_Wrong), 5), " perc.")
print('Wrong Ones are ', ones_Wrong)
print('Wrong Zeros are ', zeros_Wrong)




















