# Artificial Neural Network
# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.layers import Dropout
import pickle
# Importing the dataset
dataset = pd.read_csv('C:/Users/KPATNAIk/Desktop/data_science/ey.csv')
dataset = dataset.drop(["Key-Badge_SD","Badge earned","Initiate a badge date","Time Lapse"],axis = 1)
dataset["Key_Badge_SD"] = dataset["Domain"]+"_"+dataset["Sub Domain"]+"_"+dataset["Badge Type"]
dataset = dataset.drop(["EmployeeStatusDesc","Domain","Sub Domain","Badge Type","Badge Status","Badge Classification"],axis = 1)

X = dataset.iloc[:, 1:4]
y = dataset.iloc[:, 4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


#Create dummy variables
Country=pd.get_dummies(X['Country'])
Department=pd.get_dummies(X['Department'])
Designation=pd.get_dummies(X['Designation'])

## Concatenate the Data Frames

X=pd.concat([X,Country,Designation,Department],axis=1)

## Drop Unnecessary columns
X=X.drop(['Country','Designation','Department'],axis=1)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 96, init = 'he_uniform',activation='relu',input_dim = 48))
classifier.add(Dropout(p =0.3))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 20, init = 'he_uniform',activation='relu'))
classifier.add(Dropout(p =0.3))
# Adding the output layer
classifier.add(Dense(output_dim = 91, init = 'glorot_uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 1000, nb_epoch = 400)

print(model_history.history.keys())

with open('C:/Users/KPATNAIk/Desktop/data_science/nn_classifier.pkl', 'wb') as file:
    pickle.dump(classifier, file)
with open('C:/Users/KPATNAIk/Desktop/data_science/scaler.pkl', 'wb') as file:
    pickle.dump(sc, file)
with open('C:/Users/KPATNAIk/Desktop/data_science/encoder.pkl', 'wb') as file:
    pickle.dump(encoder, file)

k_nul = X.iloc[0:0]
k_nul.to_pickle('C:/Users/KPATNAIk/Desktop/data_science/saved_data.pkl')

root_data = pd.read_csv('C:/Users/KPATNAIk/Desktop/data_science/ey.csv')
root_data.to_pickle('C:/Users/KPATNAIk/Desktop/data_science/root_data.pkl')


#############Prediction##########################
print(model_history.history.keys())
# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
     



    


