import csv
import h5py
import numpy as np
import keras
from   keras.models import Sequential
from   keras.layers import Dense,LSTM,Dropout
from   keras        import utils

poem   = ''

with open ('frost.csv') as f:
   rows = csv.reader(f,delimiter = ',')  
   for eachrow in rows:
       if eachrow: 
          val  = eachrow[0]
          poem = poem+val+'\n'
            
        
# storing corpus charecter level
corpus  =  list(poem.lower())
chars   =  list(set(poem.lower()))

#printing lengths
print("The corpus length is: ",len(corpus),'\nNumber of unique characters: ',len(chars))

#creating a dictionary for the character to num
char2num = dict((c,i) for i,c in enumerate(chars))
num2char = dict((i,c) for i,c in enumerate(chars))

#creating a seed.
X_data , y_data = [],[]
sequence_len = 118

#prep of data.
for iter in range(0,len(corpus)-sequence_len):
      tempX = corpus[iter:iter+sequence_len]
      tempy = corpus[iter+sequence_len]
      X_data.append(tempX)
      y_data.append(tempy)

for iter in range(len(X_data)):
      for sub in range(len(X_data[iter])):
               X_data[iter][sub] = char2num[X_data[iter][sub]]

for iter in range(len(y_data)):
      y_data[iter]= char2num[y_data[iter]]



X = np.array([X_data])
y = np.array([y_data])

#preprocissing the data
X = X.reshape(len(X_data),sequence_len,1)
y = y.reshape(len(y_data),1)

X = keras.utils.to_categorical(X,len(chars))
y = keras.utils.to_categorical(y,len(chars))

#creating the net

model = Sequential()
model.add(LSTM(512,input_shape =(sequence_len,len(chars)),return_sequences = True))
model.add(Dropout(0.5))
model.add(LSTM(512))
model.add(Dropout(0.5))
model.add(Dense(len(chars),activation='softmax'))
model.compile(optimizer ='adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(X,y,epochs= 80,batch_size = 500)
score_train = model.evaluate(X,y)
print('The accuracy:', score_train[1]*100)

#save model
model.save('testpoem.h5')

# testing the sequence
print('Testing......')
sentence= '''And there's a barrel that I didn't fill
Beside it, and there may be two or three
Apples I didn't pick upon some bough.
But I am done with apple-picking now.
Essence of winter sleep is on the night,'''

print('Sentence fed: ',sentence[0:sequence_len].lower())
test    = list(sentence[0:sequence_len].lower())
tempX   = []
for iter in range(sequence_len):
       tempX.append(char2num[test[iter]])
tempx = list(tempX)       
X_test = np.array([tempx])
#to categorical
X_test = keras.utils.to_categorical(X_test,len(chars))
add_seq = 120
str = sentence[0:sequence_len].lower()

def pred(X_test):
      out  = model.predict(X_test)
      return np.argmax(out[0])

for iter in range(add_seq):
      out = pred(X_test)
      str = str+num2char[out]
      test= list(str.lower())
      tempX = []
      for j in range(iter+1,sequence_len+iter+1):
            tempX.append(char2num[test[j]])
      tempx = list(tempX)       
      X_test = np.array([tempx])
     #to categorical
      X_test = keras.utils.to_categorical(X_test,len(chars))
print('Resulting sentence:\n')
print(str)
