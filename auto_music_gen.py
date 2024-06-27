from music21 import *
import glob
from tqdm import tqdm
import numpy as np
import random
# LSTM,Dense,Input,Dropout
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Sequential,Model,load_model
from sklearn.model_selection import train_test_split


def read_files(file):
    notes=[]
    notes_to_parse=None
    #parse the midi file
    midi=converter.parse(file)
    #seperate all instruments from the file
    instrmt=instrument.partitionByInstrument(midi)

    for part in instrmt.parts:
        #fetch data only of piano
        if 'Piano' in str(part):
            #recurse is used 
            notes_to_parse=part.recurse()

            #iterate over all the parts of sub stream elements
            #check if elements's type is Note or Chord
            #if it is chord then split them into notes
            for element in notes_to_parse:
                if type(element)==note.Note:
                    notes.append(str(element.pitch))
                elif type(element) == chord.Chord:
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    #returning the notes
    return notes

#retrieve paths recursively from inside the directories/files
file_path=["/Users/aryansheel/Desktop/python/nn_music/schubert"]
#glob here is finding all the files based on the given pattern
all_files=glob.glob(file_path[0] + '/*.mid',recursive=True)


# Ensure that MIDI files are found
# if not all_files:
#     raise FileNotFoundError("No MIDI files found in the specified directory.")


#reading each midi file
notes_array = np.array([read_files(i) for i in tqdm(all_files, position=0, leave=True)], dtype=object)

#unique notes
notess=sum(notes_array,[]) # this flattens the numpy array into a single list
unique_notes=list(set(notess))
print("Unique notes = ",len(unique_notes))

#notes with their frequency
freq=dict(map(lambda x: (x,notess.count(x)),unique_notes))

#get the threshold frequency
for i in range(30,100,20):
    print(i,":",len(list(filter(lambda x:x[1]>=i,freq.items()))))

#we are taking the nodes which have freq>50
freq_notes=dict(filter(lambda x:x[1]>=50,freq.items()))

#create new nodes using the frequency notes
new_notes = []
for j in notes_array:
    filtered_sublist = []
    for i in j:
        if i in freq_notes:
            filtered_sublist.append(i)
    new_notes.append(filtered_sublist)

#dictionary having key as note index and value as note
ind2note=dict(enumerate(freq_notes))

#dictionary having key as note and value as note index
note2ind=dict(map(reversed,ind2note.items()))

#timesteps
# Now we will create input and output sequences for our model. We will be using a timestep of 50. So if we traverse 50 notes of our input sequence then the 51th note will be the output for that sequence. Let’s take an example to see how it works.
timesteps=50

#for storing the values of input and output
x=[] 
y=[]

for i in new_notes:
    for j in range(0,len(i)-timesteps):
        #input will be the current index + timestep
        #output will be the next index after timestep
        inp=i[j:j+timesteps]
        out=i[j+timesteps]

        #append the index value of respective notes
        x.append(list(map(lambda x:note2ind[x],inp)))
        y.append(note2ind[out])

x_new=np.array(x)
y_new=np.array(y)


## TRAINING AND TESTING

#reshape input and output for the model
x_new = np.reshape(x_new,(len(x_new),timesteps,1))
y_new = np.reshape(y_new,(-1,1))

#split the input and value into training and testing sets
#80% for training and 20% for testing sets
x_train,x_test,y_train,y_test = train_test_split(x_new,y_new,test_size=0.2,random_state=42)


##CREATING THE MODEL
model=Sequential()

#creating two stacked LSTM layer with the latent dimension of 256
model.add(LSTMV1(256,return_sequences=True,input_shape=(x_new.shape[1],x_new.shape[2])))
model.add(Dropout(0.2))
model.add(LSTMV1(256))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
#fully connected layer for the output with softmax activation
model.add(Dense(len(note2ind),activation='softmax'))
model.summary()


## TRAINING THE MODEL
#compile the model using Adam optimizer
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

#train the model on training sets and validate on testing sets
model.fit(x_train,y_train,batch_size=128,epochs=80,validation_data=(x_test,y_test))

# model.save("s2s")

