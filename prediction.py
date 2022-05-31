# Starter code for CS 165B HW4
from email.mime import image
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import os


data = []
label = []

for filename in os.listdir('hw4_train/0'):
    f = os.path.join('hw4_train/0',filename)

    if os.path.isfile(f):
        temp=Image.open(f)
        test = temp.getdata()
        img = []
        for p in test:
            img.append(p)
        label.append(0)
        data.append(img)

for filename in os.listdir('hw4_train/1'):
    f = os.path.join('hw4_train/1',filename)

    if os.path.isfile(f):
        temp=Image.open(f)
        test = temp.getdata()
        img = []
        for p in test:
            img.append(p)
        label.append(1)
        data.append(img)

for filename in os.listdir('hw4_train/2'):
    f = os.path.join('hw4_train/2',filename)

    if os.path.isfile(f):
        temp=Image.open(f)
        test = temp.getdata()
        img = []
        for p in test:
            img.append(p)
        label.append(2)
        data.append(img)

for filename in os.listdir('hw4_train/3'):
    f = os.path.join('hw4_train/3',filename)

    if os.path.isfile(f):
        temp=Image.open(f)
        test = temp.getdata()
        img = []
        for p in test:
            img.append(p)
        label.append(3)
        data.append(img)

for filename in os.listdir('hw4_train/4'):
    f = os.path.join('hw4_train/4',filename)

    if os.path.isfile(f):
        temp=Image.open(f)
        test = temp.getdata()
        img = []
        for p in test:
            img.append(p)
        label.append(4)
        data.append(img)

for filename in os.listdir('hw4_train/5'):
    f = os.path.join('hw4_train/5',filename)

    if os.path.isfile(f):
        temp=Image.open(f)
        test = temp.getdata()
        img = []
        for p in test:
            img.append(p)
        label.append(5)
        data.append(img)

for filename in os.listdir('hw4_train/6'):
    f = os.path.join('hw4_train/6',filename)

    if os.path.isfile(f):
        temp=Image.open(f)
        test = temp.getdata()
        img = []
        for p in test:
            img.append(p)
        label.append(6)
        data.append(img)

for filename in os.listdir('hw4_train/7'):
    f = os.path.join('hw4_train/7',filename)

    if os.path.isfile(f):
        temp=Image.open(f)
        test = temp.getdata()
        img = []
        for p in test:
            img.append(p)
        label.append(7)
        data.append(img)

for filename in os.listdir('hw4_train/8'):
    f = os.path.join('hw4_train/8',filename)

    if os.path.isfile(f):
        temp=Image.open(f)
        test = temp.getdata()
        img = []
        for p in test:
            img.append(p)
        label.append(8)
        data.append(img)

for filename in os.listdir('hw4_train/9'):
    f = os.path.join('hw4_train/9',filename)

    if os.path.isfile(f):
        temp=Image.open(f)
        test = temp.getdata()
        img = []
        for p in test:
            img.append(p)
        label.append(9)
        data.append(img)

data = np.array(data)
label = np.array(label)
#df = pd.DataFrame(arr)
#features = list(df.columns[:784])
#y = df[784]
#X = df[features]
print(tf.__version__)
data = data/255.0
print(data.shape)
print(len(label))
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(data,label,epochs=10)

test_data = []

for filename in os.listdir('hw4_test'):
    #NEED TO MAKE SURE IMAGES ARE ADDED TO TEST_DATA IN THE RIGHT ORDER
    f = os.path.join('hw4_test',filename)

    if os.path.isfile(f):
        if(f=="hw4_test/.DS_Store"):
            continue
        print(f)
        temp=Image.open(f)
        test = temp.getdata()
        img = []
        for p in test:
            img.append(p)
        test_data.append(img)

test_data = np.array(test_data)

test_data=test_data/255.0


prediction_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
                    
predictions = prediction_model.predict(data)
f = open('prediction.txt','w')
for i in range(len(predictions)):
    if(i==9999):
        f.write(str(np.argmax(predictions[i])))
    else:
        f.write(str(np.argmax(predictions[i])))
        f.write("\n")

f.close()






"""
Implement the testing procedure here. 

Inputs:
    Unzip the hw4_test.zip and place the folder named "hw4_test" in the same directory of your "prediction.py" file, your "prediction.py" need to give the following required output.

Outputs:
    A file named "prediction.txt":
        * The prediction file must have 10000 lines because the testing dataset has 10000 testing images.
        * Each line is an integer prediction label (0 - 9) for the corresponding testing image.
        * The prediction results must follow the same order of the names of testing images (0.png – 9999.png).
    Notes: 
        1. The teaching staff will run your "prediction.py" to obtain your "prediction.txt" after the competition ends.
        2. The output "prediction.txt" must be the same as the final version you submitted to the CodaLab, 
        otherwise you will be given 0 score for your hw4.


**!!!!!!!!!!Important Notes!!!!!!!!!!**
    To open the folder "hw4_test" or load other related files, 
    please use open('./necessary.file') instead of open('some/randomly/local/directory/necessary.file').

    For instance, in the student Jupyter's local computer, he stores the source code like:
    - /Jupyter/Desktop/cs165B/hw4/prediction.py
    - /Jupyter/Desktop/cs165B/hw4/hw4_test
    If he/she use os.chdir('/Jupyter/Desktop/cs165B/hw4/hw4_test'), this will cause an IO error 
    when the teaching staff run his code under other system environments.
    Instead, he should use os.chdir('./hw4_test').


    If you use your local directory, your code will report an IO error when the teaching staff run your code,
    which will cause 0 score for your hw4.
"""
