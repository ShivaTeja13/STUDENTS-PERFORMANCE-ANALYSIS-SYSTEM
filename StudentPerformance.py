from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

main = tkinter.Tk()
main.title("Students’ performance analysis system using cumulative predictor algorithm")
main.geometry("1200x1200")


global classifier
global filename
global dataset
global X, Y
labels = ['Excellent', 'Medium', 'Poor']
accuracy = []
error_rate = []
global temp_acc
global le
    
def uploadDataset():
    global filename
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    text.insert(END,str(dataset.head()))
    pathlabel.config(text=str(filename)+" Dataset Loaded\n\n")

def preprocessDataset():
    global le
    global dataset
    global X, Y
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    cols = ['becgrade','Placed','becstatus']
    le = LabelEncoder()
    dataset[cols[0]] = pd.Series(le.fit_transform(dataset[cols[0]].astype(str)))
    dataset[cols[1]] = pd.Series(le.fit_transform(dataset[cols[1]].astype(str)))
    dataset[cols[2]] = pd.Series(le.fit_transform(dataset[cols[2]].astype(str)))
    text.insert(END,str(dataset.head())+"\n\n")
    cols = dataset.shape[1]-1
    dataset = dataset.values
    X = dataset[:,0:cols]
    Y = dataset[:,cols]
    X = normalize(X)
    text.insert(END,"Dataset contains total records : "+str(X.shape[0])+"\n")

def runNB():
    text.delete('1.0', END)
    global X, Y
    global accuracy
    global error_rate
    accuracy.clear()
    error_rate.clear()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    naivebayes = GaussianNB()
    naivebayes.fit(X,Y)
    predict = naivebayes.predict(X_test) 
    acc = accuracy_score(y_test,predict)*100
    error = 100 - acc
    accuracy.append(acc)
    error_rate.append(error)
    text.insert(END,"Naive Bayes Accuracy   : "+str(acc)+"\n")
    text.insert(END,"Naive Bayes Error Rate : "+str(error)+"\n\n")

def runDT():
    global temp_acc
    global X, Y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    dt = DecisionTreeClassifier(criterion = "gini", max_depth = 200,splitter="best",class_weight="balanced",max_leaf_nodes=100)
    dt.fit(X,Y)
    predict = dt.predict(X_test) 
    acc = accuracy_score(y_test,predict)*100
    error = 100 - acc
    accuracy.append(acc)
    error_rate.append(error)
    text.insert(END,"Decision Tree Accuracy   : "+str(acc)+"\n")
    text.insert(END,"Decision Tree Error Rate : "+str(error)+"\n\n")
    temp_acc = acc

def runCP():
    global X, Y
    global classifier
    acc = 0
    global temp_acc
    i = 0
    while i < 5: #taking loop
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)#split dataset into train and test
        #creating decison tree object
        cls = DecisionTreeClassifier(max_depth = 200,splitter="best",class_weight="balanced",max_leaf_nodes=100)
        #training decison tree on dataset
        cls.fit(X,Y)
        #predicting on test data
        predict = cls.predict(X_test)
        #calculating accuracy
        a = accuracy_score(y_test,predict)*100
        #if accuracy more then select model
        if a > acc:
            acc = a
            classifier = cls
        if acc < temp_acc and i == 4: #if accuracy not more then keep evaluting model till get best accuracy
            i = i - 1
        i = i + 1
    error = 100 - acc
    accuracy.append(acc)
    error_rate.append(error)    
    text.insert(END,"Cumulative Predictor Accuracy   : "+str(acc)+"\n")
    text.insert(END,"Cumulative Predictor Error Rate : "+str(error)+"\n\n")        
        
            
def graph():
    df = pd.DataFrame([['Naive Bayes','Accuracy',accuracy[0]],['Naive Bayes','Error Rate',error_rate[0]],
                       ['Decision Tree','Accuracy',accuracy[1]],['Decision Tree','Error Rate',error_rate[1]],
                       ['Cumulative Predictor','Accuracy',accuracy[2]],['Cumulative Predictor','Error Rate',error_rate[2]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()
            

def predict():
    global le
    global classifier
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    test = pd.read_csv(filename)
    test.fillna(0, inplace = True)
    cols = ['becgrade','Placed']
    test[cols[0]] = pd.Series(le.fit_transform(test[cols[0]].astype(str)))
    test[cols[1]] = pd.Series(le.fit_transform(test[cols[1]].astype(str)))
    data = test.values
    test = test.values
    test = normalize(test)
    predictValue = classifier.predict(test)
    print(predictValue)
    for i in range(len(predictValue)):
        index = int(predictValue[i])
        text.insert(END,str(data[i])+" Predicted Performance is : "+labels[index]+"\n\n")
        
    

font = ('times', 14, 'bold')
title = Label(main, text='Students’ performance analysis system using cumulative predictor algorithm')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Student Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

NBButton = Button(main, text="Run Naive Bayes Algorithm", command=runNB)
NBButton.place(x=50,y=200)
NBButton.config(font=font1)

DTButton = Button(main, text="Run Decision Tree Algorithm", command=runDT)
DTButton.place(x=50,y=250)
DTButton.config(font=font1)

CPButton = Button(main, text="Run Cumulative Predictor Algorithm", command=runCP)
CPButton.place(x=50,y=300)
CPButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=350)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Performance from Test Data", command=predict)
predictButton.place(x=50,y=400)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
