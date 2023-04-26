import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score

def get_data():
    data = pd.read_csv('data.csv')
    data = data.drop(['id', 'Unnamed: 32'], axis=1)
    data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
    return data

def scale_data(data):
    x=data.drop(['diagnosis'],axis=1)
    y=data['diagnosis']
    
    scaler=StandardScaler()
    x=scaler.fit_transform(x)
    Xtrain,Xtest,Ytrain,Ytest=train_test_split(x,y,test_size=0.2,random_state=2)
    return Xtrain,Xtest,Ytrain,Ytest,scaler

def train_model(xtrain,ytrain):
    model=LogisticRegression()
    model.fit(xtrain,ytrain)
    return model

def test_model(model,xtest,ytest):
    ypred=model.predict(xtest)
    print("Accuracy :",accuracy_score(ytest,ypred))

def export(model,scaler):
    pickle.dump(model,open('model.pkl','wb'))
    pickle.dump(model,open('scaler.pkl','wb'))

def main():
    data=get_data()
    xtrain,xtest,ytrain,ytest,scaler=scale_data(data)
    model=train_model(xtrain,ytrain)
    export(model,scaler)
    test_model(model,xtest,ytest)




if __name__ == '__main__':
    main()