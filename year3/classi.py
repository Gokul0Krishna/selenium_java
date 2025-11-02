import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

class Customdataset(Dataset):
    def __init__(self,a,b):
        self.x=torch.tensor(a,dtype=torch.float32)
        self.y=torch.tensor(b,dtype=torch.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index],self.y[index]
    
class Classification():
    def __init__(self):
        self.sc = StandardScaler()
    
    def load_transform_data(self):
        df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Desktop\code\cp\javaproject\year3\weather_classification_data.csv')
        temp = pd.get_dummies(df['Cloud Cover'],dtype=int)
        df.drop(['Cloud Cover'],axis=1,inplace=True)
        df=pd.concat([df,temp],axis=1)
        temp = pd.get_dummies(df['Season'],dtype=int)
        df.drop(['Season'],axis=1,inplace=True)
        df=pd.concat([df,temp],axis=1)
        temp = pd.get_dummies(df['Location'],dtype=int)
        df.drop(['Location'],axis=1,inplace=True)
        df=pd.concat([df,temp],axis=1)
        y=df['Weather Type']
        df.drop(['Weather Type'],axis=1,inplace=True)
        x=df.copy()
        y=pd.Series(y)
        temp = pd.get_dummies(y,dtype=int)
        y=pd.concat([y,temp],axis=1)
        y.drop(['Weather Type'],axis=1,inplace=True)
        x = self.sc.fit_transform(X=x)
        y = self.sc.fit_transform(X=y)
        Xtrain,Xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42)
        Xtest,Xval,ytest,yval = train_test_split(Xtest,ytest,test_size=0.5,random_state=42)
        traindataset=Customdataset(a=Xtrain,b=ytrain)
        valdataset=Customdataset(a=Xval,b=yval)
        testdataset=Customdataset(a=Xtest,b=ytest)
        traindl=DataLoader(traindataset,batch_size=16,shuffle=True)
        testdl=DataLoader(testdataset,batch_size=16,shuffle=True)
        valdl=DataLoader(valdataset,batch_size=16,shuffle=True)
        
        return traindl,testdl,valdl