import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as pl

class Customdataset(Dataset):
    def __init__(self,a,b):
        self.x=torch.tensor(a,dtype=torch.float32)
        self.y=torch.tensor(b,dtype=torch.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index],self.y[index]
    
class regression(nn.Module):
    def __init__(self,x):
        super(regression,self).__init__()
        self.Linear1=nn.Linear(x.shape[1],24)
        self.Linear2=nn.Linear(24,1)

    def forward(self,x):
        x=self.Linear1(x)
        x = torch.relu(x)
        x=self.Linear2(x)
        return x


class Regression():

    def __init__(self):
        self.lc = LabelEncoder()
        self.sc = StandardScaler()

    def load_transform_data(self):
        df= pd.read_csv(r'C:\Users\ASUS\OneDrive\Desktop\code\cp\javaproject\year3\Student_Performance.csv')
        y=df.iloc[:,-1]
        x=df.iloc[:,0:5]
        label = self.lc.fit_transform(df['Extracurricular Activities'])
        x.drop('Extracurricular Activities',axis=1,inplace=True)
        x['Extracurricular Activities']=list(label)
        x=np.array(x)
        y=np.array(y).reshape(-1, 1)
        x=self.sc.fit_transform(X=x)
        y=self.sc.fit_transform(X=y)
        Xtrain,Xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42)
        Xtest,Xval,ytest,yval = train_test_split(Xtest,ytest,test_size=0.5,random_state=42)
        traindataset=Customdataset(a=Xtrain,b=ytrain)
        valdataset=Customdataset(a=Xval,b=yval)
        testdataset=Customdataset(a=Xtest,b=ytest)
        traindl=DataLoader(traindataset,batch_size=16,shuffle=True)
        testdl=DataLoader(testdataset,batch_size=16,shuffle=True)
        valdl=DataLoader(valdataset,batch_size=16,shuffle=True)
        return traindl,testdl,valdl

if __name__ == '__main__':
    obj = Regression()
    obj.load_transform_data()