import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import statistics
from sklearn.metrics import precision_score

class Customdataset(Dataset):
    def __init__(self,a,b):
        self.x=torch.tensor(a,dtype=torch.float32)
        self.y=torch.tensor(b,dtype=torch.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index],self.y[index]
    
class Classification(nn.Module):
    def __init__(self,x:np.array,y:np.array,hiddenlayer:int):
        super(Classification,self).__init__()
        self.Linear1=nn.Linear(x.shape[1],hiddenlayer)
        self.Linear2=nn.Linear(hiddenlayer,y.shape[1])
            
    def forward(self,x):
        x=self.Linear1(x)
        x=torch.relu(x)
        x=self.Linear2(x)
        return x
    
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
        self.x=df.copy()
        y=pd.Series(y)
        temp = pd.get_dummies(y,dtype=int)
        self.y=pd.concat([y,temp],axis=1)
        self.y.drop(['Weather Type'],axis=1,inplace=True)
        x = self.sc.fit_transform(X=self.x)
        y = self.sc.fit_transform(X=self.y)
        Xtrain,Xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42)
        Xtest,Xval,ytest,yval = train_test_split(Xtest,ytest,test_size=0.5,random_state=42)
        traindataset=Customdataset(a=Xtrain,b=ytrain)
        valdataset=Customdataset(a=Xval,b=yval)
        testdataset=Customdataset(a=Xtest,b=ytest)
        traindl=DataLoader(traindataset,batch_size=16,shuffle=True)
        testdl=DataLoader(testdataset,batch_size=16,shuffle=True)
        valdl=DataLoader(valdataset,batch_size=16,shuffle=True)
        
        return traindl,testdl,valdl
    
    def train(self,traindl,valdl,learing_rate,epoches,hiddenlayer):
        model=Classification(x=self.x,y=self.y,hiddenlayer=hiddenlayer)
        critetion=nn.CrossEntropyLoss()
        optimizer=optim.Adam(model.parameters(),lr=learing_rate)
        train_acc,train_loss=[],[]
        val_acc,val_loss=[],[]
        for i in range(epoches):
            l1,l2=0,0
            model.train()
            ta,tl=[],[]
            for values,labels in traindl:
                labels=torch.argmax(labels,dim=1)
                ypred = model(values)
                tloss = critetion(ypred,labels)
                optimizer.zero_grad()
                tloss.backward()        
                optimizer.step()
                l1=labels.unsqueeze(1)
                ypred=torch.argmax(ypred,dim=1) 
                tl.append(int(tloss.detach().numpy()))
                ta.append(precision_score(l1.cpu().numpy(),ypred.cpu().numpy(),average='macro'))
            train_acc.append(statistics.mean(ta))
            train_loss.append(statistics.mean(tl))
            # print('trian')
            # print(mean_absolute_error(l1,ypred.detach().numpy()))  
            # print(tloss)
            va,vl = [],[]
            model.eval()   
            with torch.no_grad():
                for values,labels in valdl:
                    labels=torch.argmax(labels,dim=1)
                    ypred = model(values)
                    tloss = critetion(ypred,labels)
                    l2=labels.unsqueeze(1)
                    ypred=torch.argmax(ypred,dim=1) 
                    vl.append(int(tloss.detach().numpy()))
                    va.append(precision_score(l2.cpu().numpy(),ypred.cpu().numpy(),average='macro'))
                val_acc.append(statistics.mean(va))
                val_loss.append(statistics.mean(vl))
                # print('test')
                # print(mean_absolute_error(l2,ypred.detach().numpy()))
                # print(tloss)

if __name__ == '__main__':
    obj = Classification()
    traindl,testdl,valdl=obj.load_transform_data()
    train_acc,train_loss,val_acc,val_loss = obj.train(traindl=traindl,valdl=valdl,learing_rate=0.001,epoches=5,hiddenlayer=24)