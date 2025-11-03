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
import statistics
import matplotlib.pyplot as plt

class Customdataset(Dataset):
    def __init__(self,a,b):
        self.x=torch.tensor(a,dtype=torch.float32)
        self.y=torch.tensor(b,dtype=torch.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index],self.y[index]
    
class regression(nn.Module):
    def __init__(self,x,hiddenlayer):
        super(regression,self).__init__()
        self.Linear1=nn.Linear(x.shape[1],hiddenlayer)
        self.Linear2=nn.Linear(hiddenlayer,1)

    def forward(self,x):
        x=self.Linear1(x)
        x = torch.relu(x)
        x=self.Linear2(x)
        return x


class Regression():

    def __init__(self):
        self.lc = LabelEncoder()
        self.sc = StandardScaler()
        self.x = None 

    def load_transform_data(self):
        df= pd.read_csv(r'C:\Users\ASUS\OneDrive\Desktop\code\cp\javaproject\year3\Student_Performance.csv')
        y=df.iloc[:,-1]
        x=df.iloc[:,0:5]
        label = self.lc.fit_transform(df['Extracurricular Activities'])
        x.drop('Extracurricular Activities',axis=1,inplace=True)
        x['Extracurricular Activities']=list(label)
        self.x=np.array(x)
        y=np.array(y).reshape(-1, 1)
        x=self.sc.fit_transform(X=self.x)
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
    
    def train(self,traindl,valdl,learing_rate,epoches,hiddenlayer):
        print('reg')
        self.model=regression(x=self.x,hiddenlayer=hiddenlayer)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learing_rate)
        critetion=nn.MSELoss()
        train_acc,train_loss=[],[]
        val_acc,val_loss=[],[]
        for i in range(epoches):
            l1,l2=0,0
            self.model.train()
            ta,tl=[],[]
            for values,labels in traindl:
                ypred = self.model(values)
                tloss = critetion(ypred,labels)
                optimizer.zero_grad()
                tloss.backward()        
                optimizer.step()
                l1=labels.unsqueeze(1)
                tl.append(int(tloss.detach().numpy()))
                ta.append(root_mean_squared_error(l1.view(-1),ypred.detach().numpy()))
            train_acc.append(statistics.mean(ta))
            train_loss.append(statistics.mean(tl))

            va,vl = [],[]
            self.model.eval()   
            with torch.no_grad():
                for values,labels in valdl:
                    ypred = self.model(values)
                    tloss = critetion(ypred,labels)
                    l2=labels.unsqueeze(1)
                    vl.append(int(tloss.detach().numpy()))
                    va.append(root_mean_squared_error(l2.view(-1),ypred.detach().numpy()))
                val_acc.append(statistics.mean(va))
                val_loss.append(statistics.mean(vl))
        return train_acc,train_loss,val_acc,val_loss
    
    def plot_train_val(self,epoches,train_loss,train_acc,val_loss,val_acc,testdl):
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(range(epoches), train_loss, color='tab:red', label='Loss')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("MSE Loss", color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax2 = ax1.twinx()
        ax2.plot(range(epoches), train_acc, color='tab:blue', label='Accuracy')
        ax2.set_ylabel("Accuracy", color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        plt.title("Training Loss and Accuracy vs Epochs")
        fig.tight_layout()
        train_plot_path = f"year3/static/plots/regression_trian_plot.png"
        plt.savefig(train_plot_path)
        print('pathsaved')
        plt.close()

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(range(epoches), val_loss, color='tab:red', label='Loss')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("MSE Loss", color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax2 = ax1.twinx()
        ax2.plot(range(epoches), val_acc, color='tab:blue', label='Accuracy')
        ax2.set_ylabel("Accuracy", color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        plt.title("validation Loss and Accuracy vs Epochs")
        val_plot_path = f"year3/static/plots/regression_val_plot.png"
        plt.savefig(val_plot_path)
        print('pathsaved')
        plt.close()

        self.model.eval()   
        with torch.no_grad():
            for values,labels in testdl:
                ypred = self.model(values)
        plt.figure(figsize=(6,6))
        plt.plot(labels.numpy(),color='blue')
        plt.plot(ypred.numpy(),color='red')
        plt.grid(True)
        plt.title("predictied vs real data")
        test_plot_path = f"year3/static/plots/regression_test_plot.png"
        plt.savefig(test_plot_path)
        plt.close()

        return train_plot_path,val_plot_path,test_plot_path

if __name__ == '__main__':
    obj = Regression()
    traindl,testdl,valdl=obj.load_transform_data()
    train_acc,train_loss,val_acc,val_loss = obj.train(traindl=traindl,valdl=valdl,learing_rate=0.001,epoches=5)
    train_plot_path,val_plot_path=obj.plot_train_val(train_acc=train_acc,train_loss=train_loss,val_acc=val_acc,val_loss=val_loss,epoches=5)