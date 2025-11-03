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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Customdataset(Dataset):
    def __init__(self,a,b):
        self.x=torch.tensor(a,dtype=torch.float32)
        self.y=torch.tensor(b,dtype=torch.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index],self.y[index]
    
class Classification(nn.Module):
    def __init__(self,x,y,hiddenlayer:int):
        super(Classification,self).__init__()
        self.Linear1=nn.Linear(x.shape[1],hiddenlayer)
        self.Linear2=nn.Linear(hiddenlayer,y.shape[1])
            
    def forward(self,x):
        x=self.Linear1(x)
        x=torch.relu(x)
        x=self.Linear2(x)
        return x
    
class Classification_model():
    def __init__(self):
        self.sc = StandardScaler()
        self.pca = PCA()
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
        self.model=Classification(x=self.x,y=self.y,hiddenlayer=hiddenlayer)
        critetion=nn.CrossEntropyLoss()
        optimizer=optim.Adam(self.model.parameters(),lr=learing_rate)
        train_acc,train_loss=[],[]
        val_acc,val_loss=[],[]
        for i in range(epoches):
            l1,l2=0,0
            self.model.train()
            ta,tl=[],[]
            for values,labels in traindl:
                labels=torch.argmax(labels,dim=1)
                ypred = self.model(values)
                tloss = critetion(ypred,labels)
                optimizer.zero_grad()
                tloss.backward()        
                optimizer.step()
                l1=labels.unsqueeze(1)
                ypred=torch.argmax(ypred,dim=1) 
                tl.append(int(tloss.detach().numpy()))
                ta.append(precision_score(l1.cpu().numpy(),ypred.cpu().numpy(),average='macro',zero_division=0))
            train_acc.append(statistics.mean(ta))
            train_loss.append(statistics.mean(tl))
            # print('trian')
            # print(mean_absolute_error(l1,ypred.detach().numpy()))  
            # print(tloss)
            va,vl = [],[]
            self.model.eval()   
            with torch.no_grad():
                for values,labels in valdl:
                    labels=torch.argmax(labels,dim=1)
                    ypred = self.model(values)
                    tloss = critetion(ypred,labels)
                    l2=labels.unsqueeze(1)
                    ypred=torch.argmax(ypred,dim=1) 
                    vl.append(int(tloss.detach().numpy()))
                    va.append(precision_score(l2.cpu().numpy(),ypred.cpu().numpy(),average='macro',zero_division=0))
                val_acc.append(statistics.mean(va))
                val_loss.append(statistics.mean(vl))
                # print('test')
                # print(mean_absolute_error(l2,ypred.detach().numpy()))
                # print(tloss)
        return train_acc,train_loss,val_acc,val_loss
    
    def plot_train_val(self,train_acc,train_loss,val_acc,val_loss,epoches,testdl):
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(range(epoches), train_loss, color='tab:red', label='Loss')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss", color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax2 = ax1.twinx()
        ax2.plot(range(epoches), train_acc, color='tab:blue', label='Accuracy')
        ax2.set_ylabel("Accuracy", color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        plt.title("Training Loss and Accuracy vs Epochs")
        fig.tight_layout()
        train_plot_path = f"year3/static/plots/classification_trian_plot.png"
        plt.savefig(train_plot_path)
        print('pathsaved')
        plt.close()

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(range(epoches), val_loss, color='tab:red', label='Loss')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss", color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax2 = ax1.twinx()
        ax2.plot(range(epoches), val_acc, color='tab:blue', label='Accuracy')
        ax2.set_ylabel("Accuracy", color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        plt.title("Validation Loss and Accuracy vs Epochs")
        fig.tight_layout()
        val_plot_path = f"year3/static/plots/classification_val_plot.png"
        plt.savefig(val_plot_path)
        print('pathsaved')
        plt.close()

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for values, labels in testdl:
                labels = torch.argmax(labels, dim=1)  # true class indices
                ypred = self.model(values)
                ypred = torch.argmax(ypred, dim=1)   # predicted class indices
                
                all_labels.append(labels.cpu().numpy())
                all_preds.append(ypred.cpu().numpy())

        # Stack all batches into single arrays
        y_true = np.hstack(all_labels)
        y_pred = np.hstack(all_preds)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        cm_path = "year3/static/plots/confusion_matrix.png"
        plt.title("Confusion Matrix on Validation Data")
        plt.savefig(cm_path)   # save to file
        plt.close()

        acc = accuracy_score(y_true,y_pred)
        return train_plot_path,val_plot_path,cm_path,acc

if __name__ == '__main__':
    obj = Classification_model()
    traindl,testdl,valdl=obj.load_transform_data()
    train_acc,train_loss,val_acc,val_loss = obj.train(traindl=traindl,valdl=valdl,learing_rate=0.001,epoches=2,hiddenlayer=10)