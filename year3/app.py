from flask import Flask, render_template,request
from reg import Regression
from classi import Classification_model


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/Regression',methods=['GET', 'POST'])
def regression():
    obj = Regression()
    traindl,testdl,valdl = obj.load_transform_data()
    if request.method == 'POST':
        epochs = int(request.form['epochs'])
        learning_rate = float(request.form['learning_rate'])
        hidden_layer = int(request.form['hidden_layer'])
        train_acc,train_loss,val_acc,val_loss = obj.train(traindl=traindl,valdl=valdl,epoches=epochs,learing_rate=learning_rate,hiddenlayer=hidden_layer)
        train_plot_path,val_plot_path,test_plot_path,train_view_plot_path = obj.plot_train_val(train_loss=train_loss,train_acc=train_acc,val_acc=val_acc,val_loss=val_loss,epoches=epochs,testdl=testdl,traindl=traindl)
        return render_template('regression.html',train_plot_path=train_plot_path,val_plot_path=val_plot_path,test_plot_path=test_plot_path,train_view_plot_path=train_view_plot_path) 
        
    return render_template('regression.html')

@app.route('/Classification',methods=['GET', 'POST'])
def classification():
    obj = Classification_model()
    traindl,testdl,valdl = obj.load_transform_data()
    if request.method == 'POST':
        epochs = int(request.form['epochs'])
        learning_rate = float(request.form['learning_rate'])
        hidden_layer = int(request.form['hidden_layer'])
        train_acc,train_loss,val_acc,val_loss = obj.train(traindl=traindl,valdl=valdl,epoches=epochs,learing_rate=learning_rate,hiddenlayer=hidden_layer)
        train_plot_path,val_plot_path,cm_path,acc = obj.plot_train_val(train_loss=train_loss,train_acc=train_acc,val_acc=val_acc,val_loss=val_loss,epoches=epochs,testdl=testdl)
        return render_template('classification.html',train_plot_path=train_plot_path,val_plot_path=val_plot_path,cm_path=cm_path,acc=acc) 
    return render_template('classification.html')  


if __name__ == '__main__':
    app.run()
