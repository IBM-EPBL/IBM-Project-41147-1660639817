import os
import numpy as np
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


app=Flask(__name__)#our flask app
model=load_model('ECG.h5')

@app.route("/") #default routC:\Users\hariharan\Documents\Projects\Untitled Foldere
def about():
    return render_template( "about.html")
@app.route("/about")
def home():
    return render_template("about.html")

@app.route("/info") #default route
def information():
    return render_template("info.html")

@app.route("/upload")
def test():
    return render_template("index6.html")

@app.route("/predict",methods=["GET","POST"])
def upload():
    if request.method=='POST':
        
        f=request.files['file']
        basepath=os.path.dirname('__file__')
        filepath=os.path.join(basepath,"uploads",f.filename)
        f.save(filepath)

        img=image.load_img(filepath,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)

        pred=(model.predict(x) > 0.5).astype("int32")
        print("Prediction",pred)

        index=['Left Bundle Branch Block','Normal','Premature Atrial Contraction','Premature Ventricular Contraction','Right Bundle Branch Block','Ventricular Fibrillation']
        result=str(index[pred[0].tolist().index(1.)])
        return render_template("base.html", name = result)  
        
    return None
    

if __name__=="__main__":
    app.run(debug=True)
    app.run(host='127.0.0.1', port=5000)
