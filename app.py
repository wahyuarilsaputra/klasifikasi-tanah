from flask import Flask,render_template,request,url_for
import numpy as np
import pickle
import os
app = Flask(__name__)

@app.route('/')
@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/penjelasan.html')
def jelas():
    return render_template('penjelasan.html')

@app.route('/Hitung.html',methods = ['POST','GET'])
@app.route('/Hasil.html',methods = ['POST','GET'])
def Hitung():
    if request.method == 'POST':
        Nitrogen = int(request.form['Nitrogen'])
        Phosphorous = int(request.form['Phosphorous'])
        Potassium = int(request.form['Potassium'])
        Temperature = float(request.form['Temperature'])
        Humidity = float(request.form['Humidity'])
        Ph = float(request.form['Ph'])
        Rainfall = float(request.form['Rainfall'])
        metode = str(request.form['metode'])
        
        #normalisasi
        model_normalisasi_N = os.path.join('normalisasiNitrogen')
        model_N = pickle.load(open(model_normalisasi_N,'rb'))
        normal_N = model_N.transform([[Nitrogen]])
        normal_P = model_N.transform([[Phosphorous]])
        normal_K = model_N.transform([[Potassium]])
        normal_Temperature = model_N.transform([[Temperature]])
        normal_Humidity = model_N.transform([[Humidity]])
        normal_Ph = model_N.transform([[Ph]])
        normal_Rainfall = model_N.transform([[Rainfall]])

        inp = np.array([[normal_N[0][0],normal_P[0][0],normal_K[0][0],normal_Temperature[0][0],normal_Humidity[0][0],normal_Ph[0][0],normal_Rainfall[0][0]]],dtype=object)
        if metode == "bayes":
            model_path = os.path.join('bayes.pickle')
            acc_path = os.path.join('accBayes.pickle')
            model = pickle.load(open(model_path, 'rb'))
            acc_model = pickle.load(open(acc_path, 'rb'))
            prediksi = model.predict(inp)
            teks = "NAIVE BAIYES"
            # str_prediksi = str(prediksi)
            return render_template('Hasil.html',hasil = prediksi[0],acc = acc_model,metode = teks)
        elif metode == "forest":
            model_path = os.path.join('randomforest.pickle')
            acc_path = os.path.join('accForest.pickle')
            model = pickle.load(open(model_path, 'rb'))
            acc_model = pickle.load(open(acc_path, 'rb'))
            prediksi = model.predict(inp)
            teks = "RANDOM FOREST"
            return render_template('Hasil.html',hasil = prediksi[0],acc = acc_model,metode = teks)
        elif metode == "tree":
            model_path = os.path.join('decisiontree.pickle')
            acc_path = os.path.join('accTree.pickle')
            model = pickle.load(open(model_path, 'rb'))
            acc_model = pickle.load(open(acc_path, 'rb'))
            prediksi = model.predict(inp)
            teks = "DECISION TREE"
            return render_template('Hasil.html',hasil = prediksi[0],acc = acc_model,metode = teks)
    else:
        return render_template('Hitung.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0')