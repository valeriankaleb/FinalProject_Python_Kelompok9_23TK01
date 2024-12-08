from flask import Flask, render_template, request, redirect
import pickle
import sklearn
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        
        with open('AI_AirClassifier.pkl','rb') as r:
            model = pickle.load(r)

        Temperature = float(request.form['Temperature'])
        Humidity = float(request.form['Humidity'])
        PM25 = float(request.form['PM25'])
        PM10 = float(request.form['PM10'])
        NO2 = float(request.form['NO2'])
        SO2 = float(request.form['SO2'])
        CO = float(request.form['CO'])
        Proximity_to_Industrial_Areas = float(request.form['Proximity_to_Industrial_Areas'])
        Population_Density = float(request.form['Population_Density'])

        datas = np.array((Temperature, Humidity, PM25, PM10, NO2, SO2, CO, Proximity_to_Industrial_Areas, Population_Density))
        datas = np.reshape(datas, (1, -1))

        airclassifier = model.predict(datas)

        return render_template('hasil.html', finalData=airclassifier)

    else:
        return render_template('index.html')
    
if __name__ == "__main__":
    app.run(debug=True)