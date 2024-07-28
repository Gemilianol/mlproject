from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

#Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Fats=float(request.form.get('Fat')),
            SFats=float(request.form.get('Saturated Fats')),
            MFats=float(request.form.get('Monounsaturated Fats')),
            PFats=float(request.form.get('Polyunsaturated Fats')),
            
            Carbs=float(request.form.get('Carbohydrates')),
            Sugars=float(request.form.get('Sugars')),
            Protein=float(request.form.get('Protein')),
            
            DFiber=float(request.form.get('Dietary Fiber')),
            Cholesterol=float(request.form.get('Cholesterol')),
            Sodium=float(request.form.get('Sodium')),
            Water=float(request.form.get('Water')),
            
            VA=float(request.form.get('Vitamin A')),
            VB1=float(request.form.get('Vitamin B1')),
            VB2=float(request.form.get('Vitamin B2')),
            VB3=float(request.form.get('Vitamin B3')),
            VB5=float(request.form.get('Vitamin B5')),
            VB6=float(request.form.get('Vitamin B6')),
            VB11=float(request.form.get('Vitamin B11')),
            VB12=float(request.form.get('Vitamin B12')),

            VC=float(request.form.get('Vitamin C')),
            VD=float(request.form.get('Vitamin D')),
            VE=float(request.form.get('Vitamin E')),
            VK=float(request.form.get('Vitamin K')),
            
            Calcium=float(request.form.get('Calcium')),
            Copper=float(request.form.get('Copper')),
            Iron=float(request.form.get('Iron')),
            Magnesium=float(request.form.get('Magnesium')),
            Manganese=float(request.form.get('Manganese')),
            Phosphorus=float(request.form.get('Phosphorus')),
            Potassium=float(request.form.get('Potassium')),
            Selenium=float(request.form.get('Selenium')),
            Zinc=float(request.form.get('Zinc')),
            ND=float(request.form.get('Nutrition Density'))
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline=PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        return render_template('home.html', results=results[0]) #return a list
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    