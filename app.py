from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import json
import pickle


import sklearn  

app = Flask(__name__)

main_cols = pickle.load(open("columns.pkl", 'rb'))

def data_encode(df):
    df = pd.get_dummies(data = df, columns=["Geography"], drop_first = False)
    for col in df.select_dtypes(include=['category','object']).columns:
        codes,_ = df[col].factorize(sort=True)    
        df[col]=codes
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    
    form_data = request.form.to_dict()
    print("form_data is printed******************************")
    print(form_data)
    df_input = pd.DataFrame.from_records([form_data])
    df_input = pd.DataFrame(df_input)
    
    sample_df = pd.DataFrame(columns = main_cols)
    clean_df = data_encode(df_input)
    main_df = sample_df.append(clean_df,sort=False)
    main_df = main_df.fillna(0)
    print(main_df)
    std_df = main_df.copy()
    
    std_df = std_df.astype(float)
    print("Data is printed *******************************************************")
    print(type(std_df))
    print("Data type is printed *******************************************************")

    clf = pickle.load(open('model.pkl', 'rb'))
    pred = clf.predict(std_df)

    print("pred is printed ******************************")
    print(pred)
    
    x = pred
    if x == 1:
        
        return render_template('index.html', 
                               predicted_value="The customer has the RISK of churn.")
    else:
        
        return render_template('index.html', 
                               predicted_value="The customer DO NOT have the RISK of churn.") 

if __name__ == '__main__':
    app.run()







