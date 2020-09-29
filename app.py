#from app import app as appl
# from waitress import serve
from flask import Flask,request
import numpy as np
import pandas as pd
import pickle
import requests
app=Flask(__name__)
@app.route('/')
def index():
    return'HI!'




np.random.seed(123)

# @app.route('/get-alm', methods=['POST'])
# def get_ALM_forecasts():
    # post_request_object = request.get_json(force=True)
    # print(post_request_object)

    # return(str(post_request_object))
    
@app.route('/get-alm-oneToOne', methods=['POST'])
def get_ALM_forecasts_oneToOne():

    post_request_object = request.get_json(force=True)
    
    py_response = {}
    
    models_all = ['RidgeCV', 'LassoCV', 'LinearRegression', 'Lasso', 'Ridge']
    model1, model2 = "LinearRegression", "LinearRegression"
    
    if post_request_object['model1'] in models_all:
        model1 = post_request_object['model1']
    if post_request_object['model2'] in models_all:
        model2 = post_request_object['model2']


    rates_spread_delta_forecast, rates_spread_delta_forecast_base, mrktShare_delta_forecast, mrktShare_delta_forecast_base = [], [], [], []


    # Case 2. Difference between shocked case and zer0-shock case is added to the fact!
    for tenor, rates_input in zip(['1w', '2w', '3w', '1m', '2m', '3m', '6m'], 
                                  np.stack((post_request_object['ftp_spread_delta_shocked'],
                                            post_request_object['mosprime_delta']), axis=1)):

        modelFitted1 = pickle.load( open("models/one-to-one/model1/" + model1 + "/" + \
                                     post_request_object['sector'] + "_" + tenor + ".pkl", "rb" )) 

        rates_spread_delta_forecast.append(np.float(modelFitted1.predict(rates_input.reshape(-1,2))))

    for tenor, rates_input in zip(['1w', '2w', '3w', '1m', '2m', '3m', '6m'], 
                                  np.stack((post_request_object['ftp_spread_delta'],
                                            post_request_object['mosprime_delta']), axis=1)):

        modelFitted1 = pickle.load( open("models/one-to-one/model1/" + model1 + "/" + \
                                     post_request_object['sector'] + "_" + tenor + ".pkl", "rb" )) 

        rates_spread_delta_forecast_base.append(np.float(modelFitted1.predict(rates_input.reshape(-1,2)))) 


    for tenor, mrktShare_input in zip(['1w', '2w', '3w', '1m', '2m', '3m', '6m'], np.array(rates_spread_delta_forecast)):

        modelFitted2 = pickle.load( open("models/one-to-one/model2/" + model2 + "/" + \
                                     post_request_object['sector'] + "_" + tenor + ".pkl", "rb" )) 

        mrktShare_delta_forecast.append(np.float(modelFitted2.predict(mrktShare_input.reshape(-1,1))))  


    for tenor, mrktShare_input in zip(['1w', '2w', '3w', '1m', '2m', '3m', '6m'], np.array(rates_spread_delta_forecast_base)):

        modelFitted2 = pickle.load( open("models/one-to-one/model2/" + model2 + "/" + \
                                     post_request_object['sector'] + "_" + tenor + ".pkl", "rb" )) 

        mrktShare_delta_forecast_base.append(np.float(modelFitted2.predict(mrktShare_input.reshape(-1,1))))   


    # adding back differences between shocked case and zero-shock case:
    rates_spread_forecast_difference = np.nan_to_num(np.array(rates_spread_delta_forecast) - np.array(rates_spread_delta_forecast_base))
    #     mrktShare_forecast_difference = np.array(mrktShare_delta_forecast) - np.array(mrktShare_delta_forecast_base)
    mrktShare_forecast_difference = np.array(mrktShare_delta_forecast) - np.array(mrktShare_delta_forecast_base)
    #     mrktShare_forecast_difference = np.nan_to_num(np.array(adjust_MrktShareDelta_To0(mrktShare_forecast_difference)))
    #________________________________________________#
    # Adjustment function adjust_MrktShare_To100() although monotonous is not smooth enough for the final result!!
    # I should think of improvement!
    #________________________________________________#


    # ΔSpread_rate = ΔMosPrime - Δrate;  
    # By construction/assumptions/constant external forecasts partial derivative of ΔMosPrime to ΔFtp is zero; 
    # =>  ΔSpread_rate = ΔMosPrime - Δrate ~ ΔSpread_rate = -Δrate or Δrate = -ΔSpread_rate 
    rates_forecast_difference = -rates_spread_forecast_difference
    rates_forecast = np.array(post_request_object['rates_fact']) + rates_forecast_difference
    mrktShare_forecast = np.array(post_request_object['mrktShare_fact']) + mrktShare_forecast_difference


    py_response['rates_spread_delta_forecast'] = list(rates_spread_delta_forecast)
    py_response['mrktShare_delta_forecast'] = list(mrktShare_delta_forecast)

    py_response['rates_forecast'] = list(rates_forecast)
    #     py_response['mrktShare_forecast'] = adjust_MrktShare_To100(mrktShare_forecast)
    py_response['mrktShare_forecast'] = list(mrktShare_forecast)
    
    return(str(py_response))
    
	
@app.route('/get-alm-manyToOne', methods=['POST'])
def get_ALM_forecasts_manyToOne():

    post_request_object = request.get_json(force=True)
    
    py_response = {}
    
    models_all = ['LinearRegression', 'Lasso', 'Ridge']

    model1, model2 = "LinearRegression", "LinearRegression"

    if post_request_object['model1'] in models_all:
        model1 = post_request_object['model1']
    if post_request_object['model2'] in models_all:
        model2 = post_request_object['model2']


    rates_spread_delta_forecast, rates_spread_delta_forecast_base, mrktShare_delta_forecast, mrktShare_delta_forecast_base = [], [], [], []

    for tenor, mosprime in zip(['1w', '2w', '3w', '1m', '2m', '3m', '6m'], post_request_object['mosprime_delta']):

        modelFitted1 = pickle.load( open("models/many-to-one/model1/" + model1 + "/" + \
                                     post_request_object['sector'] + "_" + tenor + ".pkl", "rb" )) 

        rates_spread_delta_forecast.append(np.float(modelFitted1.predict(post_request_object['ftp_spread_delta'] + [mosprime])))

        rates_spread_delta_forecast_base.append(np.float(modelFitted1.predict(post_request_object['ftp_spread_delta'] + [mosprime]))) 



    for tenor in ['1w', '2w', '3w', '1m', '2m', '3m', '6m']:

        modelFitted2 = pickle.load( open("models/many-to-one/model2/" + model2 + "/" + \
                                     post_request_object['sector'] + "_" + tenor + ".pkl", "rb" )) 

        mrktShare_delta_forecast.append(np.float(modelFitted2.predict(rates_spread_delta_forecast))) 

        mrktShare_delta_forecast_base.append(np.float(modelFitted2.predict(rates_spread_delta_forecast_base))) 


    # adding back differences between shocked case and zero-shock case:
    rates_spread_forecast_difference = np.nan_to_num(np.array(rates_spread_delta_forecast) - np.array(rates_spread_delta_forecast_base))
    mrktShare_forecast_difference = np.array(mrktShare_delta_forecast) - np.array(mrktShare_delta_forecast_base)
    #     mrktShare_forecast_difference = np.nan_to_num(np.array(adjust_MrktShareDelta_To0(mrktShare_forecast_difference)))
    #________________________________________________#
    # Adjustment function adjust_MrktShare_To100() although monotonous is not smooth enough for the final result!!
    # I should think of improvement!
    #________________________________________________#


    # ΔSpread_rate = ΔMosPrime - Δrate;  
    # By construction/assumptions/constant external forecasts partial derivative of ΔMosPrime to ΔFtp is zero; 
    # =>  ΔSpread_rate = ΔMosPrime - Δrate ~ ΔSpread_rate = -Δrate or Δrate = -ΔSpread_rate 
    rates_forecast_difference = -rates_spread_forecast_difference
    rates_forecast = np.array(post_request_object['rates_fact']) + rates_forecast_difference
    mrktShare_forecast = np.array(post_request_object['mrktShare_fact']) + mrktShare_forecast_difference


    py_response['rates_spread_delta_forecast'] = list(rates_spread_delta_forecast)
    py_response['mrktShare_delta_forecast'] = list(mrktShare_delta_forecast)

    py_response['rates_forecast'] = list(rates_forecast)
    #     py_response['mrktShare_forecast'] = adjust_MrktShare_To100(mrktShare_forecast)
    py_response['mrktShare_forecast'] = list(mrktShare_forecast)
    
    return(str(py_response))
    
if __name__=='__main__':
    app.run(threaded=True, debug=True)  
