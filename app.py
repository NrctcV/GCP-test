"""
Launch app with `streamlit run main.py --server.port 8000`.
"""


import google.oauth2.credentials
import pandas_gbq
from datetime import date
import os
from google.cloud import storage
import pandas as pd
import numpy as np
import streamlit as st
from fbprophet import Prophet
import plotly.graph_objs as go
#fsspec
#gcsfs

storage_client = storage.Client()

def is_exist(bucket_name,object):

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.get_blob(object)
    try:
        return blob.exists(storage_client)
    except:
        return False

#@st.cache(allow_output_mutation=True)  # This function will be cached
def dataset(n):

    """
    Connection to Google BigQuery
    """
    '''
    credentials = google.oauth2.credentials.Credentials(
        'xxxx')
    project_id = "al-bi-bq-prod"
    final_date = date.today()
    sql_query = f"""
        select date as ds, sum(total_installs) as y from `al-bi-bq-prod.dwh.fact_daily_stats`
        where _partitiondate between '2020-11-01' and '{final_date}'
        group by 1
        order by 1"""

    df_init = pandas_gbq.read_gbq(sql_query, project_id=project_id)
    df_init['ds'] = df_init['ds'].dt.strftime('%Y-%m-%d')'''
    df_init = pd.read_csv('gs://axiomm/installs.csv')
    df_init.drop(df_init.tail(n).index, inplace=True)
    return df_init



def prediction(dataset):
    """
    Modeling and prediction making.
    :param dataset: imported dataset
    :return: predicted metrics value for the next period, graph of the model performance
    """
    #with open('serialized_model.json', 'r') as fin:
    #   model = model_from_json(json.load(fin))  # Load model
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(dataset)
    future = model.make_future_dataframe(periods=7, freq='d')
    forecast = model.predict(future)
    forecast['ds'] = forecast['ds'].dt.strftime('%Y-%m-%d')
    #fig = fbprophet.plot.plot_plotly(model, forecast, xlabel='Date', ylabel='Metric_value')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dataset['ds'], y=dataset['y'], name='Actual', ))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prediction', ))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], name='Trend', ))
    # fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['rain'], name='Rain',))
    # fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['temp'], name='Temp',))
    # fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['holidays'], name='Holidays', ))
    # fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yearly'], name='Yearly', ))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['weekly'], name='Weekly', ))
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    return forecast, model, fig


def anomaly(data_new, forecast_old):
    df_merged = pd.merge(forecast_old, data_new, on='ds', how='left')
    df_merged['Anomaly?'] = np.where(
        (df_merged['y'] < df_merged['yhat_lower']) | (df_merged['y'] > df_merged['yhat_upper']),
        'Yes', 'No')
    df_merged = df_merged[['ds','yhat_lower','yhat_upper','yhat','y','Anomaly?']]
    df_merged.columns = ['date', 'Lowest possible value', 'Highest possible value','Actual prediction','Actual value', 'Anomaly?']
    df_merged["Lowest possible value"] = df_merged["Lowest possible value"].astype(int)
    df_merged["Highest possible value"] = df_merged["Highest possible value"].astype(int)
    df_merged["Actual prediction"] = df_merged["Actual prediction"].astype(int)
    df_merged["Actual value"] = df_merged["Actual value"].fillna("0").astype(int)

    df_merged.to_csv('forecast_merged.csv')
    storage_client.get_bucket('axiomm').blob('forecast_merged.csv').upload_from_filename(
        'forecast_merged.csv')
    return df_merged


def forecast_horizons(data_new, forecast_for_today):
    merged_table = anomaly(data_new, forecast_for_today).reset_index()
    merged_table_month = merged_table[len(merged_table) - 31:]
    merged_table_week = merged_table[len(merged_table) - 7:]
    merged_table_day = merged_table[len(merged_table) - 7: len(merged_table) - 6]
    return merged_table_day, merged_table_week, merged_table_month, merged_table

def color_survived(val):
    color = 'red' if val == 'Yes' else 'white'
    return f'background-color: {color}'




def main():

    if is_exist('axiomm','forecast.csv') == False:
            #os.path.exists('gs://axiomm/forecast.csv'):
        data_first = dataset(1)
        forecast_for_today = prediction(data_first)[0]
        #gstorage
        forecast_for_today.to_csv('forecast.csv')
        storage_client.get_bucket('axiomm').blob('forecast.csv').upload_from_filename('forecast.csv')
        main()
    else:


        #daily_iterations
        data_new = dataset(0)
        #gstorage
        forecast_for_today = pd.read_csv('gs://axiomm/forecast.csv')
        forecast_for_tomorrow = prediction(data_new)[0]
        #Safe update
        last_date1 = forecast_for_today['ds'].iloc[-1]
        last_date2 = forecast_for_tomorrow['ds'].iloc[-1]
        if last_date1 != last_date2:
            # gstorage
            forecast_for_tomorrow.to_csv('forecast.csv')
            storage_client.get_bucket('axiomm').blob('forecast.csv').upload_from_filename(
                'forecast.csv')
            # gstorage
            forecast_for_today.to_csv('forecast_for_spammers.csv')
            storage_client.get_bucket('axiomm').blob('forecast_for_spammers.csv').upload_from_filename(
                'forecast_for_spammers.csv')
        else:
            st.text('No new updates')
            # gstorage
            forecast_for_today = pd.read_csv('gs://axiomm/forecast_for_spammers.csv')


        # output

        st.write('# Today')
        st.table(forecast_horizons(data_new, forecast_for_today)[0].style.applymap(color_survived, subset=['Anomaly?']))
        st.write('# Weekly forecast')
        st.table(forecast_horizons(data_new, forecast_for_today)[1].style.applymap(color_survived, subset=['Anomaly?']))
        st.write('# Monthly performance')
        st.table(forecast_horizons(data_new, forecast_for_today)[2].style.applymap(color_survived, subset=['Anomaly?']))
        st.write('# Anomaly visual')
        st.plotly_chart(prediction(data_new)[2])






if __name__ == "__main__":
    main()


#Milestones:

# send to slack
# AWS chalice
# test on datetime
# separate model training from prediction: move it to a different function (with cross val and pickle, then load model)




