import json

import numpy as np
import pandas as pd
import pymysql
from flask import Flask, request
import flask_cors as cors
from train import main_, parse_args
from utils.for_overview import solve_overview, get_threshold_list

app = Flask(__name__)
cors.CORS(app)


@app.route('/')
def ping():
    return 'pong'


df_all = pd.read_csv('dataset/re_data.csv',
                     dtype={'company_id': str, 'port_id': str, 'polution_id': str, }).sort_values(by='timestamp')
df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], format='%d/%m/%Y %H:%M:%S')



@app.route('/api/v1/run', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'])
def run():
    if request.method != 'POST':
        return 'wrong request method'
    db = pymysql.Connect(host='47.96.116.167', port=3306, user='root', passwd='123123', db='yuheng', charset='utf8')
    cursor = db.cursor()
    meta = 'run'
    args = parse_args()
    data = request.get_data()
    data = json.loads(data)
    resp_json = dict()

    date_s_1 = pd.to_datetime(data['date_s_1'], format='%d/%m/%Y').strftime('%Y-%m-%d')
    date_e_1 = pd.to_datetime(data['date_e_1'], format='%d/%m/%Y').strftime('%Y-%m-%d')
    date_s_2 = pd.to_datetime(data['date_s_2'], format='%d/%m/%Y').strftime('%Y-%m-%d')
    date_e_2 = pd.to_datetime(data['date_e_2'], format='%d/%m/%Y').strftime('%Y-%m-%d')
    company_id = data['company_id']
    port_id = data['port_id']
    date_select = [date_s_1, date_e_1, date_s_2, date_e_2]

    selected_df_period1_data = df_all[(df_all['timestamp'] >= date_s_1) & (df_all['timestamp'] <= date_e_1) &
                                      (df_all['port_id'] == port_id) & (df_all['company_id'] == company_id)].iloc[:,
                               2:].groupby(['timestamp']).sum()
    selected_df_period2_data = df_all[(df_all['timestamp'] >= date_s_2) & (df_all['timestamp'] <= date_e_2) &
                                      (df_all['port_id'] == port_id) & (df_all['company_id'] == company_id)].iloc[:,
                               2:].groupby(['timestamp']).sum()
    selected_data = [selected_df_period1_data, selected_df_period2_data]

    for i in range(len(date_select) // 2):
        resp_json['date' + str(i + 1)] = dict()
        if selected_data[i].empty:
            resp_json['date' + str(i + 1)] = []
            continue
        resp_json['date' + str(i + 1)]['date_list'] = selected_data[i].index.strftime('%d/%m/%Y').tolist()
        resp_json['date' + str(i + 1)] = main_(args, selected_data[i], resp_json['date' + str(i + 1)], meta=meta,
                                               cursor=cursor)

    # 下面是overview(总览)的整理
    resp_json = solve_overview(resp_json, selected_data, get_threshold_list(cursor), meta)

    return resp_json


@app.route('/api/v1/selfcheck', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'])
def self_check():
    if request.method != 'POST':
        return 'wrong request method'
    meta = 'selfcheck'
    args = parse_args()
    data = request.get_data()
    data = json.loads(data)

    date_s_1 = pd.to_datetime(data['date_s_1'], format='%d/%m/%Y')
    date_e_1 = pd.to_datetime(data['date_e_1'], format='%d/%m/%Y')
    date_s_2 = pd.to_datetime(data['date_s_2'], format='%d/%m/%Y')
    date_e_2 = pd.to_datetime(data['date_e_2'], format='%d/%m/%Y')

    data = pd.DataFrame(data['data'])
    data.iloc[1:, 0] = pd.to_datetime(data.iloc[1:, 0], format='%d/%m/%Y')

    selected_df_period1_data = data.iloc[1:, :][(data.iloc[1:, 0] >= date_s_1) & (data.iloc[1:, 0] <= date_e_1)]
    selected_df_period2_data = data.iloc[1:, :][(data.iloc[1:, 0] >= date_s_2) & (data.iloc[1:, 0] <= date_e_2)]
    selected_data = [selected_df_period1_data, selected_df_period2_data]

    print('data\n', data)
    resp_json = dict()
    threshold_list = np.array(data.iloc[0, 1:]).astype(float)  # 取第一行的数据，去掉第一列的时间
    date_list = pd.to_datetime(data.iloc[1:, 0].values).strftime('%Y-%m-%d')  # 取第一列的数据，去掉第一行的时间


    for i in range(2):
        resp_json['date' + str(i + 1)] = dict()
        if selected_data[i].empty:
            resp_json['date' + str(i + 1)] = []
            continue

        if len(pd.to_datetime(selected_data[i].iloc[:, 0].values).strftime('%Y-%m-%d').tolist()) < 7:
            return {
                'error': 'too less exist date',
            }
        resp_json['date' + str(i + 1)]['date_list'] = pd.to_datetime(selected_data[i].iloc[:, 0].values).strftime(
            '%Y-%m-%d').tolist()
        resp_json['date' + str(i + 1)] = main_(args, selected_data[i].iloc[:, 1:], resp_json['date' + str(i + 1)],
                                               meta=meta,
                                               threshold_list=threshold_list, date_list=date_list)

    # 下面是overview(总览)的整理
    resp_json = solve_overview(resp_json, selected_data, threshold_list, meta)

    return resp_json


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
