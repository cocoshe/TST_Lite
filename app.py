import json
from random import random

import numpy as np
import pandas as pd
import pymysql
from flask import Flask, request
import flask_cors as cors
from self_check import run_self_check
from train import main_, parse_args

app = Flask(__name__)
cors.CORS(app)


@app.route('/')
def ping():
    return 'pong'


df_all = pd.read_csv('dataset/re_data.csv', dtype={'company_id': str, 'port_id': str, 'polution_id': str,}).sort_values(by='timestamp')
print('types: ', df_all.dtypes)
df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], format='%d/%m/%Y %H:%M:%S')

# db
db = pymysql.Connect(host='localhost', port=3306, user='root', passwd='123123', db='yuheng', charset='utf8')
cursor = db.cursor()


@app.route('/run', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'])
def run():
    if request.method != 'POST':
        return 'wrong request method'
    args = parse_args()
    data = request.get_data()
    data = json.loads(data)
    # data = pd.DataFrame(data)
    # print(data)
    print('data\n', data)
    resp_json = dict()

    date_s_1 = pd.to_datetime(data['date_s_1'], format='%d/%m/%Y').strftime('%Y-%m-%d')
    date_e_1 = pd.to_datetime(data['date_e_1'], format='%d/%m/%Y').strftime('%Y-%m-%d')
    date_s_2 = pd.to_datetime(data['date_s_2'], format='%d/%m/%Y').strftime('%Y-%m-%d')
    date_e_2 = pd.to_datetime(data['date_e_2'], format='%d/%m/%Y').strftime('%Y-%m-%d')
    company_id = data['company_id']
    port_id = data['port_id']
    # 不用选择polution_id 直接并行跑完总览
    date_select = [date_s_1, date_e_1, date_s_2, date_e_2]
    # print('date_select\n', date_select)

    # print('origin_df: ', df_all)
    # 根据传入的时间段，选择对应的数据
    # print('company_id: ', company_id, 'port_id: ', port_id)

    # print('------------------------------------------------')
    # print(company_id == df_all['company_id'].values)
    # print('------------------------------------------------')
    # print(df_all[(df_all['timestamp'] >= date_s_1) & (df_all['timestamp'] <= date_e_1) & (df_all['port_id'] == port_id)])
    # print(df_all[(df_all['timestamp'] >= date_s_2) & (df_all['timestamp'] <= date_e_2) & (df_all['port_id'] == port_id)])
    selected_df_period1_data = df_all[(df_all['timestamp'] >= date_s_1) & (df_all['timestamp'] <= date_e_1) & (df_all['port_id'] == port_id) & (df_all['company_id'] == company_id)].iloc[:, 2:].groupby(['timestamp']).sum()
    selected_df_period2_data = df_all[(df_all['timestamp'] >= date_s_2) & (df_all['timestamp'] <= date_e_2) & (df_all['port_id'] == port_id) & (df_all['company_id'] == company_id)].iloc[:, 2:].groupby(['timestamp']).sum()
    selected_data = [selected_df_period1_data, selected_df_period2_data]

    print('------------------------------------------------')
    print('timestamp: ', selected_df_period1_data.index.strftime('%d/%m/%Y'))

    print('selected_df: ', selected_df_period1_data)
    for i in range(len(date_select) // 2):
        resp_json['date' + str(i + 1)] = dict()
        resp_json['date' + str(i + 1)]['date_list'] = selected_data[i].index.strftime('%d/%m/%Y').tolist()
        resp_json['date' + str(i + 1)] = main_(args, selected_data[i] + np.random.rand(selected_data[i].shape[0], selected_data[i].shape[1]), resp_json['date' + str(i + 1)], cursor)
    # resp_json = run_self_check(data)
    return resp_json


@app.route('/selfcheck', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'])
def self_check():
    if request.method != 'POST':
        return 'wrong request method'
    args = parse_args()
    data = request.get_data()
    data = json.loads(data)
    data = data['data']
    data = pd.DataFrame(data)
    # print(data)
    print('data\n', data)
    resp_json = dict()
    resp_json = main_(args, data, resp_json)
    # resp_json = run_self_check(data)
    return resp_json


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
