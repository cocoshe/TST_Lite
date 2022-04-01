import pandas as pd
import numpy as np


def solve_overview(resp_json, selected_data, threshold_list, meta):
    resp_json['overview'] = dict()
    # 对不同指标分析增长率
    resp_json['overview']['compare_date2_date1_of_diff_features'] = (
            np.mean(selected_data[1].iloc[:, 1:].values.astype(float), axis=0) - np.mean(
        selected_data[0].iloc[:, 1:].values.astype(float), axis=0)).tolist()

    # 防止除0
    resp_json['overview']['compare_date2_date1_rate_of_diff_features'] = ((np.mean(
        selected_data[1].iloc[:, 1:].values.astype(float), axis=0) - np.mean(
        selected_data[0].iloc[:, 1:].values.astype(float), axis=0)) / (np.mean(
        selected_data[0].iloc[:, 1:].values.astype(float) +
        np.random.rand(selected_data[0].iloc[:, 1:].values.shape[0], selected_data[0].iloc[:, 1:].values.shape[1]),
        axis=0))).tolist()
    # 分析异常数量的增长率(做对比)
    if meta == 'selfcheck':
        if resp_json['date1']:
            resp_json['overview']['date1_warning_count_of_diff_features'] = np.sum(
                selected_data[0].iloc[:, 1:].values.astype(float) - resp_json['date1']['rebuild_data'] > threshold_list,
                axis=0).tolist()
        else:
            resp_json['overview']['date1_warning_count_of_diff_features'] = []
        if resp_json['date2']:
            resp_json['overview']['date2_warning_count_of_diff_features'] = np.sum(
                selected_data[1].iloc[:, 1:].values.astype(float) - resp_json['date2']['rebuild_data'] > threshold_list,
                axis=0).tolist()
        else:
            resp_json['overview']['date2_warning_count_of_diff_features'] = []
        if resp_json['date1'] and resp_json['date2']:
            resp_json['overview']['compare_date2_date1_count_of_diff_features'] = (
                    np.array(resp_json['overview']['date2_warning_count_of_diff_features']) - np.array(
                resp_json['overview']['date1_warning_count_of_diff_features'])).tolist()
        else:
            resp_json['overview']['compare_date2_date1_count_of_diff_features'] = []
            print('date1 or date2 is None')
    else:  # meta == 'run'
        # print("------------------------------------------------------")
        # print(selected_data[0].iloc[:, :].values.astype(float))
        # print("date1: ", resp_json['date1'])
        # print(resp_json['date1']['rebuild_data'])
        # print("------------------------------------------------------")
        if resp_json['date1']:
            resp_json['overview']['date1_warning_count_of_diff_features'] = np.sum(
                selected_data[0].iloc[:, :].values.astype(float) - resp_json['date1']['rebuild_data'] > threshold_list,
                axis=0).tolist()
        else:
            resp_json['overview']['date1_warning_count_of_diff_features'] = []

        if resp_json['date2']:
            resp_json['overview']['date2_warning_count_of_diff_features'] = np.sum(
                selected_data[1].iloc[:, :].values.astype(float) - resp_json['date2']['rebuild_data'] > threshold_list,
                axis=0).tolist()
        else:
            resp_json['overview']['date2_warning_count_of_diff_features'] = []

        if resp_json['overview']['date2_warning_count_of_diff_features'] and resp_json['overview'][
            'date1_warning_count_of_diff_features']:
            resp_json['overview']['compare_date2_date1_count_of_diff_features'] = (
                    np.array(resp_json['overview']['date2_warning_count_of_diff_features']) - np.array(
                resp_json['overview']['date1_warning_count_of_diff_features'])).tolist()
        else:
            resp_json['overview']['compare_date2_date1_count_of_diff_features'] = []
            print("Warning: no data in date2 or date1")
    return resp_json


def get_threshold_list(cursor):
    sql = "select threshold from threshold"
    cursor.execute(sql)
    threshold_list = cursor.fetchall()[0]
    return threshold_list
