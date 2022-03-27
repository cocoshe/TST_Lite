import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch.nn as nn
import torch
from model.Transformer import TransAm

from utils.data_prepare import get_data, create_inout_sequences, get_batch


def run_self_check(data):
    df = pd.DataFrame(data)
    print(df.head())

    model = TransAm()
    print(model)

    resp_json = reconstruct(model, data)
    return resp_json


def reconstruct(model, data):
    model = model.eval()
    criterion = nn.MSELoss()
    model_type = model.model_type
    scaler = MinMaxScaler(feature_range=(-1, 1))
    if len(data.shape) == 1:
        amplitude = scaler.fit_transform(data.to_numpy().reshape(-1, 1)).reshape(-1)
    else:
        amplitude = scaler.fit_transform(data.to_numpy())
    test_data = amplitude
    test_data = create_inout_sequences(test_data, tw=20, output_window=1)
    test_data = test_data[:-1]

    test_data = torch.cat((test_data[[0]], test_data, test_data[[-1]]), 0)
    print('data_source shape:', test_data.shape)
    print('---------------------------------')

    total_loss = 0.
    with torch.no_grad():
        for i in range(0, len(test_data) - 1):
            data, target = get_batch(test_data, i, 1, test_data)
            output = model(data)
            if i == 0:
                print('output shape:', output.shape)
                if output.shape[2] == 1:
                    test_result = torch.cat((output[0].view(-1), output[:-1].view(-1).cpu()), 0)
                    truth = target.view(-1)
                else:
                    print('output[[0]].shape:', output[[0]].shape)
                    print('output[:-1].shape:', output[:-1].shape)
                    test_result = torch.cat((output[[0]].squeeze(1), output[:-1].squeeze(1).cpu()), 0)
                    truth = target.squeeze(1)
                    # test_result = torch.cat((output[0].view(-1), test_result.view(-1).cpu()), 0)
            total_loss += criterion(output, target).item()
            if output.shape[2] == 1:
                test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
                truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
            else:
                test_result = torch.cat((test_result, output[[-1]].squeeze(1).cpu()), 0)
                truth = torch.cat((truth, target[[-1]].squeeze(1).cpu()), 0)

    if len(truth.shape) == 1:
        truth = scaler.inverse_transform(truth.reshape(-1, 1))
        test_result = scaler.inverse_transform(test_result.reshape(-1, 1))
    else:
        truth = scaler.inverse_transform(truth)
        test_result = scaler.inverse_transform(test_result)

    print('output truth shape:', truth.shape)
    print('output test_result shape:', test_result.shape)

    print('---------------------------------')
    print('test_result[0].shape, type', test_result[0].shape, test_result[0].dtype)
    print('test_result.shape, type', test_result.shape, test_result.dtype)
    # test_result = torch.cat((test_result[0], test_result), 0)
    print("loss shape: ", (test_result - truth).shape)
    print('---------------------------------')


    json_resp = dict()
    json_resp['test_result'] = test_result.T.tolist()
    json_resp['loss'] = np.abs(test_result - truth).T.tolist()

    # plt.plot(truth, color="blue")
    # plt.plot(test_result, color="red")


    for i in range(0, test_result.shape[-1]):
        plt.figure(figsize=(20, 10))

        plt.plot(truth[:, i], color="blue")
        plt.plot(test_result[:, i], color="red")
        plt.plot(test_result[:, i] - truth[:, i], color="green")

        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        # plt.xticks(ticks=range(len(truth)), labels=timestamp.values[:len(truth)], rotation=90)

        if not os.path.exists("vis"):
            os.mkdir("vis")
        plt.savefig('vis/%s_%s.png' % (model_type, i + 1))
        plt.close()

    return json_resp
