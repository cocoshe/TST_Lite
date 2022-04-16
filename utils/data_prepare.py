import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def create_inout_sequences(input_data, tw, output_window):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + output_window:i + tw + output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(np.array(inout_seq))


def get_data(args, input_window, output_window, device='cpu', data=None):
    series = data
    print('df.head():\n', series.head())
    # timestamp = series['timestamp']
    dim_name = args.dim
    if dim_name is not None:
        series = series[dim_name]
        print("dim_name: ", dim_name)
    print('series.shape:', series.shape)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    if len(series.shape) == 1:
        amplitude = scaler.fit_transform(series.to_numpy().reshape(-1, 1)).reshape(-1)
    else:
        amplitude = scaler.fit_transform(series.to_numpy())

    train_data = test_data = amplitude

    train_sequence = create_inout_sequences(train_data, input_window, output_window)
    train_sequence = train_sequence[:-output_window]

    test_data = create_inout_sequences(test_data, input_window, output_window)
    test_data = test_data[:-output_window]

    return train_sequence.to(device), test_data.to(device), scaler


def get_batch(source, i, batch_size, input_window):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]  # 这里的sql_len是指source的seq_len, 其实应该还是batch_size
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))

    if len(input.shape) == 4:
        input = input.squeeze(2)
        target = target.squeeze(2)
    return input, target
