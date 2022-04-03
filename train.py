import json
import math
import time
import pandas as pd
import torch
import torch.nn as nn
import argparse
from utils.data_prepare import get_data
from model.Transformer import TransAm
from utils.train import train
from utils.plot_and_loss import plot_and_loss
from utils.eval import evaluate


df_all = pd.read_csv('dataset/re_data.csv')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--input_window', type=int, default=5, help='Number of input steps.')
    parser.add_argument('--output_window', type=int, default=1, help='Number of prediction steps, '
                                                                     'in this model its fixed to one.')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of batch_size.')
    parser.add_argument('--weight', type=str, default="weights/best_model.pth", help='Load weight path.(default: '
                                                                                     'weightss/best_model.pth)')
    parser.add_argument('--model', type=str, default='ts', help='Wanna run which model?')

    parser.add_argument('--port_id', type=str, default=None, help='port_id.')
    parser.add_argument('--polution_id', type=str, default=None, help='polution_id.')
    parser.add_argument('--date_s', type=str, default=None, help='start of the date.')
    parser.add_argument('--date_e', type=str, default=None, help='end of the date.')
    parser.add_argument('--dim', type=str, default=None, help='choose one dim(input dim name).')
    parser.add_argument('--company_id', type=str, default=None, help='company_id.')
    # parser.add_argument("--seed", type=int, default=None, help="random seed")
    return parser.parse_args()


def main_(args, data, resp_json, meta, cursor=None, threshold_list=None, date_list=None):
    input_window = args.input_window
    output_window = args.output_window
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    choice_model = args.model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    train_data, val_data, scaler = get_data(args, input_window, output_window, device=device, data=data)
    print("----------------------------------------------------------")
    print("train_data.shape:", train_data.shape)
    print("----------------------------------------------------------")
    if choice_model == 'ts':
        model = TransAm(feature_size=train_data.shape[-1]).to(device)
    print('model_type: ', model.model_type)
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.95)

    # best_val_loss = float("inf")
    # best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(train_data, input_window, model, optimizer, criterion, scheduler, epoch, batch_size)

        if epoch % 2 == 0:
            val_loss, resp_json = plot_and_loss(model, val_data, epoch, criterion, input_window, scaler, args.dim,
                                                resp_json, meta, cursor, threshold_list=threshold_list, date_list_=date_list)
            # save_path = "weightss" + os.sep + "trained-for-" + str(epoch) + "-epoch.pth"
            # if not os.path.exists("weightss"):
            #     os.mkdir("weightss")
            # torch.save(model.state_dict(), save_path)
        else:
            val_loss = evaluate(model, val_data, criterion, input_window)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} '.format(epoch, (
                time.time() - epoch_start_time), val_loss))
        print('-' * 89)

        # if not os.path.exists('weightss'):
        #     os.mkdir('weightss')
        #
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_model = model
        #     save_path = "weights" + os.sep + "best_model.pth"
        #     torch.save(model.state_dict(), save_path)
        #     print("save successfully")

        scheduler.step()

    return resp_json


# if __name__ == "__main__":
#     args = parse_args()
# print(args)
# main(args)
