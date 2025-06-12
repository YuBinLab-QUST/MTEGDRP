import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from Model_utils import *
#from models.DbTrs_ge import DbTrs_ge
#from models.DbTrs_mut import DbTrs_mut
#from models.DbTrs_meth import DbTrs_meth
#from models.DbTrs_ge_meth_mut import DbTrs_ge_meth_mut
from models.MTEGDRP import MTEGDRP
from torch_geometric.loader import DataLoader

import datetime
import argparse
import csv

# training function at each epoch

def train(model, device, train_loader, optimizer, epoch, log_interval, model_st):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    loss_ae = nn.MSELoss()
    avg_loss = []
    weight_fn = 0.01
    weight_ae = 2


    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        if 'VAE' in model_st:
            output, _, decode, log_var, mu = model(data)
            loss = weight_fn*loss_fn(output, data.y.view(-1, 1).float().to(device)) + loss_ae(decode, data.target_mut[:,None,:].float().to(device)) + torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        elif 'AE' in model_st:
            output, _, decode = model(data)
            loss = weight_fn*loss_fn(output, data.y.view(-1, 1).float().to(device)) + loss_ae(decode, data.target_mut[:,None,:].float().to(device)) 
        else:
            output, drug_data, mut_data = model(data)
            loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()



        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.target_mut),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

    print(f'Total loss for epoch {epoch}: {sum(avg_loss)/len(avg_loss):.6f}')
    return sum(avg_loss)/len(avg_loss)


def predicting(model, device, loader, model_st):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if 'VAE' in model_st:
                    output, _, decode, log_var, mu = model(data)
            elif 'AE' in model_st:
                output, _, decode = model(data)
            else:
                output, drug_data, mut_data = model(data)

            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    preds = np.array(total_preds.cpu().detach().numpy().flatten(), dtype=np.float32)
    preds = pd.DataFrame(preds)
    preds.to_csv("MTEGDRP-main/data/data_pred/preds_.csv")

    labels = np.array(total_labels.cpu().detach().numpy().flatten(), dtype=np.float32)
    labels = pd.DataFrame(labels)
    labels.to_csv("MTrsDRP-main/data/data_pred/labels_.csv")
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

def main(modeling, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name):

    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)
    for model in modeling:
        model_st = model.__name__
        dataset = 'GDSC'
        train_losses = []
        val_losses = []
        val_pearsons = []
        test_losses = []
        test_pearsons = []
        print('\nrunning on ', model_st + '_' + dataset )
        processed_data_file_train = 'MTEGDRP-main/data/processed/' + dataset + '_train_mix'+'.pt'
        processed_data_file_val = 'MTEGDRP-main/data/processed/' + dataset + '_val_mix'+'.pt'
        processed_data_file_test = 'MTEGDRP-main/data/processed/' + dataset + '_test_mix'+'.pt'
        if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_val)) or (not os.path.isfile(processed_data_file_test))):
            print('please run create_data.py to prepare data in pytorch format!')
        else:
            train_data = TestbedDataset(root='MTEGDRP-main/data', dataset=dataset+'_train_mix')
            val_data = TestbedDataset(root='MTEGDRP-main/data', dataset=dataset+'_val_mix')
            test_data = TestbedDataset(root='MTEGDRP-main/data', dataset=dataset+'_test_mix')

            train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
            print("CPU/GPU: ", torch.cuda.is_available())
                    
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            print(device)
            model = model().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            best_mse = 1000
            best_pearson = 1
            best_epoch = -1
            model_file_name = 'model_MTEGDRP_' + model_st + '_' + dataset +  '.model'
            result_file_name = 'result_MTEGDRP_' + model_st + '_' + dataset +  '.csv'
            loss_fig_name = 'model_MTEGDRP_' + model_st + '_' + dataset + '_loss'
            pearson_fig_name = 'model_MTEGDRP_' + model_st + '_' + dataset + '_pearson'
            for epoch in range(num_epoch):
                train_loss = train(model, device, train_loader, optimizer, epoch+1, log_interval, model_st)
                G,P = predicting(model, device, val_loader, model_st)
                ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),r2(G,P),mae(G,P)]
                            
                G_test,P_test = predicting(model, device, test_loader, model_st)
                ret_test = [rmse(G_test,P_test),mse(G_test,P_test),pearson(G_test,P_test),spearman(G_test,P_test),r2(G_test,P_test),mae(G_test,P_test)]

                train_losses.append(train_loss)
                val_losses.append(ret[1])
                val_pearsons.append(ret[2])
                test_losses.append(ret_test[0])
                test_pearsons.append(ret_test[2])
                

                scheduler.step(ret[1])

                if ret_test[1]<best_mse:
                    torch.save(model.state_dict(), "MTEGDRP-main/log/model/" + model_file_name)
                    with open("MTEGDRP-main/log/result/" + result_file_name,'w') as f:
                        f.write(','.join(map(str,ret_test)))
                    best_epoch = epoch+1
                    best_mse = ret_test[1]
                    best_pearson = ret_test[2]
                    print(' rmse improved at epoch ', best_epoch, '; best_mse:', best_mse,model_st,dataset)
                else:
                    print(' no improvement since epoch ', best_epoch, '; best_mse, best pearson:', best_mse, best_pearson, model_st, dataset)
            draw_loss(train_losses, val_losses, "MTEGDRP-main/log/evaluation/" + loss_fig_name)
            draw_pearson(val_pearsons, "MTEGDRP-main/log/evaluation/" + pearson_fig_name)
            draw_loss(train_losses,test_losses, "MTEGDRP-main/log/evaluation/" + loss_fig_name)
            draw_pearson(test_pearsons, "MTEGDRP-main/log/evaluation/" + pearson_fig_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=0,     help='0: MTEGDRP')
    parser.add_argument('--train_batch', type=int, required=False, default=128,  help='Batch size training set')
    parser.add_argument('--val_batch', type=int, required=False, default=128, help='Batch size validation set')
    parser.add_argument('--test_batch', type=int, required=False, default=128, help='Batch size test set')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=200, help='Number of epoch')
    parser.add_argument('--log_interval', type=int, required=False, default=20, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default="cuda:0", help='Cuda')

    args = parser.parse_args()

    MTEGDRP=[MTEGDRP]
    modeling = MTEGDRP[args.model]
    model = [modeling]
    train_batch = args.train_batch
    val_batch = args.val_batch

    test_batch = args.test_batch
    lr = args.lr
    num_epoch = args.num_epoch
    log_interval = args.log_interval
    cuda_name = args.cuda_name

    main(model, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name)