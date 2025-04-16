# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:00:09 2024

@author: WANG Wenlong
@email: wenlongw@nus.edu.sg
"""


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def LoadOpenLoopData():
    data_list = np.load('../0 Open-loop Simulation/e_k2_data.npy')
    dev_data_list = [] #[x_k_1, x_k_2, u_k_1, u_k_2, u_k1_1, u_k1_2, e_k2_1, e_k2_2]
    for item in data_list:
        x_k_1 = (item[0] - 0.832637)*100
        x_k_2 = item[1] - 199.343
        u_k_1 = item[2] - 120
        u_k_2 = item[3] - 3.5
        u_k1_1 = item[4] - 120
        u_k1_2 = item[5] - 3.5
        e_k2_1 = abs(item[6] - 0.832637)*100 #absolute value of x_k2_1
        e_k2_2 = abs(item[7] - 199.343) #absolute value of x_k2_1
        dev_data_list.append([x_k_1, x_k_2, u_k_1, u_k_2, u_k1_1, u_k1_2, e_k2_1, e_k2_2])
    return np.array(dev_data_list)

def ProcessSimData(trj_data):
    x_k_u_k_u_k1_list = trj_data[:, :6]
    e_k2_list = trj_data[:, 6:]
    x_k_u_k_u_k1_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(x_k_u_k_u_k1_list)
    e_k2_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(e_k2_list)
    np.save('./PICNN_scaler/x_k_u_k_u_k1_scaler.npy', x_k_u_k_u_k1_scaler)
    np.save('./PICNN_scaler/e_k2_scaler.npy', e_k2_scaler)
    x_k_u_k_u_k1_list_scaled = x_k_u_k_u_k1_scaler.transform(x_k_u_k_u_k1_list)
    e_k2_list_scaled = e_k2_scaler.transform(e_k2_list)
    trj_data_scaled = np.column_stack((x_k_u_k_u_k1_list_scaled, e_k2_list_scaled))
    return trj_data_scaled 

class NonNegWeightConstraint(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module,'weight'):
            w = module.weight.data.clamp(min=0.0)
            module.weight.data = w
            
class SimDataset(Dataset):
    def __init__(self, trj_data):
        trj_data = np.expand_dims(trj_data, axis=1)
        self.x_k = trj_data[:,:,:2]
        self.u_k_u_k1 = trj_data[:,:,2:6]
        self.e_k2 = trj_data[:,:,6:]

    def __len__(self):
        return int(len(self.x_k))

    def __getitem__(self, index):
        return self.x_k[index], self.u_k_u_k1[index], self.e_k2[index]
    
class PICNN_Model_PH2(nn.Module):
    def __init__(self):
        super(PICNN_Model_PH2, self).__init__()
        input_dim_x = 2 #non-convex input dim    
        input_dim_u = 4 #convex input dim
        output_dim_h = 2 #output dim: e_k2
        
        self.W1h_u = nn.Linear(12, 24)
        self.W2h_u = nn.Linear(24, 48)
        self.W3h_u = nn.Linear(48, 12)
        self.W4h_u = nn.Linear(12, output_dim_h)
        self.W0u = nn.Linear(input_dim_u, 12)
        self.W1u = nn.Linear(input_dim_u, 24)
        self.W2u = nn.Linear(input_dim_u, 48)
        self.W3u = nn.Linear(input_dim_u, 12)
        self.W4u = nn.Linear(input_dim_u, output_dim_h)
        self.W0h_x = nn.Linear(input_dim_x, 12)
        self.W1h_x = nn.Linear(12, 24)
        self.W2h_x = nn.Linear(24, 48)
        self.W3h_x = nn.Linear(48, 12)
        self.W1xu = nn.Linear(12, 24)     
        self.W2xu = nn.Linear(24, 48)     
        self.W3xu = nn.Linear(48, 12)     
        self.W4xu = nn.Linear(12, output_dim_h)
        self.relu = nn.ReLU()
        
    def forward(self, x, u):
        #x: non-convex; u: convex
        h_u = self.relu(0 + 0 + self.W0u(u))
        h_x = self.relu(self.W0h_x(x))
        h_u = self.relu(self.W1xu(h_x*h_u) + self.W1h_u(h_u) + self.W1u(u))
        h_x = self.relu(self.W1h_x(h_x))
        h_u = self.relu(self.W2xu(h_x*h_u) + self.W2h_u(h_u) + self.W2u(u))
        h_x = self.relu(self.W2h_x(h_x))
        h_u = self.relu(self.W3xu(h_x*h_u) + self.W3h_u(h_u) + self.W3u(u))
        h_x = self.relu(self.W3h_x(h_x))
        h_u = self.relu(self.W4xu(h_x*h_u) + self.W4h_u(h_u) + self.W4u(u))
        return h_u
    
def model_evaluation(model, loss_fn, dl):
    output_list = []
    model.eval()
    for i, (x_k, u_k_u_k1, e_k2) in enumerate(dl):
        x_k = x_k.to(torch.float32).to(device)
        u_k_u_k1 = u_k_u_k1.to(torch.float32).to(device)
        e_k2 = e_k2.to(torch.float32).to(device)
        e_k2_pred = model(x_k, u_k_u_k1)
        loss = loss_fn(e_k2, e_k2_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output_list.append(loss.item())
    loss_ave = np.mean(np.array(output_list))
    return loss_ave

def plot_loss(loss_list):
    epochs = list(range(1, 1+len(loss_list)))
    plt.plot(epochs, loss_list)
    plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    #data preparation
    trj_data_original = LoadOpenLoopData()
    trj_data_scaled = ProcessSimData(trj_data_original)
    trj_data_train, trj_data_valid = train_test_split(trj_data_scaled, test_size=0.05, random_state=123)
    trj_data_train, trj_data_test = train_test_split(trj_data_train, test_size=0.0526, random_state=456)
    train_ds = SimDataset(trj_data_train)
    valid_ds = SimDataset(trj_data_valid)
    test_ds = SimDataset(trj_data_test)
    train_dl = DataLoader(train_ds, batch_size=2**11, shuffle=True, drop_last=False)
    valid_dl = DataLoader(valid_ds, batch_size=2**3, shuffle=False, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=2**3, shuffle=False, drop_last=False)
    
    #ML model setup
    model = PICNN_Model_PH2().to(device)
    NonNeg = NonNegWeightConstraint()
    loss_fn = nn.MSELoss().to(device)
    optimizer = Adam(model.parameters(), lr = 1e-3)
    loss_list = []
    epochs = 2000
    
    #ML model training
    for epoch in range(epochs):
        model.train()
        output_list = []
        for i, (x_k, u_k_u_k1, e_k2) in enumerate(tqdm(train_dl)):
            x_k = x_k.to(torch.float32).to(device)
            u_k_u_k1 = u_k_u_k1.to(torch.float32).to(device)
            e_k2 = e_k2.to(torch.float32).to(device)
            e_k2_pred = model(x_k, u_k_u_k1)
            
            #update the model parameters
            loss = loss_fn(e_k2, e_k2_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #ensure non-negative weight
            model._modules['W1h_u'].apply(NonNeg)
            model._modules['W2h_u'].apply(NonNeg)
            model._modules['W3h_u'].apply(NonNeg)
            model._modules['W4h_u'].apply(NonNeg)
            model._modules['W1xu'].apply(NonNeg)
            model._modules['W2xu'].apply(NonNeg)
            model._modules['W3xu'].apply(NonNeg)
            model._modules['W4xu'].apply(NonNeg)
            
            output_list.append(loss.item())
        loss_ave = np.mean(np.array(output_list))
        loss_list.append(loss_ave)
        if (epoch+1) % 10 == 0:
            tqdm.write(f'epoch: {epoch+1}\tloss_ave: {loss_ave}')
            valid_MSE = model_evaluation(model, loss_fn, valid_dl)
            tqdm.write(f'\nvalid_MSE: {valid_MSE}')
            
    #evaluate ML model performance
    train_MSE = model_evaluation(model, loss_fn, train_dl)
    test_MSE = model_evaluation(model, loss_fn, test_dl)
    valid_MSE = model_evaluation(model, loss_fn, valid_dl)
    print(f'train_MSE: {train_MSE}')
    print(f'valid_MSE: {valid_MSE}')
    print(f'test_MSE: {test_MSE}')
    torch.save(model, 'PICNN_e_k2.pkl')
    plot_loss(loss_list)
