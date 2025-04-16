# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:15:53 2024

@author: WANG Wenlong
@email: wenlongw@nus.edu.sg
"""

import numpy as np
import time
import torch
import warnings
warnings.filterwarnings('ignore', category=Warning)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../1 Model Training/')))
from PICNN_e_k1_training import PICNN_Model_PH1
from PICNN_e_k2_training import PICNN_Model_PH2
import pickle
from sklearn.neighbors import KDTree
from Solve_MIQP_PH2 import SolveMIQP
import win32com.client as winc

def LoadData():
    grid_xk1_list1 = np.load('../2 Space Discretization/Grid_Files/Grid_ek1_1.npy')
    K_xk1_list1 = np.load('../2 Space Discretization/Grid_Files/K_ek1_1.npy')
    grid_xk1_list2 = np.load('../2 Space Discretization/Grid_Files/Grid_ek1_2.npy')
    K_xk1_list2 = np.load('../2 Space Discretization/Grid_Files/K_ek1_2.npy')
    grid_xk2_list1 = np.load('../2 Space Discretization/Grid_Files/Grid_ek1_1.npy')
    K_xk2_list1 = np.load('../2 Space Discretization/Grid_Files/K_ek1_1.npy')
    grid_xk2_list2 = np.load('../2 Space Discretization/Grid_Files/Grid_ek1_2.npy')
    K_xk2_list2 = np.load('../2 Space Discretization/Grid_Files/K_ek1_2.npy')

    all_grid_list = [grid_xk1_list1, grid_xk1_list2, grid_xk2_list1, grid_xk2_list2]
    all_K_list = [K_xk1_list1, K_xk1_list2, K_xk2_list1, K_xk2_list2]
    return all_grid_list, all_K_list

    
def ConstructXKdTree(Sols_MPC):
    '''
    center_pt_dict = {
                     center1_x1&center1_x2: Lx1&Lx2&Ux1&Ux2,
                     center2_x1&center2_x2: Lx1&Lx2&Ux1&Ux2,
                     center3_x1&center3_x2: Lx1&Lx2&Ux1&Ux2,
                     ...
        }
    '''
    center_pt_dict = dict()
    center_pt_list = list()
    
    for k, v in Sols_MPC.items():
        x_bds_list = k.split('&')
        x_bds_list = [float(item) for item in x_bds_list]
        Lx1, Lx2, Ux1, Ux2 = x_bds_list
        x1_m = (Lx1 + Ux1)/2
        x2_m = (Lx2 + Ux2)/2
        center_pt_dict[f'{x1_m}&{x2_m}'] = k
        center_pt_list.append([x1_m, x2_m])
    xkd_tree = KDTree(center_pt_list, metric='euclidean')
    return center_pt_dict, center_pt_list, xkd_tree

def GetCandidateSols(kd_tree, Sols_MPC, x_pts):
    x1 = x_pts[0]
    x2 = x_pts[1]

    index_list = kd_tree.query_radius([[x1, x2]], r=0.707)
    candidate_region_list = []
    
    for i, idx in enumerate(index_list[0]):
        x1_m, x2_m = center_pt_list[idx]
        x_bds = center_pt_dict[f'{x1_m}&{x2_m}'].split('&')
        Lx1, Lx2, Ux1, Ux2 = [float(item) for item in x_bds]
        if Lx1 <= x1 <= Ux1 and Lx2 <= x2 <= Ux2:
            region_list = Sols_MPC[f'{Lx1}&{Lx2}&{Ux1}&{Ux2}']
            candidate_region_list.append(region_list)
            print(f'one region is found, id={i}, position={i/len(center_pt_dict)}')
            
    if len(candidate_region_list) > 1: # remove redundant candidate regions
        final_candidate_region_list = []
        known_d_list = []
        for candidate_region in candidate_region_list:
            d = candidate_region[0][0][4] - candidate_region[0][0][0]
            if d not in known_d_list:
                known_d_list.append(d)
                final_candidate_region_list.extend(candidate_region)
    else:
        final_candidate_region_list = candidate_region_list[0]
        
    region_list = []
    for item in final_candidate_region_list:
        region_list.append(item[-1])
    return region_list

def SetUpAspenDynamics(filename):
    aspen = winc.DispatchEx('AD Application')
    aspen.Visible = True
    aspen.activate()
    aspen.OpenDocument(os.path.abspath(filename))
    sim = aspen.Simulation
    return sim
 
def SetFLASH_Q(new_devi_Q):
    print(f'Change the heat duty of FLASH to {new_devi_Q + 3.5} GJ/hr')
    blocks('Heater').QR.Value = new_devi_Q + 3.5

def SetCSTR2_T(new_devi_T):
    print(f'Change the temperature setpoint of CSTR2 to {new_devi_T + 120}°C')
    blocks("CSTR-2_TC").SPRemote.Value = new_devi_T + 120

def ReadStream(streams, stream_name, substance_name):
    return (streams(stream_name).Zn(substance_name).Value - 0.832637)*100
    
def ReadFLASH_T(blocks, block_name):
    return blocks(block_name).T.Value - 199.343

def ApplyControlActions(uk):
    SetCSTR2_T(uk[0])
    SetFLASH_Q(uk[1])
  
def FP_next_state(best_uk):
    global EndTime
    ApplyControlActions(best_uk)
    print('Running the simulation for one sampling period...')
    EndTime += time_step
    sim.EndTime = EndTime
    sim.run(True)
    xk1_1 = ReadStream(streams, 'PRO-OUT', 'EB')
    xk1_2 = ReadFLASH_T(blocks, 'FLASH')
    return [xk1_1, xk1_2]

def InitialDeviation(u_init):
    global EndTime, sim
    print('1 hr steady-state evolution...', end='\t')
    init_runtime0 = 1
    EndTime += init_runtime0
    sim.EndTime = EndTime
    sim.run(True)
    print('Done!')

    print('9 hrs initial deviation...', end='\t')
    init_runtime1 = 9
    blocks("CSTR-2_TC").SPRemote.Value = u_init[0] + 120 #110~130
    blocks('Heater').QR.Value = u_init[1] + 3.5 #1.5~5.5
    EndTime += init_runtime1
    sim.EndTime = EndTime
    sim.run(True)
    print('Done!')
            

def ReviseU(u_new, u_old):
    def clamp(new, old, bound):
        delta = new - old
        if abs(delta) <= bound:
            return new
        return old + bound * (1 if delta > 0 else -1)
    
    T_bound = 3
    Q_bound = 0.5
    
    T_new, Q_new = u_new
    T_old, Q_old = u_old
    
    T_final = clamp(T_new, T_old, T_bound)
    Q_final = clamp(Q_new, Q_old, Q_bound)
    
    return [T_final, Q_final]

if __name__ == "__main__":
    print('Loading ML models...', end='\t')
    model1 = torch.load('../1 Model Training/PICNN_e_k1.pkl', map_location=torch.device('cpu'))
    model2 = torch.load('../1 Model Training/PICNN_e_k2.pkl', map_location=torch.device('cpu'))
    model_list = [model1, model2]
    print('Done!')
    
    print('Loading solutions...', end='\t')
    with open("../4 mpQP Generation/cMPC_Sols_PH2", "rb") as fp:
        Sols_MPC = pickle.load(fp)
    print('Done!')
    
    print('Loading scalers...', end='\t')
    xk_uk_scaler = np.load('../1 Model Training/PICNN_scaler/x_k_u_k_scaler.npy', allow_pickle=True).item()
    xk_uk_uk1_scaler = np.load('../1 Model Training/PICNN_scaler/x_k_u_k_u_k1_scaler.npy', allow_pickle=True).item()
    ek1_scaler = np.load('../1 Model Training/PICNN_scaler/e_k1_scaler.npy', allow_pickle=True).item()
    ek2_scaler = np.load('../1 Model Training/PICNN_scaler/e_k2_scaler.npy', allow_pickle=True).item()
    scaler_list = [xk_uk_scaler, xk_uk_uk1_scaler, ek1_scaler, ek2_scaler]
    print('Done!')
    
    #construct k-d tree
    center_pt_dict, center_pt_list, xkd_tree = ConstructXKdTree(Sols_MPC)
    # Load aspen dynamics file
    sim = SetUpAspenDynamics('AspenSimulationFile/2CSTR-1Flash.dynf')
    sim.RunMode = 'Dynamic'
    # Create handlers
    fsheet = sim.Flowsheet
    streams = fsheet.Streams
    blocks = fsheet.Blocks
    time_step = 2
    sim_step = 15
    EndTime = 0
    
    # u_init = [-8, 1.9] #initial condition 1
    u_init = [-9, -1.9] #initial condition 2
    InitialDeviation(u_init)
    u_list = [u_init]

    x_1_init = ReadStream(streams, 'PRO-OUT', 'EB')
    x_2_init = ReadFLASH_T(blocks, 'FLASH')
    
    xk = np.array([x_1_init, x_2_init])
    xk_ = xk_uk_scaler.transform([[xk[0], xk[1], 0, 0]])[0][:2]

    for iter_cnt in range(1, sim_step+1):
        print(f'Current system state: {xk}')
        time1 = time.time()
        sols_list = GetCandidateSols(xkd_tree, Sols_MPC, xk_)
        best_uk, obj_value = SolveMIQP('cMPC', xk, sols_list, model_list, scaler_list)
        time2 = time.time()
        print(f'STEP ID: {iter_cnt}\t elapsed time: {time2-time1}')
        final_uk = ReviseU(best_uk, u_list[-1])
        u_list.append(final_uk)
        xk = FP_next_state(final_uk)
        xk_ = xk_uk_scaler.transform([[xk[0], xk[1], 0, 0]])[0][:2]
        print('\n')
    