# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:15:53 2024

@author: WANG Wenlong
@email: wenlongw@nus.edu.sg
"""

import numpy as np
import time
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../1 Model Training/')))
from PICNN_e_k1_training import PICNN_Model_PH1
from PICNN_e_k2_training import PICNN_Model_PH2
import multiprocessing

import pickle
from sklearn.neighbors import KDTree
from Solve_MIQP_PH2 import SolveMIQP
import win32com.client as winc

def dMPC1_subprocess(input_queue, output_queue, stop_event, ML_items):
    print('Loading dMPC1 solutions...', end='\t')
    with open("../4 mpQP Generation/dMPC1_Sols_PH2", "rb") as fp:
        Sols_MPC = pickle.load(fp)
    print('Done!')
    center_pt_dict, center_pt_list, kd_tree = ConstructKdTree('dMPC1', Sols_MPC)
    
    while not stop_event.is_set():
        xk, guess_u = input_queue.get() # Wait for input
        if guess_u is None:
            break
        else:
            MPC_items = [kd_tree, center_pt_list, center_pt_dict, Sols_MPC, xk, guess_u]
            MPC_items.extend(ML_items)
            best_u = MPC_get_sols('dMPC1', MPC_items)
        output_queue.put(best_u) # Send result to dMPC2 (via MAIN process)

def dMPC2_subprocess(input_queue, output_queue, stop_event, ML_items):
    print('Loading dMPC2 solutions...', end='\t')
    with open("../4 mpQP Generation/dMPC2_Sols_PH2", "rb") as fp:
        Sols_MPC = pickle.load(fp)
    print('Done!')
    center_pt_dict, center_pt_list, kd_tree = ConstructKdTree('dMPC2', Sols_MPC)
    
    while not stop_event.is_set():
        xk, guess_u = input_queue.get() # Wait for input
        if guess_u is None:
            break
        else:
            MPC_items = [kd_tree, center_pt_list, center_pt_dict, Sols_MPC, xk, guess_u]
            MPC_items.extend(ML_items)
            best_u = MPC_get_sols('dMPC2', MPC_items)
        output_queue.put(best_u) # Send result to dMPC2 (via MAIN process)
    
def MPC_get_sols(MODE, item):
    kd_tree, center_pt_list, center_pt_dict, Sols_MPC, xk, guess_u, model_list, scaler_list = item
    xk_uk_scaler, xk_uk_uk1_scaler, _, _ = scaler_list
    xk_ = xk_uk_scaler.transform([[xk[0], xk[1], 0, 0]])[0][:2]
    if MODE == 'dMPC1':
        temp_ = xk_uk_uk1_scaler.transform([[0, 0, 0, guess_u[0], 0, guess_u[1]]])[0]
        guess_u_ = [temp_[3], temp_[5]]
        theta_ = [xk_[0], xk_[1], guess_u_[0], guess_u_[1]]
        theta = [xk[0], xk[1], guess_u[0], guess_u[1]]
        sols_list = GetCandidateSols_dMPC('dMPC1', kd_tree, center_pt_list, center_pt_dict, Sols_MPC, theta_)
        best_uk_uk1, obj_value = SolveMIQP('dMPC1', theta, sols_list, model_list, scaler_list)
    elif MODE == 'dMPC2':
        temp_ = xk_uk_uk1_scaler.transform([[0, 0, guess_u[0], 0, guess_u[1], 0]])[0]
        guess_u_ = [temp_[2], temp_[4]]
        theta_ = [xk_[0], xk_[1], guess_u_[0], guess_u_[1]]
        theta = [xk[0], xk[1], guess_u[0], guess_u[1]]
        sols_list = GetCandidateSols_dMPC('dMPC2', kd_tree, center_pt_list, center_pt_dict, Sols_MPC, theta_)
        best_uk_uk1, obj_value = SolveMIQP('dMPC2', theta, sols_list, model_list, scaler_list)
        
    return best_uk_uk1

def ConstructKdTree(MODE, theta_dict):
    center_pt_dict = dict()
    center_pt_list = list()
    if MODE == 'dMPC1':
        for k, v in theta_dict.items():
            theta_bds_list = k.split('&')
            theta_bds_list = [float(item) for item in theta_bds_list]
            Lx1, Lx2, Lu2, Lu2_k1, Ux1, Ux2, Uu2, Uu2_k1 = theta_bds_list
            x1_m = (Lx1 + Ux1)/2
            x2_m = (Lx2 + Ux2)/2
            u2_m = (Lu2 + Uu2)/2
            u2_k1_m = (Lu2_k1 + Uu2_k1)/2
            center_pt_dict[f'{x1_m}&{x2_m}&{u2_m}&{u2_k1_m}'] = k
            center_pt_list.append([x1_m, x2_m, u2_m, u2_k1_m])
            
    elif MODE == 'dMPC2':
        for k, v in theta_dict.items():
            theta_bds_list = k.split('&')
            theta_bds_list = [float(item) for item in theta_bds_list]
            Lx1, Lx2, Lu1, Lu1_k1, Ux1, Ux2, Uu1, Uu1_k1 = theta_bds_list
            x1_m = (Lx1 + Ux1)/2
            x2_m = (Lx2 + Ux2)/2
            u1_m = (Lu1 + Uu1)/2
            u1_k1_m = (Lu1_k1 + Uu1_k1)/2
            center_pt_dict[f'{x1_m}&{x2_m}&{u1_m}&{u1_k1_m}'] = k
            center_pt_list.append([x1_m, x2_m, u1_m, u1_k1_m])
    kd_tree = KDTree(center_pt_list, metric='euclidean')
    return center_pt_dict, center_pt_list, kd_tree

def GetCandidateSols_dMPC(MODE, kd_tree, center_pt_list, center_pt_dict, Sols_MPC, theta):
    index_list = kd_tree.query_radius([theta], r=1)
    candidate_region_list = []
    if MODE == 'dMPC1':
        x1, x2, u2, u2_k1 = theta
        for i, idx in enumerate(index_list[0]):
            x1_m, x2_m, u2_m, u2_k1_m = center_pt_list[idx]
            x_bds = center_pt_dict[f'{x1_m}&{x2_m}&{u2_m}&{u2_k1_m}'].split('&')
            Lx1, Lx2, Lu2, Lu2_k1, Ux1, Ux2, Uu2, Uu2_k1 = [float(item) for item in x_bds]
            if Lx1 <= x1 <= Ux1 and Lx2 <= x2 <= Ux2  and\
               Lu2 <= u2 <= Uu2 and Lu2_k1 <= u2_k1 <= Uu2_k1:
                region_list = Sols_MPC[f'{Lx1}&{Lx2}&{Lu2}&{Lu2_k1}&{Ux1}&{Ux2}&{Uu2}&{Uu2_k1}']
                candidate_region_list.append(region_list)
                print(f'{MODE}: one region is found , id={i}, position={i/len(center_pt_dict)}')      
    elif MODE == 'dMPC2':
        x1, x2, u1, u1_k1 = theta
        for i, idx in enumerate(index_list[0]):
            x1_m, x2_m, u1_m, u1_k1_m = center_pt_list[idx]
            x_bds = center_pt_dict[f'{x1_m}&{x2_m}&{u1_m}&{u1_k1_m}'].split('&')
            Lx1, Lx2, Lu1, Lu1_k1, Ux1, Ux2, Uu1, Uu1_k1 = [float(item) for item in x_bds]
            if Lx1 <= x1 <= Ux1 and Lx2 <= x2 <= Ux2  and\
               Lu1 <= u1 <= Uu1 and Lu1_k1 <= u1_k1 <= Uu1_k1:
                region_list = Sols_MPC[f'{Lx1}&{Lx2}&{Lu1}&{Lu1_k1}&{Ux1}&{Ux2}&{Uu1}&{Uu1_k1}']
                candidate_region_list.append(region_list)
                print(f'{MODE}: one region is found , id={i}, position={i/len(center_pt_dict)}')   
                
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
        
    sols_list = []
    for item in final_candidate_region_list:
        sols_list.append(item[-1])
    return sols_list
                
def CalculateError(guess_u1, guess_u2, new_u1, new_u2):
    delta_u1_1 = 0.9*abs(guess_u1[0]-new_u1[0])/20
    delta_u1_2 = 0.1*abs(guess_u1[1]-new_u1[1])/20
    delta_u2_1 = 0.9*abs(guess_u2[0]-new_u2[0])/4
    delta_u2_2 = 0.1*abs(guess_u2[1]-new_u2[1])/4
    return [delta_u1_1, delta_u1_2, delta_u2_1, delta_u2_2]

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
    # Temperature setpoint in °C
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
    
    print('Loading scalers...', end='\t')
    xk_uk_scaler = np.load('../1 Model Training/PICNN_scaler/x_k_u_k_scaler.npy', allow_pickle=True).item()
    ek1_scaler = np.load('../1 Model Training/PICNN_scaler/e_k1_scaler.npy', allow_pickle=True).item()
    xk_uk_uk1_scaler = np.load('../1 Model Training/PICNN_scaler/x_k_u_k_u_k1_scaler.npy', allow_pickle=True).item()
    ek2_scaler = np.load('../1 Model Training/PICNN_scaler/e_k2_scaler.npy', allow_pickle=True).item()
    scaler_list = [xk_uk_scaler, xk_uk_uk1_scaler, ek1_scaler, ek2_scaler]
    print('Done!')
    
    ML_items = [model_list, scaler_list]

    # Queues for communication
    MAIN_to_dMPC1 = multiprocessing.Queue()
    MAIN_to_dMPC2 = multiprocessing.Queue()
    dMPC1_to_MAIN = multiprocessing.Queue()
    dMPC2_to_MAIN = multiprocessing.Queue()
    # Event to signal termination
    stop_event = multiprocessing.Event()
    # Start subprocesses
    dMPC1 = multiprocessing.Process(target=dMPC1_subprocess, args=(MAIN_to_dMPC1, dMPC1_to_MAIN, stop_event, ML_items))
    dMPC2 = multiprocessing.Process(target=dMPC2_subprocess, args=(MAIN_to_dMPC2, dMPC2_to_MAIN, stop_event, ML_items))
    dMPC1.start()
    dMPC2.start()
    
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

    for step_cnt in range(1, sim_step+1):
        print(f'Current system state: {xk}')
        guess_u1 = [0, 0]
        guess_u2 = [0, 0]
        iter_cnt = 0
        time1 = time.time()
        while True:
            #print(f'STEP ID: {step_cnt}\tCurrent u: u1={guess_u1}\t u2={guess_u2}')
            iter_cnt += 1

            MAIN_to_dMPC1.put([xk, guess_u2])  # Send initial input to dMPC1
            MAIN_to_dMPC2.put([xk, guess_u1])  # Send initial input to dMPC2
            new_u1 = dMPC1_to_MAIN.get()
            new_u2 = dMPC2_to_MAIN.get()
            err_list = CalculateError(guess_u1, guess_u2, new_u1, new_u2)
            err = np.mean(np.array(err_list))
            print(f'average error: {err}')
            if err <= 0.1 or iter_cnt >= 5:
                break
            else:
                guess_u2 = new_u2
                guess_u1 = new_u1
        time2 = time.time()
        print(f'STEP ID: {step_cnt}\t elapsed time: {time2-time1}')
        best_uk =[new_u1[0], new_u2[0]]
        final_uk = ReviseU(best_uk, u_list[-1])
        u_list.append(final_uk)
        xk = FP_next_state(final_uk)
        print('\n')
        
    stop_event.set()
    MAIN_to_dMPC1.put([0, None])  # Send termination signal
    MAIN_to_dMPC2.put([0, None])  # Send termination signal