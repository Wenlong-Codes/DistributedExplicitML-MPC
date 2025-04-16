# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 16:02:58 2023

@author: WANG Wenlong
@email: wenlongw@nus.edu.sg
"""

import numpy as np
from tqdm import tqdm
import sys
import os
import torch
from PICNN_e_k1_training import PICNN_Model_PH1
import itertools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../1 Model Training/')))

def PICNN_pred(input_xk, input_uk):
    global model, OUTPUT_ID
    input_xk = np.array(input_xk).reshape(-1, 1, 2)
    input_uk = np.array(input_uk).reshape(-1, 1, 2)
    input_xk = torch.from_numpy(input_xk).to(torch.float32)
    input_uk = torch.from_numpy(input_uk).to(torch.float32)
    output_e_k1 = model(input_xk, input_uk)
    return output_e_k1.cpu().data.numpy()[:,0][:, OUTPUT_ID] 

def generate_data(bds_x1, bds_x2, bds_u1, bds_u2):
    (L_x1, U_x1) = bds_x1
    (L_x2, U_x2) = bds_x2
    (L_u1, U_u1) = bds_u1
    (L_u2, U_u2) = bds_u2
    cnt_pts = 2
    delta = (U_x1-L_x1)/cnt_pts
    x1_step = np.linspace(L_x1, U_x1, cnt_pts, endpoint=False)
    x2_step = np.linspace(L_x2, U_x2, cnt_pts, endpoint=False)
    u1_step = np.linspace(L_u1, U_u1, cnt_pts, endpoint=False)
    u2_step = np.linspace(L_u2, U_u2, cnt_pts, endpoint=False)
    
    K_list, grid_pts_list = [], []
    for x1 in x1_step:
        for x2 in x2_step:
            for u1 in u1_step:
                for u2 in u2_step:
                    l0 = (x1, x2, u1, u2)
                    l1 = (x1+delta, x2, u1, u2)
                    l2 = (x1, x2+delta, u1, u2)
                    l3 = (x1, x2, u1+delta, u2)
                    l4 = (x1, x2, u1, u2+delta)
                    grid_pts_list.append([l0, l1, l2, l3, l4])
                    
    grid_pts_list = np.array(grid_pts_list)
    for grid_pts in grid_pts_list:
        input_xk = []
        input_uk = []
        for (x1, x2, u1, u2) in grid_pts:
            input_xk.append([x1, x2])
            input_uk.append([u1, u2])
        value = PICNN_pred(input_xk, input_uk)

        #solve a system of linear equations to obtain coefficients
        W = np.column_stack((grid_pts, np.ones(grid_pts.shape[0]).reshape(-1,1)))
        W_inv = np.linalg.inv(W)
        K_list.append(np.dot(W_inv,value))
    return K_list, grid_pts_list

def grid_refine(await_refine_grid_pts):
    delta = await_refine_grid_pts[1][0]-await_refine_grid_pts[0][0]
    ref_x1 = await_refine_grid_pts[0][0]
    ref_x2 = await_refine_grid_pts[0][1]
    ref_u1 = await_refine_grid_pts[0][2]
    ref_u2 = await_refine_grid_pts[0][3]
    refine_K_list, refine_grid_pts_list = generate_data((ref_x1, ref_x1+delta), (ref_x2, ref_x2+delta), (ref_u1, ref_u1+delta), (ref_u2, ref_u2+delta))
    return refine_K_list, refine_grid_pts_list

def IsSpecialGrid(grid_pts):
    delta = grid_pts[1][0] - grid_pts[0][0] 
    A = [grid_pts[0][0], grid_pts[0][0]+delta]
    B = [grid_pts[0][1], grid_pts[0][1]+delta]
    C = [grid_pts[0][2], grid_pts[0][2]+delta]
    D = [grid_pts[0][3], grid_pts[0][3]+delta]
    E = list(itertools.product(A, B, C, D))
    for item in E:
        if all(x == 0 for x in item):
            return True
    return False
    
def screen_grid(K, grid_pts):
    global min_delta, special_min_delta, OUTPUT_ID
    delta = grid_pts[1][0] - grid_pts[0][0] 
    ref_x1 = grid_pts[0][0]
    ref_x2 = grid_pts[0][1]
    ref_u1 = grid_pts[0][2]
    ref_u2 = grid_pts[0][3]
    delta = round(delta, 5)
    
    if delta <= min_delta:
        #check if the current region is near the origin
        if IsSpecialGrid(grid_pts) and delta > special_min_delta:
            return True
        return False
    
    #the number of samples used for evaluating approxiamtion error
    #increase this number will improve approximation process but also lead to heavier computation burden
    pt_cnt = 11
    x1_step = np.linspace(ref_x1, ref_x1+delta, pt_cnt, endpoint=True)  #get x1 sampling points
    x2_step = np.linspace(ref_x2, ref_x2+delta, pt_cnt, endpoint=True)  #get x2 sampling points
    u1_step = np.linspace(ref_u1, ref_u1+delta, pt_cnt, endpoint=True)  #get u1 sampling points
    u2_step = np.linspace(ref_u2, ref_u2+delta, pt_cnt, endpoint=True)  #get u2 sampling points
    k, b = K[:-1], K[-1] #get coefficients of affine function

    true_value_list, est_value_list = [], []
    input_xk = []
    input_uk = []

    for x1 in x1_step:
        for x2 in x2_step:
            for u1 in u1_step:
                for u2 in u2_step:
                    input_xk.append([x1, x2])
                    input_uk.append([u1, u2])
                    est_value_list.append(np.array([x1, x2, u1, u2])@k+b)
    
    true_value_list = PICNN_pred(input_xk, input_uk)
    est_value_list = np.array(est_value_list)
    
    # calculate relative percentage error
    ave_err = CalculateError(true_value_list, est_value_list)
    
    if ave_err >= 5:
        return True
    else:
        return False

def CalculateError(true_value_list, est_value_list):
    true_value_list = np.array(true_value_list)
    est_value_list = np.array(est_value_list)
    
    # calculate relative percentage error
    with np.errstate(divide='ignore', invalid='ignore'):
        err_list = np.abs((est_value_list - true_value_list) / true_value_list) * 100
        err_list[true_value_list == 0] = np.inf
    ave_err = np.mean(err_list)
    return ave_err

def find_pt_area(pt, overall_grid_pts_list):
    pt_x1, pt_x2, pt_u1, pt_u2 = pt
    for i, grid_pts in enumerate(overall_grid_pts_list):
        delta = grid_pts[1][0]-grid_pts[0][0] 
        L_x1 = grid_pts[0][0] #get reference x1 point of the grid
        L_x2 = grid_pts[0][1] #get reference x2 point of the grid
        L_u1 = grid_pts[0][2] #get reference u1 point of the grid
        L_u2 = grid_pts[0][3] #get reference u2 point of the grid
        U_x1 = L_x1+delta
        U_x2 = L_x2+delta
        U_u1 = L_u1+delta
        U_u2 = L_u2+delta
        arr_bds = np.array([U_x1, -L_x1, U_x2, -L_x2, U_u1, -L_u1, U_u2, -L_u2]).reshape(-1,1)
        arr_coef = np.array([[1, 0, 0, 0],
                              [-1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, -1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, 1],
                              [0, 0, 0, -1]])
        arr_pt = np.array([pt_x1, pt_x2, pt_u1, pt_u2]).reshape(-1, 1)
        if all(arr_coef@arr_pt<=arr_bds):
            return i, [L_x1, U_x1, L_x2, U_x2, L_u1, U_u1, L_u2, U_u2]

def check_point(pt, overall_K_list, overall_grid_pts_list):
    pt_x1, pt_x2, pt_u1, pt_u2 = pt
    area_id, bds = find_pt_area(pt, overall_grid_pts_list)
    (coef_x1, coef_x2, coef_u1, coef_u2, coef_c) = overall_K_list[area_id]
    print(f'For point ({pt_x1}, {pt_x2}, {pt_u1}, {pt_u2}), it is located in the area of id:{area_id+1} (total: {len(overall_K_list)})')
    print(f'Corresponding affine function: ({coef_x1:.5}*x1)+({coef_x2:.5}*x2)+({coef_u1:.5}*u1)+({coef_u2:.5}*u2)+({coef_c:.5})')
    true_value = PICNN_pred([pt_x1, pt_x2], [pt_u1, pt_u2])[0]
    est_value = coef_x1*pt_x1 + coef_x2*pt_x2 + coef_u1*pt_u1 + coef_u2*pt_u2 + coef_c
    
    # calculate relative percentage error
    rel_err = CalculateError([true_value], [est_value])
    print(f'true_value\t:{true_value}\nest_value\t:{est_value}\nrelative_err:{rel_err:.4f}%')
    return [coef_x1, coef_x2, coef_u1, coef_u2, coef_c], bds

def discretization(K_list, grid_pts_list):
    overall_K_list, overall_grid_pts_list = [], []
    loop_cnt = 1
    while True:
        need_refine_K, need_refine_grid_pts = [], []
        for i in tqdm(range(len(grid_pts_list)),desc=f'Round {loop_cnt}: screening new area\t\t', file=sys.stdout):
            if screen_grid(K_list[i], grid_pts_list[i]):
                need_refine_grid_pts.append(grid_pts_list[i])
                need_refine_K.append(K_list[i])
            else:
                overall_grid_pts_list.append(grid_pts_list[i])
                overall_K_list.append(K_list[i])
                
        if len(need_refine_K) == 0:
            tqdm.write(f'All regions are identified! The number of regions: {len(overall_K_list)}')
            break
        else:
            K_list, grid_pts_list = [], []
            for need_refine_grid in tqdm(need_refine_grid_pts, desc=f'Round {loop_cnt}: discretizing known area', file=sys.stdout):
                temp_k, temp_grid = grid_refine(need_refine_grid)
                K_list.extend(temp_k)
                grid_pts_list.extend(temp_grid)
        loop_cnt += 1
    return overall_K_list , overall_grid_pts_list

if __name__ == "__main__":
    #load trained ML model
    model = torch.load('../1 Model Training/PICNN_e_k1.pkl', map_location=torch.device('cpu'))
    min_delta = round(2/(2*2*2), 8) #prevent over-discretization
    special_min_delta = round(2/(2*2*2*2), 8) #adopt finer grids near origin
    
    #the input space of the ML model
    l_bds = -1 
    u_bds = 1 
    
    # 0: the first output of the ML model, i.e., e_k1_1
    # 1: the second output of the ML model, i.e., e_k1_2
    OUTPUT_ID = 1
    
    LOAD = 0
    if LOAD:
        overall_K_list = np.load(f'./Grid_Files/K_ek1_{OUTPUT_ID+1}.npy')
        overall_grid_pts_list = np.load(f'./Grid_Files/Grid_ek1_{OUTPUT_ID+1}.npy')
    else:
        #start the linearization process
        K_list, grid_pts_list = generate_data((l_bds, u_bds), (l_bds, u_bds), (l_bds, u_bds), (l_bds, u_bds))
        overall_K_list, overall_grid_pts_list = discretization(K_list, grid_pts_list)
        np.save(f'./Grid_Files/Grid_ek1_{OUTPUT_ID+1}.npy', overall_grid_pts_list)
        np.save(f'./Grid_Files/K_ek1_{OUTPUT_ID+1}.npy', overall_K_list)
      
    # evluate the approximation error at a point
    pt = [0.2, 0.9, 0.4, 0.7] #x1, x2, u1, u2
    coef_list, bds = check_point(pt, overall_K_list, overall_grid_pts_list)