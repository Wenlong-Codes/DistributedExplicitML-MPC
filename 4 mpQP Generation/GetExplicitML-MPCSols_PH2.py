# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:15:46 2023

@author: WANG Wenlong
@email: wenlongw@nus.edu.sg
"""

import numpy as np
from ppopt.mpqp_program import MPQP_Program
from ppopt.mp_solvers.solve_mpqp import solve_mpqp, mpqp_algorithm
from tqdm import tqdm
from multiprocessing import Pool, RLock
import sys
import pickle
import itertools

MODE_POOL = {'cMPC', 'dMPC1', 'dMPC2'}

def Solve_MPQP(MODE, M, N, A, B, C, D, E, F, G, bds_x1, bds_x2, bds_u1, bds_u2, bds_u1_k1, bds_u2_k1):
    L_x1, U_x1 = bds_x1
    L_x2, U_x2 = bds_x2
    L_u1, U_u1 = bds_u1
    L_u2, U_u2 = bds_u2
    L_u1_k1, U_u1_k1 = bds_u1_k1
    L_u2_k1, U_u2_k1 = bds_u2_k1
    
    M1_12 = np.column_stack((E.T@M@E + B.T@M@B + N, E.T@M@F))
    M1_34 = np.column_stack((F.T@M@E, F.T@M@F + N))
    M1 = np.row_stack((M1_12, M1_34))
    
    M2 = D.T@M@D + A.T@M@A
    M3 = np.column_stack((2*(D.T@M@E + A.T@M@B), 2*D.T@M@F))
    M4 = np.column_stack((2*(G.T@M@E + C.T@M@B), 2*G.T@M@F))
    M5 = 2*(G.T@M@D + C.T@M@A)
    M6 = G.T@M@G + C.T@M@C
    
    if MODE == 'cMPC':
        Q = 2*M1
        Q_t = M2
        H_t = M3
        c = M4
        c_t = M5
        c_c = M6
        A_ = np.array([[1, 0, 0, 0],
                       [-1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 0, -1]])
        
        b = np.array([[U_u1],
                      [-L_u1],
                      [U_u2],
                      [-L_u2],
                      [U_u1_k1],
                      [-L_u1_k1],
                      [U_u2_k1],
                      [-L_u2_k1]])
                      
        F_ = np.array([[0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 0]])
        
        A_t = np.array([[1, 0],
                        [-1,0],
                        [0, 1],
                        [0,-1]])
        
        b_t = np.array([[U_x1],
                        [-L_x1],
                        [U_x2],
                        [-L_x2]])
    
    elif MODE == 'dMPC1':
        S1 = np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0]])
        
        S2 = np.array([[0, 1, 0, 0],
                       [0, 0, 0, 1]])
        
        P1 = S1@M1@S1.T
        
        P2_12 = np.column_stack((M2, 0.5*M3@S2.T))
        P2_34 = np.column_stack((0.5*(M3@S2.T).T, S2@M1@S2.T))
        P2 = np.row_stack((P2_12, P2_34))
        
        P3 = np.row_stack((M3@S1.T, 2*S2@M1@S1.T))
        P4 = M4@S1.T
        P5 = np.column_stack((M5, M4@S2.T))   
        P6 = M6
        
        Q = 2*P1
        Q_t = P2
        H_t = P3
        c = P4
        c_t = P5
        c_c = P6
        
        
        A_ = np.array([[1, 0],
                       [-1,0],
                       [0, 1],
                       [0,-1]])
        
        b = np.array([[U_u1],
                      [-L_u1],
                      [U_u1_k1],
                      [-L_u1_k1]])
                      
        F_ = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]])
        
        A_t = np.array([[1, 0, 0, 0],
                        [-1,0, 0, 0],
                        [0, 1, 0, 0],
                        [0,-1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0,-1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0,-1]])
        
        b_t = np.array([[U_x1],
                        [-L_x1],
                        [U_x2],
                        [-L_x2],
                        [U_u2],
                        [-L_u2],
                        [U_u2_k1],
                        [-L_u2_k1]])
        
    elif MODE == 'dMPC2':
        S1 = np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0]])
        
        S2 = np.array([[0, 1, 0, 0],
                       [0, 0, 0, 1]])
    
    
        P1 = S2@M1@S2.T
        
        P2_12 = np.column_stack((M2, 0.5*M3@S1.T))
        P2_34 = np.column_stack((0.5*(M3@S1.T).T, S1@M1@S1.T))
        P2 = np.row_stack((P2_12, P2_34))
        
        P3 = np.row_stack((M3@S2.T, 2*S1@M1@S2.T))    
        P4 = M4@S2.T
        P5 = np.column_stack((M5, M4@S1.T))   
        P6 = M6
        
        Q = 2*P1
        Q_t = P2
        H_t = P3
        c = P4
        c_t = P5
        c_c = P6

        A_ = np.array([[1, 0],
                       [-1,0],
                       [0, 1],
                       [0,-1]])
        
        b = np.array([[U_u2],
                      [-L_u2],
                      [U_u2_k1],
                      [-L_u2_k1]])
                      
        F_ = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]])
    
        
        A_t = np.array([[1, 0, 0, 0],
                        [-1,0, 0, 0],
                        [0, 1, 0, 0],
                        [0,-1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0,-1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0,-1]])
        
        b_t = np.array([[U_x1],
                        [-L_x1],
                        [U_x2],
                        [-L_x2],
                        [U_u1],
                        [-L_u1],
                        [U_u1_k1],
                        [-L_u1_k1]])
    
    prog = MPQP_Program(A=A_, b=b, c=c.T, H=H_t.T, Q=Q, A_t=A_t, b_t=b_t, F=F_, c_c=c_c, c_t=c_t.T, Q_t=Q_t)
    prog.process_constraints()
    sol = solve_mpqp(prog, mpqp_algorithm.combinatorial)
    return sol


def LoadData():
    grid_xk1_list1 = np.load('../2 Space Discretization/Grid_Files/Grid_ek1_1.npy')
    K_xk1_list1 = np.load('../2 Space Discretization/Grid_Files/K_ek1_1.npy')
    grid_xk1_list2 = np.load('../2 Space Discretization/Grid_Files/Grid_ek1_2.npy')
    K_xk1_list2 = np.load('../2 Space Discretization/Grid_Files/K_ek1_2.npy')
    grid_xk2_list1 = np.load('../2 Space Discretization/Grid_Files/Grid_ek2_1.npy')
    K_xk2_list1 = np.load('../2 Space Discretization/Grid_Files/K_ek2_1.npy')
    grid_xk2_list2 = np.load('../2 Space Discretization/Grid_Files/Grid_ek2_2.npy')
    K_xk2_list2 = np.load('../2 Space Discretization/Grid_Files/K_ek2_2.npy')
    all_grid_list = [grid_xk1_list1, grid_xk1_list2, grid_xk2_list1, grid_xk2_list2]
    all_K_list = [K_xk1_list1, K_xk1_list2, K_xk2_list1, K_xk2_list2]
    return all_grid_list, all_K_list

def check_point_k1(pt_x1, pt_x2, pt_u1, pt_u2, K_list, grid_pts_list):
    for i, grid_pts in enumerate(grid_pts_list):
        delta = round(grid_pts[1][0]-grid_pts[0][0], 6)
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
            (coef_x1, coef_x2, coef_u1, coef_u2, coef_c) = K_list[i]
            return [coef_x1, coef_x2, coef_u1, coef_u2, coef_c]
    
def check_point_k2(pt_x1, pt_x2, pt_u1, pt_u2, pt_u1_k1, pt_u2_k1, K_list, grid_pts_list):
    for i, grid_pts in enumerate(grid_pts_list):
        delta = round(grid_pts[1][0]-grid_pts[0][0], 5)
        L_x1 = grid_pts[0][0] #get reference x1 point of the grid
        L_x2 = grid_pts[0][1] #get reference x2 point of the grid
        L_u1 = grid_pts[0][2] #get reference u1 point of the grid
        L_u2 = grid_pts[0][3] #get reference u2 point of the grid
        L_u1_k1 = grid_pts[0][4] #get reference u1_k1 point of the grid
        L_u2_k1 = grid_pts[0][5] #get reference u2_k1 point of the grid
        
        U_x1 = L_x1+delta
        U_x2 = L_x2+delta
        U_u1 = L_u1+delta
        U_u2 = L_u2+delta
        U_u1_k1 = L_u1_k1+delta
        U_u2_k1 = L_u2_k1+delta
        
        arr_bds = np.array([U_x1, -L_x1, U_x2, -L_x2, U_u1, -L_u1, U_u2, -L_u2, U_u1_k1, -L_u1_k1, U_u2_k1, -L_u2_k1]).reshape(-1,1)
        arr_coef = np.array([[1, 0, 0, 0, 0, 0],
                             [-1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, -1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, -1, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, -1, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, -1, 0],
                             [0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, -1]])
        arr_pt = np.array([pt_x1, pt_x2, pt_u1, pt_u2, pt_u1_k1, pt_u2_k1]).reshape(-1, 1)
        if all(arr_coef@arr_pt<=arr_bds):
            (coef_x1, coef_x2, coef_u1, coef_u2, coef_u1_k1, coef_u2_k1, coef_c) = K_list[i]
            return [coef_x1, coef_x2, coef_u1, coef_u2, coef_u1_k1, coef_u2_k1, coef_c]
    
def scaler_info_PH1(input_scaler1, output_scaler1):
    x1_max, x2_max, u1_max, u2_max = input_scaler1.data_max_
    x1_min, x2_min, u1_min, u2_min = input_scaler1.data_min_
    
    x1_k1_max, x2_k1_max, = output_scaler1.data_max_
    x1_k1_min, x2_k1_min = output_scaler1.data_min_
    
    x1_diff = x1_max - x1_min
    x2_diff = x2_max - x2_min
    u1_diff = u1_max - u1_min
    u2_diff = u2_max - u2_min
    x1_k1_diff = x1_k1_max - x1_k1_min
    x2_k1_diff = x2_k1_max - x2_k1_min
    
    x1_sum = x1_max + x1_min
    x2_sum = x2_max + x2_min
    u1_sum = u1_max + u1_min
    u2_sum = u2_max + u2_min
    
    sum_list_PH1 = [x1_sum, x2_sum, u1_sum, u2_sum]
    diff_list_PH1 = [x1_diff, x2_diff, x1_k1_diff, x2_k1_diff, u1_diff, u2_diff, x1_k1_min, x2_k1_min]
    return sum_list_PH1, diff_list_PH1

def scaler_info_PH2(input_scaler2, output_scaler2):
    x1_max, x2_max, u1_max, u2_max, u1_k1_max, u2_k1_max = input_scaler2.data_max_
    x1_min, x2_min, u1_min, u2_min, u1_k1_min, u2_k1_min = input_scaler2.data_min_
    x1_k2_max, x2_k2_max, = output_scaler2.data_max_
    x1_k2_min, x2_k2_min = output_scaler2.data_min_
    
    x1_diff = x1_max - x1_min
    x2_diff = x2_max - x2_min
    u1_diff = u1_max - u1_min
    u2_diff = u2_max - u2_min
    u1_k1_diff = u1_k1_max - u1_k1_min
    u2_k1_diff = u2_k1_max - u2_k1_min
    x1_k2_diff = x1_k2_max - x1_k2_min
    x2_k2_diff = x2_k2_max - x2_k2_min
    
    x1_sum = x1_max + x1_min
    x2_sum = x2_max + x2_min
    u1_sum = u1_max + u1_min
    u2_sum = u2_max + u2_min
    u1_k1_sum = u1_k1_max + u1_k1_min
    u2_k1_sum = u2_k1_max + u2_k1_min

    sum_list_PH2 = [x1_sum, x2_sum, u1_sum, u2_sum, u1_k1_sum, u2_k1_sum]
    diff_list_PH2 = [x1_diff, x2_diff, x1_k2_diff, x2_k2_diff, u1_diff, u2_diff, u1_k1_diff, u2_k1_diff, x1_k2_min, x2_k2_min]
    return sum_list_PH2, diff_list_PH2


def coef_rescale_k1(coef_list1, coef_list2, sum_list_PH1, diff_list_PH1):
    x1_sum, x2_sum, u1_sum, u2_sum = sum_list_PH1
    x1_diff, x2_diff, x1_k1_diff, x2_k1_diff, u1_diff, u2_diff, x1_k1_min, x2_k1_min = diff_list_PH1
        
    A1_, B1_, C1_, D1_, E1_ = coef_list1
    A1 = A1_*2*x1_k1_diff/x1_diff
    B1 = B1_*2*x1_k1_diff/x2_diff
    C1 = C1_*2*x1_k1_diff/u1_diff
    D1 = D1_*2*x1_k1_diff/u2_diff
    E1 = (E1_+x1_k1_min/x1_k1_diff-A1_*x1_sum/x1_diff-B1_*x2_sum/x2_diff-C1_*u1_sum/u1_diff-D1_*u2_sum/u2_diff)*x1_k1_diff
    
    A2_, B2_, C2_, D2_, E2_ = coef_list2
    A2 = A2_*2*x2_k1_diff/x1_diff
    B2 = B2_*2*x2_k1_diff/x2_diff
    C2 = C2_*2*x2_k1_diff/u1_diff
    D2 = D2_*2*x2_k1_diff/u2_diff
    E2 = (E2_+x2_k1_min/x2_k1_diff-A2_*x1_sum/x1_diff-B2_*x2_sum/x2_diff-C2_*u1_sum/u1_diff-D2_*u2_sum/u2_diff)*x2_k1_diff  
    return [A1, B1, C1, D1, E1],  [A2, B2, C2, D2, E2]  

def coef_rescale_k2(coef_list1, coef_list2, sum_list_PH2, diff_list_PH2):
    x1_sum, x2_sum, u1_sum, u2_sum, u1_k1_sum, u2_k1_sum = sum_list_PH2
    x1_diff, x2_diff, x1_k2_diff, x2_k2_diff, u1_diff, u2_diff, u1_k1_diff, u2_k1_diff, x1_k2_min, x2_k2_min = diff_list_PH2
      
    A1_, B1_, C1_, D1_, E1_, F1_, G1_ = coef_list1
    A1 = A1_*2*x1_k2_diff/x1_diff
    B1 = B1_*2*x1_k2_diff/x2_diff
    C1 = C1_*2*x1_k2_diff/u1_diff
    D1 = D1_*2*x1_k2_diff/u2_diff
    E1 = E1_*2*x1_k2_diff/u1_k1_diff
    F1 = F1_*2*x1_k2_diff/u2_k1_diff
    G1 = (G1_+x1_k2_min/x1_k2_diff-A1_*x1_sum/x1_diff-B1_*x2_sum/x2_diff-C1_*u1_sum/u1_diff-D1_*u2_sum/u2_diff-E1_*u1_k1_sum/u1_k1_diff-F1_*u2_k1_sum/u2_k1_diff)*x1_k2_diff
    
    A2_, B2_, C2_, D2_, E2_, F2_, G2_ = coef_list2
    A2 = A2_*2*x2_k2_diff/x1_diff
    B2 = B2_*2*x2_k2_diff/x2_diff
    C2 = C2_*2*x2_k2_diff/u1_diff
    D2 = D2_*2*x2_k2_diff/u2_diff
    E2 = E2_*2*x2_k2_diff/u1_k1_diff
    F2 = F2_*2*x2_k2_diff/u2_k1_diff
    G2 = (G2_+x2_k2_min/x2_k2_diff-A2_*x1_sum/x1_diff-B2_*x2_sum/x2_diff-C2_*u1_sum/u1_diff-D2_*u2_sum/u2_diff-E2_*u1_k1_sum/u1_k1_diff-F2_*u2_k1_sum/u2_k1_diff)*x2_k2_diff
    return [A1, B1, C1, D1, E1, F1, G1],  [A2, B2, C2, D2, E2, F2, G2]
    

def GetExplicitSol(idx, input_list):
    MODE, M, N, all_grid_list, all_K_list, subdict, scaler_list = input_list
    
    input_scaler1, output_scaler1, input_scaler2, output_scaler2 =  scaler_list
    
    grid_xk1_list1, grid_xk1_list2, grid_xk2_list1, grid_xk2_list2 = all_grid_list
    K_xk1_list1, K_xk1_list2, K_xk2_list1, K_xk2_list2 = all_K_list

    sum_list_PH1, diff_list_PH1 = scaler_info_PH1(input_scaler1, output_scaler1)
    sum_list_PH2, diff_list_PH2 = scaler_info_PH2(input_scaler2, output_scaler2)

    total_length = 0
    for value in subdict.values():
        total_length += len(value)
        
    theta_space_sol_dict = dict()
    with tqdm(range(total_length), ncols=80, desc=f'Process ID:{idx+1}',
                    ascii=False, position=idx, file=sys.stdout) as pbar:
        for theta_region, bds_list_ in subdict.items():
            candidates_list = []
            for bds_ in bds_list_:
                # x1_pair: [lower bound of x1, upper bound of x1], '_' means the values are scaled
                x1_pair_ = [bds_[0], bds_[0+6]]
                x2_pair_ = [bds_[1], bds_[1+6]]
                u1_pair_ = [bds_[2], bds_[2+6]]
                u2_pair_ = [bds_[3], bds_[3+6]]
                u1_k1_pair_ = [bds_[4], bds_[4+6]]
                u2_k1_pair_ = [bds_[5], bds_[5+6]]
                
                #use the following point to locate the corresponding segement of piecewise affine function
                pt_x1_ = x1_pair_[0] + 1e-5
                pt_x2_ = x2_pair_[0] + 1e-5
                pt_u1_ = u1_pair_[0] + 1e-5
                pt_u2_ = u2_pair_[0] + 1e-5
                pt_u1_k1_ = u1_k1_pair_[0] + 1e-5
                pt_u2_k1_ = u2_k1_pair_[0] + 1e-5
                
                #retrieve the correct coefficients
                coef_list_k1_1_ = check_point_k1(pt_x1_, pt_x2_, pt_u1_, pt_u2_, K_xk1_list1, grid_xk1_list1)
                coef_list_k1_2_ = check_point_k1(pt_x1_, pt_x2_, pt_u1_, pt_u2_, K_xk1_list2, grid_xk1_list2)
                
                #convert the coefficients of the affine functons back to original magnitude
                #mpQP problems are constructed on original magnitude, not on the scaled magnitude (e.g., 0~1)
                coef_list_k1_1, coef_list_k1_2 = coef_rescale_k1(coef_list_k1_1_, coef_list_k1_2_, sum_list_PH1, diff_list_PH1)
    
                coef_list_k2_1_ = check_point_k2(pt_x1_, pt_x2_, pt_u1_, pt_u2_, pt_u1_k1_, pt_u2_k1_, K_xk2_list1, grid_xk2_list1)
                coef_list_k2_2_ = check_point_k2(pt_x1_, pt_x2_, pt_u1_, pt_u2_, pt_u1_k1_, pt_u2_k1_, K_xk2_list2, grid_xk2_list2)
                coef_list_k2_1, coef_list_k2_2 = coef_rescale_k2(coef_list_k2_1_, coef_list_k2_2_, sum_list_PH2, diff_list_PH2)


                #coefficients of the affine functon: x_{k+1} = Ax_{k} + Bu_{k} + C
                A = np.array([[coef_list_k1_1[0], coef_list_k1_1[1]],
                              [coef_list_k1_2[0], coef_list_k1_2[1]]])
                
                B = np.array([[coef_list_k1_1[2], coef_list_k1_1[3]],
                              [coef_list_k1_2[2], coef_list_k1_2[3]]])
                
                C = np.array([[coef_list_k1_1[4]],
                              [coef_list_k1_2[4]]])
         
                #coefficients of the affine functon: x_{k+2} = Dx_{k} + Eu_{k} + Fu_{k+1} + G
                D = np.array([[coef_list_k2_1[0], coef_list_k2_1[1]],
                              [coef_list_k2_2[0], coef_list_k2_2[1]]])
                
                E = np.array([[coef_list_k2_1[2], coef_list_k2_1[3]],
                              [coef_list_k2_2[2], coef_list_k2_2[3]]])
                
                F = np.array([[coef_list_k2_1[4], coef_list_k2_1[5]],
                              [coef_list_k2_2[4], coef_list_k2_2[5]]])
                
                G = np.array([[coef_list_k2_1[6]],
                              [coef_list_k2_2[6]]])
                
                x1_pair = [0.]*2
                x2_pair = [0.]*2
                u1_pair = [0.]*2
                u2_pair = [0.]*2
                u1_k1_pair = [0.]*2
                u2_k1_pair = [0.]*2
                
                a = [x1_pair_[0], x2_pair_[0], u1_pair_[0], u2_pair_[0], u1_k1_pair_[0], u2_k1_pair_[0]]
                b = [x1_pair_[1], x2_pair_[1], u1_pair_[1], u2_pair_[1], u1_k1_pair_[1], u2_k1_pair_[1]]
                
                #convert the region boundary to the original magnitude
                x1_pair[0], x2_pair[0], u1_pair[0], u2_pair[0], u1_k1_pair[0], u2_k1_pair[0] = input_scaler2.inverse_transform([a])[0]
                x1_pair[1], x2_pair[1], u1_pair[1], u2_pair[1], u1_k1_pair[1], u2_k1_pair[1] = input_scaler2.inverse_transform([b])[0]
                
                #solve the mpQP problem via PPOPT
                sol = Solve_MPQP(MODE, M, N, A, B, C, D, E, F, G, 
                          x1_pair, x2_pair, 
                          u1_pair, u2_pair,
                          u1_k1_pair, u2_k1_pair)
                
                #collect solutions
                if len(sol.critical_regions) != 0:
                    candidates_list.append([bds_,sol])
                pbar.update(1)  
                
            #gather all applicable mpQP problems for the same region
            if len(candidates_list) != 0:
                theta_space_sol_dict[theta_region] = candidates_list
    return theta_space_sol_dict

if __name__ == "__main__":
    #can change to different mode for different MPC
    MODE='dMPC1' #MODE: {cMPC, dMPC1, dMPC2}
    
    if MODE not in MODE_POOL:
        raise ValueError('MODE ERROR')
    
    all_grid_list, all_K_list = LoadData()
    theta_bds_dict = np.load(f'../3 Region Merging/theta_dict_{MODE}_PH2.npy', allow_pickle=True).item()
    input_scaler1 = np.load('../1 Model Training/PICNN_scaler/x_k_u_k_scaler.npy', allow_pickle=True).item()
    output_scaler1 = np.load('../1 Model Training/PICNN_scaler/e_k1_scaler.npy', allow_pickle=True).item()
    input_scaler2 = np.load('../1 Model Training/PICNN_scaler/x_k_u_k_u_k1_scaler.npy', allow_pickle=True).item()
    output_scaler2 = np.load('../1 Model Training/PICNN_scaler/e_k2_scaler.npy', allow_pickle=True).item()
    scaler_list = [input_scaler1, output_scaler1, input_scaler2, output_scaler2]
    
    #weight matrices in objective function
    M = np.array([[1e4, 0.0],
                  [0.0, 1e3]])
    
    N = np.array([[1, 0.0],
                  [0.0, 1]])

    #use multiprocessing to accelerate offline computation
    if MODE == 'cMPC':
        subdict_size = 6 #16 subprocesses
    elif MODE == 'dMPC1':
        subdict_size = 300 #16 subprocesses
    elif MODE == 'dMPC2':
        subdict_size = 300 #16 subprocesses
        
    it = iter(theta_bds_dict.items())
    subdicts = [dict(list(itertools.islice(it, subdict_size))) for _ in range(0, len(theta_bds_dict), subdict_size)]
    input_list = []
    for subdict in subdicts:
        input_list.append([MODE, M, N, all_grid_list, all_K_list, subdict, scaler_list])
    pool = Pool(processes=len(input_list), initargs=(RLock(),), initializer=tqdm.set_lock)
    jobs = [pool.apply_async(GetExplicitSol, args=(idx, input_)) for (idx, input_) in enumerate(input_list)]
    pool.close()
    results = [job.get() for job in jobs]

    # merge the results of sub-tasks
    ExplicitMPC_Sols = dict()
    for i in tqdm(range(len(results))):
        ExplicitMPC_Sols.update(results[i])

    # save the solutions
    with open(f"{MODE}_Sols_PH2", "wb") as fp:
        pickle.dump(ExplicitMPC_Sols, fp)
    
