# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:59:31 2024

@author: WANG Wenlong
@email: wenlongw@nus.edu.sg
"""

import pyomo.environ as pyo
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore', category=Warning)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../1 Model Training/')))
from PICNN_e_k1_training import PICNN_Model_PH1
from PICNN_e_k2_training import PICNN_Model_PH2

M = np.array([[1e4, 0.0],
              [0.0, 1e3]])

N = np.array([[1, 0.0],
              [0.0, 1]])

def PICNN_next_state1(xk_list, uk_list, model1, scaler_list):
    xk_uk_scaler, ek1_scaler = scaler_list
    sk = np.zeros((len(xk_list), 4))
    sk[:,:2] = xk_list
    sk[:,2:] = uk_list
    sk_ = xk_uk_scaler.transform(sk)
    
    input_xk_ = np.array(sk_[:,:2]).reshape(-1, 1, 2)
    input_uk_ = np.array(sk_[:,2:]).reshape(-1, 1, 2)
    
    input_xk_ = torch.from_numpy(input_xk_).to(torch.float32)
    input_uk_ = torch.from_numpy(input_uk_).to(torch.float32)
    output_ek1_ = model1(input_xk_, input_uk_).cpu().data.numpy()[:,0] 
    output_ek1 = ek1_scaler.inverse_transform(output_ek1_)
    return output_ek1

def PICNN_next_state2(xk_list, uk_list, uk1_list, model2, scaler_list):
    xk_uk_uk1_scaler, ek2_scaler = scaler_list
    sk = np.zeros((len(xk_list), 6))
    sk[:,:2] = xk_list
    sk[:,2:4] = uk_list 
    sk[:,4:] = uk1_list 
    
    sk_ = xk_uk_uk1_scaler.transform(sk)
    
    input_xk_ = np.array(sk_[:,:2]).reshape(-1, 1, 2)
    input_uk_uk1_ = np.array(sk_[:,2:]).reshape(-1, 1, 4)

    input_xk_ = torch.from_numpy(input_xk_).to(torch.float32)
    input_uk_uk1_ = torch.from_numpy(input_uk_uk1_).to(torch.float32)
    
    output_ek2_ = model2(input_xk_, input_uk_uk1_).cpu().data.numpy()[:,0] 
    output_ek2 = ek2_scaler.inverse_transform(output_ek2_)
    return output_ek2

def GetABCkb(MODE, theta, sols): 
    #cMPC: theta = [x1, x2], var = [u1, u2, u1_k1, u2_k1]
    #dMPC1: theta = [x1, x2, u2, u2_k1], var = [u1, u1_k1]
    #dMPC2: theta = [x1, x2, u1, u1_k1], var = [u2, u2_k1]
    theta = np.array(theta)
    xk_list = []
    uk_list = []
    uk1_list = []
    
    for sol in sols:
        for i in range(len(sol.critical_regions)):
            E = sol.critical_regions[i].E
            f = sol.critical_regions[i].f.reshape(-1,1)
            if all(E@theta.reshape(-1,1) <= f):
                k = sol.critical_regions[i].A
                b = sol.critical_regions[i].b
                if MODE == 'cMPC':
                    uk_uk1 = k@theta.reshape(-1, 1) + b
                    uk_uk1 = uk_uk1.reshape(-1,)
                    uk_list.append(uk_uk1[:2])
                    uk1_list.append(uk_uk1[2:])
                    xk_list.append([theta[0], theta[1]])
                    
                elif MODE == 'dMPC1':
                    uk_uk1_s = k@theta.reshape(-1, 1) + b
                    uk_uk1_s = uk_uk1_s.reshape(-1,)
                    uk_list.append([uk_uk1_s[0], theta[2]])
                    uk1_list.append([uk_uk1_s[1], theta[3]])
                    xk_list.append([theta[0], theta[1]])
                    
                elif MODE == 'dMPC2':
                    uk_uk1_s = k@theta.reshape(-1, 1) + b
                    uk_uk1_s = uk_uk1_s.reshape(-1,)
                    uk_list.append([theta[2], uk_uk1_s[0]])
                    uk1_list.append([theta[3], uk_uk1_s[1]])
                    xk_list.append([theta[0], theta[1]])
                
    xk_list = np.array(xk_list)  
    uk_list = np.array(uk_list)  
    uk1_list = np.array(uk1_list)  
    return xk_list, uk_list, uk1_list

def ClaimModel(MODE, theta, sols, model_list, scaler_list):
    xk_list, uk_list, uk1_list = GetABCkb(MODE, theta, sols)
    scaler_list_1 = [scaler_list[0], scaler_list[2]]
    scaler_list_2 = [scaler_list[1], scaler_list[3]]
    ek1_list = PICNN_next_state1(xk_list, uk_list, model_list[0], scaler_list_1)
    ek2_list = PICNN_next_state2(xk_list, uk_list, uk1_list, model_list[1], scaler_list_2)
    m = pyo.ConcreteModel()
    ClaimVar(MODE, m, theta, ek1_list, ek2_list, uk_list, uk1_list)
    ClaimConstrs(MODE, m)
    ClaimObj(m)
    return m

def ClaimVar(MODE, m, theta, ek1_list, ek2_list, uk_list, uk1_list):
    print(f'The number of binary variables in {MODE} is : {len(ek1_list)}')
    m.n = len(ek1_list)
    m.xk1_1 = pyo.Param(range(m.n), initialize={i: ek1_list[i][0] for i in range(m.n)})
    m.xk1_2 = pyo.Param(range(m.n), initialize={i: ek1_list[i][1] for i in range(m.n)})
    m.xk2_1 = pyo.Param(range(m.n), initialize={i: ek2_list[i][0] for i in range(m.n)})
    m.xk2_2 = pyo.Param(range(m.n), initialize={i: ek2_list[i][1] for i in range(m.n)})
    
    m.xk1_1_c = pyo.Var(within=pyo.Reals)
    m.xk1_2_c = pyo.Var(within=pyo.Reals)
    m.xk2_1_c = pyo.Var(within=pyo.Reals)
    m.xk2_2_c = pyo.Var(within=pyo.Reals)
    m.coef = pyo.Var(range(m.n), within=pyo.Binary)
    
    if MODE == 'cMPC':
        m.uk_1 = pyo.Param(range(m.n), initialize={i: uk_list[i][0] for i in range(m.n)})
        m.uk_2 = pyo.Param(range(m.n), initialize={i: uk_list[i][1] for i in range(m.n)})
        m.uk1_1 = pyo.Param(range(m.n), initialize={i: uk1_list[i][0] for i in range(m.n)})
        m.uk1_2 = pyo.Param(range(m.n), initialize={i: uk1_list[i][1] for i in range(m.n)})
        
        m.uk_1_c = pyo.Var(within=pyo.Reals)
        m.uk_2_c = pyo.Var(within=pyo.Reals)
        m.uk1_1_c = pyo.Var(within=pyo.Reals)
        m.uk1_2_c = pyo.Var(within=pyo.Reals)
        
    elif MODE == 'dMPC1':
        m.uk_1 = pyo.Param(range(m.n), initialize={i: uk_list[i][0] for i in range(m.n)})
        m.uk1_1 = pyo.Param(range(m.n), initialize={i: uk1_list[i][0] for i in range(m.n)})
        m.uk_1_c = pyo.Var(within=pyo.Reals)
        m.uk1_1_c = pyo.Var(within=pyo.Reals)
        m.uk_2_c = theta[2]
        m.uk1_2_c = theta[3]    
    elif MODE == 'dMPC2':
        m.uk_2 = pyo.Param(range(m.n), initialize={i: uk_list[i][1] for i in range(m.n)})
        m.uk1_2 = pyo.Param(range(m.n), initialize={i: uk1_list[i][1] for i in range(m.n)})
        m.uk_2_c = pyo.Var(within=pyo.Reals)
        m.uk1_2_c = pyo.Var(within=pyo.Reals)
        m.uk_1_c = theta[2]
        m.uk1_1_c = theta[3]
    
def ClaimObj(m):
    m.obj = pyo.Objective(expr=M[0][0]*(m.xk1_1_c**2+m.xk2_1_c**2)+\
                                M[1][1]*(m.xk1_2_c**2+m.xk2_2_c**2)+\
                                  N[0][0]*(m.uk_1_c**2+m.uk1_1_c**2)+\
                              N[1][1]*(m.uk_2_c**2+m.uk1_2_c**2), sense=pyo.minimize)
        
def ClaimConstrs(MODE, m):
    m.coef_constraint = pyo.Constraint(rule=coef_constraint_rule)
    m.state_constraint1 = pyo.Constraint(rule=state_update1)
    m.state_constraint2 = pyo.Constraint(rule=state_update2)
    m.state_constraint3 = pyo.Constraint(rule=state_update3)
    m.state_constraint4 = pyo.Constraint(rule=state_update4)
    
    if MODE == 'cMPC':
        m.control_action_constraint1 = pyo.Constraint(rule=control_action_constraint_rule1)
        m.control_action_constraint2 = pyo.Constraint(rule=control_action_constraint_rule2)
        m.control_action_constraint3 = pyo.Constraint(rule=control_action_constraint_rule3)
        m.control_action_constraint4 = pyo.Constraint(rule=control_action_constraint_rule4)
    elif MODE == 'dMPC1':
        m.control_action_constraint1 = pyo.Constraint(rule=control_action_constraint_rule1)
        m.control_action_constraint3 = pyo.Constraint(rule=control_action_constraint_rule3)
    elif MODE == 'dMPC2':
        m.control_action_constraint2 = pyo.Constraint(rule=control_action_constraint_rule2)
        m.control_action_constraint4 = pyo.Constraint(rule=control_action_constraint_rule4)


def coef_constraint_rule(m):
    return sum(m.coef[i] for i in range(m.n)) == 1.0

def state_update1(m):
    return sum(m.coef[i]*m.xk1_1[i] for i in range(m.n)) == m.xk1_1_c

def state_update2(m):
    return sum(m.coef[i]*m.xk1_2[i] for i in range(m.n)) == m.xk1_2_c

def state_update3(m):
    return sum(m.coef[i]*m.xk2_1[i] for i in range(m.n)) == m.xk2_1_c

def state_update4(m):
    return sum(m.coef[i]*m.xk2_2[i] for i in range(m.n)) == m.xk2_2_c


def control_action_constraint_rule1(m):
    return sum(m.coef[i]*m.uk_1[i] for i in range(m.n)) == m.uk_1_c

def control_action_constraint_rule2(m):
    return sum(m.coef[i]*m.uk_2[i] for i in range(m.n)) == m.uk_2_c
        
def control_action_constraint_rule3(m):
    return sum(m.coef[i]*m.uk1_1[i] for i in range(m.n)) == m.uk1_1_c

def control_action_constraint_rule4(m):
    return sum(m.coef[i]*m.uk1_2[i] for i in range(m.n)) == m.uk1_2_c
        

def SolveMIQP(MODE, theta, sols, model_list, scaler_list):
    m = ClaimModel(MODE, theta, sols, model_list, scaler_list)
    solver = pyo.SolverFactory('gurobi_direct')
    solver.options['TimeLimit'] = 36
    solver.solve(m)
    if MODE == 'cMPC':
        return np.array([pyo.value(m.uk_1_c),  pyo.value(m.uk_2_c)]), pyo.value(m.obj)
    elif MODE == 'dMPC1':
        return np.array([pyo.value(m.uk_1_c),  pyo.value(m.uk1_1_c)]), pyo.value(m.obj)
    elif MODE == 'dMPC2':
        return np.array([pyo.value(m.uk_2_c), pyo.value(m.uk1_2_c)]), pyo.value(m.obj)
    