# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:38:15 2024

@author: Wenlong WANG
@email: wenlongw@nus.edu.sg
"""

import numpy as np
from tqdm import tqdm

def ConvertToString(bds_list):
    bds_string_list = []
    for bds in bds_list:
        bds_string = ''
        for i in range(len(bds)):
            bds_string += str(bds[i]) + '&'
        bds_string_list.append(bds_string[:-1])
    return bds_string_list

def ObtainUniqueTheta(bds_list, mode):
    theta_bds_string_list = set()
    # bds : [Lx1, Lx2, Lu1, Lu2, Lu1_k1, Lu2_k1, 
    #        Ux1, Ux2, Uu1, Uu2, Uu1_k1, Uu2_k1]
    for bds in bds_list:
        
        if mode == 'cMPC':
            # theta_bds : [Lx1, Lx2,
            #              Ux1, Ux2]
            theta_bds = [bds[0], bds[1], bds[6], bds[7]]
        elif mode == 'dMPC1':
            # theta_bds : [Lx1, Lx2, Lu2, Lu2_k1,
            #              Ux1, Ux2, Uu2, Uu2_k1]
            theta_bds = [bds[0], bds[1], bds[3], bds[5], bds[6], bds[7], bds[9], bds[11]]
        elif mode == 'dMPC2':
            # theta_bds : [Lx1, Lx2, Lu1, Lu1_k1,
            #              Ux1, Ux2, Uu1, Uu1_k1]
            theta_bds = [bds[0], bds[1], bds[2], bds[4], bds[6], bds[7], bds[8], bds[10]]
            
        theta_bds_string = ''
        for i in range(len(theta_bds)):
            theta_bds_string += str(theta_bds[i]) + '&'
        theta_bds_string_list.add(theta_bds_string[:-1]) 
    return list(theta_bds_string_list)

def ConstructThetaBdsDict(bds_list, theta_bds_string_list, mode):
    '''
    theta_dict{
           theta-region_1:[u_region_11, u_region_12,...,u_region_i], #size i
           theta-region_2:[u_region_21, u_region_22,...,u_region_j], #size j
           ...
           theta-region_n:[u_region_n1, u_region_n2,...,u_region_k]  #size k
          } 
    '''

    theta_dict = dict() 
    for i, theta_bds_string in enumerate(tqdm(theta_bds_string_list)):
        theta_bds_list = theta_bds_string.split('&')
        theta_bds_list = [float(item) for item in theta_bds_list]
        for bds in bds_list:
            bds = list(bds)
            
            if mode == 'cMPC':
                current_theta_bds_list = [bds[0], bds[1],
                                          bds[6], bds[7]]
            elif mode == 'dMPC1':
                current_theta_bds_list = [bds[0], bds[1], bds[3], bds[5],
                                          bds[6], bds[7], bds[9], bds[11]]
            elif mode == 'dMPC2':
                current_theta_bds_list = [bds[0], bds[1], bds[2], bds[4],
                                          bds[6], bds[7], bds[8], bds[10]]
                
            if current_theta_bds_list == theta_bds_list:
               try:
                   exist_list = theta_dict[theta_bds_string]
                   exist_list.append(bds)
                   theta_dict[theta_bds_string] = exist_list
               except:
                   theta_dict[theta_bds_string] = [bds]             
    return theta_dict
        
if __name__ == "__main__":
    bds_list = np.load('merged_bds_list_PH2.npy')
    mode = 'dMPC1' #cMPC, dMPC1, dMPC2
    theta_bds_string_list = ObtainUniqueTheta(bds_list, mode)
    theta_dict = ConstructThetaBdsDict(bds_list, theta_bds_string_list, mode)
    np.save(f'theta_dict_{mode}_PH2.npy', theta_dict)
    
