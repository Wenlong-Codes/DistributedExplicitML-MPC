# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:55:25 2024

@author: WANG Wenlong
@email: wenlongw@nus.edu.sg
"""

import numpy as np
from tqdm import tqdm

def MergeRegions(K_list1, grid_pts_list1, K_list2, grid_pts_list2):
    bds_list1, bds_list2 = [], []
    bds_list1_vertex, bds_list2_vertex = [], []
    
    for i, grid_pts in enumerate(grid_pts_list1):
        x1, x2, u1, u2, u1_k1, u2_k1 = grid_pts[0] 
        delta = grid_pts[1][0]-x1
        bds_list1.append((x1, x2, u1, u2, u1_k1, u2_k1, delta))
        bds_list1_vertex.append((x1, x2, u1, u2, u1_k1, u2_k1))
        
    for i, grid_pts in enumerate(grid_pts_list2):
        x1, x2, u1, u2, u1_k1, u2_k1 = grid_pts[0]
        delta = grid_pts[1][0]-x1
        bds_list2.append((x1, x2, u1, u2, u1_k1, u2_k1, delta))
        bds_list2_vertex.append((x1, x2, u1, u2, u1_k1, u2_k1))
    
    # Not all regions are elementary. Some redundant large regions need to be removed.
    bds_list = list(set(bds_list1 + bds_list2)) 
    bds_list_vertex = set(bds_list1_vertex + bds_list2_vertex) 
    return bds_list, bds_list_vertex

def CenterPointNotation(final_bds_list):
    center_bds_list = []
    for bds in final_bds_list:
        x1, x2, u1, u2, u1_k1, u2_k1, delta = bds
        mid_x1 = x1 + delta/2
        mid_x2 = x2 + delta/2
        mid_u1 = u1 + delta/2
        mid_u2 = u2 + delta/2
        mid_u1_k1 = u1_k1 + delta/2
        mid_u2_k1 = u2_k1 + delta/2
        center_bds_list.append([mid_x1, mid_x2, mid_u1, mid_u2, mid_u1_k1, mid_u2_k1, delta/2])
    return center_bds_list

def GetFinalRegion(bds_list, bds_list_vertex):
    final_bds_l_list = []
    final_bds_u_list = []
    
    # remove redundant regions by checking their center points.
    # the center point of an elementary region should not be the edge points of any other regions.
    for bds_i in tqdm(bds_list):
        x1_L, x2_L, u1_L, u2_L, u1_k1_L, u2_k1_L, delta = bds_i
        x1_u = x1_L + delta
        x2_u = x2_L + delta
        u1_u = u1_L + delta
        u2_u = u2_L + delta
        u1_k1_u = u1_k1_L + delta
        u2_k1_u = u2_k1_L + delta
        
        #get center point
        x1_m = x1_L + delta/2
        x2_m = x2_L + delta/2
        u1_m = u1_L + delta/2
        u2_m = u2_L + delta/2
        u1_k1_m = u1_k1_L + delta/2
        u2_k1_m = u2_k1_L + delta/2
    
        pt_m = (x1_m, x2_m, u1_m, u2_m, u1_k1_m, u2_k1_m)
        if pt_m not in bds_list_vertex:
            final_bds_l_list.append([x1_L, x2_L, u1_L, u2_L, u1_k1_L, u2_k1_L])
            final_bds_u_list.append([x1_u, x2_u, u1_u, u2_u, u1_k1_u, u2_k1_u])
    
    final_bds_l_list = np.array(final_bds_l_list)       
    final_bds_u_list = np.array(final_bds_u_list)   
    final_bds_list = np.column_stack((final_bds_l_list, final_bds_u_list))
    return final_bds_list 

if __name__ == "__main__":
    #load grid files
    K_list1 = np.load('../2 Space Discretization/Grid_Files/K_ek2_1.npy')
    grid_pts_list1 = np.load('../2 Space Discretization/Grid_Files/Grid_ek2_1.npy')
    K_list2 = np.load('../2 Space Discretization/Grid_Files/K_ek2_2.npy')
    grid_pts_list2 = np.load('../2 Space Discretization/Grid_Files/Grid_ek2_2.npy')
    bds_list, bds_list_vertex = MergeRegions(K_list1, grid_pts_list1, K_list2, grid_pts_list2)
    final_bds_list = GetFinalRegion(bds_list, bds_list_vertex)
    np.save('merged_bds_list_PH2.npy', final_bds_list)
