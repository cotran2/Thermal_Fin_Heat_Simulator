#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 16:01:02 2019

@author: hwan
"""
import pandas as pd
import matplotlib.pyplot as plt
import dolfin as dl

from Thermal_Fin_Heat_Simulator.Utilities.thermal_fin import get_space_2D, get_space_3D
from Thermal_Fin_Heat_Simulator.Utilities.forward_solve import Fin
from Thermal_Fin_Heat_Simulator.Generate_and_Save_Thermal_Fin_Data import convert_array_to_dolfin_function
from Thermal_Fin_Heat_Simulator.Utilities.plot_3D import plot_3D

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                                File Paths                                   #
###############################################################################
if __name__ == "__main__":  
    #=== Number of Data ===#
    num_data = 1

    #=== Select True or Test Set ===#
    generate_train_data = 1
    generate_test_data = 0
    
    #===  Select Parameter Type ===#
    generate_nine_parameters = 0
    generate_varying = 1
    
    #=== Select Thermal Fin Dimension ===#
    generate_2D = 0
    generate_3D = 1
    
    #=== Defining Filenames and Creating Directories ===#
    if generate_train_data == 1:
        train_or_test = 'train'
    if generate_test_data == 1:
        train_or_test = 'test'         
    if generate_nine_parameters == 1:
        parameter_type = '_nine'        
    if generate_varying == 1:
        parameter_type = '_vary'
    if generate_2D == 1:
        fin_dimension = ''
    if generate_3D == 1:
        fin_dimension = '_3D'
    
    data_file_name = train_or_test + '_' + str(num_data) + fin_dimension + parameter_type 
      
    parameter_savefilepath = '../Datasets/Thermal_Fin/' + 'parameter_' + train_or_test + '_%d' %(num_data) + fin_dimension + parameter_type

    observation_indices_full_savefilepath = '../Datasets/Thermal_Fin/' + 'obs_indices_full' + fin_dimension
    observation_indices_bnd_savefilepath = '../Datasets/Thermal_Fin/' + 'obs_indices_bnd' + fin_dimension
    
    state_full_savefilepath = '../Datasets/Thermal_Fin/' + 'state_' + train_or_test + '_%d' %(num_data) + fin_dimension + '_full' + parameter_type
    state_bnd_savefilepath = '../Datasets/Thermal_Fin/' + 'state_' + train_or_test + '_%d' %(num_data) + fin_dimension + '_bnd' + parameter_type

###############################################################################
#                                  Plotting                                   #
###############################################################################
    #=== Fenics Solver ===#
    if generate_2D == 1:
        V, mesh = get_space_2D(40)
    if generate_3D == 1:
        V, mesh = get_space_3D(40)
    solver = Fin(V) 
    
    #=== Loading Parameter, State and Observation Indices ===#
    df_parameter = pd.read_csv(parameter_savefilepath + '.csv')
    parameter = df_parameter.to_numpy()
    
    df_state = pd.read_csv(state_full_savefilepath + '.csv')
    state = df_state.to_numpy()
    
    df_obs_indices = pd.read_csv('../Datasets/Thermal_Fin/' + 'obs_indices_bnd' + fin_dimension + '.csv')    
    obs_indices = df_obs_indices.to_numpy() 
    
    #=== Converting Into Fenics Objects ===#
    if generate_nine_parameters == 1:
        parameter_dl = solver.nine_param_to_function(parameter)
        parameter_values = parameter_dl.vector().get_local()  
    if generate_varying == 1:
        parameter_dl = convert_array_to_dolfin_function(V,parameter)
    state_dl, _ = solver.forward(parameter_dl) # generate true state for comparison
    state = state_dl.vector().get_local()    
        
    #=== Plotting ===#
    if generate_2D == 1:
        p_fig = dl.plot(parameter_dl)
    if generate_3D == 1:   
        p_fig = plot_3D(parameter_dl, 'Parameter', angle_1 = 90, angle_2 = 270)
    plt.colorbar(p_fig)
    plt.show()
    
    if generate_2D == 1:
        s_fig = dl.plot(state_dl)
    if generate_3D == 1:     
        s_fig = plot_3D(state_dl, 'Parameter', angle_1 = 90, angle_2 = 270)
    plt.colorbar(s_fig)
    plt.show()
    
