#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 19:12:38 2019

@author: Sheroze Sheriffdeen, Oden Institute of Computational Sciences and Engineering

To avoid using dolfin when training the neural network, the data generation and file is separated. Run this to generate and save thermal fin data.
"""
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import mpi4py
pdb.set_trace()
import dolfin as dl
import matplotlib as plt
from Thermal_Fin_Heat_Simulator.Utilities.gaussian_field import make_cov_chol
from Thermal_Fin_Heat_Simulator.Utilities.forward_solve import Fin
from Thermal_Fin_Heat_Simulator.Utilities.thermal_fin import get_space_2D, get_space_3D
from Thermal_Fin_Heat_Simulator.Utilities.plot_3D import plot_3D

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"
###############################################################################
#                                   Methods                                   #
############################################################################### following Sheroze's "test_thermal_fin_gradient.py" code
def generate_thermal_fin_data(data_file_name, num_data, generate_nine_parameters, generate_varying, generate_2D, generate_3D):
    #=== Generate Dolfin function space and mesh ===#
    if generate_2D == 1:
        V, mesh = get_space_2D(40)
        kern_type = 'm32'
        cov_length = 0.8
    if generate_3D == 1:    
        V, mesh = get_space_3D(40)
        cov_length = 0.8
        kern_type = 'm52'
    solver = Fin(V)
    print(V.dim())  
  
    #=== Create storage arrays ===#
    if generate_nine_parameters == 1:
        parameter_data = np.zeros((num_data, 9))
    if generate_varying == 1:
        parameter_data = np.zeros((num_data, V.dim()))
    
    obs_indices_full = list(range(V.dim()))
    state_data_full = np.zeros((num_data, V.dim()))
    
    obs_indices_bnd = list(set(sum((f.entities(0).tolist() for f in dl.SubsetIterator(solver.boundaries, 1)), []))) # entries of this vector represent which of the (V.dim() x 1) vector of domain indices correspond to the boundary; NOT the degrees of freedom  
    state_data_bnd = np.zeros((num_data, len(obs_indices_bnd)))

    # check if actually boundary points
    #mesh_coordinates = mesh.coordinates()
    #obs_coor = np.zeros((len(obs_indices_full),3))        
    #obs_counter = 0
    #for ind in obs_indices_full:
        #obs_coor[obs_counter,:] = mesh_coordinates[ind,:]
        #obs_counter = obs_counter + 1
        
    #pdb.set_trace()
    
    #=== Generating Parameters and State ===#
    for m in range(num_data):
        print('\nGenerating: ' + data_file_name)
        print('Data Set %d of %d\n' %(m+1, num_data))
        # Generate parameters
        if generate_nine_parameters == 1:
            parameter_data[m,:], parameter_dl = parameter_generator_nine_values(V, solver, generate_2D, generate_3D, kern_type, cov_length)
        if generate_varying == 1:
            parameter_data[m,:], parameter_dl = parameter_generator_varying(V, solver, generate_2D, generate_3D, cov_length)              
        # Solve PDE for state variable
        state_dl, _ = solver.forward(parameter_dl)    
        state_data = state_dl.vector().get_local()
        state_data_full[m,:] = state_data
        state_data_bnd[m,:] = state_data[obs_indices_bnd]         
        
    return parameter_data, state_data_full, state_data_bnd, obs_indices_full, obs_indices_bnd

def parameter_generator_nine_values(V, solver, generate_2D, generate_3D, kern_type = 'm32', length = 0.8):
    chol = make_cov_chol(V, generate_2D, generate_3D, kern_type, length = length)
    norm = np.random.randn(len(chol))
    generated_parameter = np.exp(0.5 * chol.T @ norm) 
    parameter_dl = convert_array_to_dolfin_function(V,generated_parameter)
    generated_parameter = solver.subfin_avg_op(parameter_dl)
    parameter_dl = solver.nine_param_to_function(generated_parameter)
    if generate_3D == 1: # Interpolation messes up sometimes and makes some values equal 0
        parameter_values = parameter_dl.vector().get_local()  
        zero_indices = np.where(parameter_values == 0)[0]
        for ind in zero_indices:
            parameter_values[ind] = parameter_values[ind-1]
        parameter_dl = convert_array_to_dolfin_function(V, parameter_values)
    
    return generated_parameter, parameter_dl

def parameter_generator_varying(V, solver, generate_2D, generate_3D, kern_type = 'm32', length = 0.8):
    chol = make_cov_chol(V, generate_2D, generate_3D, kern_type, length = length)
    norm = np.random.randn(len(chol))
    generated_parameter = np.exp(0.5 * chol.T @ norm) 
    parameter_dl = convert_array_to_dolfin_function(V,generated_parameter)
    generated_parameter = parameter_dl.vector().get_local()
    
    return generated_parameter, parameter_dl

def convert_array_to_dolfin_function(V, nodal_values):
    nodal_values_dl = dl.Function(V)
    nodal_values_dl.vector().set_local(np.squeeze(nodal_values))
    
    return nodal_values_dl  

###############################################################################
#                                  Driver                                     #
###############################################################################
if __name__ == "__main__":  

    #################################
    #   Run Options and File Names  #
    #################################     
    #=== Number of Data ===#
    num_data = 50000

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

    #############################
    #   Generate and Save Data  #
    ############################# 
    #=== Generating Data ===#   
    parameter_data, state_data_full, state_data_bnd, obs_indices_full, obs_indices_bnd = generate_thermal_fin_data(data_file_name, num_data, generate_nine_parameters, generate_varying, generate_2D, generate_3D)
    
    #=== Saving Data ===#  
    df_parameter_data = pd.DataFrame({'parameter_data': parameter_data.flatten()})
    df_parameter_data.to_csv(parameter_savefilepath + '.csv', index=False)  
    
    df_obs_indices_full = pd.DataFrame({'obs_indices': obs_indices_full})
    df_obs_indices_full.to_csv(observation_indices_full_savefilepath + '.csv', index=False)  
    df_obs_indices_bnd = pd.DataFrame({'obs_indices': obs_indices_bnd})
    df_obs_indices_bnd.to_csv(observation_indices_bnd_savefilepath + '.csv', index=False)  
    
    df_state_data_full = pd.DataFrame({'state_data': state_data_full.flatten()})
    df_state_data_full.to_csv(state_full_savefilepath + '.csv', index=False)  
    df_state_data_bnd = pd.DataFrame({'state_data': state_data_bnd.flatten()})
    df_state_data_bnd.to_csv(state_bnd_savefilepath + '.csv', index=False) 
    print('\n All Data Saved')
