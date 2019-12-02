#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 11:47:02 2019

@author: hwan - Took out relevant code from dolfin's plotting.py _plot_matplotlib code
              - To enter dolfin's own plotting code, use dl.plot(some_dolfin_object) wheresome_dolfin_object is a 3D object and an error will be thrown up
"""
import matplotlib.pyplot as plt
import dolfin.cpp as cpp

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def plot_2D(obj, title):   
    if hasattr(obj, "cpp_object"):
        obj = obj.cpp_object()
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.title(title)
    return my_mplot_function(ax, obj)
    
def my_mplot_function(ax, f, **kwargs):
    mesh = f.function_space().mesh()
    C = f.compute_vertex_values(mesh)
    mode = kwargs.pop("mode", "contourf")
    if mode == "contourf":
        levels = kwargs.pop("levels", 40)
    return ax.tricontourf(my_mesh2triang(mesh), C, levels, **kwargs)

def my_mesh2triang(mesh):
    import matplotlib.tri as tri
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
