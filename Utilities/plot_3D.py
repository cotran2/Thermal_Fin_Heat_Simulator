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

def plot_3D(obj, title, angle_1, angle_2):    
    # Importing this toolkit has side effects enabling 3d support
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    # Enabling the 3d toolbox requires some additional arguments
    plt.title(title)
    ax = plt.gca(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(angle_1, angle_2)
    
    # For dolfin.function.Function, extract cpp_object
    if hasattr(obj, "cpp_object"):
        obj = obj.cpp_object()
        
    if isinstance(obj, cpp.function.Function):
        return my_mplot_function(ax, obj,)
    elif isinstance(obj, cpp.mesh.Mesh):
        return my_mplot_mesh(ax, obj)

def my_mplot_mesh(ax, mesh):
    tdim = mesh.topology().dim()
    gdim = mesh.geometry().dim()
    if gdim == 3 and tdim == 3:
        bmesh = cpp.mesh.BoundaryMesh(mesh, "exterior", order=False)
        my_mplot_mesh(ax, bmesh)
    elif gdim == 3 and tdim == 2:
        xy = mesh.coordinates()
        return ax.plot_trisurf(*[xy[:, i] for i in range(gdim)],
                               triangles=mesh.cells())
        
def my_mplot_function(ax, f):
    mesh = f.function_space().mesh()
    gdim = mesh.geometry().dim()
    C = f.compute_vertex_values(mesh)    
    X = [mesh.coordinates()[:, i] for i in range(gdim)]
    return ax.scatter(*X, c=C)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
