#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 11:47:02 2019

@author: hwan - Took out relevant code from dolfin's plotting.py _plot_matplotlib code
"""
import matplotlib.pyplot as plt
import dolfin.cpp as cpp

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"


def plot_mesh_3D(mesh, title, angle):    
    # Importing this toolkit has side effects enabling 3d support
    from mpl_toolkits.mplot3d import axes3d  # noqa
    # Enabling the 3d toolbox requires some additional arguments
    plt.title(title)
    ax = plt.gca(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(30, angle)
    
    return my_mplot_mesh(ax, mesh)

def my_mplot_mesh(ax, mesh, **kwargs):
    tdim = mesh.topology().dim()
    gdim = mesh.geometry().dim()
    if gdim == 3 and tdim == 3:
        bmesh = cpp.mesh.BoundaryMesh(mesh, "exterior", order=False)
        my_mplot_mesh(ax, bmesh, **kwargs)
    elif gdim == 3 and tdim == 2:
        xy = mesh.coordinates()
        return ax.plot_trisurf(*[xy[:, i] for i in range(gdim)],
                               triangles=mesh.cells(), **kwargs)