import matplotlib.pyplot as plt
from dolfin import *
from mshr import Rectangle, Box, generate_mesh
from Thermal_Fin_Heat_Simulator.Utilities.plot_3D import plot_mesh_3D

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

###############################################################################
#                        Two Dimensional Thermal Fin                          #
############################################################################### 
def get_space_2D(resolution):
    #=== Defining Geometry ===#
    geometry = Rectangle(Point(2.5, 0.0),  Point(3.5, 4.0)) \
             + Rectangle(Point(0.0, 0.75), Point(2.5, 1.0)) \
             + Rectangle(Point(0.0, 1.75), Point(2.5, 2.0)) \
             + Rectangle(Point(0.0, 2.75), Point(2.5, 3.0)) \
             + Rectangle(Point(0.0, 3.75), Point(2.5, 4.0)) \
             + Rectangle(Point(3.5, 0.75), Point(6.0, 1.0)) \
             + Rectangle(Point(3.5, 1.75), Point(6.0, 2.0)) \
             + Rectangle(Point(3.5, 2.75), Point(6.0, 3.0)) \
             + Rectangle(Point(3.5, 3.75), Point(6.0, 4.0)) \

    #=== Generating Mesh ===#
    mesh = generate_mesh(geometry, resolution)    
    
    #=== Plotting Mesh ===#
    plot(mesh)
    plt.show()
    
    #=== Generating Function Space ===#
    V = FunctionSpace(mesh, 'CG', 1)

    return V, mesh

###############################################################################
#                        Three Dimensional Thermal Fin                        #
###############################################################################
def get_space_3D(resolution):
    #=== Defining Geometry ===#
    geometry = Box(Point(2.5, 0.0, 0.0),  Point(3.5, 4.0, 0.5)) \
             + Box(Point(0.0, 0.75, 0.0), Point(2.5, 1.0, 0.5)) \
             + Box(Point(0.0, 1.75, 0.0), Point(2.5, 2.0, 0.5)) \
             + Box(Point(0.0, 2.75, 0.0), Point(2.5, 3.0, 0.5)) \
             + Box(Point(0.0, 3.75, 0.0), Point(2.5, 4.0, 0.5)) \
             + Box(Point(3.5, 0.75, 0.0), Point(6.0, 1.0, 0.5)) \
             + Box(Point(3.5, 1.75, 0.0), Point(6.0, 2.0, 0.5)) \
             + Box(Point(3.5, 2.75, 0.0), Point(6.0, 3.0, 0.5)) \
             + Box(Point(3.5, 3.75, 0.0), Point(6.0, 4.0, 0.5)) \

    #=== Generating Mesh ===#
    mesh = generate_mesh(geometry, resolution)
    
    #=== Plotting Mesh ===#
    plot_mesh_3D(mesh, title = '3D Thermal Fin', angle = 180)
    plt.show()
    
    #=== Generating Function Space ===#
    V = FunctionSpace(mesh, 'CG', 1)

    pdb.set_trace()

    return V, mesh
