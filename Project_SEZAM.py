# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:55:34 2023

@author: Samuel
"""

from util import np
from quad import QuadRule, seven_point_gauss_6
from integrate import stiffness_with_diffusivity_iter, mass_with_DATA_reaction_iter, \
                      assemble_matrix_from_iterables, \
                      poisson_rhs_iter, assemble_rhs_from_iterables
                      
from solve import solve_with_dirichlet_data
from mesh import Triangulation

def fixed_point(mesh: Triangulation, quadrule: QuadRule, alpha: float, guess_data):
  """
    FEM solution of the problem

      -∆u + a g^2 u = 1    in  Ω
                  u = 0    on ∂Ω
          #with g := guess and a := alpha.
    Parameters
    ----------
    mesh: `Triangulation`
      The mesh that represents the domain Ω
    quadrule: `QuadRule`
      quadrature scheme used to assemble the system matrices and right-hand-side.
    alpha: `float`
      the parameter #TODO : check what it corresponds to
    guess_data: `np.array(nP,)`
      array containing the value of our guess at all nodes (!= quadrature points, BEWARE) #TODO : check number of points
    Returns
    -------
    returns the data of a new (hopefully better) guess.
  """
  
  # Creation of matrix A, using our paramters to do the reaction term.
  S_iter = stiffness_with_diffusivity_iter(mesh, quadrule) #TODO : check to do StiffM. once for all.
  M_iter = mass_with_DATA_reaction_iter(mesh, quadrule, alpha * guess_data)
  A = assemble_matrix_from_iterables(mesh, S_iter, M_iter)
  
  #TODO : check to do r.h.s. once for all.
  # Create the r.h.s. vector.
  f = lambda x: np.array([100])
  P_iter =  poisson_rhs_iter(mesh, quadrule, f)
  rhs = assemble_rhs_from_iterables(mesh, P_iter)

  bindices = mesh.boundary_indices
  data = np.zeros(bindices.shape, dtype=float)

  solution = solve_with_dirichlet_data(A, rhs, bindices, data)

  mesh.tripcolor(solution)
  print(solution.shape) # On va __return__ cette solution comme étant notre "nouveau guess"
  


if __name__ == '__main__':
  ALPHA = 0.1
  
  square = np.array([ [0, 0],
                      [1, 0],
                      [1, 1],
                      [0, 1] ])
  mesh = Triangulation.from_polygon(square, mesh_size=0.05)
  mesh.plot()
  quadrule = seven_point_gauss_6()
  nP = len(mesh.points)
  
  guess_data = np.zeros((nP,))
  fixed_point(mesh, quadrule, ALPHA, guess_data)
