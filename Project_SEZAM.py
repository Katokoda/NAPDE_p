# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:55:34 2023

@author: Sam (Samuel Bélisle) and Ez (Estelle Baup)
"""

from integrate import stiffness_with_diffusivity_iter, mass_with_DATA_reaction_iter, \
                      assemble_matrix_from_iterables, \
                      poisson_rhs_iter, assemble_rhs_from_iterables, \
                      newton_rhs_iter
from mesh import Triangulation
from quad import QuadRule, seven_point_gauss_6
from solve import solve_with_dirichlet_data
from util import np

import matplotlib.pylab as plt
from time import time

def fixed_point_iteration(mesh: Triangulation, quadrule: QuadRule, alpha: float, guess_data):
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
      the parameter inside of the reaction term.
    guess_data: `np.array/(nP,))`
      array containing the value of our guess at all nodes
    Returns
    -------
    returns the data of a new (hopefully better) guess.
  """
  
  # Creation of matrix A, using our paramters to do the reaction term.
  S_iter = stiffness_with_diffusivity_iter(mesh, quadrule)
  M_iter = mass_with_DATA_reaction_iter(mesh, quadrule, alpha, guess_data)
  A = assemble_matrix_from_iterables(mesh, S_iter, M_iter)
    # We could create a Stiffness matrix only one time
    # and receive it as S at each iteration from main loop.
    # then A = S + assemble_matrix_from_iterables(mesh, M_iter).
  
  # Create the r.h.s. vector.
  f = lambda x: np.array([100])
  P_iter =  poisson_rhs_iter(mesh, quadrule, f)
  rhs = assemble_rhs_from_iterables(mesh, P_iter)
    # We also could create the r.h.s. only one time.

  bindices = mesh.boundary_indices
  data = np.zeros(bindices.shape, dtype=float)

  solution = solve_with_dirichlet_data(A, rhs, bindices, data)
  return solution # Our new guess.
  
def fixed_point_method(alpha : float, treshold : float, maxIter : int, size : float):
  """
  Will use iteratively fixed_point_iteration() to update guess_data.
    Parameters
    ----------
    alpha: `float`
      the parameter as in the problem sheet.
    treshold: `float`
      the difference treshold, to decide when we consider it converged.
    maxIter: `int`
      max number of iterations allowed.
    size: `float`
      the size of the grid.
    Returns nothing.
    ----------
    It plot the convergence rate and the final guess.
    If maxIter is reached, it plots the last guesses.
  """
  assert treshold > 0
  assert maxIter > 0
  
  square = np.array([ [0, 0],
                      [1, 0],
                      [1, 1],
                      [0, 1] ])
  
  mesh = Triangulation.from_polygon(square, mesh_size = size)
  quadrule = seven_point_gauss_6()
  
  nP = len(mesh.points)
  assert (nP > 100) # To make sure we respect the assignement.
  
  guess_data = np.zeros((nP,))
  maxDiff = treshold + 1
  
  print("====== Start of fixed-point method ======")
  tStart = time()
  print("Alpha =", alpha, "on", nP, "elements")
  
  k = 0
  differences = [0]*maxIter
  while maxDiff > treshold and k < maxIter:
    old_g_data = guess_data
    guess_data = fixed_point_iteration(mesh, quadrule, alpha, old_g_data)
    maxDiff = np.max(np.abs(old_g_data - guess_data))
    differences[k] = maxDiff
    k += 1
    
    print(k, "/", maxIter, " - ", maxDiff, sep = '')
    if ((maxIter - k) < 5 and k != maxIter):
      print("Warning: reaching iteration treshold.")
      # If we feel that we don't converge fast enough,
      # we plot the last guesses to detect cycles.
      mesh.tripcolor(guess_data)
    
  mesh.tripcolor(guess_data)
  print("Final error:", maxDiff)
  print("Elapsed time:", time() - tStart, "seconds")
  print("======= End of fixed-point method =======")
  
  # ====== Plot
  differences = differences[:k]
  plt.plot(range(k), differences, ".-", label = "$||u_n - u_{n+1}||_\infty$")
  
  plt.xlabel("Iteration number")
  plt.ylabel("Infinity norm of difference")
  plt.title("Fixed-point scheme with alpha = " + str(alpha))
  plt.legend()
  plt.grid()
  plt.show()
  plt.close()


def newton_iteration(mesh: Triangulation, quadrule: QuadRule, alpha: float, guess_data):
  """
    FEM solution of the weak formulation:
      For all phi in H_0^1,
        \int grad(phi) grad(u) + (3 a w^2) phi u dx
        = \int  - grad(phi) grad(w) - phi *(aw^3 - f) dx
      u = 0    on ∂Ω
    with w := function with nodes guess_data and a := alpha.
    Parameters
    ----------
    mesh: `Triangulation`
      The mesh that represents the domain Ω
    quadrule: `QuadRule`
      quadrature scheme used to assemble the system matrices and right-hand-side.
    alpha: `float`
      the parameter inside of the reaction term.
    guess_data: `np.array((nP,))`
      array containing the value of our guess at all nodes
    Returns
    -------
    returns the solution u, used to update our guess
  """
  
  # Creation of matrix A, using our paramters to do the reaction term.
  S_iter = stiffness_with_diffusivity_iter(mesh, quadrule)
  M_iter = mass_with_DATA_reaction_iter(mesh, quadrule, 3*alpha, guess_data)
  A = assemble_matrix_from_iterables(mesh, S_iter, M_iter)
    # We could create a Stiffness matrix only one time
    # and receive it as S at each iteration from main loop.
    # then A = S + assemble_matrix_from_iterables(mesh, M_iter).
  
  # Create the r.h.s. vector.
  f = lambda x: np.array([100])
  N_iter =  newton_rhs_iter(mesh, quadrule, f, alpha, guess_data)
  rhs = assemble_rhs_from_iterables(mesh, N_iter)

  bindices = mesh.boundary_indices
  data = np.zeros(bindices.shape, dtype=float)

  solution = solve_with_dirichlet_data(A, rhs, bindices, data)
  return solution
  
def newton_method(alpha : float, treshold : float, maxIter : int, size : float):
  """
  Will use iteratively newton_method_iteration() to update guess_data.
    Parameters
    ----------
    alpha: `float`
      the parameter as in the problem sheet.
    treshold: `float`
      the difference treshold, to decide when we consider it converged.
    maxIter: `int`
      max number of iterations allowed.
    size: `float`
      the size of the grid.
    Returns nothing.
    ----------
    It plot the convergence rate and the final guess.
    If maxIter is reached, it plots the last guesses.
  """
  assert treshold > 0
  assert maxIter > 0
  
  square = np.array([ [0, 0],
                      [1, 0],
                      [1, 1],
                      [0, 1] ])
  
  mesh = Triangulation.from_polygon(square, mesh_size = size)
  quadrule = seven_point_gauss_6()
  
  nP = len(mesh.points)
  assert (nP > 100) # To make sure we respect the assignement.
  
  guess_data = np.zeros((nP,))
  maxDiff = treshold + 1
  
  print("====== Start of the Newton method ======")
  tStart = time()
  print("Alpha =", alpha, "on", nP, "elements")
  
  k = 0
  differences = [0]*maxIter
  while maxDiff > treshold and k < maxIter:
    delta_data = newton_iteration(mesh, quadrule, alpha, guess_data)
    guess_data += delta_data
    maxDiff = np.max(np.abs(delta_data))
    differences[k] = maxDiff
    k += 1
    
    print(k, "/", maxIter, " - ", maxDiff, sep = '')
    if ((maxIter - k) < 5 and k != maxIter):
      print("Warning: reaching iteration treshold.")
      # If we feel that we don't converge fast enough,
      # we plot the last guesses to detect cycles.
      mesh.tripcolor(guess_data)
    
  mesh.tripcolor(guess_data)
  print("Final error:", maxDiff)
  print("Elapsed time:", time() - tStart, "seconds")
  print("======= End of the Newton method =======")
  
  # ====== Plot
  differences = differences[:k]
  plt.plot(range(k), differences, ".-", label = "$||u_n - u_{n+1}||_\infty$")
  
  plt.xlabel("Iteration number")
  plt.ylabel("Infinity norm of difference")
  plt.title("Newton scheme with alpha = " + str(alpha))
  plt.legend()
  plt.grid()
  plt.show()
  plt.close()
  

if __name__ == '__main__':
  tTot = time()
  fixed_point_method(0.1, 10e-6, 20, 0.05)
  fixed_point_method(2, 10e-6, 500, 0.05)
  newton_method(0.1, 10e-6, 20, 0.05)
  newton_method(2, 10e-6, 20, 0.05)
  newton_method(5, 10e-6, 20, 0.05)
  print("Total elapsed time:", time() - tTot, "seconds")
  print("===== Now : just for fun, a really dense mesh =====")
  newton_method(2, 10e-6, 20, 0.02)
