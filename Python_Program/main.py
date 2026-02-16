#!/usr/bin/env python
# -----------------------------------------------------------------------------
# A. The Python program includes two source files: main.py and FDEmodel.py.
#
#    The main.py provides main() as the program's driver.
#    The FDEmodel.py declares a Python class called "model" that provides all
#    the class functions to define three dynamical systems.
#    In the model class, some functions are designed to import recursively 
#    updated information from the driver main().
# 
# B. Model: 
#
#    The Python program is designed to solve a mathematically complicated 
#    dynamical system formed by two mixed-type functional differential equations
#    (FDEs) resulting from a neoclassical growth model that features the element
#    of  "time-to-build" capital. This mixed-type FDE system is given below: 
#           
#    dk(t)/dt = A * k(t-tau)^alpha - c(t) - delta * k(t-tau)                (1)
#
#    dc(t)/dt = (1/sigma) * c(t) * ( [A * alpha * k(t)^(alpha-1) - delta]
#
#               *[c(t)/c(t+tau)]^sigma * e^(-rho*tau) - rho )               (2)
#                               
#    Independent variable: t in [0, inf), tau > 0
#
#    Dependent variables: k(t), c(t)
#
#       k(t) - capital stock at time t, a predetermined state variable.
#       c(t) - consumption at time t, a jump variable.
#
#    In the long-run steady state: 
#
#       k(t) -> k_ss at t -> infinity
#       c(t) -> c_ss at t -> infinity 
#    
#    In the present (t = 0):
#
#       k(t) = k0 > 0 at t = 0
#       c(t) at t = 0: not predetermined 
# 
#    History in the pre-shape interval [-tau, 0):
#
#       k(t) = k0 > 0 at t in [-tau, 0)
#       c(t) = no available data at t in [-tau, 0)
#
#    -------------------------------------------------------------
#    In general, for a typical mixed-type FDE system, the history
#    of the system's pre-shape interval for t in [-tau, 0) can be
#    described by a "history" function, phi(t):
#
#    y(t) = phi(t) at t in [-tau, 0), where y(t) = [k(t), c(t)]
#  
#    For the "time-to-build" growth model, we assumed that
#    
#    phi(t) = [k(t) = k0 and c(t) = unknown] for t in [-tau, 0)
#    -------------------------------------------------------------
#
#    Time domain:
#
#    To solve the FDE system, we chose tmax=250 as a proxy for infinity.
#    Thus, we truncate the semi-infinite horizon [0, inf) into a finite
#    horizon [0, tmax]. This means that the FDE system has reached 
#    its long-run steady state approximately at t = tmax.
#
#    However, To make the FDE system well-posed. we must use the expanded
#    time domain of [-tau, tmax + tau]; otherwise the FDE system would be
#    ill-posed.
#   
#    Parameters:
#
#       A    : technical shift (A > 0)                                          
#       rho  : time preference (rho > 0))                                         
#       alpha: capital share   (0 < alpha < 1)                                         
#       sigma: relative risk aversion (sigma > 0)
#              => 1/sigma = elasticity of intertemporal substitution
#       delta: capital depreciation rate (0 < delta < 1)                     
#       tau  : the delay/advance parameter (tau > 0)  
#
# C. Mixed-typ Functional Differential Equations (MFDEs):
#
#    Equations (1)-(2) represent a mixed-typed FDE system because equation (1)
#    is a "delay differential equation" (DDE) due to the delay term, k(t-tau)
#    while equation (2) is an "advance differential equation" (ADE) due to
#    the advance term, c(t+tau). 

#    FDEs are much more complicated than ODEs (ordinary differential equations) 
#    in that ODEs are not subject to either delay or advance terms.
#
#    Certainly, if parameter tau is set equal to zero, equations (1)-(2) reduce
#    to a regular ODE system, as often seen in continuous-time macroeconomic
#    models. 
# 
#    The FDE system presents a complicated boundary value problem (BVP) because
#    the system's current state is subject to non-local history and future. 
#
# D. Algorithm: 
#
#    The Python program uses a homotopic continuation method to solve the FDE
#    system. This numerical method, proposed and demonstrated in Lin (2018), is
#    to solve the FDE system (1)-(2) by solving a functional sequence of 
#    p-constructed homotopic systems with p in [0, 1]. 
#    
#    All these p-constructed homotopic systems, or the "Homotopy" for brevity, 
#    are formed by ODEs since we design an approximant (provided in the class) 
#    to parametrize the lag and lead terms that exist in the mixed-type FDE
#    system.
#
#    The homotopic-continuation algorithm requires three major steps:
#   
#    Step 1: Construct the homotopy as a convex combinations of a chosen start
#            system  G(x, y) and the target FDE system F(x, y, y_lag, y_lead),
#            where G(.) is an ODE system and F(.) is the system (1)-(2).
#
#            Such a homotopy is constructed by
#                                                                               
#            H(x, y, p) = (1-p)*G(x, y) + p*F(x, y, y_lag, y_lead))             
#                                                                               
#            where                                                           
#                                                                               
#            x is the independent time variable;
#            
#            y = [k, c] is a vector of dependent  variables;
#
#            G(x, y) is the chosen start system formed by regular ODEs;                     
#            
#            F(x, y, y_lag, y_lead) is the target FDE system; and            
#
#            p in [0, 1] is the continuation parameter.                      
#                                                                               
#            Note that as p = 0, H(x, y, p) = G(x, y). So, by letting p increase 
#            little by little, the homotopy can finally deform into the target 
#            system; that is, as p = 1, H(x, y, p) = F(x, y, y_lag, y_lead).
#    
#    Step 2: Solve G(x,y) to obtain an initial solution y(x), where x is the
#            independent time variable.
#
#    Step 3: Use the initial solution together with the pre-shape history to
#            form the approximant function to parametrize y_lag and y_lead
#            [i.e. k(t-tau) and c(t+au)] in the target FDE system. In so doing, 
#            we can treat the homotopy H(x, y, p) as a regular ODE system. 
#            
#            Then we can solve the homotopy to obtain an updated solution and
#            use it to update the approximant so that we can parametrize the
#            lead and lag terms again. As this process keeps going on, p keeps 
#            increasing toward one while solving and updating the homotopic
#            system recursively until the homotopy has deformed into the
#            target FDE system (1)-(2) at p = 1.
#
#   In short. to start the homotopic iteration process, we set p = 0 and solve
#   G(x, y) for y(x), x in [0, tmax), where tmax is a proxy for infinity. We
#   then use the solution y(x) and history to form the approximant so that 
#   the algorithm knows how to predetermine y_lag and y_lead and solves 
#   the homotopy H(x, y, p) recursively by increasing the value of p until the
#   homotopy has deformed into the target FDE system at p=1.
#  
# E. Run the code:
#
#    To run the Python program, you will need to place main.py (the driver) and
#    FDEmodel.py (the class) in the same folder (directory). On the command line,
#    enter "python main.py" and wait to see the iteration process and graphical
#    output in a second.
#
# F. Accuracy: 
#   
#    The degree of accuracy is subject to:
#
#    tol (relative error tolerance), 
#    atol (absolute error tolerance), and
#    tmax (the upper bound of the finite horizon [0,tmax]),
#
#    where tol is required for using the Python BVP solver, atol is used check
#    whether the solution for a p-specific homotopy has converged, and tmax is
#    used to construct a finite horizon as a approximation of the semi-infinite
#    [0, inf).
#   
#    In the Python program, I set tol=1e-6, atol=1e-6, and tmax=250. So, the 
#    solution for the FDE system is accurate enough. If you wish to demand more 
#    accuracy, you can reduce tol and atol to some extent. You can also raise
#    the value of tmax for a test run.
#
#    For the continuation parameter p, the interval [0, 1] is discretized into
#    a p_mesh = {0, 0.25, 0.50, 0.75, 1}. You are free to make a denser p_mesh 
#    and re-run the code to see how the homotopy deformation will transpire.
#
#    In the program, I set the delay parameter at tau=20. You are free to change
#    this parameter value as well.
#
#    Note that if tau is set at 2, the time-path cycles in transition will 
#    become almost visible. For the numerical solution to show significant cycles,
#    tau can be set at 15 or 20.   
#
# G. Reference: 
# 
#    Lin, Hwan C. (2018). Computing transitional cycles for a time-to-build
#    growth model. Computational Economics, DOI: 10.1007/s10614-016-9633-9
#               
#    Lin, Hwan C. and L.F. Shampine (2018). R&D-based calibrated growth models
#    with finite-length patents: A novel relaxation algorithm for solving an
#    autonomous FDE system of mixed type. Computational Economics, 1-36.
#    DOI:10.1007/s10614-016-9597-9.
#
# H. Author: Hwan C. Lin, Department of Economics, UNC-Charlotte 
#    Email:  hwlin@charloote.edu; hwanlin@gmail.com
#    Date:   January 15, 2026
# -----------------------------------------------------------------------------

import sys
import numpy as np
import copy
from FDEmodel import *
from scipy.integrate import solve_bvp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import time

def main():
    """
    The driver lays out model parameters and export them to the class
    called "model." We also set up and export algorithmic parameters to the 
    class. The model parameters allow the code to compute long-run steady
    states for the three dynamical systems: the simple Start system, 
    the Homotopic system, and the Target FDE system. The driver let each
    dynamical system start with the initial capital stock k0, which is set
    5% below the target system's steady-state level. The driver then
    exports information to the class to prepare the two-point boundary
    conditions and the history in the pre-shape interval [-tau, 0). Finally,
    the driver begins the homotopic deformation process recursively to
    approach the solution for the target FDE system, given two predetermined
    error-tolerance controls.
    """
    # --------------------
    # A. Model parameters
    # --------------------
    rho   = 0.05   # time preference        
    alpha = 0.3    # capital share     
    A     = 1.0    # technical shift         
    sigma = 1.5    # relative risk aversion 
    delta = 0.10   # capital depreciation rate
    tau   = 20     # The 'delay'/'advance' parameter tau >=0. 
                 
    # -----------------------------------------------------
    # B. Instantiate dydx as an object of the class "model"
    # -----------------------------------------------------
    # The object, dydx, serves as a channel to import 
    # information from the class or export information
    # to the class.
    # -----------------------------------------------------
    dydx = model(rho, alpha, A, sigma, delta, tau)

    # ---------------------------------------------------------------------
    # C. Compute steady states for G(.), H(.), and F(.)
    # ---------------------------------------------------------------------
    # As defined in the class, the code treats:
    #    1) G(self, x, y) as the Start system (ODEs);
    #    2) F(self, x, y, y_lag, y_lead) as the Target system (FDEs);
    #    3) H(self, x, y, y_lag, y_lead, p) as the homotopic system (FDEs).
    # ---------------------------------------------------------------------
    Steady_State = True
    dydx.pass_flag_ss(Steady_State)   # Pass the steady-state flag to the class. 

    k_guess = 1.5
    c_guess = 1.0
    y_guess = [k_guess, c_guess]

    y_ss_G = fsolve(dydx.G_ss, y_guess)
    y_ss_F = fsolve(dydx.F_ss, y_guess)

    H_ss = lambda p: (1-p) * y_ss_G + p * y_ss_F

    # where H_ss is a function of p so that H_ss(p) is a p-constructed 
    # homotopy's steady state. As will be defined below, p is in [0, 1]. 
    
    # --------------------------
    # D. Algorithmic parameters 
    # --------------------------
    tmax      = 250                    # Truncate [0, inf) into [0, tmax].
    M         = 25                     # Set M nodes in time mesh
    N         = 5                      # Set N nodes in pmesh.
    pmesh     = np.linspace(0, 1, N)   # pmesh in [0,1] with N nodes.
    rtol      = 1e-6                   # Relative error for solving BVP.
    max_nodes = 1000                   # Maximum nodes for scipy.solve_bvp.
    maxiter   = 500                    # For each p, maximum iterations are set 
                                       # for solving p-constructed Homotopy.

    dydx.set_time_horizon(tmax)        # Export tmax to the class.

    # -------------------------------------------------------------
    # E. Prepare information for Start system's boundary conditions
    # -------------------------------------------------------------
    # 1) Retrieve steady state 
    k_ss_G, c_ss_G = y_ss_G[0], y_ss_G[1]
    k_ss_F, c_ss_F = y_ss_F[0], y_ss_F[1]

    # 2) Set initial capital stock 5% below the target system's steady state
    k0 = 0.95*k_ss_F
    
    dydx.set_initial_capital(k0)       # Export k0 to the class.
                                       # In the code, all the dynamical systems
                                       # start with the same capital stock, k0.

    # 3) Construct boundary conditions
    bc_values = [k0, c_ss_G]

    dydx.set_bc_G(bc_values)           # Export boundary values to the class
                                       # for constructing boundary conditions.

    # ---------------------------------------------------------
    # F. Prepare collocation points and initial solution guess
    # ---------------------------------------------------------
    x_mesh = np.linspace(0, tmax, M)          
    k_mesh = np.ones(M)*k_ss_G
    c_mesh = np.ones(M)*c_ss_G
    
    y_mesh = np.array([k_mesh, c_mesh])

    # --------------------------------------
    # G. Solve the Start system (ODEs): G(.)
    # --------------------------------------
    start_time = time.time()

    Steady_State = False

    dydx.pass_flag_ss(Steady_State)    # Inform the class to return to
                                       # the transition mode.

    solution = solve_bvp( dydx.G,              # start system; p = 0 
                          dydx.bc_G,           # 2-point boundary conditions 
                          x_mesh,              # time mesh (collocation points)
                          y_mesh,              # solution guess
                          tol=rtol,            # relative error tolerance
                          max_nodes=max_nodes, # maximum nodes in time mesh
                          verbose=0 )          # verbose = 0, 1, or 2 

    print('\nFor the start system (ODEs), solution.status: ', solution.status)

    sol_G = copy.deepcopy(solution)            # Keep the solution if you plan
                                               # to plot it later. Delete this 
                                               # code line if you plan not.

    # -------------------------------------------------------------------------- 
    # H. Solve a sequence of self-updated homotopic systems recursively
    #    until the Homotopy has deformed into the Target system (FDEs).
    # -------------------------------------------------------------------------
    #    a) For the homotopic deformation, we prepare a grid called "pmesh"
    #       with pmesh = {0, ..., 1} and pmesh[0] = 0 and pmesh[N-1] = 1.
    #
    #    b) At p = pmesh[0] = 0, the Homotopy reduces to the Start system. 
    #       Its solution has been obtained above.
    #
    #    c) At p = pmesh[1] > 0, the homotopic deformation starts out in
    #       following for loop indexed by i. At step (b), the solution serves as 
    #       the very first solution guess for the p-constructed Homotopy:
    #     
    #       H(x, y, y_lag, y_lead,,p) = (1-p)*G(x,y) + p*F(x, y, y_lag, y_lead).
    #
    #       We use the solution guess to parametrize y_lag and y_lead) and treat
    #       the homotopy as a regular ODE system.
    #
    #    d) At step (c), we need to enter an inner for-loop indexed by k to
    #       a satisfactory solution that is sufficiently close the solution
    #       guess. In this for-loop, we keep solving the p-constructed homotopy,
    #       which correspond to p = pmesh[1] by updating the solution guess
    #       recursively.
    #
    #    e) Once a satisfactory solution is obtained, we move to another p,
    #       which is pmesh[2]. That is, the deformation process restarts at
    #       step (c). This iteration keeps going on and  until p reaches 
    #       pmesh[N-1] = 1. At this point, the homotopy has deformed into the 
    #       Target FDE system with success, meaning that we have solved the 
    #       mixed-type FDE system.e
    # -------------------------------------------------------------------------
    
    print ('  \n\n  **************************************************************')
    print ('  Report: Homotopic Deformation via continuation index p in [0, 1]')
    print ('  --------------------------------------------------------------')
    print ('  a) Index i: the number of homotopic deformation (continuation)')
    print ('  b) Index k: the number of iterations for solving Homotopy') 
    print ('  c) max_atol: the maximum absolute error of an iteration')
    print ('  **************************************************************')
    
    for i in range(1, N):  
         
        p = pmesh[i]                         # Initialize p, 
        dydx.set_homotopy_continuation(p)    # Pass p value to the class.

        y_ss_H = H_ss(p)                     # Compute H's steady state. 
        k_ss_H = y_ss_H[0]                   # Initialize steady state for
        c_ss_H = y_ss_H[1]                   # the p-constructed homotopy.

        dydx.pass_HomotopySS(y_ss_H)         # Pass steady state to the class.
        dydx.pass_solution(solution)         # Pass solution to the class.

        bc_values = [ k0, c_ss_H ]           # Boundary conditions.
        dydx.set_bc_H(bc_values)             # Pass boundary  conditions 
                                             # to the class. 

        y_mesh = solution.sol(x_mesh)        # Solution guess to be updated
                                             # recursively at the same
                                             # collocation points.

        atol = 1e-6    # Absolute error tolerance
                       # for the k-loop iteration to convergence.

        for k in range(1, maxiter+1):   

            solution = solve_bvp(dydx.H,           # Homotopic system
                                 dydx.bc_H,        # updated boundary conditions
                                 x_mesh,           # collocation points 
                                 y_mesh,           # self-updated solution guess
                                 tol=rtol,         # relative error tolerance
                                 max_nodes=max_nodes,
                                 verbose=0)

            # ------------------------------------------------------------------ 
            # Check on convergence
            # ------------------------------------------------------------------
            # tol: used to implement scipy.integrate.solve_bvp to obtain
            #      a BVP solution for a p-constructed homotopy corresponding to
            #      a specific k index.
            # atol: used to check if the BVP solution corresponding to the k
            #       index can satisfy the convergence condition.
            # Note: tol (atol) is relative (absolute)  error tolerance,  
            # ------------------------------------------------------------------

            max_atol = np.max(np.abs(solution.sol(x_mesh) - y_mesh))

            # print out the iteration process
            print(f'  Homotopy i = {i:2g}   p[i] = {p:4.4f}   k = {k:2g}   max_atol = {max_atol:6.6f}')

            # Determine whether the k-loop convergence condition is satisfied
            if max_atol < atol: 
                print('\n')
                break                 # Exit the k-loop and re-enter the p-loop,

            # Determine whether the number of iterations has reached maximum
            if k == maxiter:
                print ("  Unable to converge with maximum iteration")
                quit()
            
            # Do some updating and re-enter the k-loop
            y_mesh = solution.sol(x_mesh)      # Update the solution guess  
            dydx.pass_solution(solution)       # Pass "solution" to the class.


    # --------------------------------------------------------------        
    # I. End of the homotopic continuation for solving mixed-type FDEs
    # --------------------------------------------------------------
    elapsed_time_secs = time.time() - start_time
    print ('\n  Execution took: %8.4f secs (Wall clock time)' % elapsed_time_secs)
    print ('\n  For the FDE system, solution.status is: ', solution.status)
    print ('\n  Message: ', solution.message)

    sol_F = solution

    # Transition paths of k and c for the start system (ODEs)
    
    def k_G(t):
        return sol_G.sol(t)[0]

    def c_G(t): 
        return sol_G.sol(t)[1]

    # Transitional paths of k and c for the target system (FDEs)
    def k_F(t):
        return sol_F.sol(t)[0]

    def c_F(t):
        return sol_F.sol(t)[1]

    #Print steady states for Start System (G, ODEs) and Target System (F, FDEs)
    print ('\n  **************************************************')
    print ('  Computed Results for the ODE(G) and FDE(F) Systems')
    print ('  **************************************************')
    print (f'\n  Initial capital stock: k0 = {k0: 6.4f}, 5% below steady state')
    print ('\n  *** Steady State for Start System (G) ***')
    print (f'  k_ss_G    = {k_ss_G: 8.6f}  c_ss_G    = {c_ss_G: 8.6f}')
    print (f'  k_G(tmax) = {k_G(tmax): 8.6f}  c_G(tmax) = {c_G(tmax): 8.6f}')
    
    print ('\n  *** Steady State for Target System (F) ***')
    print (f'  k_ss_F    = {k_ss_F: 8.6f}     c_ss_F = {c_ss_F: 8.6f}')
    print (f'  k_F(tmax) = {k_F(tmax): 8.6f}  c_F(tmax) = {c_F(tmax): 8.6f}')
    
    print ('\n  *** Steady State for the Homotopy (H) ***')
    print (f'  k_ss_H    = {k_ss_H: 6.6f} at p=1')
    print (f'  c_ss_H    = {c_ss_H: 6.6f} at p=1')
    print ('\n  *** Check on convergence for Target system (F) ***') 
    print (f'  k_F(tmax)/k_ss_F - 1 = {k_F(tmax)/k_ss_F-1: 8.6f}')
    print (f'  c_F(tmax)/c_ss_F - 1 = {c_F(tmax)/c_ss_F-1: 8.6f}\n')

    #Plot transition paths for the Target FDE system
    t = np.linspace(0, tmax, 200)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    ax1.plot(t, k_F(t)/k_ss_F-1, 'b-', lw=2,label=r'$\hat{k}(t)=\frac{k(t)-\bar{k}}{k(t)}$')
    ax1.legend(loc='lower right', shadow=True)
    ax1.plot(t, np.zeros(np.size(t)), 'r--')
    ax1.plot([0],[k0/k_ss_F-1], 'g.', ms=20)
    ax1.set_title(r'Fig. 1: Capital Dynamics in FDE System ($\tau$ = %2g yrs)' % tau)
    ax1.text(16, 0.005, r'% deviation of capital from steady state, $\bar{k}$', color='blue', fontsize=11)
    #ax1.set_xlabel(r'time $t$')
    ax1.set_ylabel(r'$\hat{k}$')
    ax1.set_xlim([-5, 150])
    ax1.set_ylim([-0.06, 0.02])
    ax1.grid(True)

    ax2.plot(t, c_F(t)/c_ss_F-1, 'b-', lw=2, label=r'$\hat{c}(t)=\frac{c(t)-\bar{c}}{c(t)}$')
    ax2.legend(loc='lower right', shadow=True)
    ax2.plot(t, np.zeros(np.size(t)), 'r--')
    ax2.plot([0], [c_F(0)/c_ss_F-1], 'k.', ms=20)
    ax2.set_title(r'Fig. 2: Consumption Dynamics in FDE System ($\tau$ = %2g yrs)' % tau)
    ax2.text(16, 0.0018, r'% deviation of consumption from steady state, $\bar{c}$', color='blue', fontsize=11)
    ax2.set_xlabel(r'time $t$')
    ax2.set_ylabel(r'$\hat{c}$')
    ax2.set_xlim([-5,150])
    ax2.set_ylim([-0.025, 0.005])
    ax2.grid(True)

    plt.savefig('FDE_solution.png')

    plt.tight_layout()
    plt.show()

main()

