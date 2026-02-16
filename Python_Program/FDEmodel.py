import numpy as np

class model():
    
    def __init__(self, rho, alpha, A, sigma, delta, tau):
        """
        This is a constructor for the  model class, taking inputs from main().
        These inputs are class members to serve as model parameters. 
        The constructor also creates some empty objects designed to hold some
        data from main() via the following eight class functions:

            1) set_initial_capital(self, k0)

            2) set_time_horizon(self, tmax)
            
            3) set_homotopy_continuation(self, p)
            
            4) set_bc_ODEs(self, bc_values)
            
            5) set_bc_homotopy(self, bc_values)
            
            6) pass_solution(self, solution)
            
            7) pass_homotopySS(self, y_ss_homotopy)
            
            8) pass_flag_ss(self, Steady_State)

        All objects constructed by the constructor are global variables within
        the model class. That is, all class functions can access them.
        """
        # Model Parameters
        self.rho    = rho       # time preference
        self.alpha  = alpha     # capital share
        self.A      = A         # technical shift
        self.sigma  = sigma     # risk aversion
        self.delta  = delta     # depreciation rate
        self.tau    = tau       # delay in years

        # Information to be imported
        self.k0                 = []    # k0: initial capital stock
        self.tmax               = []    # tmax: a proxy for infinity
        
        # Information to be updated and imported recursively
        self.p                  = []    # p: homotopic continuation coefficient
        self.bc_values_G        = []    # boundary values for Target system G (ODEs)
        self.bc_values_H        = []    # boundary values for Homotopy H (FDEs)
        self.solution           = []    # solution to be updated in iteration 
        self.y_ss_H             = []    # steady state for Homotopy

        # Flag for steady state
        self.Steady_State = []          # Inform the class in Steady-State mode
                                        # or in Transition mode.

    # ---------------------------------------------------
    # Class functions designed to import data from main()
    # ---------------------------------------------------

    def set_initial_capital(self, k0): self.k0 = k0

    def set_time_horizon(self, tmax): self.tmax = tmax

    def set_homotopy_continuation(self, p): self.p = p

    def set_bc_G(self, bc_values): self.bc_values_G = bc_values

    def set_bc_H(self, bc_values): self.bc_values_H = bc_values

    def pass_solution(self, solution): self.solution = solution

    def pass_HomotopySS(self, y_ss_H): self.y_ss_H = y_ss_H

    def pass_flag_ss(self, Steady_State): self.Steady_State = Steady_State

    # ----------------------------------------
    # The Start system formed by regular ODEs
    # ----------------------------------------
    def G(self, x, y):  
        """
        G(.) represents the Start ODE systems.
        Input x: independent variable (time)
              y: dependent variable including 2 components (k, c)
        Output: an array of 2 time derivatives (dk/dx, dc/dx)        
        """
        rho, alpha, sigma = self.rho, self.alpha, self.sigma
        A, delta = self.A, self.delta

        k = y[0]
        c = y[1]
        
        # Time derivative of capital
        dkdx = A*k**alpha - c - delta*k

        # Time derivative of consumption
        dcdx = (1.0/sigma)*c*(A*alpha*k**(alpha-1.0) - rho - delta)
        
        return np.array([dkdx, dcdx])


    # --------------------------------------------
    # The Target system formed by mixed-type FDEs
    # --------------------------------------------
    def F(self, x, y):
        """
        F(.) represents the Target FDE system.
        This is a two-dimensional system of mixed-type FDEs.
        We parametrize the FDE system's lag and lead terms using two other
        class function: approx_(x-tau) and approx_c(x+tau). In so doing, we
        can treat the FDE system as an ODE system.

        Input x: independent variable (time)
              y: dependent variable including 2 components (k, c)
        Output: an array of 2 time derivatives (dk/dx, dc/dx)       
        """
        rho, alpha, sigma = self.rho, self.alpha, self.sigma
        A, delta, tau  = self.A, self.delta, self.tau

        Steady_State = self.Steady_State

        k = y[0]
        c = y[1]

        # Parametrization of the lag and lead terms

        if Steady_State:

            k_lag, c_lead  = k, c

        else:

            k_lag, c_lead  = self.approx(x-tau)[0], self.approx(x+tau)[1]

        # Auxiliary variables
        coeff  = np.exp(-rho*tau) * (c/c_lead)**sigma 

        # Time derivative of capital
        dkdx = A*k_lag**alpha - c - delta*k_lag
        
        # Time derivative of consumption
        dcdx = (1.0/sigma)*c*((A*alpha*k**(alpha-1.0)-delta)*coeff - rho)

        return np.array([dkdx, dcdx])


    # --------------------------
    # The p-constructed Homotopy
    # -------------------------- 
    def H(self, x, y):
        """
        This defines the homotopic system as a convex combination of the Start
        system and the Target system, subject continuation parameter p in [0,1].
        
        Input x: independent variable (time)
              y: dependent variable including 2 components (k, c)
        Output: an array of 2 time derivatives (dk/dx, dc/dx)
        
        At p = 0, the Homotopy reduces to G(.), the Start system (ODEs).
        At p = 1, the Homotopy deforms into F(.), the Target system (FDEs).
        """ 
        p = self.p
        
        Homotopy = (1 - p)*self.G(x, y) + p*self.F(x, y)
    
        return Homotopy

    # ------------------------------------------
    # Provide the two-point boundary conditions
    # ------------------------------------------
    def bc_G(self, ya, yb):
        """
        This provides the two-point boundary conditions for the Start system.
        (1) For the predetermined state variable, k, 
                ya[0] = k(t=0) = k0.
        (2) For the jump variable, c,
                yb[1] = c(tmax) = c_ss.
        """
        k0   = self.bc_values_G[0]   
        c_ss = self.bc_values_G[1]

        res = np.zeros(2)
        res[0] = ya[0] - k0
        res[1] = yb[1] - c_ss

        return res
    
    def bc_H(self, ya, yb):
        """
        This provides the two-point boundary conditions for the Homotopy.
        (1) For the predetermined state variable, k, 
                ya[0] = k(t=0) = k0.
        (2) For the jump variable, c,
                yb[1] = c(tmax) = c_ss.
        """
        k0   = self.bc_values_H[0]   
        c_ss = self.bc_values_H[1]

        res = np.zeros(2)
        res[0] = ya[0] - k0
        res[1] = yb[1] - c_ss

        return res


    # --------------------------------------------------
    # Define the  approximant function, approx(self, x).
    # --------------------------------------------------
    def approx(self, x):
        """
        This defines an approximant function designed to 
        parametrize the lag term k(x_tau) and the lead
        term c(x+tau) in the mixed-type FDE system.

        Input x: Independent variable (time)
        Output: a tuple of two components: temp_k, temp_c
        """

        tmax, k0  = self.tmax, self.k0       
        solution, y_ss_H = self.solution, self.y_ss_H                                                
                                                                                
        temp_k = np.where( x<0, k0, 
                           np.where(x > tmax, y_ss_H[0], 
                           solution.sol(x)[0] ))                         
                                                                
        temp_c = np.where( x<0, np.nan,                                         
                           np.where( x > tmax, y_ss_H[1],                
                           solution.sol(x)[1] ))                         
                                                          
        return temp_k, temp_c



    # -------------------------------------------------
    # Steady state systems for ODEs, FDEs, and Homotopy
    #--------------------------------------------------
    def G_ss(self, y):
        """
        Steady-state system for the Start system
        """        
        return self.G(100, y) # any constant will do too 

    
    def F_ss(self, y):
        """
        Steady-state system for the Target system
        """
        return self.F(100, y)

    def H_ss(self, y, p):
        """
        Steady-state system for the Homotopy
        """
        return self.H(100, y, p)

    

