import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

class EOReactor:
    def __init__(self,
                 # Reactor geometry
                 L_inert=1.0,
                 L_reaction=12.0,
                 D_tube_in=0.045,
                 D_tube_out=0.05,
                 D_shell=8.0,
                 
                 # Operating conditions
                 P_in=21.0,        # atm
                 T_g_in=210.0,     # °C
                 T_cool_in=220.0,  # °C
                 Total_F_in=8800,  # mol/s
                 Num_tubes=25000,

                 # Catalyst properties
                 rho_bulk=1260.0,
                 void_fraction=0.45,
                 d_cat=0.006,
                 
                 # Kinetic constants
                 # (example from Petrov et al., can be adapted)
                 Ea1=38744, Ea2=43568, Ea3=36786, Ea4=39635,
                 Ea5=40208, Ea6=62944, Ea7=91960, Ea8=117040,
                 Ea9=150480, Ea10=108680, Ea11=41800, Ea12=50160,
                 k1_0=6.867, k2_0=1.073e2, k3_0=1.062e1,  k4_0=3.959e1,
                 k5_0=1.346e-6, k6_0=3.109e-8, k7_0=1.029e-3, k8_0=2.001e-1,
                 k9_0=4.253e-5, k10_0=4.585e-4, k11_0=3.001e-1, k12_0=2.000e-1,
                 K13=0.19, K14=0.07, K15=0.6,
                 
                 # Reaction enthalpies [J/mol]
                 dH1=-123000, dH2=-1300000, dH3=-1170000,

                 # Physical constants
                 Rgas=8.314,
                 
                 # Additional parameters (coolant, etc.)
                 MassFlow_c=2000,  # [kg/s]
                 rho_c=734,        # [kg/m3]
                 Cp_c=2600,        # [J/(kg.K)]
                 therm_cond_w=17   # [W/(m.K)]
                ):
        """
        Initialize all reactor parameters as instance attributes.
        """

        # Store geometry
        self.L_inert = L_inert
        self.L_reaction = L_reaction
        self.L_reactor_total = L_inert * 2.0 + L_reaction
        self.D_tube_in = D_tube_in
        self.D_tube_out = D_tube_out
        self.D_shell = D_shell
        
        # Operating conditions
        self.P_in = P_in
        self.T_g_in = T_g_in
        self.T_cool_in = T_cool_in
        self.Total_F_in = Total_F_in
        self.Num_tubes = Num_tubes
        
        # Catalyst / bed properties
        self.rho_bulk = rho_bulk
        self.void_fraction = void_fraction
        self.d_cat = d_cat
        self.char_d_cat = 1.5 * d_cat
        self.interf_area = (6 * (1 - void_fraction)) / self.char_d_cat

        # Kinetic constants
        self.Ea1  = Ea1;   self.Ea2  = Ea2;   self.Ea3  = Ea3;   self.Ea4  = Ea4
        self.Ea5  = Ea5;   self.Ea6  = Ea6;   self.Ea7  = Ea7;   self.Ea8  = Ea8
        self.Ea9  = Ea9;   self.Ea10 = Ea10;  self.Ea11 = Ea11;  self.Ea12 = Ea12
        
        self.k1_0  = k1_0;   self.k2_0  = k2_0;  self.k3_0  = k3_0;   self.k4_0  = k4_0
        self.k5_0  = k5_0;   self.k6_0  = k6_0;  self.k7_0  = k7_0;   self.k8_0  = k8_0
        self.k9_0  = k9_0;   self.k10_0 = k10_0; self.k11_0 = k11_0;  self.k12_0 = k12_0
        
        self.K13 = K13;  self.K14 = K14;  self.K15 = K15
        
        # Reaction enthalpies
        self.dH1 = dH1
        self.dH2 = dH2
        self.dH3 = dH3
        
        # Physical constants
        self.Rgas = Rgas
        
        # Additional coolant properties
        self.MassFlow_c   = MassFlow_c
        self.rho_c        = rho_c
        self.Cp_c         = Cp_c
        self.therm_cond_w = therm_cond_w
        
        # Inlet composition (mole fractions)
        self.y_C2H4_in = 0.173
        self.y_O2_in   = 0.084
        self.y_CO2_in  = 0.029
        self.y_EO_in   = 0.00001
        self.y_H2O_in  = 0.005
        self.y_CH4_in  = 0.675
        self.y_Ar_in   = 0.035
        self.y_EDC_in  = 1e-8
        
        # Molecular weights [g/mol]
        self.Mw = {
            'C2H4': 28.05,
            'O2'  : 32.00,
            'CO2' : 44.01,
            'H2O' : 18.02,
            'EO'  : 44.05,
            'CH4' : 16.04,
            'Ar'  : 39.95
        }
        
        # Derived quantities
        self.Area_cross_section = np.pi * (self.D_tube_in**2) / 4.0

        # Per-tube inlet flow [mol/s]
        self.F_in = self.Total_F_in / self.Num_tubes
        
        # Inert flows
        self.nCH4 = self.F_in * self.y_CH4_in
        self.nAr  = self.F_in * self.y_Ar_in
        self.nEDC = self.F_in * self.y_EDC_in
        
        # Inlet flows
        self.nC2H4_0 = self.F_in * self.y_C2H4_in
        self.nO2_0   = self.F_in * self.y_O2_in
        self.nCO2_0  = self.F_in * self.y_CO2_in
        self.nH2O_0  = self.F_in * self.y_H2O_in
        self.nEO_0   = self.F_in * self.y_EO_in
        
        # Gas mass flow [kg/s] per tube
        mass_flow_in_g = (self.nC2H4_0*self.Mw['C2H4'] +
                          self.nO2_0  *self.Mw['O2']   +
                          self.nCO2_0 *self.Mw['CO2']  +
                          self.nH2O_0 *self.Mw['H2O']  +
                          self.nEO_0  *self.Mw['EO']   +
                          self.nCH4   *self.Mw['CH4']  +
                          self.nAr    *self.Mw['Ar'])/1000.0
        self.MassFlow_g = mass_flow_in_g
        
        self.MassFlux_g = self.MassFlow_g / (self.Area_cross_section * self.void_fraction)
        
        # Coolant mass flux
        self.MassFlux_c = (4.0*self.MassFlow_c) / (np.pi*(self.D_shell**2 - self.Num_tubes*self.D_tube_out**2))

    # ----------------------------------------------------------------------
    # Heat capacity functions (could also store as polynomials or advanced T-dependence)
    def Cp_C2H4(self, T):
        T = T/1000
        return (-6.38788 +
                184.4019 * T +
                -112.9718 * T**2 +
                28.49593 * T**3 +
                1.142886 / T**2) 
    
    def Cp_O2(self,   T):
        T = T/1000 
        return (31.32234 +
                -20.23531 * T +
                57.86644 * T**2 +
                -36.50624 * T**3 +
                -0.007374 / T**2)
    
    def Cp_CO2(self,  T):
        T = T/1000 
        return (24.99735 +
                55.18696 * T +
                -33.69137 * T**2 +
                7.948387 * T**3 +
                -0.136638 / T**2)
    def Cp_H2O(self,  T):
        T = T/1000 
        return (30.092 +
                6.832514 * T +
                6.793435 * T**2 +
                -2.53448 * T**3 +
                0.082139 / T**2)
    def Cp_EO(self,   T):
        T = T/1000 
        return (8.89811 +
                63.53625 * T +
                -36.21684 * T**2 +
                5.42944 * T**3 +
                -0.60281 / T**2)
    
    def Cp_CH4(self,  T):
        T = T/1000     
        return (-0.703029 +
                108.4773 * T +
                -42.52157 * T**2 +
                5.862788 * T**3 +
                0.678565 / T**2)
    def Cp_Ar(self,   T):
        T = T/1000
        return (20.7860 +
                2.8259e-7 * T +
                -1.4642e-7 * T**2 +
                1.0921e-8 * T**3 +
                -3.6614e-3 / T**2)
    
    # ----------------------------------------------------------------------
    # Arrhenius rate constants
    def K1_func(self, Tk, a_factor=1.0):
        return a_factor * self.k1_0 * np.exp(-self.Ea1/(self.Rgas*Tk))
    def K2_func(self, Tk, a_factor=1.0):
        return a_factor * self.k2_0 * np.exp(-self.Ea2/(self.Rgas*Tk))
    def K3_func(self, Tk, a_factor=1.0):
        return a_factor * self.k3_0 * np.exp(-self.Ea3/(self.Rgas*Tk))
    def K4_func(self, Tk, a_factor=1.0):
        return a_factor * self.k4_0 * np.exp(-self.Ea4/(self.Rgas*Tk))
    def K5_func(self, Tk, a_factor=1.0):
        return a_factor * self.k5_0 * np.exp(self.Ea5/(self.Rgas*Tk))
    def K6_func(self, Tk, a_factor=1.0):
        return a_factor * self.k6_0 * np.exp(self.Ea6/(self.Rgas*Tk))
    def K7_func(self, Tk, a_factor=1.0):
        return a_factor * self.k7_0 * np.exp(-self.Ea7/(self.Rgas*Tk))
    def K8_func(self, Tk, a_factor=1.0):
        return a_factor * self.k8_0 * np.exp(-self.Ea8/(self.Rgas*Tk))
    def K9_func(self, Tk, a_factor=1.0):
        return a_factor * self.k9_0 * np.exp(-self.Ea9/(self.Rgas*Tk))
    def K10_func(self, Tk, a_factor=1.0):
        return a_factor * self.k10_0 * np.exp(-self.Ea10/(self.Rgas*Tk))
    def K11_func(self, Tk, a_factor=1.0):
        return a_factor * self.k11_0 * np.exp(-self.Ea11/(self.Rgas*Tk))
    def K12_func(self, Tk, a_factor=1.0):
        return a_factor * self.k12_0 * np.exp(-self.Ea12/(self.Rgas*Tk))
    
    # ----------------------------------------------------------------------
    def reactor_odes(self, z, y):
        """
        ODE function: y = [nC2H4, nO2, nCO2, nH2O, nEO, T_g, T_c, P]
        """
        nC2H4 = y[0]
        nO2   = y[1]
        nCO2  = y[2]
        nH2O  = y[3]
        nEO   = y[4]
        T_g   = y[5]
        T_c   = y[6]
        P     = y[7]

        # Additional inert flows (fixed)
        nCH4 = self.nCH4
        nAr  = self.nAr
        nEDC = self.nEDC
        
        ntot_flow = (nC2H4 + nO2 + nCO2 + nH2O + nEO + nCH4 + nAr + nEDC)
        
        # Partial pressures
        y_C2H4 = nC2H4 / ntot_flow
        y_O2   = nO2   / ntot_flow
        y_EO   = nEO   / ntot_flow
        y_EDC  = nEDC  / ntot_flow
        
        p_C2H4 = y_C2H4 * P
        p_O2   = y_O2   * P
        p_EO   = y_EO   * P
        p_EDC  = y_EDC  * P
        
        # Check if gas is in inert section
        if z <= self.L_inert or z >= (self.L_inert + self.L_reaction):
            a_factor = 0.0
        else:
            a_factor = 1.0
        
        # Gas properties
        Vol_flow = (ntot_flow * self.Rgas * T_g) / (P * 101325)  # m3/s
        rho_g = self.MassFlow_g / Vol_flow  # kg/m3
        Vel_g = self.MassFlux_g / rho_g     # m/s
        
        # Gas mixture heat capacity
        Cp_mix = ((nC2H4*self.Cp_C2H4(T_g) + 
                   nO2  *self.Cp_O2(T_g)   + 
                   nCO2 *self.Cp_CO2(T_g)  + 
                   nH2O *self.Cp_H2O(T_g)  + 
                   nEO  *self.Cp_EO(T_g)   +
                   nCH4 *self.Cp_CH4(T_g)  +
                   nAr  *self.Cp_Ar(T_g)) / self.MassFlow_g)
        
        # Some property assumptions
        visc_g         = 16.22e-6       # Pa.s
        therm_cond_g   = 0.0629         # W/m.K
        Re_cat         = (self.MassFlux_g * self.char_d_cat) / visc_g
        
        therm_cond_c   = 0.1074
        Cp_c           = self.Cp_c
        visc_c         = 0.625e-3
        Re_cool        = (self.MassFlux_c / visc_c) * ((self.D_shell**2 - self.Num_tubes*self.D_tube_out**2)/self.D_shell)
        Prandt_c       = (Cp_c*visc_c)/therm_cond_c
        
        # Heat transfer coefficients
        HTC_s = 0.2 * ((Cp_mix*visc_g)/therm_cond_g)**(-0.67) * ((2.867/Re_cat) + (0.3023/(Re_cat**0.35))) * rho_g * Cp_mix * Vel_g
        HTC_in = 7.676 + 0.0279*(therm_cond_g/self.D_tube_in)*Re_cat
        HTC_out = 0.023 * (Re_cool**(-0.2)) * (Prandt_c**(-0.6)) * self.rho_c * Cp_c * (self.MassFlux_c/self.rho_c)
        
        U_HTC = 1.0 / ( (1.0/HTC_in) + ((self.D_tube_out - self.D_tube_in)/(2.0*self.therm_cond_w)) + (1.0/HTC_out) )
        
        # Solve T_s from energy balance on catalyst particle
        # We define a local function for fsolve:
        def T_s_solver(T_s_guess):
            # Reaction rates [mol/(m3 s)]
            r1 = 0.85 * ((self.K1_func(T_s_guess, a_factor)*p_C2H4*p_O2 - 
                          self.K2_func(T_s_guess, a_factor)*p_C2H4*p_O2*(p_EDC**self.K13)) 
                         / (1 + self.K5_func(T_s_guess, a_factor)*p_O2 + 
                                self.K6_func(T_s_guess, a_factor)*p_C2H4)
                        ) * self.rho_bulk * (1 - self.void_fraction) * (1000.0/3600.0)
            
            r2 = 0.132 * ((self.K3_func(T_s_guess, a_factor)*p_C2H4*p_O2 - 
                           self.K4_func(T_s_guess, a_factor)*p_C2H4*p_O2*(p_EDC**self.K14))
                          / (1 + self.K5_func(T_s_guess, a_factor)*p_O2 + 
                                 self.K6_func(T_s_guess, a_factor)*p_C2H4)
                         ) * self.rho_bulk * (1 - self.void_fraction) * (1000.0/3600.0)
            
            r3 = 298900000 * ((self.K7_func(T_s_guess, a_factor)*p_EO*p_O2 - 
                          self.K8_func(T_s_guess, a_factor)*p_O2*p_EO*(p_EDC**self.K15))
                         / ((1 + self.K9_func(T_s_guess, a_factor)*p_O2 + 
                                 self.K10_func(T_s_guess, a_factor)*(p_O2**0.5) + 
                                 self.K11_func(T_s_guess, a_factor)*p_EO + 
                                 self.K12_func(T_s_guess, a_factor)*p_EO*(p_O2**-0.5))**2)
                        ) * self.rho_bulk * (1 - self.void_fraction) * (1000.0/3600.0)
            
            # Total heat from reactions
            Q_react = -(r1*self.dH1 + r2*self.dH2 + r3*self.dH3)
            
            # Energy balance (pseudo steady-state):
            # T_s - (Q_react/(HTC_s*interf_area) + T_g) = 0  => residual
            # or rearranged as in your code:
            return (Q_react/(HTC_s * self.interf_area) + T_g) - T_s_guess
        
        T_s_init_guess = T_g + 10.0
        T_solution = fsolve(T_s_solver, x0=T_s_init_guess)[0]

        # Now that we have T_s (solid temperature), compute the reaction rates again
        r1 = 0.85 * ((self.K1_func(T_solution, a_factor)*p_C2H4*p_O2 - 
                      self.K2_func(T_solution, a_factor)*p_C2H4*p_O2*(p_EDC**self.K13)) 
                     / (1 + self.K5_func(T_solution, a_factor)*p_O2 + 
                            self.K6_func(T_solution, a_factor)*p_C2H4)
                    ) * self.rho_bulk * (1 - self.void_fraction) * (1000.0/3600.0)
        
        r2 = 0.132 * ((self.K3_func(T_solution, a_factor)*p_C2H4*p_O2 - 
                       self.K4_func(T_solution, a_factor)*p_C2H4*p_O2*(p_EDC**self.K14))
                      / (1 + self.K5_func(T_solution, a_factor)*p_O2 + 
                             self.K6_func(T_solution, a_factor)*p_C2H4)
                     ) * self.rho_bulk * (1 - self.void_fraction) * (1000.0/3600.0)
        
        r3 = 298900000 * ((self.K7_func(T_solution, a_factor)*p_EO*p_O2 - 
                      self.K8_func(T_solution, a_factor)*p_O2*p_EO*(p_EDC**self.K15))
                     / ((1 + self.K9_func(T_solution, a_factor)*p_O2 + 
                             self.K10_func(T_solution, a_factor)*(p_O2**0.5) + 
                             self.K11_func(T_solution, a_factor)*p_EO + 
                             self.K12_func(T_solution, a_factor)*p_EO*(p_O2**-0.5))**2)
                    ) * self.rho_bulk * (1 - self.void_fraction) * (1000.0/3600.0)
        
        # Species production rates
        dF_C2H4_dz = ( -r1 + -r2 + 0.0*r3 ) * self.Area_cross_section
        dF_O2_dz   = ( -0.5*r1 + -3.0*r2 + -2.5*r3 ) * self.Area_cross_section
        dF_CO2_dz  = (  0.0*r1 +  2.0*r2 +  2.0*r3 ) * self.Area_cross_section
        dF_H2O_dz  = (  0.0*r1 +  2.0*r2 +  2.0*r3 ) * self.Area_cross_section
        dF_EO_dz   = (  1.0*r1 +  0.0*r2 + -1.0*r3 ) * self.Area_cross_section
        
        # Energy balances
        dTg_dz = (1.0/(rho_g*Cp_mix*Vel_g)) * (HTC_s*self.interf_area*(T_solution - T_g) 
                                              - (4.0*U_HTC/self.D_tube_in)*(T_g - T_c))
        dTc_dz = ((4.0*U_HTC)/(self.rho_c*Cp_c* (self.MassFlux_c/self.rho_c))) * \
                 ((self.Num_tubes*self.D_tube_out)/(self.D_shell**2 - self.Num_tubes*self.D_tube_out**2)) * \
                 (T_g - T_c)
        
        # Pressure drop (Ergun eqn.)
        dP_dz = -(150.0*((visc_g*(1.0-self.void_fraction)**2)/(self.void_fraction**3*self.d_cat**2))*Vel_g +
                  1.75*((rho_g*(1.0-self.void_fraction))/(self.void_fraction**3*self.d_cat))*Vel_g**2) / 101325.0
        
        return [dF_C2H4_dz, dF_O2_dz, dF_CO2_dz, dF_H2O_dz, dF_EO_dz, dTg_dz, dTc_dz, dP_dz]

    # ----------------------------------------------------------------------
    def cat_temp_calc(self, z, y):
        """
        Function to compute catalyst temperature T_s at a point z.
        This duplicates some logic from reactor_odes to solve for T_s.
        Typically used just for post-processing.
        """
        nC2H4 = y[0]
        nO2   = y[1]
        nCO2  = y[2]
        nH2O  = y[3]
        nEO   = y[4]
        T_g   = y[5]
        T_c   = y[6]
        P     = y[7]

        # Additional inert flows
        nCH4 = self.nCH4
        nAr  = self.nAr
        nEDC = self.nEDC
        
        ntot_flow = nC2H4 + nO2 + nCO2 + nH2O + nEO + nCH4 + nAr + nEDC
        
        y_C2H4 = nC2H4 / ntot_flow
        y_O2   = nO2   / ntot_flow
        y_EO   = nEO   / ntot_flow
        y_EDC  = nEDC  / ntot_flow
        
        p_C2H4 = y_C2H4 * P
        p_O2   = y_O2   * P
        p_EO   = y_EO   * P
        p_EDC  = y_EDC  * P
        
        # Check if in inert zone
        if z <= self.L_inert or z >= (self.L_inert + self.L_reaction):
            a_factor = 0.0
        else:
            a_factor = 1.0
        
        Vol_flow = (ntot_flow*self.Rgas*T_g)/(P*101325)
        rho_g = self.MassFlow_g / Vol_flow
        Vel_g = self.MassFlux_g / rho_g
        
        Cp_mix = ((nC2H4*self.Cp_C2H4(T_g) + 
                   nO2  *self.Cp_O2(T_g)   + 
                   nCO2 *self.Cp_CO2(T_g)  + 
                   nH2O *self.Cp_H2O(T_g)  + 
                   nEO  *self.Cp_EO(T_g)   +
                   nCH4 *self.Cp_CH4(T_g)  +
                   nAr  *self.Cp_Ar(T_g)) / self.MassFlow_g)
        
        visc_g       = 16.22e-6
        therm_cond_g = 0.0629
        Re_cat       = (self.MassFlux_g * self.char_d_cat)/visc_g
        
        therm_cond_c = 0.1074
        Cp_c         = self.Cp_c
        visc_c       = 0.625e-3
        Re_cool      = (self.MassFlux_c/visc_c)*((self.D_shell**2 - self.Num_tubes*self.D_tube_out**2)/self.D_shell)
        Prandt_c     = (Cp_c*visc_c)/therm_cond_c
        
        HTC_s = 0.2 * ((Cp_mix*visc_g)/therm_cond_g)**(-0.67) * ((2.867/Re_cat) + (0.3023/(Re_cat**0.35))) * rho_g*Cp_mix*Vel_g
        HTC_in = 7.676 + 0.0279*(therm_cond_g/self.char_d_cat)*Re_cat
        HTC_out = 0.023 * Re_cool**(-0.2)*Prandt_c**(-0.6)*self.rho_c*Cp_c*(self.MassFlux_c/self.rho_c)
        
        U_HTC = 1.0 / ( (1.0/HTC_in) + ((self.D_tube_out - self.D_tube_in)/(2.0*self.therm_cond_w)) + (1.0/HTC_out) )
        
        def T_s_solver(T_s_guess):
            r1 = 0.85 * ((self.K1_func(T_s_guess, a_factor)*p_C2H4*p_O2 -
                          self.K2_func(T_s_guess, a_factor)*p_C2H4*p_O2*(p_EDC**self.K13))
                         / (1 + self.K5_func(T_s_guess, a_factor)*p_O2 + 
                                self.K6_func(T_s_guess, a_factor)*p_C2H4)
                        ) * self.rho_bulk*(1 - self.void_fraction)*(1000.0/3600.0)
            
            r2 = 0.132*((self.K3_func(T_s_guess, a_factor)*p_C2H4*p_O2 -
                         self.K4_func(T_s_guess, a_factor)*p_C2H4*p_O2*(p_EDC**self.K14))
                         / (1 + self.K5_func(T_s_guess, a_factor)*p_O2 + 
                                self.K6_func(T_s_guess, a_factor)*p_C2H4)
                       ) * self.rho_bulk*(1 - self.void_fraction)*(1000.0/3600.0)
            
            r3 = 298900000*((self.K7_func(T_s_guess, a_factor)*p_EO*p_O2 -
                         self.K8_func(T_s_guess, a_factor)*p_O2*p_EO*(p_EDC**self.K15))
                         / ((1 + self.K9_func(T_s_guess, a_factor)*p_O2 + 
                                self.K10_func(T_s_guess, a_factor)*(p_O2**0.5) + 
                                self.K11_func(T_s_guess, a_factor)*p_EO + 
                                self.K12_func(T_s_guess, a_factor)*p_EO*(p_O2**-0.5))**2)
                       ) * self.rho_bulk*(1 - self.void_fraction)*(1000.0/3600.0)
            
            Q_react = -(r1*self.dH1 + r2*self.dH2 + r3*self.dH3)
            return T_s_guess - (Q_react/(HTC_s*self.interf_area) + T_g)

        T_s_init = T_g
        T_s = fsolve(T_s_solver, x0=T_s_init)[0]
        return T_s

    # ----------------------------------------------------------------------
    def run_simulation(self):
        """
        Set up initial conditions, solve the ODE system, and return (z, sol, conversion, selectivity).
        """
        # Convert inlet T from °C to K
        Tg_0 = self.T_g_in + 273.15
        Tc_0 = self.T_cool_in + 273.15
        P0   = self.P_in

        # Initial condition vector
        y0 = [
            self.nC2H4_0,
            self.nO2_0,
            self.nCO2_0,
            self.nH2O_0,
            self.nEO_0,
            Tg_0,
            Tc_0,
            P0
        ]
        
        z_span = (0.0, self.L_reactor_total)
        z_eval = np.linspace(0, self.L_reactor_total, 100)
        
        sol = solve_ivp(
            fun=lambda z, y: self.reactor_odes(z, y),
            t_span=z_span,
            y0=y0,
            t_eval=z_eval,
            method='RK45'
        )
        
        if not sol.success:
            raise RuntimeError("Integration failed: " + sol.message)
        
        # Multiply certain flows by Num_tubes
        nC2H4_sol = sol.y[0,:] * self.Num_tubes
        nO2_sol   = sol.y[1,:] * self.Num_tubes
        nCO2_sol  = sol.y[2,:] * self.Num_tubes
        nH2O_sol  = sol.y[3,:] * self.Num_tubes
        nEO_sol   = sol.y[4,:] * self.Num_tubes
        
        nTotal_sol = (nC2H4_sol + nO2_sol + nCO2_sol + nH2O_sol + nEO_sol
                      + (self.nAr + self.nCH4)*self.Num_tubes)
        
        Tg_sol = sol.y[5,:]
        Tc_sol = sol.y[6,:]
        P_sol  = sol.y[7,:]
        
        # Catalyst temperature at each z (post-processing)
        Ts_sol = np.array([self.cat_temp_calc(z_pt, sol.y[:,i]) 
                           for i, z_pt in enumerate(sol.t)])
        
        # Compute conversion and selectivity
        conv = (nC2H4_sol[0] - nC2H4_sol[-1]) / nC2H4_sol[0] * 100.0
        sel  = ((nEO_sol[-1] - nEO_sol[0]) / (nC2H4_sol[0] - nC2H4_sol[-1])) * 100.0
        
        return sol.t, nC2H4_sol, nO2_sol, nCO2_sol, nH2O_sol, nEO_sol, nTotal_sol, Tg_sol, Tc_sol, Ts_sol, P_sol, conv, sel

# ------------------------------------------------------------------------------
# Example "main" function showing how to use the EOReactor class
def main():
    # Create a reactor instance with default parameters (or pass your own)
    reactor = EOReactor(
        L_inert=1.0,
        L_reaction=12.0,
        D_tube_in=0.045,
        D_tube_out=0.05,
        D_shell=8.0,
        P_in=21.0,
        T_g_in=210.0,
        T_cool_in=220.0,
        Total_F_in=8800,
        Num_tubes=25000
    )
    
    # Run the simulation
    (z_points,
     nC2H4_sol,
     nO2_sol,
     nCO2_sol,
     nH2O_sol,
     nEO_sol,
     nTotal_sol,
     Tg_sol,
     Tc_sol,
     Ts_sol,
     P_sol,
     conv,
     sel) = reactor.run_simulation()
    
    # Print results
    print(f"Final C2H4 conversion = {conv:.2f}%")
    print(f"Selectivity to EO     = {sel:.2f}%")
    
    # Quick summary
    print(f"C2H4 in and out  = [{nC2H4_sol[0]:.2f}, {nC2H4_sol[-1]:.2f}] mol/s")
    print(f"O2 in and out    = [{nO2_sol[0]:.2f}, {nO2_sol[-1]:.2f}] mol/s")
    print(f"CO2 in and out   = [{nCO2_sol[0]:.2f}, {nCO2_sol[-1]:.2f}] mol/s")
    print(f"H2O in and out   = [{nH2O_sol[0]:.2f}, {nH2O_sol[-1]:.2f}] mol/s")
    print(f"EO in and out    = [{nEO_sol[0]:.2f}, {nEO_sol[-1]:.2f}] mol/s")
    
    # Plots
    fig, axs = plt.subplots(3, 1, figsize=(7,8))
    
    axs[0].plot(z_points, nC2H4_sol/nTotal_sol, label='C2H4')
    axs[0].plot(z_points, nO2_sol/nTotal_sol,   label='O2')
    axs[0].plot(z_points, nCO2_sol/nTotal_sol,  label='CO2')
    axs[0].plot(z_points, nH2O_sol/nTotal_sol,  label='H2O')
    axs[0].plot(z_points, nEO_sol/nTotal_sol,   label='EO')
    axs[0].set_xlabel("Reactor length z [m]")
    axs[0].set_ylabel("Mole fraction")
    axs[0].set_title("Species Profiles")
    axs[0].legend(loc='best')
    
    axs[1].plot(z_points, Tg_sol - 273.15, 'r-', label='Gas Temp')
    axs[1].plot(z_points, Tc_sol - 273.15, 'b-', label='Coolant Temp')
    axs[1].plot(z_points, Ts_sol - 273.15, 'g-', label='Catalyst Temp')
    axs[1].set_xlabel("Reactor length z [m]")
    axs[1].set_ylabel("Temperature [°C]")
    axs[1].set_title("Temperature Profiles")
    axs[1].legend(loc='best')
    
    axs[2].plot(z_points, P_sol, 'g-', label='Pressure')
    axs[2].set_xlabel("Reactor length z [m]")
    axs[2].set_ylabel("Pressure [atm]")
    axs[2].set_title("Pressure Profile")
    
    plt.tight_layout()
    plt.show()


def sensitivity_analysis_on_reactor_length():
    lengths = [5, 8, 10, 12, 15]  # example lengths for L_reaction
    conversions = []
    selectivities = []

    for Lr in lengths:
        reactor = EOReactor(L_inert=1.0, L_reaction=Lr, T_g_in=210.0, T_cool_in=220.0)
        _, _, _, _, _, _, _, _, _, _, _, conv, sel = reactor.run_simulation()
        conversions.append(conv)
        selectivities.append(sel)

    # Plot the results
    plt.figure()
    #plt.plot(lengths, conversions, 'o-', label="Conversion (%)")
    plt.plot(lengths, selectivities, 's-', label="Selectivity (%)")
    plt.xlabel("Reactive Length [m]")
    plt.ylabel("Value (%)")
    plt.title("Sensitivity: Reactor Length vs. Conversion/Selectivity")
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------------------------------------------------------------------

def sensitivity_analysis_on_temperatures():
    """
    Vary both the gas inlet temperature (T_g_in) and the coolant inlet temperature (T_cool_in)
    and record the conversion and selectivity at each combination.
    """
    # Choose your temperature ranges (in °C)
    T_g_in_values   = np.arange(190, 261, 20)  # 190, 200, 210, 220, 230
    T_cool_in_values = np.arange(200, 271, 20) # 200, 210, 220, 230, 240
    
    # Initialize arrays to hold conversion and selectivity results
    # We'll make them 2D, with shape (len(T_g_in_values), len(T_cool_in_values))
    conv_matrix = np.zeros((len(T_g_in_values), len(T_cool_in_values)))
    sel_matrix  = np.zeros((len(T_g_in_values), len(T_cool_in_values)))
    
    # Loop over each combination
    for i, Tg_in in enumerate(T_g_in_values):
        for j, Tc_in in enumerate(T_cool_in_values):
            # Create a reactor instance, varying only T_g_in and T_cool_in
            reactor = EOReactor(
                L_inert=1.0,
                L_reaction=12.0,
                D_tube_in=0.045,
                D_tube_out=0.05,
                D_shell=8.0,
                P_in=21.0,
                T_g_in=Tg_in,     # <--- Gas inlet temperature
                T_cool_in=Tc_in,  # <--- Coolant inlet temperature
                Total_F_in=8800,
                Num_tubes=25000
            )
            
            # Run the simulation
            (z_points,
             nC2H4_sol,
             nO2_sol,
             nCO2_sol,
             nH2O_sol,
             nEO_sol,
             nTotal_sol,
             Tg_sol,
             Tc_sol,
             Ts_sol,
             P_sol,
             conv,
             sel) = reactor.run_simulation()
            
            # Store results in our matrices
            conv_matrix[i, j] = conv
            sel_matrix[i, j]  = sel
    
    # At this point, conv_matrix[i, j] holds the conversion at:
    #   T_g_in = T_g_in_values[i] and T_cool_in = T_cool_in_values[j]
    # Similarly for sel_matrix
    
    # -------------------------
    # Make a contour plot (or other 2D map) of conversion
    # -------------------------
    plt.figure()
    # X-axis: T_cool_in_values, Y-axis: T_g_in_values
    # You can pick whichever orientation you prefer
    Tg_grid, Tc_grid = np.meshgrid(T_cool_in_values, T_g_in_values)
    
    plt.contourf(Tg_grid, Tc_grid, conv_matrix)
    plt.colorbar(label='Conversion [%]')
    plt.xlabel("Coolant Inlet Temperature [°C]")
    plt.ylabel("Gas Inlet Temperature [°C]")
    plt.title("Conversion vs. Gas & Coolant Inlet Temperatures")
    plt.show()
    
    # -------------------------
    # Make a contour plot (or other 2D map) of selectivity
    # -------------------------
    plt.figure()
    plt.contourf(Tg_grid, Tc_grid, sel_matrix)
    plt.colorbar(label='Selectivity [%]')
    plt.xlabel("Coolant Inlet Temperature [°C]")
    plt.ylabel("Gas Inlet Temperature [°C]")
    plt.title("Selectivity vs. Gas & Coolant Inlet Temperatures")
    plt.show()
    
    # Optionally, you can return conv_matrix, sel_matrix if you want to do further analysis
    return conv_matrix, sel_matrix, T_g_in_values, T_cool_in_values

def multi_parameter_sensitivity(num_samples=20):
    """
    Perform a sensitivity study on multiple reactor parameters:
      1) Inlet Pressure (P_in)
      2) Reactor Catalyst Length (L_reaction)
      3) Coolant Inlet Temperature (T_cool_in)
      4) Gas Inlet Temperature (T_g_in)
      5) Number of Tubes (Num_tubes)

    We sample each parameter randomly within a specified range,
    run the reactor simulation, and record conversion and selectivity.
    """

    # 1) Define parameter ranges (adjust based on your design/plant limits)
    P_in_range          = (15.0, 25.0)    # atm
    L_reaction_range    = (5.0, 15.0)     # m
    T_cool_in_range     = (190.0, 250.0)  # °C
    T_g_in_range        = (190.0, 250.0)  # °C
    num_tubes_range     = (20000, 30000)  # integer range for number of tubes

    # 2) Prepare a list or array for results
    # We'll store [P_in, L_reaction, T_cool_in, T_g_in, num_tubes, conversion, selectivity]
    results = []

    # 3) Randomly sample the parameter space
    for _ in range(num_samples):
        P_in_i         = np.random.uniform(*P_in_range)
        L_reaction_i   = np.random.uniform(*L_reaction_range)
        T_cool_in_i    = np.random.uniform(*T_cool_in_range)
        T_g_in_i       = np.random.uniform(*T_g_in_range)
        num_tubes_i    = np.random.randint(num_tubes_range[0], num_tubes_range[1]+1)

        # 4) Instantiate the reactor with these parameters
        #    (In your EOReactor, pass them to __init__)
        #    We'll assume default values for other parameters, e.g., L_inert=1.0, etc.
        reactor = EOReactor(
            L_inert=1.0,
            L_reaction=L_reaction_i,
            D_tube_in=0.045,
            D_tube_out=0.05,
            D_shell=8.0,
            P_in=P_in_i,
            T_g_in=T_g_in_i,        # Gas inlet temperature
            T_cool_in=T_cool_in_i,  # Coolant inlet temperature
            Total_F_in=8800.0,      # keep total feed constant
            Num_tubes=num_tubes_i
        )

        # 5) Run the simulation and capture conversion & selectivity
        try:
            # run_simulation returns a tuple, with the last two elements = conv, sel
            *_, conv_i, sel_i = reactor.run_simulation()
        except Exception as e:
            print(f"Simulation failed: {e}")
            conv_i, sel_i = np.nan, np.nan

        # 6) Store results

        results.append((P_in_i, L_reaction_i, T_cool_in_i, T_g_in_i, num_tubes_i, conv_i, sel_i))

        print("result number is ", len(results))

    # Convert to NumPy array for easy slicing => shape (num_samples, 7)
    results_array = np.array(results, dtype=float)

    # Separate columns for convenience
    P_in_vals        = results_array[:, 0]
    L_reaction_vals  = results_array[:, 1]
    T_cool_in_vals   = results_array[:, 2]
    T_g_in_vals      = results_array[:, 3]
    num_tubes_vals   = results_array[:, 4]
    conv_vals        = results_array[:, 5]
    sel_vals         = results_array[:, 6]

    # 7) Example Visualizations
    # Because we have five dimensions, we show some 2D scatter plots with a third dimension as color.

    # (A) Scatter of Conversion vs Selectivity, color by T_cool_in
    plt.figure(figsize=(7,5))
    sc1 = plt.scatter(conv_vals, sel_vals, c=T_cool_in_vals, cmap='plasma')
    cbar1 = plt.colorbar(sc1)
    cbar1.set_label('Coolant Inlet Temperature [°C]')
    plt.xlabel('Conversion [%]')
    plt.ylabel('Selectivity [%]')
    plt.title('Conv vs. Sel (Colored by Coolant Inlet Temp)')
    plt.grid(True)
    plt.show()

    # (B) Scatter of Conversion vs Selectivity, color by Gas Inlet Temperature
    plt.figure(figsize=(7,5))
    sc2 = plt.scatter(conv_vals, sel_vals, c=T_g_in_vals, cmap='viridis')
    cbar2 = plt.colorbar(sc2)
    cbar2.set_label('Gas Inlet Temperature [°C]')
    plt.xlabel('Conversion [%]')
    plt.ylabel('Selectivity [%]')
    plt.title('Conv vs. Sel (Colored by Gas Inlet Temp)')
    plt.grid(True)
    plt.show()

    # (D) Scatter of Conversion [%] vs. Selectivity, color by Reactor Length
    plt.figure(figsize=(7,5))
    sc4 = plt.scatter(conv_vals, sel_vals, c=L_reaction_vals, cmap='rainbow')
    cbar4 = plt.colorbar(sc4)
    cbar4.set_label('Reactor Length [m]')
    plt.xlabel('Conversion [%]')
    plt.ylabel('Selectivity [%]')
    plt.title('Selectivity vs. Conversion [%] (Colored by L_reaction)')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(7,5))
    sc5 = plt.scatter(conv_vals, sel_vals, c=num_tubes_vals, cmap='cool')
    cbar5 = plt.colorbar(sc5)
    cbar5.set_label('Number of Tubes')
    plt.xlabel('Conversion [%]')
    plt.ylabel('Selectivity [%]')
    plt.title('Conversion vs. Selectivity (Colored by #Tubes)')
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(7,5))
    sc2 = plt.scatter(conv_vals, sel_vals, c=P_in_vals, cmap='viridis')
    cbar2 = plt.colorbar(sc2)
    cbar2.set_label('Inlet Pressure [atm]')
    plt.xlabel('Conversion [%]')
    plt.ylabel('Selectivity [%]')
    plt.title('Conv vs. Sel (Colored by Inlet Pressure [atm])')
    plt.grid(True)
    plt.show()

    return results_array


def tubes_length_sweep():
    """
    Fix T_g_in, T_cool_in, P_in to original values.
    Sweep over (Num_tubes) and (L_reaction) in a systematic grid.
    Identify the combination that gives the highest EO yield.
    """

    # 1) Fixed operating conditions (original values)
    fixed_P_in      = 21.0    # atm
    fixed_T_g_in    = 210.0   # °C
    fixed_T_cool_in = 220.0   # °C
    fixed_Total_F_in= 8800.0  # example feed rate, if that's original
    L_inert         = 1.0     # keep inert length if that was original

    # 2) Define parameter ranges for the sweep
    #    (customize these to your scenario)
    num_tubes_range   = np.arange(20000, 30001, 2000)  # e.g. 20,000 to 30,000 in steps of 2,000
    L_reaction_range  = np.arange(6.0, 15.1, 1.0)      # 6.0 m to 15.0 m in steps of 1.0

    # Prepare to store results: we will store a 2D array with shape:
    #   (len(num_tubes_range), len(L_reaction_range))
    # representing yield(%) for each combination
    yield_matrix = np.zeros((len(num_tubes_range), len(L_reaction_range)))

    # Keep track of maximum yield and best parameters
    max_yield = -1.0
    best_num_tubes = None
    best_L_reaction = None

    # 3) Double loop over number of tubes and length
    for i, n_tubes in enumerate(num_tubes_range):
        for j, L_react in enumerate(L_reaction_range):

            # 4) Create reactor instance with fixed T, P, but varying #tubes & length
            reactor = EOReactor(
                L_inert = L_inert,
                L_reaction = L_react,
                P_in = fixed_P_in,
                T_g_in = fixed_T_g_in,
                T_cool_in = fixed_T_cool_in,
                Total_F_in = fixed_Total_F_in,
                Num_tubes = n_tubes
            )

            # 5) Run the simulation
            try:
                (*_, conv, sel) = reactor.run_simulation()
            except Exception as e:
                print(f"Solver failed: {e}")
                # If the solver fails, assign yield as NaN or some sentinel
                yield_matrix[i, j] = np.nan
                continue

            # 6) Compute yield (in %)
            yield_ij = conv * sel / 100.0   # conv and sel are in %, so yield in %
            yield_matrix[i, j] = yield_ij

            # 7) Check if this is the max
            if yield_ij > max_yield:
                max_yield = yield_ij
                best_num_tubes = n_tubes
                best_L_reaction = L_react

    # 8) Print or store the best combination
    print(f"Max yield found: {max_yield:.2f}%")
    print(f" -> Best Num_tubes = {best_num_tubes}, Best L_reaction = {best_L_reaction:.1f} m")

    # 9) Plot yield as a contour or heatmap vs. (#tubes, L_reaction)
    #    We'll do a meshgrid style plot.  X= L_reaction, Y= #Tubes
    X, Y = np.meshgrid(L_reaction_range, num_tubes_range)

    plt.figure(figsize=(8,5))
    ctf = plt.contourf(X, Y, yield_matrix, 20, cmap='viridis')
    cbar = plt.colorbar(ctf)
    cbar.set_label('Yield of EO [%]')
    plt.xlabel('Reactor Length [m]')
    plt.ylabel('Number of Tubes')
    plt.title('Yield(%) as a function of #Tubes and L_reaction')
    plt.show()

    # 10) Return results, or do further analysis
    return num_tubes_range, L_reaction_range, yield_matrix, (max_yield, best_num_tubes, best_L_reaction)

def grid_search_selectivity():
    # Fixed parameters (baseline values)
    fixed_P_in = 21.0        # atm
    fixed_T_g_in = 210.0     # °C
    fixed_T_cool_in = 220.0  # °C
    fixed_Total_F_in = 8800.0

    # Define grid ranges for L_reaction and Num_tubes
    L_reaction_values = np.linspace(8.0, 30.0, 5)  # from 8 m to 15 m (15 points)
    num_tubes_values = np.linspace(20000, 35000, 5, dtype=int)  # 15 discrete values

    # Prepare an array to hold the selectivity for each combination.
    selectivity_grid = np.zeros((len(L_reaction_values), len(num_tubes_values)))

    # Loop over the grid
    for i, L_reaction in enumerate(L_reaction_values):
        for j, num_tubes in enumerate(num_tubes_values):
            # Create a reactor instance with fixed temperatures/pressure
            reactor = EOReactor(
                L_inert=1.0,             # fixed inert length
                L_reaction=L_reaction,
                D_tube_in=0.045,
                D_tube_out=0.05,
                D_shell=9.0,
                P_in=fixed_P_in,
                T_g_in=fixed_T_g_in,
                T_cool_in=fixed_T_cool_in,
                Total_F_in=fixed_Total_F_in,
                Num_tubes=num_tubes
            )
            try:
                # Run the simulation; run_simulation returns (z, ..., conv, sel)
                *_, conv, sel = reactor.run_simulation()
                selectivity_grid[i, j] = sel
            except Exception as e:
                print(f"Simulation failed at L_reaction={L_reaction}, num_tubes={num_tubes}: {e}")
                selectivity_grid[i, j] = np.nan

    # Find the maximum selectivity in the grid and its corresponding parameters
    max_index = np.unravel_index(np.nanargmax(selectivity_grid), selectivity_grid.shape)
    optimal_L_reaction = L_reaction_values[max_index[0]]
    optimal_num_tubes = num_tubes_values[max_index[1]]
    max_selectivity = selectivity_grid[max_index]

    print("Optimal Reactor Conditions for Maximum Selectivity:")
    print(f"Reactor Catalyst Length (L_reaction): {optimal_L_reaction:.2f} m")
    print(f"Number of Tubes (Num_tubes): {optimal_num_tubes}")
    print(f"Maximum Selectivity: {max_selectivity:.2f} %")

    # Create a contour plot
    L_mesh, tubes_mesh = np.meshgrid(num_tubes_values, L_reaction_values)
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(tubes_mesh, L_mesh, selectivity_grid, cmap='viridis', levels=20)
    plt.colorbar(cp, label="Selectivity (%)")
    plt.ylabel("Number of Tubes")
    plt.xlabel("Reactor Catalyst Length [m]")
    plt.title("Selectivity vs. Reactor Length & Number of Tubes")
    #plt.scatter(optimal_num_tubes, optimal_L_reaction, color='red', marker='o', label='Optimal')
    plt.legend()
    plt.show()

    return L_reaction_values, num_tubes_values, selectivity_grid



if __name__ == "__main__":
    main()
    #sensitivity_analysis_on_reactor_length()
    #sensitivity_analysis_on_temperatures()
    results_data = multi_parameter_sensitivity(num_samples=100)
    #tubes, lengths, yield_mat, best_info = tubes_length_sweep()
    
    # best_info is (max_yield, best_num_tubes, best_L_reaction)
    #print("Best parameters from the analysis:", best_info)    
    #grid_search_selectivity()
