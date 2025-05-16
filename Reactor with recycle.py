import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, anderson, dual_annealing
from scipy.optimize._nonlin import NoConvergence
import optuna
import os

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

                 # Inlet composition (mole fractions)
                 y_C2H4_in = 0.173,
                 y_O2_in   = 0.084,
                 y_CO2_in  = 0.029,
                 y_EO_in   = 1e-5,
                 y_H2O_in  = 0.005,
                 y_CH4_in  = 0.675,
                 y_Ar_in   = 0.035,
                 y_EDC_in  = 1e-8,
                 
                 # Kinetic constants
                 Ea1=38744, Ea2=43568, Ea3=36786, Ea4=39635,
                 Ea5=40208, Ea6=62944, Ea7=91960, Ea8=117040,
                 Ea9=150480, Ea10=108680, Ea11=41800, Ea12=50160,
                 k1_0=6.867, k2_0=1.073e2, k3_0=1.062e1,  k4_0=3.959e1,
                 k5_0=1.346e-6, k6_0=3.109e-8, k7_0=1.029e-3, k8_0=2.001e-1,
                 k9_0=4.253e-5, k10_0=4.585e-4, k11_0=3.001e-1, k12_0=2.000e-1,
                 K13=0.19, K14=0.07, K15=0.6,
                 
                 # Reaction enthalpies [J/mol]
                 dH1=-123000, dH2=-1300000, dH3=-1170000,
            


                 
                 # Additional parameters (coolant, etc.)
                 MassFlow_c=2000,  # [kg/s]
                 rho_c=734,        # [kg/m3]
                 Cp_c=2600,        # [J/(kg.K)]
                 therm_cond_w=17   # [W/(m.K)]
                ):
        """
        Initialize all reactor parameters as instance attributes.
        """
        # Reactor geometry
        self.L_inert = L_inert
        self.L_reaction = L_reaction
        self.L_reactor_total = 2 * L_inert + L_reaction
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

        # Kinetic constants (store as attributes)
        self.Ea1, self.Ea2, self.Ea3, self.Ea4 = Ea1, Ea2, Ea3, Ea4
        self.Ea5, self.Ea6, self.Ea7, self.Ea8 = Ea5, Ea6, Ea7, Ea8
        self.Ea9, self.Ea10, self.Ea11, self.Ea12 = Ea9, Ea10, Ea11, Ea12

        self.k1_0, self.k2_0, self.k3_0, self.k4_0 = k1_0, k2_0, k3_0, k4_0
        self.k5_0, self.k6_0, self.k7_0, self.k8_0 = k5_0, k6_0, k7_0, k8_0
        self.k9_0, self.k10_0, self.k11_0, self.k12_0 = k9_0, k10_0, k11_0, k12_0

        self.K13, self.K14, self.K15 = K13, K14, K15

        # Reaction enthalpies
        self.dH1, self.dH2, self.dH3 = dH1, dH2, dH3

        # Physical constants
        self.Rgas = 8.314

        # Coolant properties
        self.MassFlow_c = MassFlow_c
        self.rho_c = rho_c
        self.Cp_c = Cp_c
        self.therm_cond_w = therm_cond_w

        # Inlet composition (mole fractions)
        self.y_C2H4_in = y_C2H4_in
        self.y_O2_in   = y_O2_in  
        self.y_CO2_in  = y_CO2_in 
        self.y_EO_in   = y_EO_in  
        self.y_H2O_in  = y_H2O_in 
        self.y_CH4_in  = y_CH4_in 
        self.y_Ar_in   = y_Ar_in  
        self.y_EDC_in  = y_EDC_in 

        # Molecular weights [g/mol]
        self.Mw = {'C2H4': 28.05, 'O2': 32.00, 'CO2': 44.01,
                   'H2O': 18.02, 'EO': 44.05, 'CH4': 16.04, 'Ar': 39.95}

        # Derived quantities
        self.Area_cross_section = np.pi * (D_tube_in ** 2) / 4.0

        # Per-tube inlet flow [mol/s]
        self.F_in_tube = Total_F_in / Num_tubes

        # Inert flows
        self.nCH4 = self.F_in_tube * self.y_CH4_in
        self.nAr = self.F_in_tube * self.y_Ar_in
        self.nEDC = self.F_in_tube * self.y_EDC_in

        # Inlet flows
        self.nC2H4_0 = self.F_in_tube * self.y_C2H4_in
        self.nO2_0 = self.F_in_tube * self.y_O2_in
        self.nCO2_0 = self.F_in_tube * self.y_CO2_in
        self.nH2O_0 = self.F_in_tube * self.y_H2O_in
        self.nEO_0 = self.F_in_tube * self.y_EO_in

        # Gas mass flow [kg/s] per tube
        self.MassFlow_g = (self.nC2H4_0 * self.Mw['C2H4'] +
                          self.nO2_0 * self.Mw['O2'] +
                          self.nCO2_0 * self.Mw['CO2'] +
                          self.nH2O_0 * self.Mw['H2O'] +
                          self.nEO_0 * self.Mw['EO'] +
                          self.nCH4 * self.Mw['CH4'] +
                          self.nAr * self.Mw['Ar']) / 1000.0
        

        self.MassFlux_g = self.MassFlow_g / (self.Area_cross_section * self.void_fraction)
        
        # Coolant mass flux
        self.MassFlux_c = (4.0 * self.MassFlow_c) / (np.pi * (D_shell**2 - Num_tubes * D_tube_out**2))


    def set_inlet_flows(self, species_flows):
        """
        Set the inlet flows of each species in [mol/s] (TOTAL for a single tube).
        species_flows is a dict, e.g.:
        {
            'C2H4': value,
            'O2':   value,
            ...
        }
        The sum of these flows is the total per-tube flow F_in.
        """
        # If a species is not given, assume 0
        self.nC2H4_0 = species_flows.get('C2H4', 0.0)/self.Num_tubes
        self.nO2_0   = species_flows.get('O2', 0.0)/self.Num_tubes
        self.nCO2_0  = species_flows.get('CO2', 0.0)/self.Num_tubes
        self.nH2O_0  = species_flows.get('H2O', 0.0)/self.Num_tubes
        self.nEO_0   = species_flows.get('EO', 0.0)/self.Num_tubes
        self.nCH4    = species_flows.get('CH4', 0.0)/self.Num_tubes
        self.nAr     = species_flows.get('Ar', 0.0)/self.Num_tubes
        self.nEDC    = species_flows.get('EDC', 0.0)/self.Num_tubes

        # Recompute flow per tube
        self.F_in_tube = (self.nC2H4_0 + self.nO2_0 + self.nCO2_0 + 
                     self.nH2O_0 + self.nEO_0 + self.nCH4 + 
                     self.nAr + self.nEDC)
        # Recompute total flow rate

        self.Total_F_in = self.F_in_tube * self.Num_tubes

        # Recompute mass flow per tube
        self.MassFlow_g = ((self.nC2H4_0 * self.Mw['C2H4']) +
                           (self.nO2_0   * self.Mw['O2']) +
                           (self.nCO2_0  * self.Mw['CO2']) +
                           (self.nH2O_0  * self.Mw['H2O']) +
                           (self.nEO_0   * self.Mw['EO']) +
                           (self.nCH4    * self.Mw['CH4']) +
                           (self.nAr     * self.Mw['Ar'])) / 1000.0

        self.MassFlux_g = self.MassFlow_g / (self.Area_cross_section * self.void_fraction)

    # ---------------------------
    # Helper methods to reduce repeated code
    def _calc_gas_properties(self, y, z):
        """
        Calculate total molar flow, partial pressures, a_factor, and gas properties.
        """
        nC2H4, nO2, nCO2, nH2O, nEO, T_g, T_c, P = y
        nCH4, nAr, nEDC = self.nCH4, self.nAr, self.nEDC
        ntot_flow = nC2H4 + nO2 + nCO2 + nH2O + nEO + nCH4 + nAr + nEDC

        # Partial pressures
        p_C2H4 = (nC2H4 / ntot_flow) * P
        p_EO   = max((nEO/ntot_flow)*P, 1e-8)
        p_O2   = max((nO2/ntot_flow)*P, 1e-8)
        p_EDC  = max((nEDC/ntot_flow)*P, 1e-12)


        # Determine if reaction is active
        a_factor = 0.0 if (z <= self.L_inert or z >= (self.L_inert + self.L_reaction)) else 1.0

        # Gas properties
        Vol_flow = (ntot_flow * self.Rgas * T_g) / (P * 101325)
        rho_g = self.MassFlow_g / Vol_flow
        Vel_g = self.MassFlux_g / rho_g
        Cp_mix = ((nC2H4 * self.Cp_C2H4(T_g) +
                   nO2   * self.Cp_O2(T_g) +
                   nCO2  * self.Cp_CO2(T_g) +
                   nH2O  * self.Cp_H2O(T_g) +
                   nEO   * self.Cp_EO(T_g) +
                   nCH4  * self.Cp_CH4(T_g) +
                   nAr   * self.Cp_Ar(T_g)) / self.MassFlow_g)
        return (p_C2H4, p_O2, p_EO, p_EDC, a_factor, rho_g, Vel_g, Cp_mix)

    def _calc_reaction_rates(self, T, p_C2H4, p_O2, p_EO, p_EDC, a_factor):
        """
        Calculate reaction rates r1, r2, r3 given temperature and partial pressures.
        """
        r1 = 0.85 * ((self.K1_func(T, a_factor) * p_C2H4 * p_O2 -
                      self.K2_func(T, a_factor) * p_C2H4 * p_O2 * (p_EDC ** self.K13))
                     / (1 + self.K5_func(T, a_factor) * p_O2 + self.K6_func(T, a_factor) * p_C2H4)
                    ) * self.rho_bulk * (1 - self.void_fraction) * (1000.0 / 3600.0)

        r2 = 0.132 * ((self.K3_func(T, a_factor) * p_C2H4 * p_O2 -
                       self.K4_func(T, a_factor) * p_C2H4 * p_O2 * (p_EDC ** self.K14))
                      / (1 + self.K5_func(T, a_factor) * p_O2 + self.K6_func(T, a_factor) * p_C2H4)
                     ) * self.rho_bulk * (1 - self.void_fraction) * (1000.0 / 3600.0)

        r3 = 3e8 * ((self.K7_func(T, a_factor) * p_EO * p_O2 -
                           self.K8_func(T, a_factor) * p_O2 * p_EO * (p_EDC ** self.K15))
                          / ((1 + self.K9_func(T, a_factor) * p_O2 +
                              self.K10_func(T, a_factor) * (p_O2 ** 0.5) +
                              self.K11_func(T, a_factor) * p_EO +
                              self.K12_func(T, a_factor) * p_EO * (p_O2 ** -0.5)) ** 2)
                         ) * self.rho_bulk * (1 - self.void_fraction) * (1000.0 / 3600.0)
        r1 = max(r1, 0.0)
        r2 = max(r2, 0.0)
        r3 = max(r3, 0.0)

        return r1, r2, r3

    def _calc_HTCs(self, Cp_mix, Vel_g, rho_g):
        """
        Calculate heat transfer coefficients and overall U_HTC.
        """
        # Some assumed properties for gas and coolant
        visc_g = 16.22e-6
        therm_cond_g = 0.0629
        Re_cat = (self.MassFlux_g * self.char_d_cat) / visc_g

        therm_cond_c = 0.1074
        Cp_c = self.Cp_c
        visc_c = 0.625e-3
        Re_cool = (self.MassFlux_c / visc_c) * ((self.D_shell**2 - self.Num_tubes * self.D_tube_out**2) / self.D_shell)
        Prandt_c = (Cp_c * visc_c) / therm_cond_c

        HTC_s =  ((Cp_mix * visc_g) / therm_cond_g) ** (-0.67) * \
                ((2.867 / Re_cat) + (0.3023 / (Re_cat ** 0.35))) * rho_g * Cp_mix * Vel_g
        HTC_in = 7.676 + 0.0279 * (therm_cond_g / self.D_tube_in) * Re_cat
        HTC_out = 0.023 * (Re_cool ** (-0.2)) * (Prandt_c ** (-0.6)) * self.rho_c * Cp_c * (self.MassFlux_c / self.rho_c)
        U_HTC = 1.0 / ((1.0 / HTC_in) +
                       ((self.D_tube_out - self.D_tube_in) / (2.0 * self.therm_cond_w)) +
                       (1.0 / HTC_out))
        return HTC_s, U_HTC

    # ---------------------------
    # Arrhenius rate constant functions (unchanged for clarity)
    def K1_func(self, Tk, a_factor=1.0): return a_factor * self.k1_0 * np.exp(-self.Ea1/(self.Rgas*Tk))
    def K2_func(self, Tk, a_factor=1.0): return a_factor * self.k2_0 * np.exp(-self.Ea2/(self.Rgas*Tk))
    def K3_func(self, Tk, a_factor=1.0): return a_factor * self.k3_0 * np.exp(-self.Ea3/(self.Rgas*Tk))
    def K4_func(self, Tk, a_factor=1.0): return a_factor * self.k4_0 * np.exp(-self.Ea4/(self.Rgas*Tk))
    def K5_func(self, Tk, a_factor=1.0): return a_factor * self.k5_0 * np.exp(self.Ea5/(self.Rgas*Tk))
    def K6_func(self, Tk, a_factor=1.0): return a_factor * self.k6_0 * np.exp(self.Ea6/(self.Rgas*Tk))
    def K7_func(self, Tk, a_factor=1.0): return a_factor * self.k7_0 * np.exp(-self.Ea7/(self.Rgas*Tk))
    def K8_func(self, Tk, a_factor=1.0): return a_factor * self.k8_0 * np.exp(-self.Ea8/(self.Rgas*Tk))
    def K9_func(self, Tk, a_factor=1.0): return a_factor * self.k9_0 * np.exp(-self.Ea9/(self.Rgas*Tk))
    def K10_func(self, Tk, a_factor=1.0): return a_factor * self.k10_0 * np.exp(-self.Ea10/(self.Rgas*Tk))
    def K11_func(self, Tk, a_factor=1.0): return a_factor * self.k11_0 * np.exp(-self.Ea11/(self.Rgas*Tk))
    def K12_func(self, Tk, a_factor=1.0): return a_factor * self.k12_0 * np.exp(-self.Ea12/(self.Rgas*Tk))

    # Heat capacity functions (from NIST webbook)
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

    # ---------------------------
    def _solve_catalyst_temp(self, T_g, HTC_s, interf_area, p_C2H4, p_O2, p_EO, p_EDC, a_factor):
        """
        Solve for the catalyst temperature T_s via fsolve.
        """

        def T_s_solver(T_s_guess):
            r1, r2, r3 = self._calc_reaction_rates(T_s_guess, p_C2H4, p_O2, p_EO, p_EDC, a_factor)
            Q_react = -(r1 * self.dH1 + r2 * self.dH2 + r3 * self.dH3)
            return (Q_react / (HTC_s * interf_area) + T_g) - T_s_guess

        T_s_init_guess = T_g + 10.0
        try:
            T_solution = fsolve(T_s_solver, x0=T_s_init_guess)[0]
        except RuntimeError:
            print("catalyst temp could not be solved")

        return T_solution

    # ---------------------------
    def reactor_odes(self, z, y):
        """
        ODE function: y = [nC2H4, nO2, nCO2, nH2O, nEO, T_g, T_c, P]
        """
        nC2H4, nO2, nCO2, nH2O, nEO, T_g, T_c, P = y

        if P<0 or np.isnan(P):
            print("Error, Pressure is ", P)
            raise RuntimeError("Negative Pressure / NaN in Reactor")

        # Get gas properties and partial pressures
        p_C2H4, p_O2, p_EO, p_EDC, a_factor, rho_g, Vel_g, Cp_mix = self._calc_gas_properties(y, z)

        # Heat transfer coefficients
        HTC_s, U_HTC = self._calc_HTCs(Cp_mix, Vel_g, rho_g)

        # Solve for catalyst temperature using fsolve
        T_s = self._solve_catalyst_temp(T_g, HTC_s, self.interf_area, p_C2H4, p_O2, p_EO, p_EDC, a_factor)

        # Compute reaction rates at T_s
        r1, r2, r3 = self._calc_reaction_rates(T_s, p_C2H4, p_O2, p_EO, p_EDC, a_factor)

        # Species production rates
        dF_C2H4_dz = (-r1 - r2) * self.Area_cross_section
        dF_O2_dz   = (-0.5 * r1 - 3.0 * r2 - 2.5 * r3) * self.Area_cross_section
        dF_CO2_dz  = (0.0 * r1 + 2.0 * r2 + 2.0 * r3) * self.Area_cross_section
        dF_H2O_dz  = (0.0 * r1 + 2.0 * r2 + 2.0 * r3) * self.Area_cross_section
        dF_EO_dz   = (1.0 * r1 + 0.0 * r2 - 1.0 * r3) * self.Area_cross_section

        # Energy balances
        dTg_dz = (1.0 / (rho_g * Cp_mix * Vel_g)) * \
                 (HTC_s * self.interf_area * (T_s - T_g) - (4.0 * U_HTC / self.D_tube_in) * (T_g - T_c))
        dTc_dz = (4.0 * U_HTC / (self.rho_c * self.Cp_c * (self.MassFlux_c / self.rho_c))) * \
                 ((self.Num_tubes * self.D_tube_out) / (self.D_shell**2 - self.Num_tubes * self.D_tube_out**2)) * \
                 (T_g - T_c)

        # Pressure drop (Ergun eqn.)
        visc_g = 16.22e-6
        dP_dz = -(150.0 * ((visc_g * (1.0 - self.void_fraction)**2) /
                (self.void_fraction**3 * self.d_cat**2)) * Vel_g +
                1.75 * ((rho_g * (1.0 - self.void_fraction)) /
                (self.void_fraction**3 * self.d_cat)) * Vel_g**2) / 101325.0

        return [dF_C2H4_dz, dF_O2_dz, dF_CO2_dz, dF_H2O_dz, dF_EO_dz, dTg_dz, dTc_dz, dP_dz]

    # ---------------------------
    def cat_temp_calc(self, z, y):
        """
        Calculate the catalyst temperature T_s at a given z by reusing helper methods.
        """
        # Reuse the same gas property calculations as in reactor_odes
        p_C2H4, p_O2, p_EO, p_EDC, a_factor, rho_g, Vel_g, Cp_mix = self._calc_gas_properties(y, z)
        HTC_s, _ = self._calc_HTCs(Cp_mix, Vel_g, rho_g)
        T_s = self._solve_catalyst_temp(y[5], HTC_s, self.interf_area, p_C2H4, p_O2, p_EO, p_EDC, a_factor)
        return T_s

    # ---------------------------
    def run_simulation(self):
        Tg_0 = self.T_g_in + 273.15
        Tc_0 = self.T_cool_in + 273.15
        P0   = self.P_in

        y0 = [
            self.nC2H4_0, self.nO2_0, self.nCO2_0, 
            self.nH2O_0,  self.nEO_0,
            Tg_0, Tc_0, P0
        ]

        z_span = (0.0, self.L_reactor_total)
        z_eval = np.linspace(z_span[0], z_span[1], 100)

        sol = solve_ivp(
            fun=lambda z, y: self.reactor_odes(z, y),
            t_span=z_span, y0=y0, t_eval=z_eval, method='RK45'
        )
        if not sol.success:
            print("ode solve error")
            raise RuntimeError("Integration failed: " + sol.message)

        nC2H4_sol = sol.y[0, :] * self.Num_tubes
        nO2_sol   = sol.y[1, :] * self.Num_tubes
        nCO2_sol  = sol.y[2, :] * self.Num_tubes
        nH2O_sol  = sol.y[3, :] * self.Num_tubes
        nEO_sol   = sol.y[4, :] * self.Num_tubes

        nTotal_sol = (nC2H4_sol + nO2_sol + nCO2_sol + nH2O_sol + nEO_sol +
                      (self.nAr + self.nCH4) * self.Num_tubes)

        Tg_sol = sol.y[5, :]
        Tc_sol = sol.y[6, :]
        P_sol  = sol.y[7, :]

        # Catalyst temperature
        Ts_sol = np.array([
            self.cat_temp_calc(z_pt, sol.y[:, i]) 
            for i, z_pt in enumerate(sol.t)
        ])

        conv = (nC2H4_sol[0] - nC2H4_sol[-1]) / nC2H4_sol[0] * 100.0
        sel  = ((nEO_sol[-1] - nEO_sol[0]) /
               (nC2H4_sol[0] - nC2H4_sol[-1])) * 100.0

        return (sol.t, sol.y, nC2H4_sol, nO2_sol, nCO2_sol, nH2O_sol, nEO_sol,
                nTotal_sol, Tg_sol, Tc_sol, Ts_sol, P_sol, conv, sel)


# Define a fixed order for species so that we can convert between dictionaries and vectors.
SPECIES_LIST = ["C2H4", "O2", "CO2", "H2O", "EO", "CH4", "Ar", "EDC"]

def dict_to_vector(flow_dict):
    """Convert a species-flow dict into a NumPy array with species in SPECIES_LIST order."""
    return np.array([flow_dict.get(sp, 0.0) for sp in SPECIES_LIST], dtype=float)

def vector_to_dict(x):
    """Convert a NumPy array back into a dict using the SPECIES_LIST order."""
    return {sp: val for sp, val in zip(SPECIES_LIST, x)}

def iteration_function(reactor, fresh_feed, purge_fraction, x_current):

    """
    This function encapsulates one “iteration” of the reactor-seperation-recycle loop.
    
    Parameters:
      reactor       : an instance of EOReactor.
      fresh_feed    : dict of fresh feed flows (per tube).
      purge_fraction: the purge fraction used in the recycle.
      x_current     : current guess for the inlet flows (vector form).
    
    Steps:
      1. Convert the input vector to a dictionary.
      2. Set the reactor inlet flows.
      3. Run the reactor simulation.
      4. Use the reactor outlet to compute the lean gas, 
         apply separation/purge adjustments, and form the new inlet.
      5. Return the new inlet flows (as a vector) along with conversion and selectivity.
    """
    # Convert current guess into a dictionary.
    inlet_dict_current = vector_to_dict(x_current)
    
    # Set reactor inlet flows.
    reactor.set_inlet_flows(inlet_dict_current)
    
    # Run reactor simulation.
    (z_pts, y_sol,
     nC2H4_sol, nO2_sol, nCO2_sol, nH2O_sol, nEO_sol,
     nTotal_sol, Tg_sol, Tc_sol, Ts_sol, P_sol, conv, sel) = reactor.run_simulation()
    
    # Reactor outlet flows.
    outlet_dict = {
        'C2H4': nC2H4_sol[-1],
        'O2'  : nO2_sol[-1],
        'CO2' : nCO2_sol[-1],
        'H2O' : nH2O_sol[-1],
        'EO'  : nEO_sol[-1],
        'CH4' : inlet_dict_current['CH4'],  # inert
        'Ar'  : inlet_dict_current['Ar'],
        'EDC' : 0.0, # assume EDC is neglibgible
    }
    
    # Compute “lean gas” from the outlet.
    lean_gas = {
        'C2H4': outlet_dict['C2H4'],
        'O2'  : outlet_dict['O2'],
        'CO2' : outlet_dict['CO2'],
        'H2O' :(outlet_dict['H2O'] + 4545.83) * 0.0105,
        'EO'  : outlet_dict['EO'] * 7.8e-4,
        'CH4' : outlet_dict['CH4'],
        'Ar'  : outlet_dict['Ar']
    }
    
    # Form the recycle stream by applying purge and additional adjustments.
    recycle_stream = {sp: (1 - purge_fraction) * lean_gas[sp] for sp in lean_gas}

    # Carbon capture removal
    recycle_stream['C2H4'] -= 1.25
    recycle_stream['CO2']  -= 42.778
    recycle_stream['CH4']  -= 0.6944
    recycle_stream['H2O']  -= 0.222
    recycle_stream = {sp: (0.0 if recycle_stream[sp]<0 else recycle_stream[sp]) for sp in recycle_stream }
    
    # Combine fresh feed and recycle stream to generate the new inlet.
    new_inlet_dict = {}
    for sp in SPECIES_LIST:
        fresh_val = fresh_feed.get(sp, 0.0)
        rec_val = recycle_stream.get(sp, 0.0)
        new_inlet_dict[sp] = fresh_val + rec_val
    
    # Convert the updated inlet flows back to vector format.
    x_new = dict_to_vector(new_inlet_dict)

    # Calculate species leaving system
    EO_prod = outlet_dict['EO'] * (1- 7.8e-4)

    return x_new, conv, sel, EO_prod

def run_reactor_with_anderson(reactor, fresh_feed, purge_fraction):
    """
    Solve the steady-state reactor inlet (recycle stream) problem using Anderson mixing.
    
    """
    # set the initial guess.
    init_guess_dict = {
        'C2H4': 2420.5143355918367, 
        'O2': 586.6103912330229, 
        'CO2': 710.3077065492872, 
        'H2O': 48.15885277714581, 
        'EO': 0.09007785790898179, 
        'CH4': 6554.972493547981, 
        'Ar': 252.04560364694726,
        "EDC":  1e-5
    }

    x_init = dict_to_vector(init_guess_dict)

    
    
    # Define the fixed-point operator F(x) and hence f(x) = F(x) - x.
    def F(x):
        x_new, conv, sel, _ = iteration_function(reactor, fresh_feed, purge_fraction, x)
        return x_new
    def f(x):
        return F(x) - x
    try: 
        x = anderson(f, x_init, verbose=False, maxiter=100)
        # Final evaluation: run iteration_function once more to obtain conversion and selectivity.
        x_reactor_inlet, conv, sel, EO_prod = iteration_function(reactor, fresh_feed, purge_fraction, x)
    except NoConvergence: # Penalise no convergence
        x_reactor_inlet = np.array([0,0,0,0,0,0,0,0])
        conv = 0
        sel = 0
        EO_prod = 0
        print("Error: Did not converge within specified iterations")

    
    
    return vector_to_dict(x_reactor_inlet), conv, sel, EO_prod


def objective(x, base_reactor):

    # ------- 1. unpack and assign ------------------------------------

    # Variables to optimise
    purge   = (x[0])         
    L_rxn   = (x[1])

    # Feed_C2H4 = x[2]
    # Feed_O2 = x[3]
    # Feed_CH4 = x[4]
    # Feed_Ar = Feed_O2/49
    

    # # Fresh feed flowrate [mol/s]
    # fresh_feed = {
    #     "C2H4": Feed_C2H4,
    #     "O2":   Feed_O2,
    #     "CO2":  0.0,
    #     "H2O":  0.0,
    #     "EO":   0.0,
    #     "CH4":  Feed_CH4,
    #     "Ar":   Feed_Ar,
    #     "EDC":  1e-5
    # }
    # Base fresh feed values
    fresh_feed = {
        "C2H4": 179.94587524094948,
        "O2":   159.4390638987996,
        "CO2":  0.0,
        "H2O":  0.0,
        "EO":   0.0,
        "CH4":  34.435596598980446,
        "Ar":   159.4390638987996/49,
        "EDC":  1e-5
    }
    # Species price data ($/ kg)
    price = {
        'C2H4': 0.69,
        'EO'  : 1.39,
        'O2'  : 0.1,
        'CH4' : 0.15,
    }

    # Molecular weights [g/mol]
    Mw = {'C2H4': 28.05, 'O2': 32.00, 'CO2': 44.01,
                   'H2O': 18.02, 'EO': 44.05, 'CH4': 16.04, 'Ar': 39.95}


    # ------- 2. clone reactor & over‑write design vars  ----------------
    r = EOReactor()
    r.L_reaction = L_rxn
    r.L_reactor_total = 2*r.L_inert + L_rxn
    # ------- 3. run recycle solver with the candidate purge ------------
    try:
        inlet, conv, sel, EO_prod = run_reactor_with_anderson(
            r, fresh_feed=fresh_feed, purge_fraction=purge
        )
    except RuntimeError:
        # model failed to converge 
        print("Runtime Error, could not converge \n")
        return np.nan

    # ------- 4. economics ---------------------------------------------
    if EO_prod == 0:
        print("No Convergence / 0 EO prod")
        return np.nan

    
    mass_EO   = EO_prod * Mw['EO'] / 1000          # [kg/s]
    revenue   = price["EO"] * mass_EO              # [$/s]

    # example raw‑material cost
    mass_C2H4 = fresh_feed['C2H4'] * Mw['C2H4']/1000 # [kg/s]
    mass_O2 = fresh_feed['O2'] * Mw['O2']/1000       # [kg/s]
    mass_CH4 = fresh_feed['CH4'] * Mw['CH4']/1000       # [kg/s]

    cost_raw  = (mass_C2H4 * price["C2H4"] + mass_O2 * price['O2'] + mass_CH4 * price['CH4']) # [$/s]

    profit = revenue - cost_raw # [$/s]

    # ------ 5.  penalties (safety, limits) ---------------------
    
    if inlet['O2'] / (sum(inlet.values())) > 0.09:  # flammability envelope
         print("Flammability criteria exceeded\n")
         return -profit/2

    print("profit = ", profit,"purge = ", purge, "length reactor = ", L_rxn,"\n")

    return -profit        # minimisers want a *minimum*



def main():
    # Create reactor instance 
    reactor = EOReactor(L_reaction=14.39, P_in=18.227)
    
    # Define the fresh feed as a dictionary.
    fresh_feed = {
        "C2H4": 168.9,
        "O2":   157.0,
        "CO2":  0.0,
        "H2O":  0.0,
        "EO":   0.0,
        "CH4":  47.13,
        "Ar":   157.0/49,
        "EDC":  1e-5
    }
    
    # Run the anderson-based recycle loop solver.
    final_inlet, conv, sel, EO_prod = run_reactor_with_anderson(
        reactor,
        fresh_feed,
        purge_fraction=0.0107420000,
    )
    
    print("\nFinal steady-state inlet flows (mol/s):")
    total_flowrate = 0
    for sp, val in final_inlet.items():
        print(f"{sp}: {val:.3f}")
        total_flowrate += val
    print(f"Total reactor flow rate {total_flowrate:.2f}")
    print(f"Conversion = {conv:.2f}%   Selectivity = {sel:.2f}%")
    
def optimisation_main():

    reactor_base = EOReactor()

    x0 = np.array([0.01, 12])
    
    bounds = [(0.006, 0.05),
              (8,14),
              ]   
    
    tol = np.array([5e-5,   # purge tolerance
                    5e-2])  # length tolerance

    _prev = {"x": None}
    def stop_when_converged(x, f, context):
        """Return True to stop if x hasn't moved by more than tol."""
        if _prev["x"] is not None:
            if np.all(np.abs(x - _prev["x"]) < tol):
                return True
        _prev["x"] = x.copy()
        return False

    res = dual_annealing(
        func=objective,
        bounds=bounds,
        x0=x0,
        args=(reactor_base,),
        maxiter=1,
        callback=stop_when_converged
    )
    
    print("Stopped because:", res.message)
    print("Best profit: ", -res.fun)
    print("Optimal raw x: ",     res.x)


def objective_fun_test():

    reactor_base = EOReactor()
    x = [0.03143731341787988, 11.694990791392724]


    val = objective(x, reactor_base)

    print(val)
    return val

def surface_plot():
    # bounds
    x1_min, x1_max = 0.006, 0.05
    x2_min, x2_max = 8, 14
    num_points = 15

    # Create a grid for x1 and x2
    x1 = np.linspace(x1_min, x1_max, num_points)
    x2 = np.linspace(x2_min, x2_max, num_points)
    X1, X2 = np.meshgrid(x1, x2)


    reactor_base = EOReactor()
    # Compute objective values over the grid
    Z = np.zeros_like(X1)
    for i in range(num_points):
        for j in range(num_points):
            Z[i, j] = objective(np.array([X1[i, j], X2[i, j]]), base_reactor=reactor_base)

    # Plot the 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Objective Value')
    plt.show()

    np.savez('objective_data_2.npz', X1=X1, X2=X2, Z=Z)
    print("Saved X1, X2, and Z to 'objective_data_2.npz'")

def hinge_penalty(value, lo, hi, scale):
    """
    Returns zero inside [lo, hi], else a quadratic penalty that grows
    smoothly and keeps the optimiser on the right side of the fence.
    """
    if lo <= value <= hi:
        return 0.0
    elif value < lo:
        return scale*(lo - value)**2
    else:                              # value > hi
        return scale*(value - hi)**2


def optuna_objective(trial):
        # ------- 1. unpack and assign ------------------------------------

    # Variables to optimise
    purge   = trial.suggest_float("purge", 0.006, 0.05)        
    L_rxn   = trial.suggest_float("R_length", 8, 16)
    P_in = trial.suggest_float("P_in", 15, 21)
    Feed_C2H4 = trial.suggest_float("Feed_C2H4", 140, 200)
    Feed_O2 = trial.suggest_float("Feed_O2", 130, 170)
    Feed_CH4 = trial.suggest_float("Feed_CH4", 30, 100)
    Feed_Ar = Feed_O2/49
    

    # Fresh feed flowrate [mol/s]
    fresh_feed = {
        "C2H4": Feed_C2H4,
        "O2":   Feed_O2,
        "CO2":  0.0,
        "H2O":  0.0,
        "EO":   0.0,
        "CH4":  Feed_CH4,
        "Ar":   Feed_Ar,
        "EDC":  1e-5
    }
    # # Base fresh feed values
    # fresh_feed = {
    #     "C2H4": 167,
    #     "O2":   138.2,
    #     "CO2":  0.0,
    #     "H2O":  0.0,
    #     "EO":   0.0,
    #     "CH4":  55,
    #     "Ar":   2.84,
    #     "EDC":  1e-5
    # }
    # ----------  economic constants  ----------
    DISCOUNT_RATE = 0.08      # 8 % p.a.
    PLANT_LIFE    = 15        # years
    PLANT_UPTIME  = 8000      # hours per year
    CRF = DISCOUNT_RATE*(1+DISCOUNT_RATE)**PLANT_LIFE /((1+DISCOUNT_RATE)**PLANT_LIFE-1)


    # Species price data ($/ kg)
    price = {
        'C2H4': 0.69,
        'EO'  : 1.39,
        'O2'  : 0.1,
        'CH4' : 0.15,
        'Ag'  : 1600,

    }

    # ------- 2. clone reactor & over‑write design vars  ----------------
    r = EOReactor()
    r.P_in = P_in
    r.L_reaction = L_rxn
    r.L_reactor_total = 2*r.L_inert + L_rxn
    # ------- 3. run recycle solver with the candidate purge ------------
    try:
        inlet, conv, sel, EO_prod = run_reactor_with_anderson(
            r, fresh_feed=fresh_feed, purge_fraction=purge
        )
    except RuntimeError:
        # model failed to converge 
        print("Runtime Error, could not converge \n")
        return np.nan
    
    if EO_prod == 0:
        print("No Convergence / 0 EO prod")
        return np.nan
    # ------- 4. economics ---------------------------------------------

    Mw = r.Mw
    total_F_in = (sum(inlet.values()))
    inlet_fraction = {key: sp/total_F_in for key, sp in inlet.items()}


    # 4.1 raw material and product
    mass_EO   = EO_prod * Mw['EO'] / 1000          # [kg/s]
    revenue   = price["EO"] * mass_EO * 3600 * PLANT_UPTIME    # [$/yr]

    mass_C2H4 = fresh_feed['C2H4'] * Mw['C2H4']/1000 # [kg/s]
    mass_O2 = fresh_feed['O2'] * Mw['O2']/1000       # [kg/s]
    mass_CH4 = fresh_feed['CH4'] * Mw['CH4']/1000       # [kg/s]

    C_raw  = (mass_C2H4 * price["C2H4"] + mass_O2 * price['O2'] + mass_CH4 * price['CH4'])* 3600 * PLANT_UPTIME # [$/yr]

    #4.2 catalyst cost
    A_tube   = np.pi * r.D_tube_in**2 / 4          # m² per tube
    V_bed    = A_tube * r.L_reaction * r.Num_tubes     # m³
    rho_bulk   = r.rho_bulk                            # kg·m-3 of packed bed
    cat_void_frac  = r.void_fraction                       # void

    mass_cat = V_bed * (1 - cat_void_frac) * rho_bulk         # kg catalyst solids
    Ag_wt = 0.25                                           # 25 wt-% silver
    mass_Ag  = mass_cat * Ag_wt                                  # kg silver metal

    p_mfg  = 15          # $/kg of whole catalyst (support, plant fees)
    r_rec  = 0.95       # fraction of silver recovered and sold
    Life_cat  = 3           # yr

    CAPEX_silver = mass_Ag * price['Ag']                 # $   (initial charge)
    CAPEX_mfg    = mass_cat * p_mfg               # $   (support, impregnation)

    salvage_credit = mass_Ag * r_rec * price['Ag']       # $   when catalyst is pulled

    C_cat = ((CAPEX_silver + CAPEX_mfg - salvage_credit)
                            / Life_cat)       #annualised cat cost $·yr-¹

    r_vol = r.L_reactor_total*(np.pi*8**2)/4
    C_cap_r = 14000 + 15400*r_vol**0.7

    tac = C_cap_r*CRF + C_raw  + C_cat 

    # ------ 5.  penalties (safety, limits) ---------------------
    EO_kt_yr = mass_EO * (3600/1000000) * PLANT_UPTIME 
    penalty_EO_prod = hinge_penalty(EO_kt_yr, 160, 170, 1e7)
    penalty_O2 = hinge_penalty(inlet_fraction['O2'], 0.04, 0.09, 1e10)
    penalty_C2H4 = hinge_penalty(inlet_fraction['C2H4'], 0.15, 0.40, 1e9)
    penalty_conv = hinge_penalty(conv, 7, 15, 1e7)

    tac_penalised = tac + penalty_C2H4 + penalty_O2 + penalty_conv + penalty_EO_prod
         


    return tac_penalised        # minimisers want a *minimum*

def optuna_optimise():

    storage_url = "sqlite:///TAC.db"

    study = optuna.create_study(
        study_name="TAC_04/05",
        storage=storage_url,
        direction="minimize",
        load_if_exists=True,
    )

    study.optimize(optuna_objective, n_trials=1)

    # print results
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return 0

def manual_objective():
        # ------- 1. unpack and assign ------------------------------------

    # Variables to optimise
    purge   = 0.009     
    L_rxn   = 12
    P_in = 21
    Feed_C2H4 = 176.67
    Feed_O2 = 158.89
    Feed_CH4 = 55.56
    Feed_Ar = Feed_O2/49
    

    # Fresh feed flowrate [mol/s]
    fresh_feed = {
        "C2H4": Feed_C2H4,
        "O2":   Feed_O2,
        "CO2":  0.0,
        "H2O":  0.0,
        "EO":   0.0,
        "CH4":  Feed_CH4,
        "Ar":   Feed_Ar,
        "EDC":  1e-5
    }
    # # Base fresh feed values
    # fresh_feed = {
    #     "C2H4": 167,
    #     "O2":   138.2,
    #     "CO2":  0.0,
    #     "H2O":  0.0,
    #     "EO":   0.0,
    #     "CH4":  55,
    #     "Ar":   2.84,
    #     "EDC":  1e-5
    # }
    # ----------  economic constants  ----------
    DISCOUNT_RATE = 0.08      # 8 % p.a.
    PLANT_LIFE    = 15        # years
    PLANT_UPTIME  = 8000      # hours per year
    CRF = DISCOUNT_RATE*(1+DISCOUNT_RATE)**PLANT_LIFE /((1+DISCOUNT_RATE)**PLANT_LIFE-1)


    # Species price data ($/ kg)
    price = {
        'C2H4': 0.69,
        'EO'  : 1.39,
        'O2'  : 0.1,
        'CH4' : 0.15,
        'Ag'  : 1600,

    }

    # ------- 2. clone reactor & over‑write design vars  ----------------
    r = EOReactor()
    r.P_in = P_in
    r.L_reaction = L_rxn
    r.L_reactor_total = 2*r.L_inert + L_rxn
    # ------- 3. run recycle solver with the candidate purge ------------
    try:
        inlet, conv, sel, EO_prod = run_reactor_with_anderson(
            r, fresh_feed=fresh_feed, purge_fraction=purge
        )
    except RuntimeError:
        # model failed to converge 
        print("Runtime Error, could not converge \n")
        return np.nan
    
    if EO_prod == 0:
        print("No Convergence / 0 EO prod")
        return np.nan
    # ------- 4. economics ---------------------------------------------

    Mw = r.Mw
    total_F_in = (sum(inlet.values()))
    inlet_fraction = {key: sp/total_F_in for key, sp in inlet.items()}


    # 4.1 raw material and product
    mass_EO   = EO_prod * Mw['EO'] / 1000          # [kg/s]
    revenue   = price["EO"] * mass_EO * 3600 * PLANT_UPTIME    # [$/yr]

    mass_C2H4 = fresh_feed['C2H4'] * Mw['C2H4']/1000 # [kg/s]
    mass_O2 = fresh_feed['O2'] * Mw['O2']/1000       # [kg/s]
    mass_CH4 = fresh_feed['CH4'] * Mw['CH4']/1000       # [kg/s]

    C_raw  = (mass_C2H4 * price["C2H4"] + mass_O2 * price['O2'] + mass_CH4 * price['CH4'])* 3600 * PLANT_UPTIME # [$/yr]

    #4.2 catalyst cost
    A_tube   = np.pi * r.D_tube_in**2 / 4          # m² per tube
    V_bed    = A_tube * r.L_reaction * r.Num_tubes     # m³
    rho_bulk   = r.rho_bulk                            # kg·m-3 of packed bed
    cat_void_frac  = r.void_fraction                       # void

    mass_cat = V_bed * (1 - cat_void_frac) * rho_bulk         # kg catalyst solids
    Ag_wt = 0.25                                           # 25 wt-% silver
    mass_Ag  = mass_cat * Ag_wt                                  # kg silver metal

    p_mfg  = 15          # $/kg of whole catalyst (support, plant fees)
    r_rec  = 0.95       # fraction of silver recovered and sold
    Life_cat  = 3           # yr

    CAPEX_silver = mass_Ag * price['Ag']                 # $   (initial charge)
    CAPEX_mfg    = mass_cat * p_mfg               # $   (support, impregnation)

    salvage_credit = mass_Ag * r_rec * price['Ag']       # $   when catalyst is pulled

    C_cat = ((CAPEX_silver + CAPEX_mfg - salvage_credit)
                            / Life_cat)       #annualised cat cost $·yr-¹

    r_vol = r.L_reactor_total*(np.pi*8**2)/4
    C_cap_r = 14000 + 15400*r_vol**0.7

    tac = C_cap_r*CRF + C_raw  + C_cat 

    # ------ 5.  penalties (safety, limits) ---------------------
    EO_kt_yr = mass_EO * (3600/1000000) * PLANT_UPTIME 
    penalty_EO_prod = hinge_penalty(EO_kt_yr, 160, 170, 1e7)
    penalty_O2 = hinge_penalty(inlet_fraction['O2'], 0.04, 0.09, 1e10)
    penalty_C2H4 = hinge_penalty(inlet_fraction['C2H4'], 0.15, 0.40, 1e9)
    penalty_conv = hinge_penalty(conv, 7, 15, 1e7)

    tac_penalised = tac + penalty_C2H4 + penalty_O2 + penalty_conv + penalty_EO_prod
         


    return tac_penalised        # minimisers want a *minimum*

if __name__ == "__main__":
    #
    main()
    #objective_fun_test()
    #optimisation_main()
    #surface_plot()
    #optuna_optimise()
    manual_objective()

   
