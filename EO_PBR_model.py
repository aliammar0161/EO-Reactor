import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Global variables/constants

# Reactor geometry
L_reactor = 8         # Reactor length [m] (Galen, 7.5 total, 6.5 with catalyst)
D_tube    = 0.05        # Tube inner diameter [m] 

Area_cross_section = np.pi * (D_tube**2) / 4.0

# Operating conditions
P_in       = 21.0       # Inlet pressure [atm]
T_in       = 250.0      # Inlet temperature [C]
T_cool     = 220.0      # Coolant temperature [C] (shell side)
U          = 220.0      # Overall heat transfer coefficient [W/m2-K]
Total_F_in = 30000.0    # Total molar flowrate at inlet [kmol/h]
Num_tubes  = 7000
F_in       = Total_F_in/Num_tubes     

# Composition (mole fractions) at inlet:
y_C2H4_in  = 0.173       # Ethylene
y_O2_in    = 0.084      # Oxygen
y_CO2_in   = 0.029
y_EO_in    = 0.0
y_H2O_in   = 0.005
y_CH4_in  = 0.675         # Methane
y_Ar_in   = 0.035
# inert flowrate will remain constant
nCH4 = F_in*y_CH4_in
nAr = F_in*y_Ar_in


# Catalyst properties
rho_bulk     = 1260.0   # Bulk density of catalyst bed [kg-cat / m3-bed] (From Aryana plant data)
void_fraction= 0.45      # Void fraction of the bed (Galen)
dp           = 0.0039    # Catalyst particle diameter [m] (Vandervoort uses this value from Cornelio et al)
char_dp      = 1.5*dp   # Characteristic diameter of catalyst (Vandervoort)
# If you want to incorporate time-dependent deactivation:
reactor_uptime   = 0    # Set to non-zero to account for deactivation

# Kinetic constants (example from  Petrov et al 1984, 1988 and Eliyas 1988)
# We'll assume the same 3-reaction scheme:
#   1) C2H4 + 0.5 O2 -> EO      (desired)
#   2) C2H4 + 3   O2 -> 2 CO2 + 2 H2O  (complete combustion)
#   3) EO   + 2.5 O2 -> 2 CO2 + 2 H2O  (EO oxidation)

Ea1 = 38744  # J/mol, activation energy for K1
Ea2 = 43568  # J/mol, K2
Ea3 = 36786  # J/mol, K3 
Ea4 = 39635  # J/mol, K4
Ea5 = 40208  # J/mol, K5
Ea6 = 62944  # J/mol, K6
Ea7 = 91960  # J/mol, K7
Ea8 = 117040  # J/mol, K8
Ea9 = 150480  # J/mol, K9
Ea10 = 108680  # J/mol, K10
Ea11 = 41800  # J/mol, K11
Ea12 = 50160  # J/mol, K12

k1_0 = 6.867     # Pre-exponential, reaction 1
k2_0 = 1.073e2     # ...
k3_0 = 1.062e1     # ...
k4_0 = 3.959e1     # ...
k5_0 = 1.346e-6     # ...
k6_0 = 3.109e-8     # ...
k7_0 = 1.029e-3     # ...
k8_0 = 2.001e-1     # ...
k9_0 = 4.253e-5    # ...
k10_0 = 4.585e-4     # ...
k11_0 = 3.001e-1    # ...
k12_0 = 2.000e-1     # ...

K13 = 0.19  #These rate constants do not vary with temperature
K14 = 0.07
K15 = 0.6



Rgas = 8.314     # [J/mol-K]

# Arrhenius rate constants:
def K1_func(Tk, a_factor=1.0):
    # Temperature must be given in [K]
    # a_factor is for catalyst deactivation, if needed
    return a_factor * k1_0 * np.exp(-Ea1/(Rgas*Tk))   ### CHHECK IF Tk needs to be replaces with a different temperature

def K2_func(Tk, a_factor=1.0):
    return a_factor * k2_0 * np.exp(-Ea2/(Rgas*Tk))

def K3_func(Tk, a_factor=1.0):
    return a_factor * k3_0 * np.exp(-Ea3/(Rgas*Tk))

def K4_func(Tk, a_factor=1.0):
    return a_factor * k4_0 * np.exp(-Ea4/(Rgas*Tk))

def K5_func(Tk, a_factor=1.0):
    return a_factor * k5_0 * np.exp(Ea5/(Rgas*Tk))

def K6_func(Tk, a_factor=1.0):
    return a_factor * k6_0 * np.exp(Ea6/(Rgas*Tk))

def K7_func(Tk, a_factor=1.0):
    return a_factor * k7_0 * np.exp(-Ea7/(Rgas*Tk))

def K8_func(Tk, a_factor=1.0):
    return a_factor * k8_0 * np.exp(-Ea8/(Rgas*Tk))

def K9_func(Tk, a_factor=1.0):
    return a_factor * k9_0 * np.exp(-Ea9/(Rgas*Tk))

def K10_func(Tk, a_factor=1.0):
    return a_factor * k10_0 * np.exp(-Ea10/(Rgas*Tk))

def K11_func(Tk, a_factor=1.0):
    return a_factor * k11_0 * np.exp(-Ea11/(Rgas*Tk))

def K12_func(Tk, a_factor=1.0):
    return a_factor * k12_0 * np.exp(-Ea12/(Rgas*Tk))

# Reaction enthalpies [J/mol]
#   From Aryana:
#   1) dH1 = -123000 J/mol
#   2) dH2 = -1300000 J/mol
#   3) dH3 = -1300000 + 123000 (since it's EO -> CO2) = -1170000 J/mol
dH1 = -1.23e5
dH2 = -1.3e6
dH3 = -1.17e6

# Physical constants
Mw = {
    'C2H4' : 28.05,
    'O2'   : 32.00,
    'CO2'  : 44.01,
    'H2O'  : 18.02,
    'EO'   : 44.05,
    'CH4'  : 16.04,
    'Ar'   : 39.95,
}

# Heat capacities [J/(mol K)], approximate constants or polynomials
def Cp_C2H4(T):
    return 80.0
def Cp_O2(T):
    return 36.0
def Cp_CO2(T):
    return 54.0
def Cp_H2O(T):
    return 75.0
def Cp_EO(T):
    return 60.0
def Cp_CH4(T): 
    return 35.0 
def Cp_Ar(T): 
    return 20.8 


# We'll define the differential model in the function below.
# z will be the spatial coordinate from 0 to L_reactor.
# y-vector:  y[0] = nC2H4, y[1] = nO2, y[2] = nCO2, y[3] = nH2O, y[4] = nEO, y[5] = T, y[6] = P 

def reactor_odes(z,y):
    # unpack state
    # flows are in kmol/h, convert to SI if needed
    # T is in [K], we must keep track carefully
    # P is in [bar]
    nC2H4 = y[0]
    nO2   = y[1]
    nCO2  = y[2]
    nH2O  = y[3]
    nEO   = y[4]
    T     = y[5]
    P     = y[6]
   
    # keeping T in K
    Tk = T

    # total flow
    ntot = (nC2H4 + nO2 + nCO2 + nH2O + nEO + nCH4 + nAr)  

    # partial pressures (assuming ideal gas):
    #  Pressure = total pressure * (n_i / ntot)
    #  if we want partial pressures in bar or Pa
    #  note: F_in was in kmol/h, let's do dimensionally consistent partial pressure approach:
    #  We will use fraction approach as well:
    y_C2H4 = nC2H4 / ntot
    y_O2   = nO2   / ntot
    y_CO2  = nCO2  / ntot
    y_H2O  = nH2O  / ntot
    y_EO   = nEO   / ntot
    y_CH4  = nCH4  / ntot
    y_Ar   = nAr   / ntot

    p_C2H4 = y_C2H4 * P
    p_O2   = y_O2   * P
    p_EO   = y_EO   * P
    p_EDC  = 1e-6   * P  ### change this value

    # Reaction rates
    # optional: time-depend. deactivation factor:
    a_factor = 1.0
    if reactor_uptime == 0:
        # You could define a(t) function or store t somewhere else.
        # For a demonstration we keep it =1.0 
        pass
    
    # Reaction equations [mol / (g_cat h)] 
    r1 = -((K1_func(Tk)*p_C2H4*p_O2 - K2_func(Tk)*p_C2H4*p_O2*(p_EDC**K14)) / (1 + K5_func(Tk)*p_O2 + K6_func(Tk)*p_C2H4)) ### Find out why this work when this eq is negative
    r2 = ((K3_func(Tk)*p_C2H4*p_O2 - K4_func(Tk)*p_C2H4*p_O2*(p_EDC**K14)) / (1 + K5_func(Tk)*p_O2 + K6_func(Tk)*p_C2H4)) 
    r3 = ((K7_func(Tk)*p_EO*p_O2 - K8_func(Tk)*p_C2H4*p_EO*(p_EDC**K15))/((1 + K9_func(Tk)*p_O2 + K10_func(Tk)*(p_O2**0.5) + K11_func(Tk)*p_EO + K12_func(Tk)*p_EO*(p_O2**-0.5))**2)) 

    # Production rates per bed volume [mol/ (g_cat h)] ### check these units for some reason aryana and galen use different units
    R_C2H4 = -1.0 * r1 + (-1.0)*r2 + 0.0*r3
    R_O2   = -0.5 * r1 + (-3.0)*r2 + (-2.5)*r3
    R_CO2  =  0.0 * r1 + (+2.0)*r2 + (+2.0)*r3
    R_H2O  =  0.0 * r1 + (+2.0)*r2 + (+2.0)*r3
    R_EO   = +1.0 * r1 +   0.0*r2 + (-1.0)*r3

    # Then multiply by cross-sectional area [m^2], catalyst density [kg/m^3] and bed fraction for total [kmol/ (m h)] since derivative is w.r.t. z
    dF_C2H4_dz = R_C2H4 * Area_cross_section * rho_bulk  * (1-void_fraction)
    dF_O2_dz   = R_O2   * Area_cross_section * rho_bulk  * (1-void_fraction)
    dF_CO2_dz  = R_CO2  * Area_cross_section * rho_bulk  * (1-void_fraction)
    dF_H2O_dz  = R_H2O  * Area_cross_section * rho_bulk  * (1-void_fraction)
    dF_EO_dz   = R_EO   * Area_cross_section * rho_bulk  * (1-void_fraction)


    dT_dz = 0
    dP_dz = 0 


    return [dF_C2H4_dz, dF_O2_dz, dF_CO2_dz, dF_H2O_dz, dF_EO_dz, dT_dz, dP_dz]

def main():
    # Convert inlet T_in from [C] to [K]
    T0 = T_in + 273.15
    P0 = P_in

    nC2H4_0 = F_in * y_C2H4_in
    nO2_0   = F_in * y_O2_in
    nCO2_0  = F_in * y_CO2_in
    nH2O_0  = F_in * y_H2O_in
    nEO_0   = F_in * y_EO_in

        # initial condition vector
    y0 = [nC2H4_0, nO2_0, nCO2_0, nH2O_0, nEO_0, T0, P0]

    z_span = (0.0, L_reactor)
    z_eval = np.linspace(0, L_reactor, 20)

    sol = solve_ivp(reactor_odes, z_span, y0, t_eval=z_eval, method='RK45')
    if not sol.success:
        print("Integration failed with message:", sol.message)
        return
        # extract solution
    z_points  = sol.t
    nC2H4_sol= sol.y[0,:] * Num_tubes
    nO2_sol  = sol.y[1,:] * Num_tubes
    nCO2_sol = sol.y[2,:] * Num_tubes
    nH2O_sol = sol.y[3,:] * Num_tubes
    nEO_sol  = sol.y[4,:] * Num_tubes
    T_sol    = sol.y[5,:]

    # compute conversion and selectivity
    # define inlet moles
    C2H4_in = nC2H4_0 * Num_tubes
    # final moles at end
    C2H4_out = nC2H4_sol[-1]
    # conversion
    conv = (C2H4_in - C2H4_out)/C2H4_in * 100.0

    # selectivity wrt EO = (moles EO formed)/(moles C2H4 consumed)
    # final EO - initial
    EO_out = nEO_sol[-1]
    sel = (EO_out - nEO_0 * Num_tubes)/(C2H4_in - C2H4_out)*100.0
    print(f"Final C2H4 conversion = {conv:.2f}%")
    print(f"Selectivity to EO     = {sel:.2f}%")

    # Final inlet and outlet values
    print(f"C2H4 in and out  = [{nC2H4_sol[0]:.2f}, {nC2H4_sol[-1]:.2f}] kmol/hr")
    print(f"O2 in and out    = [{nO2_sol[0]:.2f}, {nO2_sol[-1]:.2f}] kmol/hr")
    print(f"CO2 in and out   = [{nCO2_sol[0]:.2f}, {nCO2_sol[-1]:.2f}] kmol/hr")
    print(f"H2O in and out   = [{nH2O_sol[0]:.2f}, {nH2O_sol[-1]:.2f}] kmol/hr")
    print(f"EO in and out    = [{nEO_sol[0]:.2f}, {nEO_sol[-1]:.2f}] kmol/hr")

    # plotting
    fig, axs = plt.subplots(2, 1, figsize=(7,8))
    axs[0].plot(z_points, nC2H4_sol, label='C2H4')
    axs[0].plot(z_points, nO2_sol  , label='O2')
    axs[0].plot(z_points, nCO2_sol , label='CO2')
    axs[0].plot(z_points, nH2O_sol , label='H2O')
    axs[0].plot(z_points, nEO_sol  , label='EO')
    axs[0].legend(loc='best')
    axs[0].set_xlabel("Reactor length z [m]")
    axs[0].set_ylabel("Flow [kmol/h]")
    axs[0].set_title("Species profiles")

    axs[1].plot(z_points, T_sol-273.15, 'r-', label='Temperature')
    axs[1].set_xlabel("Reactor length z [m]")
    axs[1].set_ylabel("Temperature [C]")
    axs[1].set_title("Temperature profile along reactor")
    axs[1].legend(loc='best')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()