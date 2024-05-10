 
""" 

Simple fly motor neuron model with NA/K pump as described in
Megwa, Pascual, Gunay, Pulver, Prinz (2023)
(called 'the paper' in comments below) 

units in this code: unless otherwise noted, 
- times are in ms
- voltages are in mV
- currents are in pA
- conductances are in nS
- concentrations are in M (molar)
- capacitance is in pF

"""

# =============================================================================
# Initalization
# =============================================================================
import pylab as plt
from scipy.integrate import odeint
import numpy as np
from scipy.signal import find_peaks
from scipy import signal
from datetime import datetime
import csv

# =============================================================================
# Global Cell Parameters
# =============================================================================
### Conductances
sf = 1.0 
""" A scaling factor to scale all the conductances up or down"""
g_Ks = 50.0 * sf
"""Slow Potassium (K) maximum conductance, in nS"""
g_Kf = 15.1 * sf
"""Fast Potassium (K) maximum conductance, in nS"""
g_NaP = 0.80 * sf
"""Persistent Sodium (Na) maximum conductance, in nS"""
g_NaT = 100.0 * sf
"""Transient Sodium (Na) maximum conductance, in nS"""
g_leak_Na = 1.2 * sf
""" Sodium (Na) leak conductance, in nS""" 
g_leak_K = 3.75 * sf 
""" Potassium (Na) leak conductance, in nS""" 

### Membrane Constants
C_m = 4.0
"""Membrane capacitance, in pF"""
E_K = -80.0
"""Potassium (K) reversal potential, in mV"""
F = 96485.3329e15
"""Faraday's constant, in  pA*ms/mol"""
volume = 5.4994e-13 #corresponding to 549 um^3
"""Cell volume, in L"""
nao = 0.135 # or 135 milimolar (mM)
"""External [Na], in M, from O'Dowd and Aldrich (1988)"""

# =============================================================================
# Switches for the Model Settings and Injectors
# =============================================================================
### Model Version Settings Corresponding to the Paper

# [1,0,0] "ConCon" ~ constant sodium concentration, constant sodium reversal
# [1,1,0] "DynCon" ~ dynamic sodium concentration, constant sodium reversal
# [1,1,1] "DynDyn" ~ dynamic sodium concentration, dynamic sodium reversal

pumpswitch = 1      # If pumpswitch = 0, then pump is absent
                    # If pumpswitch = 1, then pump is present

NaiSwitch = 1   # If NaiSwitch = 0, then sodium concentration is constant
                # If NaiSwitch = 1, then sodium concentration is dynamic

dynrevswitch = 1    # If dynrevsitch = 0, then sodium reversal is constant
                    # If dynrevsitch = 1, then sodium reversal is dynamic
                    
model_switchsettings = [pumpswitch, NaiSwitch, dynrevswitch]

def E_NaSwitch(nai):
    switch = dynrevswitch
    if switch == 0:
        return 31.2 + 0 * np.log(nao/nai) # returns constant sodium reversal potential of 31.2
    elif switch == 1:
        return 25.694 * np.log(nao/nai) # returns the Na Nernst potential based on momentary sodium concentration and temperature 26 Celsius

### Switches for the different injection current types
# Switch ON = 1, Switch OFF = 0
inj_switch = 1      # for step current injection
ramp_switch = 0     # for ramp current injection
TP_switch = 0       # for test pulse injection
zap_switch = 0      # for zap current injection

current_switches = [inj_switch, TP_switch, ramp_switch, zap_switch]

# =============================================================================
# Time Vector (Global)
# =============================================================================
dt = .05 # default time increment, in milliseconds
inv_dt = int(1/dt) # inverse of dt

tHold = 5*1000 # pre-injection duration, time in ms 
tPulse = 5*1000 # injection duration, time in ms
tPulseEnd = tHold+tPulse # end of current injection, time in ms
tPost = 15*1000 # duration of the simulation after the current injection, time in ms
t_end = tHold+tPulse+tPost # full simulation time in ms

t_vec = np.arange(t_end, step=dt) # vectorizing the simulation time, containing indexes

# =============================================================================
# Current Clamp Parameters
# =============================================================================
I_hold = 0.0 # Constant Holding Current, in pA

inj_start = 50.0 # Initial Step Current in Step Current Loop, if enabled
inj_int = 500 # Step Current Increment in Step Current Loop, if enabled
inj_last = 50 # Last Step Current Step in Step Current Loop, if enabled

inj = np.arange(inj_start, inj_last + 0.001, inj_int) 
# Full Array of Step Currents for Loop, format is (start, end, increment)'''

# =============================================================================
# Initial Conditions of Dynamic Variables
# =============================================================================
# This starting condition has all the activation gates de-activated (set to 0.0)
# and all the inactivation gates de-inactivated (set to 1.0)
V0 = -59.9312 # membrane voltage, based on resting membrane potential of DynDyn model with default pump parameters
mNaT0 = 0.0 # NaT activation
hNaT0 = 1.0 # NaT inactivation
mNaP0 = 0.0 # NaP activation
n0 = 0.0 # Ks activation
dmKfdt0 = 0.00 # Kf activation
dhKf1dt0 = 1.0 # Kf1 inactivation 
dhKf2dt0 = 1.0 # Kf2 inactivation
Nai0 = 0.0400811 # internal sodium concentration, based on resting sodium concentration of DynDyn model with default pump parameters
param0 = np.array([V0, mNaT0, hNaT0, mNaP0, n0, dmKfdt0, dhKf1dt0 ,dhKf2dt0, Nai0])
""" param0 creates array of initial conditions to be used in integration function"""

# =============================================================================
# The Leak Currents (Sodium and Potassium)
# =============================================================================
def I_leak_NA(V, nai): # Sodium Component of Leak Membrane current (in pA)
    return  g_leak_Na * (V - E_NaSwitch(nai))

def I_leak_K(V): # Potassium Component of Leak Membrane current (in pA)
    return  g_leak_K * (V - E_K)

# =============================================================================
# NaT Current (Transient Sodium) - uses equations from Lin et al, 2012 (which is cited in 'the paper')    
# =============================================================================
def minf_NaT(V):
    return  1 / (1 + np.exp((V + 29.13) / (-8.922)))
def mtau_NaT(V):
    return 3.861 - 3.434 / (1.0 + np.exp((V + 51.35) / (-5.98))) 
def hinf_NaT(V):
    return 1 / (1 + np.exp((V + 40.0) / 6.048)) # adjusted from Lin et al, 2012 
def htau_NaT(V):
    return 2.834 - 2.371 / (1.0 + np.exp((V + 21.9) / (-2.641))) # adjusted from Gunay et al, 2015 (which is cited in 'the paper')
def I_NaT(V, mNaT, hNaT, nai):
    return g_NaT * mNaT**3 * hNaT * (V - E_NaSwitch((nai))) # adjusted from Gunay et al, 2015 (which is cited in 'the paper')

# =============================================================================
# NaP Current (Persistent Na)
# =============================================================================
def minf_NaP(V):
    return 1 / (1 + np.exp((V + 48.77)/(-3.68)))
def mtau_NaP(V):
    return 1
def I_NaP(V, mNaP, nai):
    return g_NaP * mNaP * (V - E_NaSwitch(nai))

# =============================================================================
# Ks Current (Slow Potassium)
# =============================================================================
def ninf_Ks(V):
    return 1 / (1 + np.exp((V + 12.85)/(-19.91)))
def ntau_Ks(V):
    return 2.03 + 1.96 /(1 + np.exp((V - 29.83)/3.32))
def I_Ks(V, n):
    return g_Ks * n**4 * (V - E_K)

# =============================================================================
# Kf Current (Fast Potassium)
# =============================================================================    
def minf_Kf(V):
    return 1 / (1 + np.exp((V + 17.55)/(-7.27)))
def mtau_Kf(V):
    return 1.94 + 2.66 / (1 + np.exp((V - 8.12)/7.96))
def hinf1_Kf(V):
    return 1 / (1 + np.exp((V + 45.0)/6.0))
def htau_Kf(V):
    return 1.79 + 515.8 / (1 + np.exp((V + 147.4)/(28.66)))
def hinf2_Kf(V):
    return 1 / (1 + np.exp((V + 44.2) / 1.5))
def I_Kf(V, mKf, hKf1, hKf2):
    return g_Kf * mKf**4 * (0.95*hKf1 + 0.05*hKf2) * (V - E_K)

# =============================================================================
# Na/K Pump Current
# =============================================================================
Imaxpump = 75.0 # maximal pump current, in pA
Imaxpump_start = 75.0 # Initial Imaxpump in Imaxpump Loops, if enabled
    
naih = 40 * 10**(-3) # sodium concentration of pump half activation, in M (molar)
naiH_start = 25.0 * 10**(-3) # Initial naiH in naiH Loops, if enabled
 
nais = 10.0 * 10**(-3) # pump slope factor, in M (molar)
nais_start = 10.0 * 10**(-3) # Initial nais in nais Loops, if enabled

### Default pump parameters in 'the paper' are Imaxpump = 75pA , naih = 40 mM, nais = 10 mM

# =============================================================================
# Simulation Settings - For Loops
# =============================================================================

# ### Uncomment the 3 lines below for multiple IMaxPump simulations
# Imaxpump_end = 200 + 0.001            # Final Imaxpump in Imaxpump Loop, if enabled
# Imaxpump_step = 5                     # Imaxpump increment in Imaxpump Loop, if enabled
# Imaxpump_range = np.arange(Imaxpump_start , Imaxpump_end, Imaxpump_step)

# ### Uncomment lines the 3 lines below for multiple nais simulations
# nais_end = (40 + .0001) * 10**(-3)    # Final nais in nais Loop, if enabled
# nais_step = 0.5 * 10**(-3)            # nais increment in nais Loop, if enabled
# nais_range = np.arange(nais_start , nais_end,nais_step)

# ### Uncomment lines the 3 lines below for multiple naih simulations
naiH_end = (90 + .0001) * 10**(-3)      # Final naih in naih Loop, if enabled
naiH_step = 1000.0 * 10**(-3)           # naih increment in naih Loop, if enabled
naiH_range = np.arange(naiH_start , naiH_end, naiH_step)

# =============================================================================
# CSV File Output Initialization
# =============================================================================
# This section provides functionality to export simulation results to Excel .csv spreadsheets
# and examples to construct the export file names.
# Comment in or out and edit as needed.

filetime = datetime.now()
#sweep_filename = "your file name here" + str(model_switchsettings) + ", Inj." + str(inj[0]) + " to " + str(inj[-1]) + "pA --" + filetime.strftime("%H_%M_%S, %m-%d-%Y.csv")
#sweep_filename = "ADAPT LMR ZAP" + str(model_switchsettings) + ", 2.05 pA Inj - " + filetime.strftime("%H_%M_%S, %m-%d-%Y.csv")
#sweep_filename = "LMU Sweep Data" + str(model_switchsettings) + ", Inj." + str(inj[0]) + " to " + str(inj[-1]) + "pA --" + filetime.strftime("%H_%M_%S, %m-%d-%Y.csv")

### Comment in a title and the rest of the lines below in order to create a data file for simulation loop
# sweep_fields = ['I_Max (pA)', 'NaiH (M)',   'NaiS (M)',   
#                 'Inj.(pA)',
#                 'f0 IFR',     'Mean IFR',   'Final IFR',
#                 'G9 AVG (Hz)', 'G19 AVG (Hz)',
#                 'Adapt Slope', 'SPK INT %',
#                 'AHP50 (ms)',     'AHP Amp.(mV)',
#                 'Pre-Voltage','Pre-[Na+]','Pre-Pump','Pre-Reversal']

# with open(sweep_filename, 'w') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(sweep_fields) 
    


# =============================================================================
# Run Simulation
# =============================================================================

### This code has loop functions that allow sweeping through different parameter settings.
### Comment in and out as needed.
### Make sure to comment in the corresponding line at the end of the loop.

#while sf < sf_end:                     ### commented out, this would sweep scaling factor sf
#while Imaxpump < Imaxpump_end:         ### commented out, this would sweep Imaxpump
#while nais < nais_end:                 ### commented out, this would sweep nais
while naih < naiH_end:
    ### Creating a set of data arrays 
        vmin = [] # minimum voltage during simulation 
        ahp_amp = [] # amplitude (mV) of after-hyperpolarization
        ahp_halfdur = [] # half duration of AHP
        ahp_25dur = [] # quarter duration of AHP
        ahp_75dur = [] # three-quarters duration of AHP 
        mean_ifrs = [] # mean of all instantaneous firing rates (Hz)
        mean_Vcell = [] # mean voltage (mV) of the trough between spikes
        delay = [] # delay to first spike
    
        for i in range(0, len(inj)): # loop that steps through different current injections, if enabled
            global I_pulse
            I_pulse = inj[i]
            
            def I_pump(nai): # defines Na/K pump current, function of sodium concentration
                return Imaxpump / (1 + (np.exp((naih - nai) / nais)))
            
            def I_inj(t): # defines holding current injection
                return I_hold*(t<tHold) + (I_hold + I_pulse)*(t>=tHold)*(t<tPulseEnd) + I_hold*(t>=tPulseEnd)*(t<t_end)
            
            def I_testpulse(t): # defines test pulse injection
                 """ This function allows for a single test pulse BEFORE the main step current injection and a SET of
                 test pulses AFTER the main step current injection. 
                 """
                 global I_test
                 I_test = 22.0 # Test Pulse Current, in pA
                 
                 ### Defining times for a test pulse BEFORE the main step current injection
                 tTestPDur = .2*1000 # Test Pulse Duration in ms
                 tTestPDelayPre = 5*1000 # TP time BEFORE main step current injection in ms
                 tPreTestPStart = tHold - tTestPDelayPre - tTestPDur # Start time of the pre-injection test in ms
                 tPreTestPEnd = tPreTestPStart + tTestPDur # End time of the pre-inject test pulse in ms
                 
                 ### Defining times for test pulses AFTER the current injection
                 global tTestPDelayPost
                 tTestPDelayPost = 55*1000 # Amount of Delay / Wait Time AFTER the main step current injection
                 tPostTestStart = tPulseEnd + tTestPDelayPost # Time of first test pulse start after main step current injection
                 tPostTestPPeriod = 10000*1000 # Time between post-injection test pulses
                 
                 return (I_test*(t>=tPreTestPStart)*(t<tPreTestPEnd) 
                    + I_test*((t-tPostTestStart)>=0)*(((t-tPostTestStart)%tPostTestPPeriod)<tTestPDur))
                
            ### Defining the ramp current
            ramp0 = 90 # Ramp Current Amplitude, in pA
            tRampDur = 2*1000 # Ramp Duration (Up and Down), in ms
            
            ### Defining times for a ramp current BEFORE the main current injection
            tRampDelayPre = .5*1000 # Ramp Delay before the current injection time, in ms
            tPreRampStart = tHold - tRampDelayPre - tRampDur # Start Time of the Pre-Injection Ramp
            tPreRampPeakTime = tPreRampStart + (tRampDur/2) # Time to the peak of the Ramp from the beginning of the simulation
            tPreRampEnd = tPreRampStart + tRampDur # Time to the end of the ramp

            ### Defining times for a ramp current AFTER the main current injection
            tRampDelayPost = 4000*1000 # Ramp Delay after the end of the current injection duration
            tPostRampStart = tPulseEnd + tRampDelayPost # Start time of the post-injection Ramp
            tPostRampPeakTime = tPostRampStart + (tRampDur/2) # Time to the Peak of the Post-Injection ramp
            tPostRampEnd = tPostRampStart + tRampDur # Time to the end of the post-injection ramp
                
            def I_ramp(t):
                """ This function allows for a ramp up and ramp down current injection """
                return  (ramp0*((t-tPreRampStart)/(tRampDur/2))*(t>=tPreRampStart)*(t<tPreRampPeakTime) 
                         + ramp0*(1 - (t-tPreRampPeakTime)/(tRampDur/2))*(t>=tPreRampPeakTime)*(t<tPreRampEnd)
                         + ramp0*((t-tPostRampStart)/(tRampDur/2))*(t>=tPostRampStart)*(t<tPostRampPeakTime) 
                         + ramp0*(1 - (t-tPostRampPeakTime)/(tRampDur/2))*(t>=tPostRampPeakTime)*(t<tPostRampEnd))
            
            ### Zap Current Properties
            Zap_fMin = 0.1 / 1000 # Zap minimum Frequency in Hz / 1000
            Zap_fMax = 5.0 / 1000  # Zap maximum Frequency in Hz / 1000
            pi_phase0 = np.pi # inital phase shift, cos(pi) = 0
            Zap_tmax = 40 * 1000 # zap duration in milliseconds
            Zap_thalf = Zap_tmax * 0.5 # half of the zap duration
            
            ZapStartTime = 20 * 1000 # Start time for zap current (in milliseconds)
            ZapPeakTime = ZapStartTime + (Zap_thalf) # Time half way through the full zap duration
            ZapEndTime = ZapStartTime + Zap_tmax # Time to the end of the zap
                
            I_ZapMax = 22.0 #Zap Current Amplitude in pA
            Lambda = (np.log(Zap_fMax/Zap_fMin)) / (Zap_thalf) #Lambda is the exponenial rising factor, see 'the paper'
            
            ### Creating Zap Function
            def ZapFreq(t):
                TimeFunc = t - (ZapStartTime + 1e-10) # instantaneous time of zap up in ms
                TimeFunc0 = t - (ZapPeakTime + 1e-10) # instantaneeous time of zap down in ms
                T_map = (-1 * TimeFunc0) + Zap_thalf # maps the times of the up swing, for creating mirrored frequency down swing           
                FreqUP = (Zap_fMin / (Lambda*TimeFunc)) * (np.exp(Lambda*TimeFunc)-1) 
                FreqDOWN = (Zap_fMin / (Lambda*T_map) ) * (np.exp(Lambda*T_map)-1)
                return (FreqUP * (t>=ZapStartTime)*(t<ZapPeakTime) + 
                        FreqDOWN * (t>=ZapPeakTime)*(t<ZapEndTime))

            def I_Zap(t):
                TimeFunc = t - (ZapStartTime + 1e-10) # instanteous time of up current in ms
                TimeFunc0 = t - (ZapPeakTime + 1e-10) # instanteous time of down current in ms
                T_map = (-1 * TimeFunc0) + Zap_thalf # maps the times of the up swing, for creating mirrored frequency down swing        
                FreqUP = (Zap_fMin / (Lambda*TimeFunc)) * (np.exp(Lambda*TimeFunc)-1) 
                FreqDOWN = (Zap_fMin / (Lambda*T_map) ) * (np.exp(Lambda*T_map)-1)
                ZapEQ_Up = (I_ZapMax * (0.5 + 0.5*np.cos(2*np.pi*FreqUP*TimeFunc + pi_phase0)))
                ZapEQ_Down = (I_ZapMax * (0.5 + 0.5*np.cos(2*np.pi*FreqDOWN*T_map + pi_phase0)))
                return ( ZapEQ_Up * (t>=ZapStartTime)*(t<ZapPeakTime) + 
                         (ZapEQ_Down * (t>=ZapPeakTime)*(t<ZapEndTime) ) )   
                 
        ### ODEint (Runge-Kutta 4) Solver  
            def dALLdt(param_vec, t):
                """
                Integrate
                | :param param_vec:
                | :param t:
                | :return: array of 
                0- membrane potential, mV
                1- dmNaTdt    (NaT activation)
                2- dhNaTdt    (NaT inactivation)
                3- dmNaPdt    (NaP activation)
                4- dndt       (Ks activation) [aka nKs in 'the paper']
                5- dmKfdt     (Kf activation)
                6- dhKf1dt    (Kf-1 inactivation)
                7- dhKf2dt    (Kf-2 inactivation)
                8- dNaidt     (Internal [Na]), M/s (mol/(L*s))
                """
            
                V, mNaT, hNaT, mNaP, n, mKf, hKf1, hKf2, nai = param_vec
                
                dVdt        = (-1/C_m) * (I_Kf(V, mKf, hKf1, hKf2) + I_Ks(V, n)                        
                            + I_NaP(V, mNaP, nai) + I_NaT(V, mNaT, hNaT, nai)
                            + (I_leak_NA(V, nai) + I_leak_K(V)) + pumpswitch*I_pump(nai)
                            - inj_switch*I_inj(t) - TP_switch*I_testpulse(t) - ramp_switch*I_ramp(t)
                            - zap_switch*I_Zap(t))
                
                dmNaTdt     = (minf_NaT(V) - mNaT) /  mtau_NaT(V) 
                dhNaTdt     = (hinf_NaT(V) - hNaT) /  htau_NaT(V)
                dmNaPdt     = (minf_NaP(V) - mNaP) /  mtau_NaP(V)
                dndt        = (ninf_Ks(V)  - n)    /  ntau_Ks(V)
                dmKfdt      = (minf_Kf(V)  - mKf)  /  mtau_Kf(V)
                dhKf1dt     = (hinf1_Kf(V) - hKf1) /  htau_Kf(V)
                dhKf2dt     = (hinf2_Kf(V) - hKf2) /  116 
                
                dNaidt  = NaiSwitch * (-1/(F*volume)) * ( I_NaT(V, mNaT, hNaT, nai) 
                        + I_NaP(V, mNaP, nai)                      
                        + I_leak_NA(V, nai) 
                        + 3*pumpswitch*I_pump(nai))
                
                return np.array([dVdt, dmNaTdt, dhNaTdt, dmNaPdt, dndt, dmKfdt, dhKf1dt, dhKf2dt, dNaidt])
            
            ### adds up all the applied current injections
            Inj_injectors = inj_switch*I_inj(t_vec) 
            + TP_switch*I_testpulse(t_vec) 
            + ramp_switch*I_ramp(t_vec) 
            + zap_switch*I_Zap(t_vec)
                    
            #Integrate
            y1 = odeint(dALLdt, param0, t_vec, rtol=1e-10, h0=.05, hmax=100.0)
            
            # Arrays of dynamic variables at each time point
            Vcell = y1[:,0] 
            mNaT  = y1[:,1]
            hNaT  = y1[:,2]
            mNaP  = y1[:,3]
            n     = y1[:,4]
            mKf   = y1[:,5]
            hKf1  = y1[:,6]
            hKf2  = y1[:,7]
            Nai   = y1[:,8]
            
            # Create 'baseline' = the membrane potential (mV) at 50 ms pre-injection
            V_preinj = Vcell[(tHold - 50)*inv_dt] 
            
            # Baseline values for other dynamic variables
            NaConc_preinj = Nai[(tHold - 50)*inv_dt]
            dynENA_preinj = 25.694 * np.log(nao/(NaConc_preinj))
            pump_preinj   = Imaxpump / (1 + (np.exp((naih - (NaConc_preinj)) / nais)))
            
            # AHP Peak Amplitude (find the minimum after current injection)
            Vcell_post = Vcell[tPulseEnd*inv_dt:]
            vmin_i = min(Vcell_post) # variable to store Hyperpolarization trough value
            vmin.append(vmin_i)
            ahp_amp_i = vmin_i - V_preinj 
            ahp_amp.append(ahp_amp_i)
            
            # Find time of AHP peak in ms
            z = Vcell.tolist() # make Vcell into a list
            vmin_idx = z.index(vmin_i) # find the index of time of AHP trough
            vmin_t = vmin_idx*dt # vmin time in ms
            
            # time of half of AHP trough amplitude
            ahp_half_i = ahp_amp_i * 0.5 # half of ahp trough amplitude in mV
            v_ahp_half_i = vmin_i - ahp_half_i # membrane voltage at half AHP trough, in mV
            vmin_thalf_i = np.argmax(Vcell[vmin_idx:] >v_ahp_half_i) # idx of half AHP post trough
            vmin_thalf_idx_i = vmin_idx + vmin_thalf_i # idx of half trough for entire simulation
            ahp_halfdur_i = vmin_thalf_i*dt # time to reach half trough in ms
            ahp_halfdur.append(ahp_halfdur_i)
            
            # Making AHP 75 and AHP 25 
            ahp_25_i = ahp_amp_i * 0.25 # quarter of ahp trough amplitude in mV
            v_ahp_25_i = vmin_i - ahp_25_i # membrane voltage at quarter AHP trough, in mV
            vmin_t_25_i = np.argmax(Vcell[vmin_idx:] > v_ahp_25_i) # idx of quarter AHP post trough
            vmin_t_25_idx_i = vmin_idx + vmin_t_25_i # idx of quarter trough for entire simulation
            ahp_25dur_i = vmin_t_25_i*dt # latency to reach quarter trough in ms
            ahp_25dur.append(ahp_25dur_i)
            
            ahp_75_i = ahp_amp_i * 0.75 # three-quarters of ahp trough amplitude in mV
            v_ahp_75_i = vmin_i - ahp_75_i # membrane voltage at quarter AHP trough, in mV
            vmin_t_75_i = np.argmax(Vcell[vmin_idx:] > v_ahp_75_i) # idx of three-quarters AHP post trough
            vmin_t_75_idx_i = vmin_idx + vmin_t_75_i # idx of three-quarters trough for entire simulation
            ahp_75dur_i = vmin_t_75_i*dt # latency to reach three-quarters trough in ms
            ahp_75dur.append(ahp_75dur_i)

            # Find voltage Peaks
            x = Vcell
            peaks, _ = find_peaks(x, height=-25) # height = threshold in mV 
            # find_peaks function outputs an index
            spktimes = peaks*dt # turns index into time, in ms
            spk_idx = np.arange(1, len(spktimes), 1)

            # removing artifactual spikes (spikes before stimulus starts)
            realspktimes = [] 
            for h in range(0,len(spktimes)):
                if(spktimes[h]>=ZapStartTime):   # use only spikes after the stimulus injection starts
                    realspktimes.append(spktimes[h])   # put those into realspktimes array
                h = h+1
            realspktimes_array = np.array(realspktimes)
             
            # Calculate ISIs (Interspike Intervals) in ms
            isi = [] 
            for h in range(0,len(realspktimes)-1):
                isi.append((realspktimes[h+1]-realspktimes[h]))
                h = h+1
            
            # Removing last entry from realspktimes
            realspktimesminusone = [] 
            for h in range(0,len(realspktimes)-1):
                realspktimesminusone.append(realspktimes[h])
                h = h+1
            realspktimesminusone_array = np.array(realspktimesminusone)
            
            # Calculate IFR (Instantaneous Firing Rate), in Hz (Inverse of the ISI)
            realspktimes_s = realspktimes_array/1000 # from ms to s
            ifr = []
            mean_ifr_i = []
            if len(realspktimes) > 1:
                for j in range(0, len(realspktimes)-1):
                    ifr.append(1/(realspktimes_s[j+1]-realspktimes_s[j]))
                    mean_ifr_i = np.mean(ifr)
                    j = j+1
            else:
                mean_ifr_i = 0 
                ifr = [0]
            mean_ifrs.append(mean_ifr_i)   
            ifr_array = np.asarray(ifr)
            
            ### Ramp Current Array
            rampcurrent_list = I_ramp(t_vec).tolist() 
            instantRAMP = [] 
            for h in range(0, len(peaks)-1):
                instantRAMP.append(rampcurrent_list[peaks[h]])
                h = h+1
            instantRAMP_array = np.array(instantRAMP)
            
            ### Zap Current Array
            zapcurrent_list = I_Zap(t_vec).tolist() 
            instantZAP = [] 
            for h in range(0,len(peaks)-1):
                instantZAP.append(zapcurrent_list[peaks[h]])
                h = h+1
            instantZAP_array = np.array(instantZAP)
            
            # Calculate Delay to Spike, in ms
            delay_i = []
            if len(realspktimes) == 0:
                delay_i = 0 # if no spikes, set to zero
            else:
                delay_i = (realspktimes[0] - tHold) # calculates delay from injection start to first spike in ms
            delay.append(delay_i) 
         
            # Find Troughs
            min_idx = tHold*inv_dt # index of current injection start
            max_idx = tPulseEnd*inv_dt # index of current injection end
            cur_range = [min_idx, max_idx]
            
            trough_idx = signal.argrelextrema(Vcell, np.less) # find local voltage minima
            trough_idx = trough_idx[0]
            trough_ms  = trough_idx*dt # in ms
            trough_lim = np.clip(trough_idx, min_idx, max_idx) # limit trough time range
            trough_clp = trough_lim[~np.in1d(trough_lim, cur_range).reshape(trough_lim.shape)] 
            
            # Find average of cell voltage for F-V curve
            mean_trough_i = []
            if len(trough_clp) == 0:
                mean_trough_i = Vcell[(tPulseEnd-100)*inv_dt] # indexing Vcell 100 ms before Pulse end
            else:
                mean_trough_i = np.mean(Vcell[trough_clp]) # average troughs
            mean_Vcell.append(mean_trough_i)
    
            ### Adaptation Slope and Spiking Integrity
            
            # Making the First Group of 9 IFRs for Adapt Slope Average
            Fin_IFRGroup_list = ifr_array[-10:].tolist()
            n_g1 = 1
            del Fin_IFRGroup_list[-n_g1:]
            try:
                AVG_Fin = sum(Fin_IFRGroup_list) / len(Fin_IFRGroup_list)
            except ZeroDivisionError:
                    AVG_Fin = "N/A"
            ### Note: The Middle Element is Num. 6
            
            # Making the Second Group of 9 IFRs for Adapt Slope Average
            Second_IFRGroup_list = ifr_array[-19:].tolist()
            n_g2 = 10
            del Second_IFRGroup_list[-n_g2:]
            try:
                AVG_2nd = sum(Second_IFRGroup_list) / len(Second_IFRGroup_list)
            except ZeroDivisionError:
                AVG_2nd = "N/A" 
            ### The Middle Element is Num. 15
            
            try:
                lastIFRdiff = AVG_2nd - AVG_Fin
            except TypeError:
                lastIFRdiff = "N/A"
           
            try:
                lastIFR_Timediff = (realspktimes_s[-15] - realspktimes_s[-6]) 
            except (TypeError, IndexError):
                lastIFR_Timediff = "N/A"
           
            try:
                AdaptSlope = lastIFRdiff/lastIFR_Timediff
            except TypeError:
                AdaptSlope = "N/A"
           
            ### Spiking Integrity is the percentage of time that the model spikes during the step injection
            try:
                SpkINT = (((realspktimes_array[-1] - tHold) / tPulse) * 100)
            except IndexError:
                SpkINT = 0.0
               
            ### Pandas File Saving 
            ### Saving the Temporal Dynamics of Individual Traces
            ### COMMENTED OUT, Used only for Data Collection
            
            ### Voltage-Time Saved Data
            # DF_short = pd.DataFrame({'Time (ms)': t_vec, 
            #                     'Voltage (pA)': Vcell, 
            #                     'Injection': Inj_injectors})
            # DF_short.to_csv("LMR" + str(model_switchsettings) + "Voltage Data " + str(I_pulse) + "pA," + filetime.strftime("%H_%M_%S, %m-%d-%Y.csv"), 
            #           index = False)
            
            ### Test Pulses Saved Data / Short Temporal Data 
            # DF_TP = pd.DataFrame({'Time (ms)': t_vec, 
            #                     'Voltage (pA)': Vcell, 
            #                     'Injection': Inj_injectors,
            #                     '[Na]in Conc.': Nai,
            #                     'Pump Current': I_pump(Nai)})
            # DF_TP.to_csv("LaMouche " + str(model_switchsettings) + "," + str(I_pulse) + "pA Short Temporal Data " + filetime.strftime("%H_%M_%S, %m-%d-%Y.csv"), 
            #           index = False)
            ### Other Heading for Test Pulses
            # DF_TP.to_csv("LMU TP " + str(I_test) + "pA, " + str(tTestPDelayPost/1000) +" sec Delay Data " + filetime.strftime("%H_%M_%S, %m-%d-%Y.csv"), 
            #            index = False)
               
            ### Figure 2N Full Simulation Temporal Dynamics Saved Data
            # DFLong = pd.DataFrame({'Time (ms)': t_vec, 
            #                         'Voltage (mV)': Vcell, 
            #                         'Injection (pA)': Inj_injectors,
            #                         '[Na]in Conc.': Nai,
            #                         'Dyn E_Na':E_NaSwitch(Nai),
            #                         'Transient [Na]': I_NaT(Vcell, mNaT, hNaT, Nai),
            #                         'Persistent [Na]': I_NaP(Vcell, mNaP, Nai),
            #                         'Slow [K]': I_Ks(Vcell,n),
            #                         'Fast [K]': I_Kf(Vcell, mKf, hKf1, hKf2),
            #                         'Na+ Leak Current': I_leak_NA(Vcell, Nai),
            #                         'K+ Leak Current':I_leak_K(Vcell),
            #                         'Pump Current': I_pump(Nai)})
            ### Title 1 for Default
            # DFLong.to_csv("LMU Standard" + str(model_switchsettings) + str(I_pulse) + "pA," "Full Dynamics Data " + filetime.strftime("%H_%M_%S, %m-%d-%Y.csv"), 
            #               index = False)
            ### Title 2 for Zap
            # DFLong.to_csv("LMU ZAP" + str(model_switchsettings) + str(I_ZapMax) + "pA," "Full Dynamics Data " + filetime.strftime("%H_%M_%S, %m-%d-%Y.csv"), 
            #               index = False)
            ### Titile 3 for Figure 8
            #DFLong.to_csv("LMU Fig8,EX1" + str(model_switchsettings) + str(I_pulse) + "pA," "Full Dynamics Data " + filetime.strftime("%H_%M_%S, %m-%d-%Y.csv"), 
                          # index = False)
              
            
            # ### IFR and InstantINJ Data for Ramping Current
            # DF_fr = pd.DataFrame({'SpkTime (ms)': realspktimesminusone_array, 
            #                       'IFR (Hz)':ifr_array,
            #                       'InstantINJ (pA)': instantRAMP_array})
            # DF_fr.to_csv("LMR" + str(model_switchsettings) + "IFR, InstINJ Data " + str(tRampDur/1000) + "sec RampDur, " + filetime.strftime("%H_%M_%S, %m-%d-%Y.csv"), 
            #           index = False)
            
            
            # ### IFR and InstantINJ Data for Zap Current
            # DF_fr = pd.DataFrame({'SpkTime (ms)': realspktimesminusone_array, 
            #                       'IFR (Hz)':ifr_array,
            #                       'InstantINJ (pA)': instantZAP_array})
            # DF_fr.to_csv("LMU " + str(model_switchsettings) + "IFR ZAP" + str(I_ZapMax) + " Data " + filetime.strftime("%H_%M_%S, %m-%d-%Y.csv"), 
            #           index = False)
              
            
            # ### IFR Data for Normal Injection Current
            # DF_fr = pd.DataFrame({'SpkTime (ms)': realspktimesminusone_array, 
            #                           'IFR (Hz)':ifr_array,})
            # # DF_fr.to_csv("LMR Default" + str(model_switchsettings) + " IFR Data at" + str(I_pulse) + filetime.strftime("%H_%M_%S, %m-%d-%Y.csv"), 
            # #                index = False)
            # #Alternative Title for Figure 8
            # DF_fr.to_csv("LMU Fig8,EX1" + str(model_switchsettings) + " IFR Data at" + str(I_pulse) + filetime.strftime("%H_%M_%S, %m-%d-%Y.csv"), 
            #               index = False)
            
            ### Plotting Simulation Functions
            # Plot Voltage and stars the Peaks
            def plotPeaks():
                plt.figure(figsize=(12,9))
                plt.plot(x)
                plt.plot(peaks, x[peaks], "x")
                # plt.xlim((tHold-100)*inv_dt, (tPulseEnd+5000)*inv_dt) ## 100 ms before, 200 ms after current injection
                # plt.plot(np.zeros_like(x), "--", color="gray")
                plt.title('V vs T \n I_inj = %2.1f pA' %I_pulse)
                plt.xlabel('time in microseconds')
                plt.ylabel('mV')   
                plt.show()
            #plotPeaks() #Comment Out if you don't want to output the graphs
            
            # Plot the Voltage and stars troughs
            def plotTroughs():
                x = Vcell
                plt.figure(figsize=(12,9))
                plt.plot(x)
                plt.plot(trough_clp, x[trough_clp], "x")
                plt.title('V vs T \n I_inj = %2.1f pA' %I_pulse)
                plt.xlabel('time in microseconds')
                plt.ylabel('mV')  
                plt.show()
            #plotTroughs() #Comment Out if you don't want to output the graphs
            
            # Plot VM (Membrane Voltage) vs Time, Main Voltage Plotting Function
            def plotVCell():
                plt.figure(figsize=(12,9))
                plt.plot(t_vec, Vcell)
                plt.title('V_m vs Time \n I_inj = %2.2f pA' %I_pulse
                          + '\n IMAX = %2.2f' %Imaxpump
                         # + '\n ZAP = %2.4f' %I_ZapMax
                          + '\n NaiH = %2.4f' %naih + 'NaiS = %2.4f' %nais) ### Added extra labels
                plt.xlabel('T (ms)')
                plt.ylabel('mV')
                plt.ylim(-70, 10)
                #axes = plt.gca()  
                #axes.yaxis.grid()
                plt.show()
            plotVCell() #Comment Out if you don't want to output the graphs
            
            # Plot Current Injection (all the different current injectors)
            def plotInjCurrent():
                plt.figure(figsize=(12,9))
                plt.plot(t_vec,    inj_switch*I_inj(t_vec) 
                                 + TP_switch*I_testpulse(t_vec)
                                 + zap_switch*I_Zap(t_vec)
                                 + ramp_switch*I_ramp(t_vec) ) 
                plt.xlabel('time (ms)')
                plt.ylabel('Inj Current, pA')
                plt.title('Injection Current vs Time \n I_inj = %2.1f pA' %I_pulse
                          + '\n Imaxpump = %3.0f' %Imaxpump)
                #plt.ylim(0,50) 
                plt.show()
            #plotInjCurrent() #Comment Out if you don't want to output the graph
            
            # Plot Zap Frequency
            def plotZapFreq():
                plt.figure(figsize=(12,9))
                plt.plot(t_vec, ZapFreq(t_vec))
                plt.xlabel('time (ms)')
                plt.ylabel('Frequency (Hz)')
                plt.title('Zap Frequency vs Time \n I_inj = %2.1f pA' %I_pulse
                          + '\n Imaxpump = %3.0f' %Imaxpump)
                #plt.ylim(0,50) 
                plt.show()
            #plotZapFreq() #Comment Out if you don't want to output the graph
            
            # Plot Internal Sodium Concentration [Na]
            def plotNaint():
                plt.figure(figsize=(12,9))
                plt.plot(t_vec, Nai)
                plt.title('Internal [Na] vs Time \n '
                          +'I_inj = %2.1f pA \n' %I_pulse 
                              + 'Imaxpump = %3.0f' %Imaxpump)
                plt.xlabel('time (ms)')
                plt.ylabel(r'$[Na]_{int}$ in M')
                #plt.ylim(0.0245, 0.0265) #Commented out to get an auto-generated range
                plt.show()
            #plotNaint() #Comment Out if you don't want to output the graphs
        
            # Plot Pump Current
            def plotIpump_i():
                plt.figure(figsize=(12,9))
                plt.plot(t_vec, I_pump(Nai))
                plt.xlabel('time (ms)')
                plt.ylabel('Pump Current, pA')
                plt.title('Pump Current vs Time \n I_inj = %2.1f pA' %I_pulse
                          + '\n Imaxpump = %3.0f' %Imaxpump)
                # plt.ylim(0 , 1.1*Imaxpump) 
                plt.show()
            #plotIpump_i() #Comment Out if you don't want to output the graph
        
            # Plot Dynamic E_Na
            def plotDynENa():
                plt.figure(figsize=(12,9))
                plt.plot(t_vec, E_NaSwitch(Nai))
                plt.xlabel('time (ms)')
                plt.ylabel('Na Reversal Potential (mV)')
                plt.title('Dynamic Sodium Reversal Potential (E_Na) vs T')
                plt.show()
            #plotDynENa() #Comment Out if you don't want to output the graph
            
            # Plot Instantenous Frequency
            def plotISF():
                "Plot Instantenous Spike Frequnecy over the simulation time"
                plt.figure()
                plt.plot(realspktimesminusone_array, ifr_array, 'o-')
                plt.title('IFR Curve' + 'I_inj = %2.2f pA' %I_pulse               
                  + '\n Imaxpump = %3.0f' %Imaxpump               
                  + ', nais = %1.4f' %nais)
                plt.xlabel('time in ms')
                plt.ylabel('IFR (Hz)')
            #plotISF()
            
            # Plot Full Leak Current 
            def plot_fullI_leak():
                plt.figure(figsize=(12,9))
                plt.plot(t_vec, I_leak_NA(Vcell, Nai) + I_leak_K(Vcell)) #remember to insert all the called parameters!
                plt.xlabel('time (ms)')
                plt.ylabel('Leak Current, pA')
                plt.title('Full Leak Current vs Time \n I_inj = %2.1f pA' %I_pulse
                          + '\n Imaxpump = %3.0f' %Imaxpump)
                #plt.ylim(0,50) 
                plt.show()
            #plot_fullI_leak() #Comment Out if you don't want to output the graph
            
            # Plod Sodium [Na] Leak Current
            def plotI_leakNA():
                plt.figure(figsize=(12,9))
                plt.plot(t_vec, I_leak_NA(Vcell, Nai)) #remember to insert all the called parameters!
                plt.xlabel('time (ms)')
                plt.ylabel('Leak Current, pA')
                plt.title('Sodium Component Leak Current vs Time \n I_inj = %2.1f pA' %I_pulse
                          + '\n Imaxpump = %3.0f' %Imaxpump)
                #plt.ylim(0,50) 
                plt.show()
            #plotI_leakNA() #Comment Out if you don't want to output the graph
            # Plot Full Leak Current 

            def plotI_leakK():
                plt.figure(figsize=(12,9))
                plt.plot(t_vec, I_leak_K(Vcell)) #remember to insert all the called parameters!
                plt.xlabel('time (ms)')
                plt.ylabel('Leak Current, pA')
                plt.title('Potassium Component Leak Current vs Time \n I_inj = %2.1f pA' %I_pulse
                          + '\n Imaxpump = %3.0f' %Imaxpump)
                #plt.ylim(0,50) 
                plt.show()
            #plotI_leakK() #Comment Out if you don't want to output the graph
            
            # Plot Fast Potassium [Kf] Current
            def plotI_Kf():
                plt.figure(figsize=(12,9))
                plt.plot(t_vec, I_Kf(Vcell, mKf, hKf1, hKf2)) #remember to insert all the called parameters!
                plt.xlabel('time (ms)')
                plt.ylabel('Fast Potassium Current, pA')
                plt.title('Kf Current vs Time \n I_inj = %2.1f pA' %I_pulse
                          + '\n Imaxpump = %3.0f' %Imaxpump)
                #plt.ylim(0,50) 
                plt.show()
            #plotI_Kf() #Comment Out if you don't want to output the graph
            
            # Plot Slow Potassium [Ks] Current
            def plotI_Ks():
                plt.figure(figsize=(12,9))
                plt.plot(t_vec, I_Ks(Vcell,n)) #remember to insert all the called parameters!
                plt.xlabel('time (ms)')
                plt.ylabel('Slow Potassium Current, pA')
                plt.title('Ks Current vs Time \n I_inj = %2.1f pA' %I_pulse
                          + '\n Imaxpump = %3.0f' %Imaxpump)
                #plt.ylim(0,50) 
                plt.show()
            #plotI_Ks() #Comment Out if you don't want to output the graph
            
            # Plot Transient Sodium [NaT] Current
            def plotI_NaT():
                plt.figure(figsize=(12,9))
                plt.plot(t_vec, I_NaT(Vcell, mNaT, hNaT, Nai)) #remember to insert all the called parameters!
                plt.xlabel('time (ms)')
                plt.ylabel('Transient Sodium, pA')
                plt.title('NaT Current vs Time \n I_inj = %2.1f pA' %I_pulse
                          + '\n Imaxpump = %3.0f' %Imaxpump)
                #plt.ylim(0,50) 
                plt.show()
            #plotI_NaT() #Comment Out if you don't want to output the graph
            
            # Plot Persistent Sodium [NaP]
            def plotI_NaP():
                plt.figure(figsize=(12,9))
                plt.plot(t_vec, I_NaP(Vcell, mNaP, Nai)) #remember to insert all the called parameters!
                plt.xlabel('time (ms)')
                plt.ylabel('Persistent Sodium, pA')
                plt.title('NaP Current vs Time \n I_inj = %2.1f pA' %I_pulse
                          + '\n Imaxpump = %3.0f' %Imaxpump)
                #plt.ylim(0,50) 
                plt.show()
            #plotI_NaP() #Comment Out if you don't want to output the graph
            
        # Plot Frequency-Injection 
        def plotFI():
             "Plot initial, final and mean IFR per current injection"
             plt.figure()
             plt.plot(inj, mean_ifrs, 'o-b', label = 'mean ifr')
             plt.plot(inj, ifr[0]   , 'o-r', label = 'initial ifr')
             plt.plot(inj, ifr[-1]  , 'o-g', label = 'final ifr')
             plt.legend(loc='upper left')
             plt.title('FI Curve'              
                       + '\n Imaxpump = %3.0f' %Imaxpump               
                       + '\n nais = %1.4f' %nais
                       + '\n NaiH = %1.4f' %naih)
             plt.xlabel('pA Injection')
             plt.ylabel('IFR (Hz)')
        #plotFI()
        
        ### Sweeps
        #Imaxpump = Imaxpump + Imaxpump_step
        #nais = nais + nais_step
        naih = naih + naiH_step

#Imaxpump = Imaxpump_start
#nais = nais_start
naih = naih + naiH_start


# =============================================================================
# Finalization: Printing Lines for the Simulation
# =============================================================================
print("Your simulations are successfully complete!")
# print("Current Injections of the Simulation were" + inj + "pA")
# print("IMaxPump Values of the Simulation were" + Imaxpump_range + "pA")  
# print("NaiH Values of the Simulation were" + naiH_range + "M")  
# print("NaiS Values of the Simulation were" + nais_range + "M")  

# In[ ]:
# =============================================================================
# Plot: Functions of individual Current Steps
# =============================================================================   
""" 
Plots figures from the last [i] (instance) of the current injection range. 
Can manually call these functions by typing 'plot____()' in the IPython Console. 
"""
def plotPeakAHP():
    "AHP Peak Amplitude per current injection step"
    plt.figure()
    plt.plot(inj, ahp_amp, 'o-b')
    plt.title('AHP Peak Amplitude vs Current Injection'
              +'\n Imaxpump = %3.0f' %Imaxpump               
              + ', nais = %1.4f' %nais)
    plt.xlabel('pA injection')
    plt.ylabel('mV')
def plotHalfAHPTime():
    "Time it takes for AHP to return halfway to baseline, per current injection"
    plt.figure()
    plt.plot(inj, ahp_halfdur, 'o-b')
    plt.title('Half Duration of AHP vs Current Injection'               
              + '\n Imaxpump = %3.0f' %Imaxpump               
              + ', nais = %1.4f' %nais
              + ',\n naih = %1.4f' %naih)
    plt.xlabel('pA injection')
    plt.ylabel('ms') 
def plotDelay():
    "Delay to First Spike per current injection"
    plt.figure()
    plt.plot(inj, delay, 'or')
    plt.title('Delay to First Spike \n Isopotential, '               
              + '\n Imaxpump = %3.0f' %Imaxpump               
              + ', nais = %1.4f' %nais)
    plt.xlabel('I_inj(pA)')
    plt.ylabel('Delay (ms)')
def plotMeanVcell():
    "Average voltage at AP troughs per current injection"
    plt.figure()
    plt.plot(inj, mean_Vcell, 'oy')
    plt.title('Average Membrane Voltage'               
              + '\n Imaxpump = %3.0f' %Imaxpump 
              + ', nais = %1.4f' %nais)
    plt.xlabel('I_inj (pA)')
    plt.ylabel('Average Membrane Voltage (mV)')  
def plotAHPTroughIFR():
    "Plotting AHP Trough Amplitudes versus Mean Spiking Frequency"
    plt.figure()
    plt.plot(mean_ifrs, ahp_amp, 'o-b')
    plt.title('AHP Trough Amplitudes vs Mean Spiking Frequency'
              +'\n Imaxpump = %3.0f' %Imaxpump               
              + ', nais = %1.4f' %nais)
    plt.xlabel('Average Spike Frequency (Hz)')
    plt.ylabel('AHP Amplitudes (mV)')
def plotAHPTimeIFR():
    "Plotting AHP Half Duration versus Mean Spiking Frequency"
    plt.figure()
    plt.plot(mean_ifrs, ahp_halfdur, 'o-b')
    plt.title('AHP Half Duration vs. Mean Spiking Frequency'
              +'\n Imaxpump = %3.0f' %Imaxpump               
              + ', nais = %1.4f' %nais)
    plt.xlabel('Average Spike Frequency (Hz)')
    plt.ylabel('AHP Half Duration Time in ms')

# =============================================================================
# Plot: Functions relating to most recent simulation over time
# =============================================================================   
def plotIpump():
    "Pump Current vs time of the final i within current injection range"
    plt.figure()
    plt.plot(t_vec, I_pump(Nai))
    plt.xlabel('time (ms)')
    plt.ylabel('Pump Current, pA')
    plt.title('Pump Current vs T \n I_inj = %2.1f pA' %I_pulse)   
def plotVcell():
    "Membrane Voltage of most recent simulation"    
    plt.figure(figsize=(20,10))
    plt.plot(t_vec, Vcell)
    plt.title('V vs T \n I_inj = %2.1f pA' %I_pulse)
    plt.xlabel('T (ms)')
    plt.ylabel('mV')
    plt.xlim(9900, 20000)
    axes = plt.gca()
    axes.yaxis.grid()
def plotFV():
    "Frequency-Voltage Curve"
    plt.figure()
    plt.plot(mean_Vcell, mean_ifrs, 'ok')
    plt.title('F-V Curve'               
              + '\n Imaxpump = %3.0f' %Imaxpump 
              + ', nais = %1.4f' %nais)
    plt.xlabel('Average Membrane Voltage (mV)')
    plt.ylabel('Average IFR (Hz)')
