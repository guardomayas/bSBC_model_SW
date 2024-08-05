# imports
import os
# os.chdir('C:/Users/Documents/Schwartz/NEURON/bSbC Paper Model/IndiraModel2/')
import csv
import numpy
import scipy.io

from neuron import h
from neuron.units import ms, mV
from neuron import gui
h.load_file('stdrun.hoc')

import matplotlib.pyplot as plt

from HybridCell import HybridCell

temp = 32
h.celsius = temp

vLeak = -60
v_init = -61.5
NaVRatio = [0.4, 0]
NaDensity =  [0.003452431, 0.002592035] 
KDensity = 0.003873771
SomaDiam = 17.5
DendLen = 508
HillLen = 24
AISLen = [22, 16]
Multiplier = 30
Factor = 0.4

h.dt = 1/10 #1/ms


OFFsACell = HybridCell(vLeak, NaVRatio[0], Multiplier, NaDensity[0], KDensity, SomaDiam, DendLen, HillLen, AISLen[0])
OFFsACell_bSbCparam = HybridCell(vLeak, NaVRatio[1], Multiplier, NaDensity[0], KDensity, SomaDiam, DendLen, HillLen, AISLen[0])
bSbCCell_OFFsAparam = HybridCell(vLeak, NaVRatio[0], Multiplier, NaDensity[1], KDensity, SomaDiam, DendLen, HillLen, AISLen[1])
bSbCCell = HybridCell(vLeak, NaVRatio[1], Multiplier, NaDensity[1], KDensity, SomaDiam, DendLen, HillLen, AISLen[1])

# measure IR
ImpA = h.Impedance()
ImpA.loc(0.5, sec=OFFsACell.soma)
ImpA.compute(0)
IR_val_A = ImpA.input(0.5, sec=OFFsACell.soma)
print(IR_val_A)

ImpA_B = h.Impedance()
ImpA_B.loc(0.5, sec=OFFsACell_bSbCparam.soma)
ImpA_B.compute(0)
IR_val_A_B = ImpA_B.input(0.5, sec=OFFsACell_bSbCparam.soma)
print(IR_val_A_B)

ImpB_A = h.Impedance()
ImpB_A.loc(0.5, sec=bSbCCell_OFFsAparam.soma)
ImpB_A.compute(0)
IR_val_B_A = ImpB_A.input(0.5, sec=bSbCCell_OFFsAparam.soma)
print(IR_val_B_A)

ImpB = h.Impedance()
ImpB.loc(0.5, sec=bSbCCell.soma)
ImpB.compute(0)
IR_val_B = ImpB.input(0.5, sec=bSbCCell.soma)
print(IR_val_B)


# load mat files of conductances
# filepath = 'C:/Users/Documents/Schwartz/analysis/DynamicClampConductances/'
filepath = "Z:/Rig_Related/Dynamic Clamp/Conductances/SRW_bSbCproject"
Alpha_n100_exc = scipy.io.loadmat(filepath+'Sophia_Alpha_cm100_Exc'+'.mat')
Alpha_n100_inh = scipy.io.loadmat(filepath+'Sophia_Alpha_cm100_Inh'+'.mat')
#bSbC_n100_exc = scipy.io.loadmat(filepath+'Sophia_Bursty_cm100_Exc'+'.mat')
#bSbC_n100_inh = scipy.io.loadmat(filepath+'Sophia_Bursty_cm100_Inh'+'.mat')

A_Exc = h.Vector(1000/Alpha_n100_exc['conductances'][0]) * Factor # needs to be in MOhms
A_Inh = h.Vector(1000/Alpha_n100_inh['conductances'][0]) * Factor
#B_Exc = h.Vector(1000/bSbC_n100_exc['conductances'][0]) * Factor
#B_Inh = h.Vector(1000/bSbC_n100_inh['conductances'][0]) * Factor

# add in the conductances measured for dynamic clamp and input them
OFFsA_E = h.SEClamp(OFFsACell.soma(0.4))
OFFsA_E.dur1 = 1e9
OFFsA_E.amp1 = 0 # set to reversal potential
OFFsA_I = h.SEClamp(OFFsACell.soma(0.6))
OFFsA_I.dur1 = 1e9
OFFsA_I.amp1 = -70

OFFsA_B_E = h.SEClamp(OFFsACell_bSbCparam.soma(0.4))
OFFsA_B_E.dur1 = 1e9
OFFsA_B_E.amp1 = 0 # set to reversal potential
OFFsA_B_I = h.SEClamp(OFFsACell_bSbCparam.soma(0.6))
OFFsA_B_I.dur1 = 1e9
OFFsA_B_I.amp1 = -70

bSbC_A_E = h.SEClamp(bSbCCell_OFFsAparam.soma(0.4))
bSbC_A_E.dur1 = 1e9
bSbC_A_E.amp1 = 0 # set to reversal potential
bSbC_A_I = h.SEClamp(bSbCCell_OFFsAparam.soma(0.6))
bSbC_A_I.dur1 = 1e9
bSbC_A_I.amp1 = -70

bSbC_E = h.SEClamp(bSbCCell.soma(0.4))
bSbC_E.dur1 = 1e9
bSbC_E.amp1 = 0 # set to reversal potential
bSbC_I = h.SEClamp(bSbCCell.soma(0.6))
bSbC_I.dur1 = 1e9
bSbC_I.amp1 = -70

# run conductances
A_Exc.play(OFFsA_E._ref_rs, h.dt)
A_Inh.play(OFFsA_I._ref_rs, h.dt)
A_Exc.play(OFFsA_B_E._ref_rs, h.dt)
A_Inh.play(OFFsA_B_I._ref_rs, h.dt)
A_Exc.play(bSbC_A_E._ref_rs, h.dt)
A_Inh.play(bSbC_A_I._ref_rs, h.dt)
A_Exc.play(bSbC_E._ref_rs, h.dt)
A_Inh.play(bSbC_I._ref_rs, h.dt)

TimeVec = numpy.linspace(0, 2.5, num=25001, endpoint=True) # sample rate 10000
SomaV_A = h.Vector().record(OFFsACell.soma(0.5)._ref_v)
SomaV_A_B = h.Vector().record(OFFsACell_bSbCparam.soma(0.5)._ref_v)
SomaV_B_A = h.Vector().record(bSbCCell_OFFsAparam.soma(0.5)._ref_v)
SomaV_B = h.Vector().record(bSbCCell.soma(0.5)._ref_v)
h.finitialize(-60 * mV)
h.continuerun(2500 * ms)

plt.figure(1)
plt.plot(TimeVec, list(SomaV_A))
plt.xlabel('t (s)')
plt.ylabel('v (mV)')
plt.title('OFFsA, OFFsA params')
plt.figure(2)
plt.plot(TimeVec, list(SomaV_A_B))
plt.xlabel('t (s)')
plt.ylabel('v (mV)')
plt.title('OFFsA, bSbC params')
plt.figure(3)
plt.plot(TimeVec, list(SomaV_B_A))
plt.xlabel('t (s)')
plt.ylabel('v (mV)')
plt.title('bSbC, OFFsA params')
plt.figure(4)
plt.plot(TimeVec, list(SomaV_B))
plt.xlabel('t (s)')
plt.ylabel('v (mV)')
plt.title('bSbC, bSbC params')


scipy.io.savemat('Voltages_ContrastResponse_NaVRatio.mat', dict(OFFsA = numpy.array(SomaV_A), bSbC = numpy.array(SomaV_B), OFFsA_bSbCparam = numpy.array(SomaV_A_B), bSbC_OFFsAparam = numpy.array(SomaV_B_A), TimeVec = numpy.array(TimeVec)))
print('done')
