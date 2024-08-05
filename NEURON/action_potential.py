#Simulate and AP
import matplotlib.pyplot as plt
from neuron import h
from neuron.units import ms, mV
h.load_file("stdrun.hoc")
axon = h.Section(name='axon')
h.hh.insert(axon)
ic = h.IClamp(axon(0))
ic.delay = 2 #ms
ic.dur = 0.1 #ms
ic.amp = 1000 #nA

t= h.Vector().record(h._ref_t) #_ref_t or v is to record when it changes
v= h.Vector().record(axon(0.5)._ref_v)

h.finitialize(-65 *mV)
h.continuerun(25 *ms)

plt.plot(t,v)
plt.show()