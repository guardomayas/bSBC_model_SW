# Na/K ATPase Models

## Ground truth: some kind of post albers 6 step reaction.
1. Lots of parameters.
2. Well explained in mathematical physiology  book. 

References: 
1. This is the paper the book cites. Development of models of active ion transport for whole-cell modelling: cardiac sodium–potassium pump as a case study: https://www.sciencedirect.com/science/article/pii/S0079610704000215?fr=RR-2&ref=pdf_download&rr=88a2c24b9bb822fc 

2. Importance of the Voltage Dependence of Cardiac Na/K ATPase Isozymes

## Sodium dependent model

1. Temporal dynamics of na/k pump mediated memory traces: insights from conductance-based models of drosophila neurons
2. https://elifesciences.org/articles/19322#equ4

$I_{\text{pump}} = \frac{I_{\text{pump}}^{\text{max}}}{1 + \exp\left(\frac{[\text{Na}]_{\text{ih}} - [\text{Na}]_{\text{i}}}{[\text{Na}]_{\text{is}}}\right)}$

Derivation: https://www.frontiersin.org/articles/10.3389/fncom.2017.00085/full


## Sodium and potassium dependent model

1. Ion concentration dynamics as a mechanism for neuronal bursting: https://link.springer.com/article/10.1007/s10867-010-9212-6
 
 ${\tilde I}_{\rm pump}=\rho {\left( {1 +     \exp \left( {\frac{{25 - {{\rm [Na]}_i}}}{3}} \right)} \right)^{-1}} \left( {\frac{1}{{1 + \exp \left( {5.5 - {{\rm [K]}_o}} \right)}}} \right)$


2. The influence of sodium and potassium dynamics on excitability, seizures, and the stability of persistent states: I. Single neuron dynamics https://link.springer.com/article/10.1007/s10827-008-0132-4

3. A mathematical model of recurrent spreading depolarizations https://link.springer.com/article/10.1007/s10827-017-0675-3


$I_{P} = \rho_{N} \left( \frac{[\mathrm{K}^{+}]_{e}}{2 + [\mathrm{K}^{+}]_{e}}\right)^{2}\left( \frac{[\text{Na}^{+}]_{i}}{7.7 + [\text{Na}^{+}]_{i}}\right)^{3}$

## There is also a Voltage dependent model
The main advantage would be not having the need to model Na concentration. Howeverr, I'm concerned about peformarce

- Clarifying the composition of the ATP consumption factors required for maintaining ion homeostasis in mouse rod photoreceptors uses that model
    - Inspired by Gabriel's paper (https://rupress.org/jgp/article-standard/153/2/e202012687/211728/Loss-of-the-K-channel-Kv2-1-greatly-reduces). 

# More functional papers

- Spike integration and cellular memory in a rhythmic network from Na+/K+ pump current dynamics. https://www.nature.com/articles/nn.2444

- Review: Sodium pump regulation of locomotor control circuits https://journals.physiology.org/doi/full/10.1152/jn.00066.2017 

- Biophysics of Adaptation in a Computational Model of the Leech T Neuron. https://web.archive.org/web/20190223220507id_/http://pdfs.semanticscholar.org/4c24/2d9111fb990f28d2d1dd84f4178971ca486a.pdf 

- Sodium pumps adapt spike bursting to stimulus statistics https://www.nature.com/articles/nn1982#Sec9

- Intrinsic Mechanisms for Adaptive Gain Rescaling in Barrel Cortex. https://www.jneurosci.org/content/28/3/696

- Bursting Dynamics Based on the Persistent Na1 and Na1/K1 Pump Currents: A Dynamic Clamp
Approach

# More computational

- Nonlinear Interaction between Shunting and Adaptation Controls a Switch between Integration and Coincidence Detection in Pyramidal Neurons. https://www.jneurosci.org/content/26/36/9084.short

- Diversity of Gain Modulation by Noise in Neocortical Neurons: Regulation by the Slow Afterhyperpolarization Conductance. https://www.jneurosci.org/content/26/34/8787 

- Intrinsic Gain Modulation and Adaptive Neural Coding. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000119#pcbi.1000119-Higgs1

