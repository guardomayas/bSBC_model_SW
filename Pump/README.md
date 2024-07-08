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