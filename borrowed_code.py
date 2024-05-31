# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:43:29 2020

@author: Stefano rini
"""

import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import LorentzianModel, QuadraticModel

test = np.loadtxt('spectra.txt')

xdat = test[0, :]
ydat = test[1, :]

def add_peak(prefix, center, amplitude=0.005, sigma=0.05):
    peak = LorentzianModel(prefix=prefix)
    pars = peak.make_params()
    pars[f'{prefix}center'].set(center)
    pars[f'{prefix}amplitude'].set(amplitude)
    pars[f'{prefix}sigma'].set(sigma, min=0)
    return peak, pars

model = QuadraticModel(prefix='bkg_')
params = model.make_params(a=0, b=0, c=0)

rough_peak_positions = (0.61, 0.76, 0.85, 0.99, 1.10, 1.40, 1.54, 1.7)
for i, cen in enumerate(rough_peak_positions):
    peak, pars = add_peak('lz%d_' % (i+1), cen)
    model = model + peak
    params.update(pars)

init = model.eval(params, x=xdat)
result = model.fit(ydat, params, x=xdat)
comps = result.eval_components()

print(result.fit_report(min_correl=0.5))

plt.plot(xdat, ydat, label='data')
plt.plot(xdat, result.best_fit, label='best fit')
for name, comp in comps.items():
    plt.plot(xdat, comp, '--', label=name)
plt.legend(loc='upper right')
plt.show()