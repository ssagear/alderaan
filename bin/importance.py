import numpy as np
import matplotlib.pyplot as plt
import astropy

from astropy.table import Table
from astropy.io import fits
from corner import corner


import sys
sys.path.append('../')
from alderaan.Results import *
import sys


koi_cumul = Table.read('/Users/ssagear/UFL Dropbox/Sheila Sagear/Research/github/alderaan/bin/Catalogs/cumulative_2025.05.30_14.09.29.csv', format='csv', comment='#')
berger = Table.read('/Users/ssagear/UFL Dropbox/Sheila Sagear/Research/github/alderaan/bin/Catalogs/gaia_kepler_berger_2020_tab2_output.mrt', format='mrt')

nkoi = sys.argv[1]
plno = int(sys.argv[2])

figure_direct = '/Users/ssagear/UFL Dropbox/Sheila Sagear/Research/github/alderaan/bin/Figures/06_09_25/' + str(nkoi) + '/' + str(nkoi) + '_pl' + str(plno) + '-'
results_direct = '/Users/ssagear/UFL Dropbox/Sheila Sagear/Research/github/alderaan/bin/Results/06_09_25/' + str(nkoi) + '/' + str(nkoi) + '_pl' + str(plno) + '-'

kep_tab_system = koi_cumul[koi_cumul['kepoi_sys_name'] == nkoi]
kep_tab_system.sort('koi_period')
kepid = kep_tab_system['kepid'].value[0]


kep_tab_planet = kep_tab_system[plno]
koin = kep_tab_planet['kepoi_name']
print(koin)

def koi_to_kepid(nkoi):

    koi_cumul = Table.read('/Users/ssagear/UFL Dropbox/Sheila Sagear/Research/github/alderaan/bin/Catalogs/cumulative_2025.05.30_14.09.29.csv', format='csv', comment='#')

    kep_tab_system = koi_cumul[koi_cumul['kepoi_sys_name'] == nkoi]
    kepid = kep_tab_system['kepid'].value[0]

    return kepid

res = Results(target=nkoi, data_dir='/Users/ssagear/UFL Dropbox/Sheila Sagear/Research/github/alderaan/bin/Results/06_09_25/',)


plt.clf()
res.plot_lightcurve()
plt.savefig(figure_direct + 'lc_pl' + str(plno) + '.png')

plt.clf()
res.plot_folded(plno)
plt.savefig(figure_direct + 'folded_pl' + str(plno) + '.png')

res.summary()
np.savetxt(results_direct + 'summary.txt', res.summary(), fmt='%s')

res.samples(plno)
np.savetxt(results_direct + 'samples_pl' + str(plno) + '.txt', res.samples(plno), fmt='%s')

fs = np.vstack((res.samples(plno)['PERIOD'], res.samples(plno)['T0'], res.samples(plno)['ROR'], res.samples(plno)['IMPACT'], res.samples(plno)['DUR14'], res.samples(plno)['LD_U1'], res.samples(plno)['LD_U2'])).T


per_range = np.percentile(res.samples(plno)['PERIOD'], [1,99])
t0_range = np.percentile(res.samples(plno)['T0'], [1,99])
ror_range = np.percentile(res.samples(plno)['ROR'], [1,98])
impact_range = np.percentile(res.samples(plno)['IMPACT'], [1,99])
dur_range = np.percentile(res.samples(plno)['DUR14'], [1,99])
ld1_range = [0, 1]
ld2_range = [0, 1]


truths = [kep_tab_planet['koi_period'], kep_tab_planet['koi_time0bk'], kep_tab_planet['koi_ror'], \
          kep_tab_planet['koi_impact'], kep_tab_planet['koi_duration']/24, \
            kep_tab_planet['koi_ldm_coeff1'], kep_tab_planet['koi_ldm_coeff2']]



range = [per_range,t0_range, ror_range, impact_range, dur_range, ld1_range, ld2_range]

plt.clf()
corner(fs, labels=['per', 't0', 'ror', 'impact', 'dur', 'ld1', 'ld2'], show_titles=True, plot_contours=True, range=range, truths=truths, truth_color='red');
plt.savefig(figure_direct + 'corner_pl' + str(plno) + '.png')
plt.close()


mstar_boot = np.random.normal(loc=berger[berger['KIC'] == kepid]['Mstar'], scale=berger[berger['KIC'] == kepid]['E_Mstar'], size=1000)
rstar_boot = np.random.normal(loc=berger[berger['KIC'] == kepid]['Rstar'], scale=berger[berger['KIC'] == kepid]['E_Rstar'], size=1000)
rho_mann = mstar_boot/rstar_boot**3
rho_mean, rho_std = np.mean(rho_mann), np.std(rho_mann)
print(rho_mean, rho_std)



# Do the e-w weight fitting.
def calc_rhostar_samp(P, e, w, T14, RpRs, b):

    import scipy.constants as c

    # Period to seconds
    Ps = P*86400
    # Duration to seconds
    T14s = T14*86400
    # Omega must be in radians

    term1 = (3*np.pi) / (c.G * Ps**2)

    numer = (1 + RpRs)**2 - b**2

    subnumer = T14s*np.pi * (1 + e*np.sin(w))
    subdenom = Ps * np.sqrt(1-e**2)
    denom = np.sin(subnumer/subdenom)**2

    term2 = ( (numer / denom) + b**2 )**(3/2.)

    # Returns rho star in solar density
    return (term1 * term2) / 1408

print('Importance Sampling...')

period = res.samples(0)['PERIOD']
T14 = res.samples(0)['DUR14']
rprs = res.samples(0)['ROR']
b = res.samples(0)['IMPACT']


test_es = np.random.uniform(0, 0.99, size=len(T14))
test_ws = np.random.uniform(-np.pi/2, 3*(np.pi/2), size=len(T14))

rho_star_samp = calc_rhostar_samp(period, test_es, test_ws, T14, rprs, b)


def log_like_rho(rho_samp, rho_true, sigma_rho_true):
    return -0.5 * ( (rho_samp - rho_true) / sigma_rho_true )**2


kic_rho_star = [rho_mean, rho_std]
log_likes = log_like_rho(rho_star_samp, kic_rho_star[0], kic_rho_star[1])


total_like = np.sum(np.exp(log_likes))
weights = np.exp(log_likes) / total_like

np.savetxt(results_direct + 'es.txt', test_es)
np.savetxt(results_direct + 'ws.txt', test_ws)
np.savetxt(results_direct + 'weights.txt', weights)
np.savetxt(results_direct + 'kic_rho_star.txt', kic_rho_star)
np.savetxt(results_direct + 'rho_star_samp.txt', rho_star_samp)
np.savetxt(results_direct + 'log_likes.txt', log_likes)

plt.clf()
plt.figure(figsize=(5, 5))
plt.hist2d(test_ws, test_es, bins=[np.arange(-np.pi/2, 3*np.pi/2, 0.4), np.arange(0,0.92,0.08)], weights=weights);
plt.xlabel('w')
plt.ylabel('e')
plt.title(koin)
plt.savefig(figure_direct + 'ew.png')
plt.close()


plt.clf()
plt.figure(figsize=(5, 4))
plt.hist(test_es, weights=weights);
plt.xlabel('e')
plt.title(koin)
plt.savefig(figure_direct + 'ehist.png')
plt.close()