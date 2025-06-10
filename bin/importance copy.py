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


koi_cumul = Table.read('/Users/ssagear/UFL Dropbox/Sheila Sagear/Research/github/alderaan-fork/alderaan/bin/Catalogs/cumulative_2025.05.30_14.09.29.csv', format='csv', comment='#')
berger = Table.read('/Users/ssagear/UFL Dropbox/Sheila Sagear/Research/github/alderaan-fork/alderaan/bin/Catalogs/gaia_kepler_berger_2020_tab2_output.mrt', format='mrt')

nkoi = sys.argv[1]
plno = int(sys.argv[2])

figure_direct = '/Users/ssagear/UFL Dropbox/Sheila Sagear/Research/github/alderaan-fork/alderaan/bin/Figures/06_01_25/' + str(nkoi) + '/' + str(nkoi) + '_pl' + str(plno) + '-'
results_direct = '/Users/ssagear/UFL Dropbox/Sheila Sagear/Research/github/alderaan-fork/alderaan/bin/Results/06_01_25/' + str(nkoi) + '/' + str(nkoi) + '_pl' + str(plno) + '-'

kep_tab_system = koi_cumul[koi_cumul['kepoi_sys_name'] == nkoi]
kep_tab_system.sort('koi_period')
kepid = kep_tab_system['kepid'].value[0]


kep_tab_planet = kep_tab_system[plno]
koin = kep_tab_planet['kepoi_name']
print(koin)

def koi_to_kepid(nkoi):

    koi_cumul = Table.read('/Users/ssagear/UFL Dropbox/Sheila Sagear/Research/github/alderaan-fork/alderaan/bin/Catalogs/cumulative_2025.05.30_14.09.29.csv', format='csv', comment='#')

    kep_tab_system = koi_cumul[koi_cumul['kepoi_sys_name'] == nkoi]
    kepid = kep_tab_system['kepid'].value[0]

    return kepid

res = Results(target=nkoi, data_dir='/Users/ssagear/UFL Dropbox/Sheila Sagear/Research/github/alderaan-fork/alderaan/bin/Results/06_01_25/',)


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
plt.savefig(figure_direct + 'raw_corner_pl' + str(plno) + '.png')
plt.close()


mstar_boot = np.random.normal(loc=berger[berger['KIC'] == kepid]['Mstar'], scale=berger[berger['KIC'] == kepid]['E_Mstar'], size=1000)
rstar_boot = np.random.normal(loc=berger[berger['KIC'] == kepid]['Rstar'], scale=berger[berger['KIC'] == kepid]['E_Rstar'], size=1000)
rho_mann = mstar_boot/rstar_boot**3
rho_mean, rho_std = np.mean(rho_mann), np.std(rho_mann)
print(rho_mean, rho_std)


import pandas as pd
from   scipy import stats
from   scipy.interpolate import interp1d, RectBivariateSpline
import astropy.constants as apc


pi = np.pi
BIGG = 6.6743 * 10**(-8)   # Newton's constant; cm^3 / (g * s^2)

__all__ = ['calc_rho_star',
           'get_e_omega_obs_priors',
           'imp_sample_rhostar'
          ]


def calc_aRs(P, rho):
    """
    P : period [days]
    rho : stellar density [g/cm3]
    """
    P_   = P*86400.       # [seconds]
    rho_ = rho*1000.      # [kg/m3]
    G    = apc.G.value    # Newton's constant

    return ((G*P_**2*rho_)/(3*pi))**(1./3)
    

def calc_rho_star(P, T14, b, ror, ecc, omega):
    '''
    Inverting T14 equation from Winn 2010 
    
    Args:
        P: period in units of days
        T14: duration in units of days
        b: impact parameter
        ror: radius ratio
        ecc: eccentricity
        omega: argument of periastron in radians
    Out:
        rho_star: stellar density in units of g/cc
    '''
    per = P * 86400.
    dur = T14 * 86400.

    con = (3*pi) / (BIGG * per**2)
    num = (1+ror)**2 - b**2
    arg = (pi*dur/per) * (1+ecc*np.sin(omega)) / np.sqrt(1-ecc**2)
    den = np.sin(arg)**2
    
    return con * (num/den + b**2) ** 1.5


def get_e_omega_obs_priors(N, ecut):
    '''
    Get N random draws of ecc [0, ecut] and omega [-pi/2, 3pi/2],
    using the transit observability prior 
    (see: https://github.com/gjgilbert/notes/blob/main/calculate_e-omega_grid.ipynb)
    '''
    ngrid = 101
    ndraw = int(N)

    e_uni = np.linspace(0,ecut,ngrid)
    z_uni = np.linspace(0,1,ngrid)

    omega_grid = np.zeros((ngrid,ngrid))

    for i, e_ in enumerate(e_uni):
        x = np.linspace(-0.5*pi, 1.5*pi, int(1e4))
        y = (1 + e_*np.sin(x))/(2*pi)

        cdf = np.cumsum(y)
        cdf -= cdf.min()
        cdf = cdf/cdf.max()
        inv_cdf = interp1d(cdf, x)

        omega_grid[i] = inv_cdf(z_uni)

    RBS = RectBivariateSpline(e_uni, z_uni, omega_grid)

    e_draw = np.random.uniform(0, ecut, ndraw)
    z_draw = np.random.uniform(0, 1, ndraw)
    w_draw = RBS.ev(e_draw, z_draw)
    
    return e_draw, w_draw


def imp_sample_rhostar(period, dur, rprs, impact, rho_star, norm=True, return_log=False, ecut=None, ew_obs_prior=False, distr='uniform', params=[], upsample=1):
    '''
    Perform standard importance sampling from {IMPACT, ROR, PERIOD, DUR14} --> {ECC, OMEGA}
    
    Args
    ----
    samples [dataframe]: pandas dataframe of sampled data which includes: IMPACT, ROR, PERIOD, DUR14
    rho_star [tuple]: values of the true stellar density and its uncertainty
    norm [bool]: True to normalize weights before output (default=True)
    return_log [bool]: True to return ln(weights) instead of weights (default=False)
    ecut [float]: upper bound on the ecc prior between (0,1); default None will set to a/Rs * (1-e) > 1
    ew_obs_prior [bool]: bool flag indicating whether or not to use the ecc-omega transit obs prior (default False)
    distr [str]: name of the distribution shape to sample ECC from; defaults to uniform
    params [list]: list of values to be used as parameters for the indicated distribution
    upsample [int]: integer factor to increase the number of samples (default = 1)
    
    Output:
    weights [array]: importance sampling weights
    data [dataframe]: pandas dataframe containing all input and derived data, including: 
                      ECC: random values drawn from 0 to 'ecut' according to 'distr' and 'params'
                      OMEGA: random values drawn from -pi/2 to 3pi/2 (with transit obs prior if 'ew_obs_prior'=True)
                      IMPACT: inputs values
                      ROR: inputs values
                      PERIOD: inputs values
                      DUR14: inputs values
                      RHOSTAR: derived values
                      WEIGHTS (or LN_WT): importance weights
    '''
    P   = np.repeat(period, upsample)
    T14 = np.repeat(dur, upsample)    
    ror = np.repeat(rprs, upsample)
    b   = np.repeat(impact, upsample)
    
    N = len(b)

    if ecut is None:
        ecut = 1 - 1/np.mean(calc_aRs(P, rho_star[0]))

    if ew_obs_prior == True:
        ecc, omega = get_e_omega_obs_priors(N, ecut)

    else:
        if distr == 'uniform':
            ecc = np.random.uniform(0., ecut, N)

        elif distr == 'rayleigh':
            sigma = params[0]
            ecc = np.random.rayleigh(sigma, size=N)
            while np.any(ecc >= ecut):
                ecc[ecc >= ecut] = np.random.rayleigh(sigma, size=np.sum(ecc >= ecut))

        elif distr == 'beta':
            alpha_mu, beta_mu = params
            ecc = np.random.beta(alpha_mu, beta_mu, size=N)
            while np.any(ecc >= ecut):
                ecc[ecc >= ecut] = np.random.beta(alpha_mu, beta_mu, size=np.sum(ecc >= ecut))

        elif distr == 'half-gaussian':
            sigma = params[0]
            ecc = np.random.normal(loc=0, scale=sigma, size=N)
            while np.any((ecc >= ecut)|(ecc < 0)):
                ecc[(ecc >= ecut)|(ecc < 0)] = np.random.normal(loc=0, scale=sigma, size=np.sum((ecc >= ecut)|(ecc < 0)))

        omega = np.random.uniform(-0.5*np.pi, 1.5*np.pi, N)
        
        
    rho_samp = calc_rho_star(P, T14, b, ror, ecc, omega)
    log_weights = -np.log(rho_star[1]) - 0.5*np.log(2*pi) - 0.5 * ((rho_samp - rho_star[0]) / rho_star[1]) ** 2
    
    # flag weights that are NaN-valued or below machine precision
    bad = np.isnan(log_weights) + (log_weights < np.log(np.finfo(float).eps))

    print(np.sum(bad))

    if np.sum(bad)/len(bad) < 0.05:
        raise ValueError("Fraction of viable samples is below 5%")
    
    # prepare outputs
    data = pd.DataFrame()
    data['PERIOD']  = P[~bad]
    data['ROR']     = ror[~bad]
    data['IMPACT']  = b[~bad]
    data['DUR14']   = T14[~bad]
    data['ECC']     = ecc[~bad]
    data['OMEGA']   = omega[~bad]
    data['RHOSTAR'] = rho_samp[~bad]

    if return_log:       
        data['LN_WT'] = log_weights[~bad]
        return log_weights, data

    else:
        weights = np.exp(log_weights[~bad] - np.max(log_weights[~bad]))
        
        if norm:
            weights /= np.sum(weights)
        data['WEIGHTS'] = weights
        return weights, data
    

period = res.samples(plno)['PERIOD']
rprs = res.samples(plno)['ROR']
impact = res.samples(plno)['IMPACT']
duration = res.samples(plno)['DUR14']

rho_star = (rho_mean, rho_std)

w, d = imp_sample_rhostar(period, duration, rprs, impact, rho_star)

fs = np.vstack((d['PERIOD'], d['ROR'], d['IMPACT'], d['DUR14'], d['OMEGA'], d['ECC'])).T

per_range = np.percentile(d['PERIOD'], [1,99])
ror_range = np.percentile(d['ROR'], [1,98])
impact_range = np.percentile(d['IMPACT'], [1,99])
dur_range = np.percentile(d['DUR14'], [1,99])
ecc_range = np.percentile(d['ECC'], [1,99])
omega_range = np.percentile(d['OMEGA'], [1,99])


plt.clf()
range = [per_range, ror_range, impact_range, dur_range, omega_range, ecc_range]
corner(fs, labels=['per', 'ror', 'impact', 'dur', 'omega', 'ecc'], show_titles=True, plot_contours=True, range=range);
plt.savefig(figure_direct + 'imp_corner_pl' + str(plno) + '.png')
plt.close()


np.savetxt(results_direct + 'importance_samples.txt', d)
np.savetxt(results_direct + 'weights.txt', w)

plt.clf()
plt.figure(figsize=(5, 5))
plt.hist2d( d['OMEGA'], d['ECC'], bins=[np.arange(-np.pi/2, 3*np.pi/2, 0.4), np.arange(0,0.92,0.08)], cmap='Blues');
plt.xlabel('w')
plt.ylabel('e')
plt.title(koin)
plt.savefig(figure_direct + 'ew.png')
plt.close()


plt.clf()
plt.figure(figsize=(5, 4))
plt.hist(d['ECC'], bins=np.arange(0, 0.92, 0.08));
plt.xlabel('e')
plt.title(koin)
plt.savefig(figure_direct + 'ehist.png')
plt.close()