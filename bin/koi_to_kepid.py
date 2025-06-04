from astropy.table import Table
import sys

koi_cumul = Table.read('/Users/ssagear/UFL Dropbox/Sheila Sagear/Research/github/alderaan/bin/Catalogs/cumulative_2025.05.30_14.09.29.csv', format='csv', comment='#')

nkoi = sys.argv[1]

kep_tab_system = koi_cumul[koi_cumul['kepoi_sys_name'] == nkoi]
kep_tab_system.sort('koi_period')
kepid = kep_tab_system['kepid'].value[0]

print(kepid)