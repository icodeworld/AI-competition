# %%
from netCDF4 import Dataset as D
import numpy as np
f = D('tcdata/enso_round1_train_20210201/SODA_label.nc')
va = ['nino']
for v in range(len(va)):
    data0 = f.variables[va[v]][:]
    data1 = data0[0:99]
    data2 = data0[1:100, -12:]
    print(data1.shape, data2.shape)
    data3 = np.concatenate((data1, data2), axis=1)
    print(data3.shape)
    locals()[va[v]] = data3
# %%
ncfile = D('data_48/SODA_label_48.nc', 'w', format='NETCDF4')

tdim = ncfile.createDimension('time', len(nino))
tdim = ncfile.createDimension('member', 48)

tvar = ncfile.createVariable('time', np.int, 'time')
tvar.setncattr_string('long_name', 'time')
tvar.setncattr_string('units', 'days since 0000-01-01')
tvar.calendar = "standard"
tvar[:] = np.arange(len(nino))

tvar = ncfile.createVariable('member', np.int, 'member')
tvar.setncattr_string('long_name', 'member')
tvar[:] = range(48)

for i in range(len(va)):
    var = ncfile.createVariable(va[i], np.float, ('time', 'member'))
    var.setncattr_string('long_name', va[i])
    var.setncattr_string('units', ' ')
    var[:] = locals()[va[i]]

#Create netCDF file
ncfile.close()
print('NC is OK')
