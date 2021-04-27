# %%
from netCDF4 import Dataset as D
import numpy as np
f = D('tcdata/enso_round1_train_20210201/CMIP_label.nc')
va = ['nino']
for v in range(len(va)):
    data0 = f.variables[va[v]][:]
    a = []
    for k in range(15):
        print(k)
        data1 = data0[k * 151:150 + k * 151]
        data2 = data0[k * 151 + 1:151 + k * 151, -12:]
        print(data1.shape, data2.shape)
        data3 = np.concatenate((data1, data2), axis=1)
        print(data3.shape)
        a.append(data3)
    cmip6 = np.concatenate(tuple(a), 0)
    print(cmip6.shape)

    a = []
    for k in range(17):
        print(k)
        data1 = data0[2265 + k * 140:2265 + 139 + k * 140]
        data2 = data0[2265 + k * 140 + 1:2265 + 140 + k * 140, -12:]
        print(data1.shape, data2.shape)
        data3 = np.concatenate((data1, data2), axis=1)
        print(data3.shape)
        a.append(data3)
    cmip5 = np.concatenate(tuple(a), 0)
    print(cmip5.shape)

    cmip = np.concatenate((cmip6, cmip5), axis=0)
    print(cmip.shape)
    locals()[va[v]] = cmip

# %%

ncfile = D('data_48/CMIP_label_48.nc', 'w', format='NETCDF4')

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
