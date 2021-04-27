# %%
from netCDF4 import Dataset as D
import numpy as np
f = D('tcdata/enso_round1_train_20210201/CMIP_train.nc')
lat = f.variables['lat'][:]
lon = f.variables['lon'][:]
varity = ['sst', 't300', 'ua', 'va']
for v in range(4):
    data0 = f.variables[varity[v]][:]
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
    locals()[varity[v]] = cmip
# %%
ncfile = D('data_48/CMIP_train_48.nc', 'w', format='NETCDF4')
#Add dimensions
xdim = ncfile.createDimension('lon', 72)
ydim = ncfile.createDimension('lat', 24)
tdim = ncfile.createDimension('time', len(sst))
tdim = ncfile.createDimension('member', 48)

#Add variables
var = ncfile.createVariable('lon', np.float32, 'lon')
var.setncattr_string('long_name', 'longitude')
var.setncattr_string('units', 'degrees_east')
var[:] = lon

var = ncfile.createVariable('lat', np.float32, 'lat')
var.setncattr_string('long_name', 'latitude')
var.setncattr_string('units', 'degrees_north')
var[:] = lat

tvar = ncfile.createVariable('time', np.int, 'time')
tvar.setncattr_string('long_name', 'time')
tvar.setncattr_string('units', 'days since 0000-01-01')
tvar.calendar = "standard"
tvar[:] = np.arange(len(sst))

tvar = ncfile.createVariable('member', np.int, 'member')
tvar.setncattr_string('long_name', 'member')
tvar[:] = range(48)

for i in range(4):
    var = ncfile.createVariable(str(varity[i]), np.float, ('time', 'member', 'lat', 'lon'))
    var.setncattr_string('long_name', str(varity[i]))
    var.setncattr_string('units', ' ')
    var[:] = locals()[varity[i]]

#Create netCDF file
ncfile.close()
print('NC is OK')
