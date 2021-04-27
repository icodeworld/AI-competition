# %%
from netCDF4 import Dataset as D
import numpy as np
f = D('tcdata/enso_round1_train_20210201/SODA_train.nc')
lat = f.variables['lat'][:]
lon = f.variables['lon'][:]
varity = ['sst', 't300', 'ua', 'va']
for v in range(4):
    data0 = f.variables[varity[v]][:]
    data1 = data0[0:99]
    data2 = data0[1:100, -12:]
    print(data1.shape, data2.shape)
    data3 = np.concatenate((data1, data2), axis=1)
    print(data3.shape)
    locals()[varity[v]] = data3
# %%
ncfile = D('data_48/SODA_train_48.nc', 'w', format='NETCDF4')
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
    var = ncfile.createVariable(varity[i], np.float, ('time', 'member', 'lat', 'lon'))
    var.setncattr_string('long_name', varity[i])
    var.setncattr_string('units', ' ')
    var[:] = locals()[varity[i]]

#Create netCDF file
ncfile.close()
print('NC is OK')
