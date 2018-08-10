import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from mpl_toolkits.basemap import Basemap
plt.switch_backend('agg')

rad2deg = 180/np.pi

# Pre-defined plotting boxes
Western_Atlantic = np.array([-98.186645, -59.832744, 7.791301 ,45.942453])
Contiguous_US = np.array([-132.0,-59.832744,7.791301,51.0])
North_America = np.array([-175.0,-60.0,7.5,72.0])
Entire_Globe = np.array([-180,180,-90,90])


##############################################################
# Functions
##############################################################

def plot_field(lon,lat,data,var,mesh_name,nfig,i,nmesh,plot_box,bounds=''):

    print "   plotting field: " + var
    
    m = Basemap(projection='cyl',llcrnrlat=plot_box[2],urcrnrlat=plot_box[3],\
                llcrnrlon=plot_box[0],urcrnrlon=plot_box[1],resolution='c')    

    lon_plot,lat_plot,data_plot = get_data_inside_box(lon,lat,data,plot_box)

    plt.figure(nfig)
    plt.subplot(nmesh,1,i)
    if bounds:
        levels = np.linspace(bounds[0],bounds[1],100)
    else:
        levels = np.linspace(np.amin(data_plot),np.amax(data_plot),100)
    plt.tricontourf(lon_plot,lat_plot,data_plot,levels=levels)
    m.fillcontinents(color='grey',lake_color='white')
    m.drawcoastlines()
    m.drawparallels(np.arange(plot_box[2],plot_box[3],20),labels=[True, False, False, False])
    m.drawmeridians(np.arange(plot_box[0],plot_box[1],60),labels=[False,False, False, True])
    plt.axis(plot_box)
    cb = plt.colorbar()
    cb.set_label(var.lower())
    plt.title(mesh_name)
    plt.suptitle(var,y=1.05,fontweight='bold')
    plt.tight_layout()
    plt.savefig(var.replace(' ','')+'_field.png',bbox_inches='tight')

##############################################################

def plot_hist(data,var,mesh_name,nfig,i,nmesh,bounds):

    print "   plotting histogram: " + var

    plt.figure(nfig)
    plt.subplot(nmesh,1,i)
    plt.hist(data,'auto')
    plt.title(mesh_name)
    plt.ylabel('count')
    plt.xlabel(var.lower())
    plt.xlim([bounds[0],bounds[1]])
    plt.ylim([bounds[2],bounds[3]])
    plt.suptitle(var,y=1.05,fontweight='bold')
    plt.tight_layout()
    plt.savefig(var.replace(' ','')+'_hist.png',bbox_inches='tight')

##############################################################

def plot_latavg(lat,data,var,mesh_name,binsize,nfig,i,nmesh,bounds):

    print "   plotting lat average: " + var

    lat_bins = np.arange(-90.0,90.0,binsize)
    lat_avg = np.zeros(lat_bins.shape)
    lat_min = np.zeros(lat_bins.shape)
    lat_max = np.zeros(lat_bins.shape)

    for j in range(lat_bins.size-1):
        
        idx = np.where((lat > lat_bins[j]) & (lat <= lat_bins[j+1]))
        lat_data = data[idx] 
        if lat_data.size > 0:
            lat_avg[j] = np.mean(lat_data)
            lat_min[j] = np.min(lat_data)
            lat_max[j] = np.max(lat_data)
        else:
            lat_avg[j] = np.nan
            lat_min[j] = np.nan
            lat_max[j] = np.nan

    plt.figure(nfig)
    plt.subplot(nmesh,1,i)
    plt.fill_between(lat_bins,lat_min,lat_max,alpha=.5)
    plt.plot(lat_bins,lat_avg)
    plt.title(mesh_name)
    plt.xlabel('latitude')
    plt.ylabel('average '+var.lower())
    plt.xlim([bounds[0],bounds[1]])
    plt.ylim([bounds[2],bounds[3]])
    plt.suptitle(var,y=1.05,fontweight='bold')
    plt.tight_layout()
    plt.savefig(var.replace(' ','')+'_lat.png',bbox_inches='tight')

#############################################################

def get_data_inside_box(lon,lat,data,box):

    # Find indicies of coordinates inside bounding box
    lon_idx, = np.where((lon > box[0]) & (lon < box[1]))
    lat_idx, = np.where((lat > box[2]) & (lat < box[3]))
    
    # Get region data inside bounding box
    latlon_idx = np.intersect1d(lat_idx,lon_idx)
    lon_region = lon[latlon_idx]
    lat_region = lat[latlon_idx]
    z_region = data[latlon_idx]

    return (lon_region,lat_region,z_region)

##############################################################
# Configuration Section
##############################################################

meshes = [
           {'file':'/users/sbrus/scratch1/MPAS-O_testing/ocean/global_ocean/EC60to30/spin_up/mesh_metrics/mesh_with_metrics.nc',
            'name':'EC60to30V3'},
           {'file':'/users/sbrus/scratch1/MPAS-O_testing/ocean/global_ocean/EC60to30JS/init/mesh_metrics/mesh_with_metrics.nc',
            'name':'EC60to30V4 (Jigsaw)'}
         ]

plot_box = Entire_Globe 

##############################################################
# Main Program
##############################################################


nmesh = len(meshes)
for i,mesh in enumerate(meshes):
   
    print "Mesh: " + mesh['name']
 
    # Open mesh and read in cell coordinates and data
    ds = Dataset(mesh['file'],'r')
    loncell = (np.mod(ds.variables['lonCell'][:]+np.pi,2*np.pi)-np.pi)*rad2deg
    latcell = ds.variables['latCell'][:]*rad2deg
    areaCell = ds.variables['areaCell'][:]
    cellQuality = ds.variables['cellQuality'][:]
    

    # Plot fields on globe
    plot_field(loncell,latcell,areaCell,'Cell Area',mesh['name'],1,i+1,nmesh,plot_box,[5.1e8,4.0e9]) 
    plot_field(loncell,latcell,cellQuality,'Cell Quality',mesh['name'],2,i+1,nmesh,plot_box,[0,1]) 

    # Plot histograms
    plot_hist(areaCell,'Cell Area',mesh['name'],3,i+1,nmesh,[0,4.0e9,0,60000])
    plot_hist(cellQuality,'Cell Quality',mesh['name'],4,i+1,nmesh,[0,1.1,0,8000])

    # Plot latitude averages
    binsize = 1.0
    plot_latavg(latcell,areaCell,'Cell Area',mesh['name'],binsize,5,i+1,nmesh,[-90,90,0,4.5e9])
    plot_latavg(latcell,cellQuality,'Cell Quality',mesh['name'],binsize,6,i+1,nmesh,[-90,90,0,1.1])





