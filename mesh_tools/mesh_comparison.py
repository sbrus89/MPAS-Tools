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
class comparison_plots:

  def __init__(self,nmesh,var,savepaths):

      # Initialize varaiables common to all methods
      self.nmesh = nmesh
      self.var = var
      self.savepaths = savepaths

  def plot_field(self,lon,lat,data,mesh_name,i,plot_box,bounds=''):
  
      print "   plotting field: " + self.var
      
      # Get the data inside the plotting box
      lon_plot,lat_plot,data_plot = self.get_data_inside_box(lon,lat,data,plot_box)
  
      # Initialize the figure and create subplot
      if i == 1:
        self.field = plt.figure()
      else:
        plt.figure(self.field.number)
      ax = self.field.add_subplot(nmesh,1,i)
      
      # Set the color scale bounds
      if bounds:
          levels = np.linspace(bounds[0],bounds[1],100)
      else:
          levels = np.linspace(np.amin(data_plot),np.amax(data_plot),100)

      # Plot the data contours
      cplot = ax.tricontourf(lon_plot,lat_plot,data_plot,levels=levels)

      # Plot the land, coastlines, and lat/lon lines
      m = Basemap(projection='cyl',llcrnrlat=plot_box[2],urcrnrlat=plot_box[3],\
                  llcrnrlon=plot_box[0],urcrnrlon=plot_box[1],resolution='c')    
      m.fillcontinents(color='grey',lake_color='white')
      m.drawcoastlines()
      m.drawparallels(np.arange(plot_box[2],plot_box[3],20),labels=[True, False, False, False])
      m.drawmeridians(np.arange(plot_box[0],plot_box[1],60),labels=[False,False, False, True])

      # Add the colorbar
      cb = self.field.colorbar(cplot,ax=ax)
      cb.set_label(self.var.lower())
      
      # Add axis and figure lables
      ax.set_title(mesh_name)
      ax.axis(plot_box)
      self.field.suptitle(self.var,y=1.05,fontweight='bold')
      self.field.tight_layout()

      # Save the figure
      for path in self.savepaths:
        self.field.savefig(path+self.var.replace(' ','')+'_field.png',bbox_inches='tight')
  
  ##############################################################
  
  def plot_hist(self,data,mesh_name,i,bounds='',bins=''):
  
      print "   plotting histogram: " + self.var
 
      # Initialize the figure and create subplot
      if i == 1: 
        self.hist = plt.figure()
      else:
        plt.figure(self.hist.number)
      ax = self.hist.add_subplot(nmesh,1,i)

      # Plot a histogram of the data
      if bins:
        ax.hist(data,bins)
      else:
        ax.hist(data,'auto')

      # Add axis and figure labels
      ax.set_title(mesh_name + '\n(Number of cells = '+str(data.size)+')')
      ax.set_ylabel('count')
      ax.set_xlabel(self.var.lower())
      self.hist.suptitle(self.var,y=1.05,fontweight='bold')
      self.hist.tight_layout()

      # Adjust the axis limits
      if bounds:
        ax.set_xlim([bounds[0],bounds[1]])
        ax.set_ylim([bounds[2],bounds[3]])
      else:
        self.make_axis_same(self.hist)
    
      # Add percentage axis 
      ax2 = ax.twinx()
      ax_ylim = ax.get_ylim()
      ax2.set_ylim([0,ax_ylim[1]/data.size*100])
      ax2.set_ylabel('percentage')

      # Save the figure
      for path in self.savepaths:
        self.hist.savefig(path+self.var.replace(' ','')+'_hist.png',bbox_inches='tight')

  ##############################################################
  
  def plot_latavg(self,lat,data,mesh_name,i,bounds='',binsize=1.0):
  
      print "   plotting lat average: " + self.var
  
      # Initialize the latitude vectors
      lat_bins = np.arange(-90.0,90.0,binsize)
      lat_avg = np.zeros(lat_bins.shape)
      lat_min = np.zeros(lat_bins.shape)
      lat_max = np.zeros(lat_bins.shape)
 
      for j in range(lat_bins.size-1):
          
          # Find cells in this bin
          idx = np.where((lat > lat_bins[j]) & (lat <= lat_bins[j+1]))
          lat_data = data[idx] 

          # Find average, minimum, and maximum of the data in this bin 
          # (if no data is in the bin, set to NaN so it is ignored in the plot)
          if lat_data.size > 0:
              lat_avg[j] = np.mean(lat_data)
              lat_min[j] = np.min(lat_data)
              lat_max[j] = np.max(lat_data)
          else:
              lat_avg[j] = np.nan
              lat_min[j] = np.nan
              lat_max[j] = np.nan
 
      # Initialize the figure and add subplot 
      if i == 1: 
        self.latavg = plt.figure()
      else:
        plt.figure(self.latavg.number)
      ax = self.latavg.add_subplot(nmesh,1,i)
 
      # Plot the average, minimum and maximum
      ax.fill_between(lat_bins,lat_min,lat_max,alpha=.5)
      ax.plot(lat_bins,lat_avg)

      # Add axis and figure labels
      ax.set_title(mesh_name)
      ax.set_xlabel('latitude')
      ax.set_ylabel('average '+self.var.lower())
      self.latavg.suptitle(self.var,y=1.05,fontweight='bold')
      self.latavg.tight_layout()

      # Adjust the axis limits
      if bounds:
        ax.set_xlim([bounds[0],bounds[1]])
        ax.set_ylim([bounds[2],bounds[3]])
      else:
        self.make_axis_same(self.latavg)

      # Save the figure
      for path in self.savepaths:
        self.latavg.savefig(path+self.var.replace(' ','')+'_lat.png',bbox_inches='tight')
  
  #############################################################
  
  def get_data_inside_box(self,lon,lat,data,box):
  
      # Find indicies of coordinates inside bounding box
      lon_idx, = np.where((lon > box[0]) & (lon < box[1]))
      lat_idx, = np.where((lat > box[2]) & (lat < box[3]))
      
      # Get region data inside bounding box
      latlon_idx = np.intersect1d(lat_idx,lon_idx)
      lon_region = lon[latlon_idx]
      lat_region = lat[latlon_idx]
      z_region = data[latlon_idx]
  
      return (lon_region,lat_region,z_region)

  #############################################################

  def make_axis_same(self,fig):
    
    # Initialize the axis limits
    xlim_all = [1e99,-1e99]
    ylim_all = [1e99,-1e99]

    # Find the lower and upper bounds of the subplot axes limits
    axes = fig.get_axes()
    for ax in axes:
      if ax.get_ylabel() != 'percentage':
        xlim = ax.get_xlim()
        xlim_all[0] = min(xlim_all[0],xlim[0])
        xlim_all[1] = max(xlim_all[1],xlim[1])
        ylim = ax.get_ylim()
        ylim_all[0] = min(ylim_all[0],ylim[0])
        ylim_all[1] = max(ylim_all[1],ylim[1])

    # Set all axes limits to the lower and upper bounds
    for ax in axes:
      if ax.get_ylabel() != 'percentage':
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

##############################################################
# Configuration Section
##############################################################

meshes = [
           #{'path':'/users/sbrus/scratch1/MPAS-O_testing/ocean/global_ocean/EC60to30/spin_up/mesh_metrics/',
           # 'file':'mesh_with_metrics.nc',
           # 'name':'EC60to30V3'},
           #{'path':'/users/sbrus/scratch1/MPAS-O_testing/ocean/global_ocean/EC60to30JS/init/mesh_metrics/',
           # 'file':'mesh_with_metrics.nc',
           # 'name':'EC60to30V4 (Jigsaw)'},

           #{'path':'/users/sbrus/scratch1/MPAS-O_testing/ocean/global_ocean/USNAEC60to30cr10/init/mesh_metrics/',
           # 'file':'mesh_with_metrics.nc',
           # 'name':'USNAEC60to30cr10'},

           #{'path':'/users/sbrus/scratch1/MPAS-O_testing/ocean/global_ocean/USNARRS30to10cr10/init/mesh_metrics/',
           # 'file':'mesh_with_metrics.nc',
           # 'name':'USNARRS30to10cr10'}
           #{'path':'/users/sbrus/scratch1/MPAS-O_testing/ocean/global_ocean/USNARRS30to10cr5/init/mesh_metrics/',
           # 'file':'mesh_with_metrics.nc',
           # 'name':'USNARRS30to10cr5'},
           #{'path':'/users/sbrus/scratch1/MPAS-O_testing/ocean/global_ocean/USNARRS30to10cr1/init/mesh_metrics/',
           # 'file':'mesh_with_metrics.nc',
           # 'name':'USNARRS30to10cr1'},

           {'path':'/users/sbrus/scratch1/MPAS-O_testing/ocean/global_ocean/USDEQU120cr1/init/mesh_metrics/',
            'file':'mesh_with_metrics.nc',
            'name':'USDEQU120cr1'},
           {'path':'/users/sbrus/scratch1/MPAS-O_testing/ocean/global_ocean/USDEQU240cr1/init/mesh_metrics/',
            'file':'mesh_with_metrics.nc',
            'name':'USDEQU240cr1'},

           #{'path':'/users/sbrus/scratch1/MPAS-O_testing/ocean/global_ocean/USDERRS30to10cr1/init/mesh_metrics/',
           # 'file':'mesh_with_metrics.nc',
           # 'name':'USDERRS30to10cr1'},
           #{'path':'/users/sbrus/scratch1/MPAS-O_testing/ocean/global_ocean/USDERRS30to10cr500m/init/mesh_metrics/',
           # 'file':'mesh_with_metrics.nc',
           # 'name':'USDERRS30to10cr500m'},
           #{'path':'/users/sbrus/scratch1/MPAS-O_testing/ocean/global_ocean/USDERRS30to10cr100m/init/mesh_metrics/',
           # 'file':'mesh_with_metrics.nc',
           # 'name':'USDERRS30to10cr100m'}
         ]

plot_box = Entire_Globe 
cell_width_bounds = [0.9,275.0]

##############################################################
# Main Program
##############################################################

# Initialize comparison_plots class instances for each mesh metric
nmesh = len(meshes)
savepaths = [mesh['path'] for mesh in meshes]
cell_size = comparison_plots(nmesh,'Cell Width',savepaths)
cell_qual = comparison_plots(nmesh,'Cell Quality',savepaths)

for i,mesh in enumerate(meshes):
   
    print "Mesh: " + mesh['name']
 
    # Open mesh and read in cell coordinates and data
    ds = Dataset(mesh['path']+mesh['file'],'r')
    loncell = (np.mod(ds.variables['lonCell'][:]+np.pi,2*np.pi)-np.pi)*rad2deg
    latcell = ds.variables['latCell'][:]*rad2deg
    areaCell = ds.variables['areaCell'][:]
    cellQuality = ds.variables['cellQuality'][:]

    # Estimate cell width based on cell area using equal-area circle diameter
    cellWidth = 2.0*np.sqrt(areaCell/np.pi)/1000
    
    # Plot fields on globe
    cell_size.plot_field(loncell,latcell,cellWidth,  mesh['name'],i+1,plot_box,cell_width_bounds) 
    cell_qual.plot_field(loncell,latcell,cellQuality,mesh['name'],i+1,plot_box,[0.0,1.0]) 

    # Plot histograms
    cell_size.plot_hist(cellWidth,  mesh['name'],i+1,bins=100) 
    cell_qual.plot_hist(cellQuality,mesh['name'],i+1,bins=20)

    # Plot latitude averages
    cell_size.plot_latavg(latcell,cellWidth,  mesh['name'],i+1) 
    cell_qual.plot_latavg(latcell,cellQuality,mesh['name'],i+1)





