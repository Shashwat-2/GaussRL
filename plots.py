import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

def obs_data_count(trj):
    counts = [0 for _ in range(120)]
    for i in trj.obs[:,0]:
        counts[int(i)-1900] += 1
    plt.scatter(1900+np.arange(30), counts[:30])
    plt.plot(1900+np.arange(30), counts[:30])
    plt.xlabel('Year', fontweight='bold')
    plt.ylabel('No. of data points', fontweight='bold')
    plt.title('Year-wise total field intensity count', fontweight='bold')
    plt.savefig(f'Plots/{trj.year}/obs_data_count.png',dpi=600)
    plt.close()

def obs_data_loc(trj):

    fig = plt.figure()
    m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='lightgray')
    m.drawmapboundary()
    lat = [i for i,j in trj.catalog.keys()]
    lon = [j for i,j in trj.catalog.keys()]
    x,y = m(lon,lat)
    m.scatter(x,y,c='r',s=10)
    plt.title(f'Year {trj.year}', fontweight='bold')
    plt.xlabel('Longitudes', fontweight='bold')
    plt.ylabel('Latitudes', fontweight='bold')
    plt.savefig(f'Plots/{trj.year}/obs_data_loc_{trj.year}.png',dpi=600)
    plt.close()

def contour_plot(trj, pred_coeff, year, lat_spaceing = 1000, lon_spacing = 1500):

    fig, (ax1,ax2) = plt.subplots(2,1, figsize=(11,9), constrained_layout=True)
    m1 = Basemap(projection='merc',llcrnrlat=-65,urcrnrlat=65,\
            llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='i',ax=ax1)
    m1.drawcoastlines()
    m1.drawmapboundary()
    m2 = Basemap(projection='merc',llcrnrlat=-65,urcrnrlat=65,\
            llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='i',ax=ax2)
    m2.drawcoastlines()
    m2.drawmapboundary()

    x1,y1,z1 = trj.contour_plot(lat_spaceing,lon_spacing)
    x2,y2,z2 = trj.contour_plot(lat_spaceing,lon_spacing,pred_coeff)
    x1,y1 = m1(x1,y1)
    x2,y2 = m2(x2,y2)
    _c1 = m1.contourf(x1,y1,z1)
    _c2 = m2.contourf(x2,y2,z2)
    ax1.set_title(f'Original DGRF ({year})', fontweight='bold')
    ax2.set_title(f'Predicted GRF ({year})', fontweight='bold')
    fig.colorbar(_c2, ax=[ax1,ax2])
    fig.savefig(f'Contour_Plots/{year}/contour_plot_{year}.png',dpi=1200)
    plt.close()

def gauss_coeff_plot(trj, pred_coeff, year):

    plt.plot(abs(trj.coeff(year)), label='DGRF', c='b')
    plt.plot(abs(pred_coeff), label='Predicted', c='r')
    plt.xlabel('Gauss coefficient ID', fontweight='bold')
    plt.ylabel('|Gauss coefficient value|', fontweight='bold')
    plt.legend()
    plt.title('Absolute Gauss Coefficients (original scale)', fontweight='bold')
    plt.savefig(f'Plots/orig_vs_pred(norm)_{year}.png',dpi=600)
    plt.close()

    plt.plot(np.log(abs(pred_coeff)), label='Predicted', c='r')
    plt.plot(np.log(abs(trj.coeff(year))), label='DGRF', c='b', alpha=0.6)
    plt.xlabel('Gauss coefficient ID', fontweight='bold')
    plt.ylabel('log(|Gauss coefficient value|)', fontweight='bold')
    plt.legend()
    plt.title('Gauss Coefficients (logarithmic scale)', fontweight='bold')
    plt.savefig(f'Plots/orig_vs_pred(log)_{year}.png',dpi=600)
    plt.close()

def obs_diff(trj, pred_coeff, N=10):

    locs = [f'{i}' for i in trj.catalog.keys()][:N]
    diffs = {'pred':[abs(i-j) for i,j in trj.var_catalog(pred_coeff).values()][:N],
             'dgrf':[abs(i-j) for i,j in trj.catalog.values()][:N]}

    bar_width = 0.35
    index = np.arange(len(locs))
    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(index + bar_width, diffs['dgrf'], bar_width, label='DGRF')
    bar2 = ax.bar(index, diffs['pred'], bar_width, label='Pred')

    ax.set_xlabel('Locations (Lat,Lon)', fontweight='bold')
    ax.set_ylabel('Absolute Difference |F - F_obs|', fontweight='bold')
    ax.set_title('Deviations from Observed Data', fontweight='bold')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(locs, rotation=45)
    ax.legend()

    fig.tight_layout()
    fig.savefig(f'Plots/{trj.year}/obs_diff_{trj.year}.png', dpi=600)
    plt.close()

def div_plot(trj, true, pred):
    divs = list()
    for i in [true, pred]:
        LON, LAT, (F, B_r, B_theta, B_phi) = trj.contour_plot(10,10,i)
        div = trj.divergence(6371.2, 90-LAT, LON, B_r, B_theta, B_phi)
        divs.append(div)