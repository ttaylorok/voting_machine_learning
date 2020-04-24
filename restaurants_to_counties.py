import geopandas as gp
import shapely as sp
from shapely.geometry import Point
from shapely.geometry import asMultiPoint
import geopy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import time
start_time = time.time()



states = gp.read_file('C:\\Users\\comp\\Documents\\OSM\\tl_2017_us_state\\tl_2017_us_state.shp')
counties = gp.read_file('C:\\Users\\comp\\Documents\\OSM\\tl_2017_us_county\\tl_2017_us_county.shp')
#r = gp.read_file('C:\\Users\\comp\\Documents\\OSM\\combined_restaurant_and_ff.csv')
r =gp.read_file('all_restaurants_and_ff_compiled_with_cuisine.csv')
#r['geometry'] = Point(r['lon'],r['lat'])

r['geometry'] = gp.points_from_xy(x=r.lon.astype(float), y=r.lat.astype(float))
r.crs = 'epsg:4269'
#locs = geopandas.tools.geocode(counties)

tex = states[states['NAME'] == 'Texas']

rc = r.within(tex)

# r['state2'] = ''
# r['state2'][rc] = 'hehe'

voting = pd.read_csv('countypres_2000-2016.csv').dropna()
voting['FIPS'] = voting['FIPS'].astype(int)
voting = voting[voting['year'] == 2016]
vg = voting.groupby(['state','FIPS','party'])['candidatevotes'].sum()
vgu = vg.unstack()
vgu.reset_index(inplace = True)
vgu['party'] = np.where(vgu['democrat'] > vgu['republican'], 'democrat', 'republican')
vgu['FIPS2'] = np.where(vgu['FIPS'] < 10000, '0'+vgu['FIPS'].astype(str), vgu['FIPS'].astype(str))


rs = pd.merge(r,states,left_on = 'state',right_on = 'NAME')
rs.set_geometry(col='geometry_x', inplace = True)
rs.plot(column='state')

r['county'] = ''
for i,row in counties.iterrows():
    print(row['NAME'])
    rc = r.within(row['geometry'])
    r['county'][rc] = row['GEOID']
    
    
r.to_csv('restaurants_with_counties.csv')

fig, ax = plt.subplots()
ax.set_aspect('equal')
counties.plot(ax=ax)
r.plot(ax=ax, column='county')
plt.show()


rg = r.groupby('county')['county'].count()
m = pd.merge(rg,vgu,left_index=True,right_on='FIPS2')
s = m.sort_values('county', ascending = False)

r[r['county'] == '39017']

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#rs['STATEFP'].unique()

print("--- %s seconds ---" % (time.time() - start_time))

# c = counties[counties['STATEFP'] == '31']

# rc = r.within(counties)

# fig, ax = plt.subplots(1, 1)
# r.plot()
# counties.plot()

