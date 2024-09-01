'''
this file contains a function to cluster fields into groups. Each groups will be suggested to plant the same variety in the same period

'''

import json
import numpy as np
from sklearn.cluster import KMeans
from .utils.utils_field_size import cal_field_size
from .utils.utils_routing import get_route, get_inert_coeff
import os
import pandas as pd

# for visualization
import matplotlib.pyplot as plt

def _process_geojson(geojson_path, cal_size = True):
    '''
    function to process location of each field to (lon,lat) array

    param: geojson_path: geojson field filepath
    param: cal_size, boolean: for development, if False, we will use the precalculated field_size_array to reduce calculation time

    output: field_size_array: numpy array with shape m where m is the number fo fields
    output: field_coord_array: numpy array with shape m*2 where m is the number of fields
    '''
    
    f = open(geojson_path, encoding='utf8')
    data = json.load(f, encoding='utf8')
    m = len(data["features"])
    field_coord_array = np.empty(shape=(m, 2))
    field_size_array = np.empty(shape=(m,))
    for i, polygon in enumerate(data["features"]):
        coord_list = polygon["geometry"]["coordinates"][0]
        coord_array = np.array(coord_list)[:, :2] # remove altitude
        coord = np.mean(coord_array, axis=0)
        field_coord_array[i,:] = coord
        if cal_size:
            field_size_array[i] = cal_field_size(coord_array)

    if cal_size:
        size_array_filename = os.path.splitext(geojson_path)[0]
        field_size_array.tofile(f'{size_array_filename}.dat')


    return field_size_array, field_coord_array

def _get_harvest_capacity(harvester_def_path, working_hour):
    '''
    internal function to get total available harvester (planting) capacity
    param: harvester_def_path: string of path of harvester info json
    param: working_hour: float, user defined working hour (default = 8)

    output: total_cap: float, total capacity [hec/day]
    output: num_machine: int, total number of machine
    '''
    f = open(harvester_def_path, encoding='utf8')
    data = json.load(f, encoding='utf8')
    total_cap = 0.0
    num_machine = 0
    for harvester_type in data.keys():
        harvester = data[harvester_type]
        total_cap += harvester["number"] * harvester['harvest_speed'] * harvester['width'] * 0.36 * working_hour # hec/hr * hr = hec/day
        num_machine += harvester["number"]
    
    return total_cap, int(num_machine)


def _get_plant_capacity(planter_def_path, working_hour):
    '''
    internal function to get total available harvester (planting) capacity
    param: planter_def_path: string of path of harvester info json
    param: working_hour: float, user defined working hour (default = 8)

    output: total capacity [hec/day]
    output: num_machine: int, total number of machine
    '''
    f = open(planter_def_path, encoding='utf8')
    data = json.load(f, encoding='utf8')
    total_cap = 0.0
    num_machine = 0
    for planter_type in data.keys():
        planter = data[planter_type]
        total_cap += planter["number"] * planter['plant_speed'] * planter['width'] * 0.36 * working_hour # hec/hr * hr = hec/day
        num_machine += planter["number"]

    return total_cap, int(num_machine)


def _get_cluster_num(total_cap_harvester, total_cap_planter, field_size_array, variety_num, inert_coeff_harvester, inert_coeff_planter, cluster_max_planting_range = 7):
    '''
    function to get the proper cluster number that match the available capacity
    param: total_cap_harvester: float, total actual cap for harvester in one day (hec/day) 
    param: total_cap_planter: float, total actual cap for planter in one day (hec/day) 
    param: field_size_array, array of field sizes
    param: variety_num, int: user defined number of variety they want to plant
    param: inert_coeff_harvester: float, ratio to substitute the other wasted capacity such as traveling between fields, cleaning, mantainance, rest time 
    param: inert_coeff_planter: float, ratio to substitute the other wasted capacity such as traveling between fields, cleaning, mantainance, rest time
    param: cluster_max_planting_range: float, user defined group (how many days that one cluster should be planted within) (default = 7 days)
   
    output: cluster_num: int, number of cluster needed
    output: total_cap_planter_inert: float, total cap for planter in one day (hec/day) with inert panelty from the traveling time
    output: total_cap_harvester_inert: float, total cap for harvester in one day (hec/day) with inert panelty from the traveling time
    '''
    total_cap_harvester_inert = total_cap_harvester * inert_coeff_harvester #hec/day
    total_cap_planter_inert = total_cap_planter * inert_coeff_planter #hec/day

    max_area_per_cluster_harvester = total_cap_harvester_inert*cluster_max_planting_range
    max_area_per_cluster_planter = total_cap_planter_inert*cluster_max_planting_range

    max_area_per_cluster = min(max_area_per_cluster_harvester, max_area_per_cluster_planter)

    total_area = np.sum(field_size_array)
    cluster_num = round(total_area/max_area_per_cluster)

    # use maximum cluster_num of user-defined variety needed or the capacity-calculated one
    cluster_num = max(variety_num, cluster_num)

    return cluster_num, total_cap_planter_inert, total_cap_harvester_inert


def _get_clustering(field_coord_array, field_size_array, n_clusters, visualize):
    '''
    param: field_coord_array: numpy array, array of centroids (lon,lat) of each field
    param: field_size_array: numpy array, array of sizes of each field
    param: n_clusters: int, number of clusters
    param: visualize, boolean, visualize the clustering for development purpose

    output: segment_array, numpy array, array of cluster index
    '''
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X=field_coord_array,  y=None, sample_weight=field_size_array)
    cluster_array = kmeans.labels_

    if visualize:
        plt.scatter(field_coord_array[:,0], field_coord_array[:,1], c = cluster_array, s=field_size_array*500, alpha=0.5)
        plt.show()

    return cluster_array

def field_clustering(geojson_path, harvester_def_path, planter_def_path, truck_def_path, variety_num, working_hour = 7, cluster_max_planting_range = 7, inert_coeff=None, cal_size=False, visualize=False):
    '''
    param: geojson_path: string of geojson field filepath
    param: harvester_def_path: string of path of harvester info json
    param: planter_def_path: string of path of planter info json
    param: truck_def_path: string of path of truck info json
    param: variety_num: int, user defined number of variety they want to plant
    param: working_hour: float, user defined working hour (default = 8)
    param: cluster_max_planting_range: float, user defined group (how many days that one cluster should be planted within) (default = 7 days)
    param: inert_coeff: float, ratio to substitute the other wasted capacity such as traveling between fields, cleaning, mantainance, rest time (default=None (= means auto calculated) (recommended)) 
    param: cal_size: boolean, for development, if False, we will use the precalculated field_size_array to reduce calculation time
    
    output: cluster_df, dataframe
    output: cluster_num: int, number of cluster needed
    output: total_cap_planter: float, total actual cap for planter in one day (hec/day)
    output: total_cap_harvester: float, total actual cap for harvester in one day (hec/day)
    output: num_planter: int, total number of planter
    output: num_harvester: int, total number of harvester
    '''

    # process geojson
    field_size_array, field_coord_array = _process_geojson(geojson_path=geojson_path, cal_size=cal_size)
    if not cal_size:
        size_array_filename = os.path.splitext(geojson_path)[0]
        field_size_array = np.fromfile(f'{size_array_filename}.dat', dtype=float)

    # get actual capacity
    total_cap_harvester, num_harvester = _get_harvest_capacity(harvester_def_path, working_hour) #hec/day
    total_cap_planter, num_planter = _get_plant_capacity(planter_def_path, working_hour) #hec/day

    # set inert coefficient
    if inert_coeff != None: # manual input
        inert_coeff_planter = inert_coeff
        inert_coeff_harvester = inert_coeff
    else:
        # temp dataframe for routing
        data_routing = {'field_id': np.arange(field_size_array.shape[0]),
                'lon': field_coord_array[:,0],
                'lat': field_coord_array[:,1],
                'size': field_size_array,
                }
        cluster_df_routing = pd.DataFrame(data_routing)

        total_area = field_size_array.sum()

        inert_coeff_planter, inert_coeff_harvester = get_inert_coeff(cluster_df_routing, 
                                                                    total_area,
                                                                    working_hour, 
                                                                    truck_def_path, 
                                                                    total_cap_planter, 
                                                                    total_cap_harvester)
    
    # get cluster number
    n_clusters, _, _ = _get_cluster_num(total_cap_harvester=total_cap_harvester, 
                                        total_cap_planter=total_cap_planter,
                                        field_size_array=field_size_array, 
                                        variety_num=variety_num, 
                                        inert_coeff_harvester=inert_coeff_harvester,
                                        inert_coeff_planter=inert_coeff_planter,
                                        cluster_max_planting_range = cluster_max_planting_range, 
                                        )

    # do clustering
    cluster_array = _get_clustering(field_coord_array, field_size_array, n_clusters, visualize=visualize)

    # put processed data in dataframe
    # columns: [field_id, lon (centroid), lat (centroid), size, cluster_idx]
    data = {'field_id': np.arange(field_size_array.shape[0]),
            'lon': field_coord_array[:,0],
            'lat': field_coord_array[:,1],
            'size': field_size_array,
            'cluster_idx': cluster_array,
            }
    cluster_df = pd.DataFrame(data)

    return cluster_df, n_clusters, total_cap_planter, total_cap_harvester, num_planter, num_harvester



if __name__ == '__main__':
    # geojson_path = 'farm_def/2022-01-07_naorice.geojson'
    geojson_path = 'farm_def/2022-01-05_Ohta_Paddy_data_flat_2.geojson'
    harvester_def_path = 'farm_def/harvester_multiple.json'
    planter_def_path = 'farm_def/planter_multiple.json'
    cal_size = False
    variety_num = 3
    visualize = True

    cluster_df = field_clustering(geojson_path=geojson_path, harvester_def_path=harvester_def_path, planter_def_path=planter_def_path, variety_num=variety_num, cal_size=cal_size, visualize=visualize)