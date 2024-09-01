# def get_route(field_coord_array, planter_def_path, harvester_def_path, truck_def_path, start_point = None, dest_point = None):
#    '''
#     solve route and the travel capacity needed through all fields in the given array
#     the solution of traveling is done by looking at it as travelling salemans problem
#     '''

#     pass


import pyproj
from ortools.constraint_solver import pywrapcp

import pandas as pd
import json
import random
import plotly.graph_objs as go


def distance_callback_index(cluster_df, from_index, to_index):

    """Returns the distance between the two nodes by their index, used as in _distance_callback, and also in route_planning.py"""

    # 2点間の距離を計算
    g = pyproj.Geod(ellps='WGS84')
    distance = g.inv(
        cluster_df['lon'][from_index], cluster_df['lat'][from_index], 
        cluster_df['lon'][to_index], cluster_df['lat'][to_index]
    )[2]
    return distance

    
def _get_solution(cluster_df, machine_num, machine_acc_cap, start_local_idx_list, end_local_idx_list, solution_limit_num=3):
    # create index manager
    manager = pywrapcp.RoutingIndexManager(len(cluster_df), machine_num, start_local_idx_list, end_local_idx_list)

    # create routing model
    routing = pywrapcp.RoutingModel(manager)

    # distance callback function to be registered in routing model
    def _distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Indexの変換
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        distance = distance_callback_index(cluster_df, from_node, to_node)
        return distance


    # create capacity callback function to be registered in routing model
    def _demand_callback(index):
        """Returns the demand of the node."""
        node = manager.IndexToNode(index)
        return cluster_df['size'][node] * 10e5 # the solver accept only integer constraint, so we change the unit to 10e5 hec

    # register distance callback
    transit_callback_index = routing.RegisterTransitCallback(_distance_callback)

    # travel cost setting
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # register capacity model
    demand_callback_index = routing.RegisterUnaryTransitCallback(_demand_callback)

    # # capacity constraint
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        machine_acc_cap,  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')
    capacity = routing.GetDimensionOrDie('Capacity')

    # excute optimization
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # search_parameters.time_limit.seconds = 10
    search_parameters.solution_limit = solution_limit_num # https://developers.google.com/optimization/routing/routing_tasks#solution_limits
    solution = routing.SolveWithParameters(search_parameters)

    return manager, routing, capacity, solution


def _get_solution_df(manager, routing, capacity, solution, machine_num):
    """Get optimization results"""
    result = dict()
    for vehicle_id in range(machine_num):
        index = routing.Start(vehicle_id)
        route_distance = 0
        route_load = 0
        IsEnd_boo = False
        while not IsEnd_boo:
            node_index = manager.IndexToNode(index)
            route_load = solution.Value(capacity.CumulVar(index)) / 10e5
            previous_index = index

            if not routing.IsEnd(index):
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
            else: 
                IsEnd_boo = True
            
            result[previous_index] = {'field_local_idx': node_index, 'acc_cap': route_load, 'acc_dist': route_distance, 'vehicle_id': vehicle_id}

    result_table = pd.DataFrame(result).T
    return result_table

def get_route(cluster_df, machine_num, machine_max_acc_cap, start_field_id_list, end_field_id_list, solution_limit_num = 3, visualize=False):

    # get local index instead of field_id
    if start_field_id_list != None and end_field_id_list != None:
        start_local_idx_list = []
        end_local_idx_list = []
        for [start_field_id, end_field_id] in zip(start_field_id_list, end_field_id_list): 
            start_local_idx_list.append(cluster_df.index[cluster_df['field_id'] == start_field_id].tolist()[0])
            end_local_idx_list.append(cluster_df.index[cluster_df['field_id'] == end_field_id].tolist()[0])
    else:
        start_local_idx_list = [0 for _ in range(machine_num)]
        end_local_idx_list = [0 for _ in range(machine_num)]

    machine_max_acc_cap = [cap * 10e5 for cap in machine_max_acc_cap] # the solver accept only integer constraint, so we change the unit to 10e5 hec
    manager, routing, capacity, solution = _get_solution(cluster_df, machine_num, machine_max_acc_cap, start_local_idx_list, end_local_idx_list, solution_limit_num)
    total_distance = solution.ObjectiveValue()
    result_table = _get_solution_df(manager, routing, capacity, solution, machine_num)

    if visualize:
        result_table['arrival_seq'] = result_table.groupby('vehicle_id').cumcount()

        trace = []
        # color_list = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])for i in range(machine_num)]
        color_list = ['red', 'blue', 'green', 'brown', 'purple', 'orange', 'yellow']
        for idx in range(machine_num):
            route_v0 = result_table.query(f'vehicle_id=={idx}').merge(cluster_df, left_on='field_local_idx', right_on=cluster_df.index)
            route_v0 = route_v0.append(route_v0.iloc[0]) # return as cycle

            depot_v0 = route_v0.iloc[0:1]
            end_v0 = route_v0.iloc[-1:]
            depot_v0['acc_dist'] = 0

            
            for route, color, name in zip([route_v0, depot_v0, end_v0], 
                                        [color_list[idx], 'black', 'green'], 
                                        [f'vehicle_{idx}', f'depot_{idx}', f'end_{idx}']):

                trace_route = go.Scattermapbox(
                    lon = route['lon'],
                    lat = route['lat'],
                    mode = 'markers + text + lines',
                    text = 'field_local_idx:'+ route['field_local_idx'].astype(str) + '<br>' + 
                        'field_id:'+ route['field_id'].astype(str) + '<br>' + 
                        'arrival_seq: ' + route['arrival_seq'].astype(str) + '<br>' + 
                        'accumulated_cap: ' + route['acc_cap'].astype(str) + '<br>' + 
                        '次地点到着までのacc_dist: ' + route['acc_dist'].astype(str),
                    marker = dict(size=route['size']),
                    line = dict(color=color),
                    name=name
                )
                trace.append(trace_route)

        data = trace

        fig = go.Figure(data)
        fig.update_layout(mapbox_style="open-street-map",
                        mapbox_center_lat=34.884,
                        mapbox_center_lon=136.886,
                        mapbox={'zoom': 10},
                        margin={"r":0,"t":0,"l":0,"b":0},
        )
        fig.show()

    return total_distance, result_table


def get_inert_coeff(cluster_df_routing, total_area, working_hour, truck_def_path, total_cap_planter, total_cap_harvester):
    '''
    solve the time to travel through all fields (regardless of the clusters) and compare the ratio of traveling time (=cap in area unit) and harvest time, and then plus some user-defined inert such as resting
    the solution of traveling is done by looking at it as travelling salemans problem

    param: cluster_df_routing: dataframe, temp dataframe for routing
    param: total_area: float, total area of all of interest fields (hec)
    param: working_hour: float, user defined working hour (default = 8)
    param: truck_def_path: string of path of truck info json
    param: total_cap_planter: float, total actual cap for planter in one day (hec/day)
    param: total_cap_harvester: float, total actual cap for harvester in one day (hec/day)

    output: inert_coeff_planter: float, value in range [0,1] for the inert of capacity considering the traveling time by truck
    output: inert_coeff_harvester: float, value in range [0,1] for the inert of capacity considering the traveling time by truck
    '''
    machine_max_acc_cap = [total_area]

    total_distance, _ = get_route(cluster_df=cluster_df_routing, 
                                machine_num=1, 
                                machine_max_acc_cap=machine_max_acc_cap, 
                                start_field_id_list=None, 
                                end_field_id_list=None, 
                                visualize=False)

    f = open(truck_def_path, encoding='utf8')
    truck = json.load(f, encoding='utf8')
    truck_speed = truck['travel_speed'] # m/s
    total_time = total_distance / truck_speed /(60*60*working_hour) # day
    total_travel_cap_planter = total_time * total_cap_planter # day * hec/day = hec
    total_travel_cap_harvester = total_time * total_cap_harvester
    inert_coeff_planter = total_area / (total_area + total_travel_cap_planter)
    inert_coeff_harvester = total_area / (total_area + total_travel_cap_harvester)
    
    # rest time by constant inert penalty -0.1
    return inert_coeff_planter - 0.1, inert_coeff_harvester - 0.1