from program.utils.utils_routing import get_route, distance_callback_index

def _get_intercluster_nearest_fields(cluster_df, from_cluster_idx, to_cluster_idx):
    # get list of field_id in each cluster
    from_cluster_field_id_list = cluster_df.loc[cluster_df['cluster_idx'] == from_cluster_idx]['field_id']
    to_cluster_field_id_list = cluster_df.loc[cluster_df['cluster_idx'] == to_cluster_idx]['field_id']

    from_index_best = None
    to_index_best = None
    distance_best = None
    for from_index in from_cluster_field_id_list:
        for to_index in to_cluster_field_id_list:
            distance = distance_callback_index(cluster_df, from_index, to_index)

            if distance_best == None or distance < distance_best:
                distance_best = distance
                from_index_best = from_index
                to_index_best = to_index
    
    return from_index_best, to_index_best
        



def get_route_planning_depot(cluster_df, machine_num, area_size_array, cluster_plant_seq_list, depot_loc, solution_limit_num = 100, visualize=False):
    '''
    param: cluster_df: DataFrame
    param: machine_num: int, number of machine (planter or harvester)
    param: area_size_array: numpy array, of shape n_clusters storing total area for each cluster
    param: cluster_plant_seq_list: list of cluster_idx order by planting (or harvesting) day
    param: depot_loc: list, list of [lon, lat] of depot
    param: solution_limit_num: int, used in or tools to limit the minimum number of valid search (default=100)
    param: visualize: visualize the route (default=False)

    output: route_dict
    '''

    route_dict = {}
    depot_field_id = len(cluster_df)+1
    for cluster_idx in cluster_plant_seq_list:
        max_acc_cap_const = area_size_array[cluster_idx] / machine_num
        machine_max_acc_cap = [max_acc_cap_const*1.5 for _ in range(machine_num)] # multiply 1.5 for flexibility and feasibility

        cluster_df_routing = cluster_df.loc[cluster_df['cluster_idx'] == cluster_idx].reset_index(drop=True)
        # manually define imagined depot and append to the dataframe
        cluster_df_routing = cluster_df_routing.append({'field_id': depot_field_id, 
                                                    'lon': depot_loc[0],
                                                    'lat': depot_loc[1],
                                                    'size': 0}, ignore_index = True).reset_index(drop=True)

        # check the start point and end point (current assumption start at depot and end at depot)
        start_field_id_list = [depot_field_id for _ in range(machine_num)]
        end_field_id_list = [depot_field_id for _ in range(machine_num)]
   
        _, result_table = get_route(cluster_df_routing, machine_num, machine_max_acc_cap, start_field_id_list, end_field_id_list, solution_limit_num = solution_limit_num, visualize=visualize)
        
        route_dict[f'{cluster_idx}'] = result_table

    return route_dict

def get_route_planning_condition(cluster_df, machine_num, area_size_array, cluster_plant_seq_list, depot_loc, solution_limit_num = 100, visualize=False):
    '''
    param: cluster_df: DataFrame
    param: machine_num: int, number of machine (planter or harvester)
    param: area_size_array: numpy array, of shape n_clusters storing total area for each cluster
    param: cluster_plant_seq_list: list of cluster_idx order by planting (or harvesting) day
    param: depot_loc: list, list of [lon, lat] of depot
    param: solution_limit_num: int, used in or tools to limit the minimum number of valid search (default=100)
    param: visualize: visualize the route (default=False)

    output: route_dict
    '''

    route_dict = {}
    depot_field_id = len(cluster_df)+1
    current_field_id = depot_field_id
    for i, cluster_idx in enumerate(cluster_plant_seq_list):
        max_acc_cap_const = area_size_array[cluster_idx] / machine_num
        machine_max_acc_cap = [max_acc_cap_const*1.5 for _ in range(machine_num)] # multiply 1.5 for flexibility and feasibility

        cluster_df_routing = cluster_df.loc[cluster_df['cluster_idx'] == cluster_idx].reset_index(drop=True)
        # manually define imagined depot and append to the dataframe
        cluster_df_routing = cluster_df_routing.append({'field_id': depot_field_id, 
                                                    'lon': depot_loc[0],
                                                    'lat': depot_loc[1],
                                                    'size': 0}, ignore_index = True).reset_index(drop=True)

        # add current_field_id as the starting point
        if current_field_id != depot_field_id:
            cluster_df_routing = cluster_df_routing.append({'field_id': current_field_id, 
                                                        'lon': cluster_df['lon'][current_field_id],
                                                        'lat': cluster_df['lat'][current_field_id],
                                                        'size': 0}, ignore_index = True).reset_index(drop=True)

        # check the start point and end point (current assumption start at depot and end at depot)
        # check from and to field_id
        start_field_id_list = [current_field_id for _ in range(machine_num)]

        print(cluster_df_routing)


        if i+1 > len(cluster_plant_seq_list)-1:
            end_field_id_list = [depot_field_id for _ in range(machine_num)]
        else:
            next_cluster_idx = cluster_plant_seq_list[i+1]
            from_index_best, to_index_best = _get_intercluster_nearest_fields(cluster_df, cluster_idx, next_cluster_idx)
            end_field_id_list = [from_index_best for _ in range(machine_num)]

        current_field_id = from_index_best
   
        _, result_table = get_route(cluster_df_routing, machine_num, machine_max_acc_cap, start_field_id_list, end_field_id_list, solution_limit_num = solution_limit_num, visualize=visualize)
        
        route_dict[f'{cluster_idx}'] = result_table

    return route_dict