from program.field_clustering import field_clustering
from program.rough_tuning import process_cropsim_output, precal_harvest_day_dist, get_required_cap_case, get_plant_day_base_list, get_obj_func_value
from program.fine_tuning import finetune_plant_day_chosen
from program.route_planning import get_route_planning_depot, get_route_planning_condition
import numpy as np
import pandas as pd
import itertools
from program.utils.utils_routing import get_inert_coeff, get_route


def main():
    '''
    ==================== assumption and user input ============================================
    '''
    # assumption and clustering

    geojson_path = 'farm_def/ricefield1.geojson'
    harvester_def_path = 'farm_def/harvester_multiple.json'
    planter_def_path = 'farm_def/planter_multiple.json'
    truck_def_path = 'farm_def/truck.json'
    working_hour = 7
    cluster_max_planting_range = 7
    inert_coeff = None
    cal_size = False
    variety_list = ['ishikari', 'koshihikari', 'nipponbare']
    variety_price_list = [250700, 234167, 240000]   
    variety_num = len(variety_list)
    visualize = False

    # rough tuning
    crop_sim_dir = 'crop_sim/simriw_crop_sim_data'    

    scenario_num = 200
    search_range = 40
    plant_day_search_start = 2022091
    store_best_rough_num = 5

    # fine tuning
    finetine_range = 20
    pop_size = 60
    num_generations = 80
    parent_ratio = 0.5

    # route planning
    depot_loc = [136.902288, 34.918178]
    always_return_depot = True

    '''
    ======================== step 0) clustering ================================================
    '''
    print('processing the clustering..')

    cluster_df, n_clusters, total_cap_planter, total_cap_harvester, num_planter, num_harvester = field_clustering(geojson_path=geojson_path, 
                                                                                                                    harvester_def_path=harvester_def_path, 
                                                                                                                    planter_def_path=planter_def_path,
                                                                                                                    truck_def_path=truck_def_path,
                                                                                                                    variety_num=variety_num, 
                                                                                                                    working_hour = working_hour,
                                                                                                                    cluster_max_planting_range = cluster_max_planting_range,
                                                                                                                    inert_coeff = inert_coeff,
                                                                                                                    cal_size=cal_size, 
                                                                                                                    visualize=True)

    # calculate total size of each cluster
    area_size_array = np.empty(shape=(n_clusters,))
    for i in range(n_clusters):
        area_size_array[i] = cluster_df.loc[cluster_df['cluster_idx'] == i, 'size'].sum()

    '''
    ========================= step 1) rough tuning ============================================
    '''

    print('processing cropsim data..')

    yield_brown_array, plant_day_array, harvest_day_array = process_cropsim_output(crop_sim_dir, variety_list, scenario_num, search_range, plant_day_search_start)

    # precalculate the norm distribution fitting for the distribution of maturity date (harvesting date) for all planting dates
    harvest_day_dist_array = precal_harvest_day_dist(harvest_day_array, search_range)

    # get the ideal planting_day_chosen for each variety
    plant_day_base_list = get_plant_day_base_list(yield_brown_array)

    # calculate the inert_coeff_harvester and inert_coeff_planter for each cluster
    inert_coeff_planter_list = []
    inert_coeff_harvester_list = []
    for cluster_idx in range(n_clusters):

        total_area = area_size_array[cluster_idx]
        cluster_df_routing = cluster_df.loc[cluster_df['cluster_idx'] == cluster_idx].reset_index(drop=True)

        inert_coeff_planter, inert_coeff_harvester = get_inert_coeff(cluster_df_routing, 
                                                                    total_area,
                                                                    working_hour, 
                                                                    truck_def_path, 
                                                                    total_cap_planter, 
                                                                    total_cap_harvester)

        inert_coeff_planter_list.append(inert_coeff_planter)
        inert_coeff_harvester_list.append(inert_coeff_harvester)


    # iterate rough tuning by brute force, store all obj_func in a dataframe
    print('rough tuning..')
    chosen_list_df = pd.DataFrame(columns=['variety_chosen_list', 'plant_day_chosen_list', 'obj_func_value'])

    for variety_chosen_list in list(itertools.product(np.arange(variety_num), repeat=n_clusters)):
        plant_day_chosen_list = [plant_day_base_list[i] for i in variety_chosen_list]

        obj_func_value = get_obj_func_value(plant_day_chosen_list=plant_day_chosen_list,
                                            variety_chosen_list=variety_chosen_list,
                                            cluster_max_planting_range=cluster_max_planting_range,
                                            variety_price_list=variety_price_list,
                                            total_cap_planter=total_cap_planter,
                                            total_cap_harvester=total_cap_harvester,
                                            area_size_array=area_size_array,
                                            yield_brown_array=yield_brown_array,
                                            plant_day_array=plant_day_array,
                                            harvest_day_array=harvest_day_array,
                                            harvest_day_dist_array=harvest_day_dist_array,
                                            inert_coeff_planter_list=inert_coeff_planter_list,
                                            inert_coeff_harvester_list=inert_coeff_harvester_list,
                                            visualize=False)

        chosen_list_df = chosen_list_df.append({'variety_chosen_list': variety_chosen_list, 
                                                'plant_day_chosen_list':plant_day_chosen_list,
                                                'obj_func_value':obj_func_value}, ignore_index = True)


    chosen_list_df = chosen_list_df.sort_values(by = 'obj_func_value', ascending=False).head(store_best_rough_num)


    # just for visualization
    # _, _, _ = get_required_cap_case(yield_brown_array=yield_brown_array,
    #                                 plant_day_array=plant_day_array,
    #                                 harvest_day_array=harvest_day_array,
    #                                 harvest_day_dist_array=harvest_day_dist_array,
    #                                 variety_chosen_list=chosen_list_df.iloc[0]['variety_chosen_list'],
    #                                 plant_day_chosen_list=chosen_list_df.iloc[0]['plant_day_chosen_list'],
    #                                 area_size_array=area_size_array,
    #                                 cluster_max_planting_range = cluster_max_planting_range,
    #                                 total_cap_planter = total_cap_planter,
    #                                 total_cap_harvester = total_cap_harvester,
    #                                 inert_coeff_planter_list=inert_coeff_planter_list,
    #                                 inert_coeff_harvester_list=inert_coeff_harvester_list,
    #                                 visualize=True)


    '''
    ======================== step 2) fine tuning =====================================================
    '''
    print('finetuning..')
    # finetune the planting date by GA optimizer

    chosen_list_df = finetune_plant_day_chosen(chosen_list_df=chosen_list_df, 
                                                finetune_range=20, 
                                                cluster_max_planting_range=cluster_max_planting_range, 
                                                variety_price_list=variety_price_list, 
                                                n_clusters=n_clusters, 
                                                total_cap_planter=total_cap_planter, 
                                                total_cap_harvester=total_cap_harvester, 
                                                area_size_array=area_size_array, 
                                                search_range=search_range, 
                                                yield_brown_array=yield_brown_array, 
                                                plant_day_array=plant_day_array, 
                                                harvest_day_array=harvest_day_array, 
                                                harvest_day_dist_array=harvest_day_dist_array, 
                                                inert_coeff_planter_list=inert_coeff_planter_list, 
                                                inert_coeff_harvester_list=inert_coeff_harvester_list, 
                                                pop_size = pop_size, 
                                                num_generations = num_generations, 
                                                parent_ratio = parent_ratio,
                                                visualize=visualize)

    # just for visualization and save fig in result_tuning
    _, _, _ = get_required_cap_case(yield_brown_array=yield_brown_array,
                                    plant_day_array=plant_day_array,
                                    harvest_day_array=harvest_day_array,
                                    harvest_day_dist_array=harvest_day_dist_array,
                                    variety_chosen_list=chosen_list_df.iloc[0]['variety_chosen_list'],
                                    plant_day_chosen_list=chosen_list_df.iloc[0]['plant_day_chosen_list'],
                                    area_size_array=area_size_array,
                                    cluster_max_planting_range = cluster_max_planting_range,
                                    total_cap_planter = total_cap_planter,
                                    total_cap_harvester = total_cap_harvester,
                                    inert_coeff_planter_list=inert_coeff_planter_list,
                                    inert_coeff_harvester_list=inert_coeff_harvester_list,
                                    visualize=visualize,
                                    saved_name='best_roughtuning')

    _, plant_required_cap_case_best, harvest_required_cap_case_best = get_required_cap_case(yield_brown_array=yield_brown_array,
                                                                                            plant_day_array=plant_day_array,
                                                                                            harvest_day_array=harvest_day_array,
                                                                                            harvest_day_dist_array=harvest_day_dist_array,
                                                                                            variety_chosen_list=chosen_list_df.iloc[0]['variety_chosen_list'],
                                                                                            plant_day_chosen_list=chosen_list_df.iloc[0]['plant_day_chosen_finetune_list'],
                                                                                            area_size_array=area_size_array,
                                                                                            cluster_max_planting_range = cluster_max_planting_range,
                                                                                            total_cap_planter = total_cap_planter,
                                                                                            total_cap_harvester = total_cap_harvester,
                                                                                            inert_coeff_planter_list=inert_coeff_planter_list,
                                                                                            inert_coeff_harvester_list=inert_coeff_harvester_list,
                                                                                            visualize=visualize,
                                                                                            saved_name='best_finetuning')


    print(chosen_list_df)
        
    '''
    ======================== step 3) route planning =====================================================
    '''
    cluster_plant_seq_list = np.argsort(chosen_list_df.iloc[0]['plant_day_chosen_finetune_list'])

    if always_return_depot:
        route_dict = get_route_planning_depot(cluster_df, num_planter, area_size_array, cluster_plant_seq_list, depot_loc, solution_limit_num = 50, visualize=True)
    else:
        route_dict = get_route_planning_condition(cluster_df, num_planter, area_size_array, cluster_plant_seq_list, depot_loc, solution_limit_num = 50, visualize=True)


if __name__ == '__main__':
    main()