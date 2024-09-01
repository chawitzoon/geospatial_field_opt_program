from program.rough_tuning import get_obj_func_value
from program.utils.utils_ga_optimizer import GA
import functools

def _get_output_func(variety_chosen_list, cluster_max_planting_range, variety_price_list, total_cap_planter, total_cap_harvester, area_size_array, yield_brown_array, plant_day_array, harvest_day_array, harvest_day_dist_array, inert_coeff_planter_list, inert_coeff_harvester_list):
    '''
    function to create the objective function with only interest optimized variable (used only once for each setting, so we can neglect the computation efficiency)
    '''
    output_func = functools.partial(get_obj_func_value,
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
                                    )
    return output_func



def finetune_plant_day_chosen(chosen_list_df, finetune_range, cluster_max_planting_range, variety_price_list, n_clusters, total_cap_planter, total_cap_harvester, area_size_array, search_range, yield_brown_array, plant_day_array, harvest_day_array, harvest_day_dist_array, inert_coeff_planter_list, inert_coeff_harvester_list, pop_size = 50, num_generations = 50, parent_ratio = 0.5, visualize=False):
    
    '''
    param: chosen_list_df: DataFrame, storing data with columns ['variety_chosen_list', 'plant_day_chosen_list', 'obj_func_value']
    param: finetune_range: int, range of number of shifted dates from the base platning date
    param: cluster_max_planting_range: int, day range for planting per cluster
    param: variety_price_list: list, list of prices for each variety in yen/ton
    param: n_clusters: int, number of clusters
    param: total_cap_planter: float, total cap for planter in one day (hec/day)
    param: total_cap_harvester: float, total cap for harvester in one day (hec/day)
    param: area_size_array: numpy array, of shape n_clusters storing total area for each cluster
    param: search_range: int, range of dates to try start planting
    param: yield_brown_array: numpy array, with shape variety_num*search_range 
    param: plant_day_array: numpy array, with shape search_range storing relative date from plant_day_search_start # for visualization
    param: harvest_day_array:  numpy array, with shape variety_num*search_range*scenario_num storing relative maturity date from plant_day_search_start # for visualization only
    param: harvest_day_dist_array: numpy array, array of shape variety_num*search_range*2 storing mean and std of the fitted distribution
    param: inert_coeff_planter_list: numpy array, with shape n_clusters storing ratio to substitute the other wasted capacity such as traveling between fields, cleaning, mantainance, rest time for each cluster
    param: inert_coeff_harvester_list: numpy array, with shape n_clusters storing ratio to substitute the other wasted capacity such as traveling between fields, cleaning, mantainance, rest time for each cluster
    param: visualize: boolean (default=False)

    param: pop_size: int, for GA (default=50)
    param: num_generations: int, for GA (default = 50)
    param: parent_ratio: int, for GA (default = 0.5)

    output: chosen_list_df: DataFrame, storing data with columns ['variety_chosen_list', 'plant_day_chosen_list', 'obj_func_value', 'plant_day_chosen_finetune_list', 'obj_func_value_finetune'] sorted by the obejctive function value
    '''
    chosen_list_df = chosen_list_df.reset_index() 

    # do ga optimization for each variety_chosen_list (each row in chosen_list_df)
    plant_day_chosen_finetune_list = []
    obj_func_value_finetune = []
    for index, row in chosen_list_df.iterrows():
        print(f'finetuning chosen variety sample {index}..')
        optimized_var_range = [[max(0, int(row['plant_day_chosen_list'][i]-finetune_range/2)), min(search_range, int(row['plant_day_chosen_list'][i]+finetune_range/2))] for i in range(n_clusters)]

        output_func = _get_output_func(variety_chosen_list=row['variety_chosen_list'],
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
                                        inert_coeff_harvester_list=inert_coeff_harvester_list)

        ga = GA(optimized_var_range=optimized_var_range,
                optimized_var_len=n_clusters,
                output_func=output_func,
                pop_size = pop_size, 
                num_generations = num_generations, 
                parent_ratio = parent_ratio,
                extremum = 'maximize', 
                save_fname= f'finetuning_variety_chosen_list_{index}')

        best_var, cost_max = ga.update_pop(plot=visualize)
        ga.save_result()

        obj_func_value_finetune.append(cost_max)
        plant_day_chosen_finetune_list.append(best_var.astype(int))

    chosen_list_df['plant_day_chosen_finetune_list'] = plant_day_chosen_finetune_list
    chosen_list_df['obj_func_value_finetune'] = obj_func_value_finetune

    chosen_list_df = chosen_list_df.sort_values(by = 'obj_func_value_finetune', ascending=False)

    return chosen_list_df