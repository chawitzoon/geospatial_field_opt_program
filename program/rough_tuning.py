from .utils.utils_crop_sim_csv import crop_sim_find_best
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d



def process_cropsim_output(crop_sim_dir, variety_list, scenario_num, search_range, plant_day_search_start):
    '''
    function to process the simriw outout date stored as csv
    param: crop_sim_dir: string, string of directory storing simrix output csv files
    param: variety_list: list, list of strings of name of variety
    param: scenario_num: int, number of weather scenario runs
    param: search_range: int, range of dates to try start planting
    param: plant_day_search_start, string or int in the form of yyyyddd

    output: yield_brown_array: numpy array, with shape variety_num*search_range 
    output: plant_day_array: numpy array, with shape search_range storing relative date from plant_day_search_start
    output: harvest_day_array:  numpy array, with shape variety_num*search_range*scenario_num storing relative maturity date from plant_day_search_start
    '''
    variety_num = len(variety_list)
    yield_brown_array = np.empty(shape=(variety_num, search_range, scenario_num))
    plant_day_array = np.empty(shape=(variety_num, search_range, scenario_num))
    harvest_day_array = np.empty(shape=(variety_num, search_range, scenario_num))

    for variety_idx, variety in enumerate(variety_list):
        for scenario_idx in range(scenario_num):
            yield_brown, plant_day, harvest_day = crop_sim_find_best(crop_sim_dir, variety, scenario_idx, plant_day_search_start = plant_day_search_start, plant_range=search_range)
            
            yield_brown_array[variety_idx, :, scenario_idx] = yield_brown
            plant_day_array[variety_idx, :, scenario_idx] = plant_day
            harvest_day_array[variety_idx, :, scenario_idx] = harvest_day


    # planting_day_array for all varieties, all scenario are the same. So, get only one
    plant_day_array = plant_day_array[0, :, 0]

    # average the yield
    yield_brown_array = yield_brown_array.mean(axis=2)

    return yield_brown_array, plant_day_array, harvest_day_array



def precal_harvest_day_dist(harvest_day_array, search_range):
    '''
    function to precalculate the norm distribution fitting for the distribution of maturity date (harvesting date) for all planting dates
    param: harvest_day_array:  numpy array, with shape variety_num*search_range*scenario_num storing relative maturity date from plant_day_search_start
    param: search_range: int, range of dates to try start planting

    output: harvest_day_dist_array: numpy array, array of shape variety_num*search_range*2 storing mean and std of the fitted distribution
    '''
    harvest_day_dist_array = np.empty(shape=(harvest_day_array.shape[0], search_range, 2))
    for variety_idx in range(harvest_day_array.shape[0]):
        for plant_day_chosen in range(search_range):
            mu, std = norm.fit(harvest_day_array[variety_idx, plant_day_chosen, :])

            harvest_day_dist_array[variety_idx, plant_day_chosen, 0] = mu
            harvest_day_dist_array[variety_idx, plant_day_chosen, 1] = std

    return harvest_day_dist_array


# def cropsim_dist_visualize(yield_brown_array, plant_day_array, harvest_day_array, variety_list, plant_day_chosen):
#     '''
#     not used, just for visualization
#     function to visualize average yield and distribution of maturity date (harvesting date) for chosen planting date
#     param: yield_brown_array: numpy array, with shape variety_num*search_range 
#     param: plant_day_array: numpy array, with shape search_range storing relative date from plant_day_search_start
#     param: harvest_day_array:  numpy array, with shape variety_num*search_range*scenario_num storing relative maturity date from plant_day_search_start
#     param: variety_list: list, list of strings of name of variety
#     param: plant_day_chosen: int, relative date from plant_day_search_start

#     '''
#     harvest_day_range = np.arange(np.amin(harvest_day_array), np.amax(harvest_day_array) + 1, 1)

#     # visualize yield of a scenario
#     variety_num = len(variety_list)
#     fig, axs = plt.subplots(variety_num, 2, figsize=(20,10))
#     fig.suptitle('average yield and maturity date wrt. planting date')

#     for i, variety in enumerate(variety_list):
#         axs[i, 0].plot(plant_day_array, yield_brown_array[i,:])
#         axs[i, 0].axvline(x=plant_day_chosen, color='red', label = 'chosen planting date')
#         axs[i, 0].set(xlabel='relative date', ylabel='yield (t/ha)')
#         axs[i, 0].set_title(label= f'average yield of {variety}')
#         axs[i, 0].legend()

#         harvest_day_chosen = harvest_day_array[i, plant_day_chosen, :]
#         axs[i, 1].hist(harvest_day_chosen, bins=harvest_day_range, density=True, facecolor='g', alpha=0.75)
#         # np.arange(min(harvest_day_chosen), max(harvest_day_chosen) + 1, 1)
#         axs[i, 1].set(xlabel='relative date', ylabel='frequency')
#         axs[i, 1].set_title(label= f'maturity date distribution when plant on the chosen planting date for {variety}')


#     plt.tight_layout()
#     plt.show() 

def get_plant_day_base_list(yield_brown_array):
    '''
    function to find ideal planting date for each variety: do smoothing and find maximum points
    param: yield_brown_array: numpy array, with shape variety_num*search_range 

    output: plant_day_base_list: list, list of ideal planting date
    '''
    yield_brown_smooth_array = np.empty(shape=yield_brown_array.shape)
    for i in range(yield_brown_array.shape[0]):
        yield_brown_smooth_array[i,:] = gaussian_filter1d(yield_brown_array[i,:], sigma=4)

        # plt.plot(yield_brown_array[i,:], 'k', label='original data')
        # plt.plot(yield_brown_smooth_array[i,:], '--', label='filtered, sigma=4')
        # plt.legend()
        # plt.grid()
        # plt.show()

    plant_day_base_list = np.argmax(yield_brown_smooth_array, axis=1)
    return plant_day_base_list    



def _get_cropsim_dist(yield_brown_array, plant_day_array, harvest_day_dist_array, variety_idx, plant_day_chosen):
    '''
    function to get average yield and distribution of maturity date (harvesting date) for chosen planting date
    param: yield_brown_array: numpy array, with shape variety_num*search_range 
    param: plant_day_array: numpy array, with shape search_range storing relative date from plant_day_search_start
    param: harvest_day_dist_array: numpy array, array of shape variety_num*search_range*2 storing mean and std of the fitted distribution
    param: variety_idx: int, index of interest variety in variety_list
    param: plant_day_chosen: int, relative date from plant_day_search_start

    output: yield_average_chosen: float (t/ha)
    output: harvest_day_dist_chosen: numpy array, [mu, std] fitted for the maturity date for chosen planting date
    '''
    yield_average_chosen = yield_brown_array[variety_idx, plant_day_chosen]
    harvest_day_dist_chosen = harvest_day_dist_array[variety_idx, plant_day_chosen, :]
    
    return yield_average_chosen, harvest_day_dist_chosen


def _get_cropsim_dist_case(yield_brown_array, plant_day_array, harvest_day_dist_array, variety_chosen_list, plant_day_chosen_list):
    '''
    function to get average yield and distribution of maturity date (harvesting date) for chosen planting date for each clustered area
    param: yield_brown_array: numpy array, with shape variety_num*search_range 
    param: plant_day_array: numpy array, with shape search_range storing relative date from plant_day_search_start # for visualization
    param: harvest_day_dist_array: numpy array, array of shape variety_num*search_range*2 storing mean and std of the fitted distribution
    param: variety_chosen_list: list, list of strings of name of variety for a case
    param: plant_day_chosen_list: list, list of relative dates from plant_day_search_start for a case

    # output: yield_average_chosen_array: numpy array, with shape n_clusters storing the expected yield of each area (t/ha)
    # output: harvest_day_dist_chosen_array: numpy array, with shape n_clusters*2 store (mu, std) of harvesting date distribution of each area
    '''
    n_clusters = len(variety_chosen_list)

    # get the output
    yield_average_chosen_array = np.empty(shape=(n_clusters,))
    harvest_day_dist_chosen_array = np.empty(shape=(n_clusters, 2))
    for i, [variety_chosen, plant_day_chosen] in enumerate(zip(variety_chosen_list, plant_day_chosen_list)):
        yield_average_chosen, harvest_day_dist_chosen = _get_cropsim_dist(yield_brown_array=yield_brown_array, 
                                                                    plant_day_array=plant_day_array, 
                                                                    harvest_day_dist_array=harvest_day_dist_array, 
                                                                    variety_idx=variety_chosen, 
                                                                    plant_day_chosen=plant_day_chosen)

        yield_average_chosen_array[i] = yield_average_chosen
        harvest_day_dist_chosen_array[i,:] = harvest_day_dist_chosen


    
    return yield_average_chosen_array, harvest_day_dist_chosen_array


def get_required_cap_case(yield_brown_array, plant_day_array, harvest_day_array, harvest_day_dist_array, variety_chosen_list, plant_day_chosen_list, area_size_array, cluster_max_planting_range, total_cap_planter, total_cap_harvester, inert_coeff_planter_list, inert_coeff_harvester_list, visualize=False, saved_name=None):
    '''
    function to get total capacity needed for planting and harvesting

    param: yield_brown_array: numpy array, with shape variety_num*search_range 
    param: plant_day_array: numpy array, with shape search_range storing relative date from plant_day_search_start # for visualization
    param: harvest_day_array:  numpy array, with shape variety_num*search_range*scenario_num storing relative maturity date from plant_day_search_start # for visualization only
    param: harvest_day_dist_array: numpy array, array of shape variety_num*search_range*2 storing mean and std of the fitted distribution
    param: variety_chosen_list: list, list of strings of name of variety for a case
    param: plant_day_chosen_list: list, list of relative dates from plant_day_search_start for a case
    param: area_size_array: numpy array, of shape n_clusters storing total area for each cluster
    param: cluster_max_planting_range, int, day range for planting per cluster
    param: total_cap_planter: float, total cap for planter in one day (hec/day)
    param: total_cap_harvester: float, total cap for harvester in one day (hec/day)
    param: inert_coeff_planter_list: numpy array, with shape n_clusters storing ratio to substitute the other wasted capacity such as traveling between fields, cleaning, mantainance, rest time for each cluster
    param: inert_coeff_harvester_list: numpy array, with shape n_clusters storing ratio to substitute the other wasted capacity such as traveling between fields, cleaning, mantainance, rest time for each cluster
    param: visualize: boolean
    param: saved_name: string or None, name of saved image file in result_tuning folder (default=None)

    output: yield_average_chosen_array: numpy array, with shape n_clusters storing the expected yield of each area (t/ha)
    output: plant_required_cap_case, numpy array with shape n_clusters*search_range storing planting capacity needed
    output: harvest_required_cap_case, numpy array with shape n_clusters*harvest_day_range.shape[0] storing harvesting capacity needed (the index is the relative date from first harvest date in all possible simulations)
    '''
    
    yield_average_chosen_array, harvest_day_dist_chosen_array = _get_cropsim_dist_case(yield_brown_array=yield_brown_array,
                                                                                        plant_day_array=plant_day_array, 
                                                                                        harvest_day_dist_array=harvest_day_dist_array, 
                                                                                        variety_chosen_list=variety_chosen_list, 
                                                                                        plant_day_chosen_list=plant_day_chosen_list,)

    # # calculate array storing planting cap and harvesting cap required per day for each area
    # cap_per_day_array = np.divide(area_size_array, cluster_max_planting_range)

    # consider the inert coefficient by putting it in the area_size_array 
    area_size_array_inert_planter = np.divide(area_size_array, inert_coeff_planter_list)
    area_size_array_inert_harvester = np.divide(area_size_array, inert_coeff_harvester_list)

    # calculate number of planting days and harvesting days needed for each field
    num_plant_day_range_array = np.divide(area_size_array_inert_planter,total_cap_planter)
    last_day_plant_cap_per_day_array = [(num_plant_day % 1)*total_cap_planter for num_plant_day in num_plant_day_range_array]
    num_plant_day_range_array = np.ceil(num_plant_day_range_array).astype(int)
    num_harvest_day_range_array = np.divide(area_size_array_inert_harvester, total_cap_harvester)
    last_day_harvest_cap_per_day_array = [(num_harvest_day % 1)*total_cap_harvester for num_harvest_day in num_harvest_day_range_array]
    num_harvest_day_range_array = np.ceil(num_harvest_day_range_array).astype(int)


    # output array
    n_clusters = len(variety_chosen_list)
    search_range = plant_day_array.shape[0]
    harvest_day_range = np.arange(np.amin(harvest_day_array), np.amax(harvest_day_array) + 1, 1)
    # cluster_max_planting_range_half = int(cluster_max_planting_range/2)

    plant_required_cap_case = np.zeros(shape=(n_clusters, search_range))
    harvest_required_cap_case = np.zeros(shape=(n_clusters, harvest_day_range.shape[0]))

    for i, [plant_day_chosen, num_plant_day_range, last_day_plant_cap, num_harvest_day_range, last_day_harvest_cap] in enumerate(zip(plant_day_chosen_list, num_plant_day_range_array, last_day_plant_cap_per_day_array, num_harvest_day_range_array, last_day_harvest_cap_per_day_array)):

        num_plant_day_range_half = int(num_plant_day_range/2)
        # num_harvest_day_range_half = int(num_harvest_day_range/2)
        

        start_harvest_date = round(harvest_day_dist_chosen_array[i,0]) - int(np.amin(harvest_day_array))# mu of fitted normal distribution
        harvest_required_cap_case[i, start_harvest_date: start_harvest_date+num_harvest_day_range].fill(total_cap_harvester)
        harvest_required_cap_case[i, start_harvest_date+num_harvest_day_range-1] = last_day_harvest_cap

        if plant_day_chosen < num_plant_day_range_half:

            plant_required_cap_case[i, plant_day_chosen:plant_day_chosen+num_plant_day_range].fill(total_cap_planter)
            plant_required_cap_case[i, plant_day_chosen+num_plant_day_range-1]  = last_day_plant_cap
        
        elif plant_day_chosen > search_range - num_plant_day_range_half:

            plant_required_cap_case[i, plant_day_chosen-num_plant_day_range:plant_day_chosen].fill(total_cap_planter)
            plant_required_cap_case[i, plant_day_chosen-1]  = last_day_plant_cap
        
        else:
            if num_plant_day_range%2==0:
                plant_required_cap_case[i, plant_day_chosen-num_plant_day_range_half+1:plant_day_chosen+num_plant_day_range_half].fill(total_cap_planter)
                plant_required_cap_case[i, plant_day_chosen+num_plant_day_range_half] = last_day_plant_cap

            else:
                plant_required_cap_case[i, plant_day_chosen-num_plant_day_range_half:plant_day_chosen+num_plant_day_range_half].fill(total_cap_planter)
                plant_required_cap_case[i, plant_day_chosen+num_plant_day_range_half] = last_day_plant_cap

    # visualize yield of a scenario
    if saved_name != None or visualize:
        fig, axs = plt.subplots(n_clusters+1, 2, figsize=(20,10))
        fig.suptitle('average yield and maturity date wrt. planting date')

        for i, [variety_idx, plant_day_chosen] in enumerate(zip(variety_chosen_list, plant_day_chosen_list)):
            axs[i, 0].plot(plant_day_array, yield_brown_array[variety_idx,:])
            axs[i, 0].axvline(x=plant_day_chosen, color='red', label = 'chosen planting date')

            ax1 = axs[i, 0].twinx()
            ax1.bar(plant_day_array, height=plant_required_cap_case[i,:], alpha=0.5, color='green',label='plant cap needed')
            axs[i, 0].set(xlabel='relative date', ylabel='yield (t/ha)')
            axs[i, 0].set_title(label= f'average yield of area {i}, planting {variety_idx}')
            ax1.set(ylabel='PlantCapNeed(ha)')
            axs[i, 0].legend()

            harvest_day_chosen = harvest_day_array[variety_idx, plant_day_chosen, :]
            axs[i, 1].hist(harvest_day_chosen, bins=harvest_day_range, density=True, facecolor='g', alpha=0.75)

            # plot normal distribution
            p = norm.pdf(harvest_day_range, harvest_day_dist_chosen_array[i,0], harvest_day_dist_chosen_array[i,1])
            axs[i, 1].plot(harvest_day_range, p, 'k', linewidth=2)
            axs[i, 1].axvline(x=harvest_day_dist_chosen_array[i,0], color='orange', label = 'chosen planting date')

            ax2 = axs[i, 1].twinx()
            ax2.bar(harvest_day_range, height=harvest_required_cap_case[i,:], alpha=0.5, color='orange', label='harvest cap needed')

            axs[i, 1].set(xlabel='relative date', ylabel='frequency')
            axs[i, 1].set_title(label= f'maturity date distribution when plant on the chosen planting date \n for area {i}, planting {variety_idx}')
            ax2.set(ylabel='HarvestCapNeed(ha)')


        axs[-1, 0].bar(plant_day_array, height=np.sum(plant_required_cap_case, axis=0), alpha=0.5,  color='green', label='plant cap needed')
        axs[-1, 0].set(ylabel='TotalPlantCapNeed(ha)')  

        axs[-1, 1].bar(harvest_day_range, height=np.sum(harvest_required_cap_case, axis=0), alpha=0.5, color='orange', label='harvest cap needed')
        axs[-1, 1].set(ylabel='TotalHarvestCapNeed(ha)')



        plt.tight_layout()
        if saved_name != None:
            plt.savefig(f'result_tuning/{saved_name}.png')
        if visualize:
            plt.show()

    return yield_average_chosen_array, plant_required_cap_case, harvest_required_cap_case

def cal_obj_func(yield_average_chosen_array, plant_required_cap_case, harvest_required_cap_case, area_size_array, variety_chosen_list, variety_price_list):
    '''
    param: yield_average_chosen_array: numpy array, with shape n_clusters storing the expected yield of each area (t/ha)
    param: plant_required_cap_case, numpy array with shape n_clusters*search_range storing planting capacity needed
    param: harvest_required_cap_case, numpy array with shape n_clusters*harvest_day_range.shape[0] storing harvesting capacity needed (the index is the relative date from first harvest date in all possible simulations)
    param: area_size_array: numpy array, of shape n_clusters storing total area for each cluster
    param: variety_chosen_list: list, list of strings of name of variety for a case
    param: variety_price_list, list, list of prices for each variety in yen/ton

    output: objective_value: int
    Objective value = revenue - harvest capacity overused (in yen) - planting capacity overused (in yen)
    '''
    # process price_list to be for each area
    area_price_list = [variety_price_list[i] for i in variety_chosen_list]

    # calculate revenue
    revenue = 0
    plant_overused = 0
    harvest_overused = 0
    for yield_average, area, price in zip(yield_average_chosen_array, area_size_array, area_price_list):
        revenue += yield_average*area*price #[ton/ha]*[ha]*[yen/ton]

    # looking through each day
    for i in range(plant_required_cap_case.shape[1]):
        nonzero_indices = np.nonzero(plant_required_cap_case[:,i])
        if nonzero_indices[0].shape[0] != 0:
            average_price = 0
            average_cap = 0
            average_yield = 0
            for j in nonzero_indices[0]:
                average_price += area_price_list[j]
                average_cap += plant_required_cap_case[j,i]
                average_yield += yield_average_chosen_array[j]
            
            average_price = average_price/nonzero_indices[0].shape[0]
            average_cap = average_cap/nonzero_indices[0].shape[0]
            average_yield = average_yield/nonzero_indices[0].shape[0]
            plant_overused += (nonzero_indices[0].shape[0] - 1) * average_yield * average_cap * average_price # assume we can plant only one area, the other (all - 1) is the panelty with average price


    # looking through each day
    for i in range(harvest_required_cap_case.shape[1]):
        nonzero_indices = np.nonzero(harvest_required_cap_case[:,i])
        if nonzero_indices[0].shape[0] > 1:
            average_price = 0
            average_cap = 0
            average_yield = 0
            for j in nonzero_indices[0]:
                average_price += area_price_list[j]
                average_cap += harvest_required_cap_case[j,i]
                average_yield += yield_average_chosen_array[j]

            average_price = average_price/nonzero_indices[0].shape[0]
            average_cap = average_cap/nonzero_indices[0].shape[0]
            average_yield = average_yield/nonzero_indices[0].shape[0]
            harvest_overused += (nonzero_indices[0].shape[0] - 1) * average_yield * average_cap * average_price # assume we can plant only one area, the other (all - 1) is the panelty with average price

    objective_value = revenue - plant_overused - harvest_overused
    return objective_value
    

def get_obj_func_value(plant_day_chosen_list, variety_chosen_list, cluster_max_planting_range, variety_price_list, total_cap_planter, total_cap_harvester, area_size_array, yield_brown_array, plant_day_array, harvest_day_array, harvest_day_dist_array, inert_coeff_planter_list, inert_coeff_harvester_list, visualize=False):
    '''
    function to calculate objective function value from plant_day_chosen_list and variety_chosen_list in main program

    param: plant_day_chosen_list: list, list of relative dates from plant_day_search_start for a case
    param: variety_chosen_list: list, list of strings of name of variety for a case
    param: cluster_max_planting_range, int, day range for planting per cluster
    param: variety_price_list, list, list of prices for each variety in yen/ton
    param: total_cap_planter: float, total cap for planter in one day (hec/day)
    param: total_cap_harvester: float, total cap for harvester in one day (hec/day)
    param: area_size_array: numpy array, of shape n_clusters storing total area for each cluster
    param: yield_brown_array: numpy array, with shape variety_num*search_range 
    param: plant_day_array: numpy array, with shape search_range storing relative date from plant_day_search_start # for visualization
    param: harvest_day_array:  numpy array, with shape variety_num*search_range*scenario_num storing relative maturity date from plant_day_search_start # for visualization only
    param: harvest_day_dist_array: numpy array, array of shape variety_num*search_range*2 storing mean and std of the fitted distribution
    param: inert_coeff_planter_list: numpy array, with shape n_clusters storing ratio to substitute the other wasted capacity such as traveling between fields, cleaning, mantainance, rest time for each cluster
    param: inert_coeff_harvester_list: numpy array, with shape n_clusters storing ratio to substitute the other wasted capacity such as traveling between fields, cleaning, mantainance, rest time for each cluster
    param: visualize: boolean (default=False)

    output: objective_value: int
    '''
    
    yield_average_chosen_array, plant_required_cap_case, harvest_required_cap_case = get_required_cap_case(yield_brown_array=yield_brown_array,
                                                                                                            plant_day_array=plant_day_array,
                                                                                                            harvest_day_array=harvest_day_array,
                                                                                                            harvest_day_dist_array=harvest_day_dist_array,
                                                                                                            variety_chosen_list=variety_chosen_list,
                                                                                                            plant_day_chosen_list=plant_day_chosen_list,
                                                                                                            area_size_array=area_size_array,
                                                                                                            cluster_max_planting_range = cluster_max_planting_range,
                                                                                                            total_cap_planter = total_cap_planter,
                                                                                                            total_cap_harvester = total_cap_harvester,
                                                                                                            inert_coeff_planter_list=inert_coeff_planter_list,
                                                                                                            inert_coeff_harvester_list=inert_coeff_harvester_list,
                                                                                                            visualize=visualize)

    obj_func_value = cal_obj_func(yield_average_chosen_array, plant_required_cap_case, harvest_required_cap_case, area_size_array, variety_chosen_list, variety_price_list)
    return obj_func_value

if __name__ == '__main__':

    pass