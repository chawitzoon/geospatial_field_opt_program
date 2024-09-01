import numpy as np
import pyproj
from shapely.ops import transform
from shapely.geometry import Point, point


def _find_utm_band(coord):
    '''
    internal function to get current epsg code

    param: coord, numpy array, list, tuple of (lon,lat)
    output: epsg_code, string
    '''
    lon = coord[0]
    lat = coord[1]
    utm_band = str(int((np.floor((lon + 180) / 6 ) % 60) + 1))
    if len(utm_band) == 1:
        utm_band = '0'+ utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return epsg_code

def _convert_wgs_to_utm(coord):
    '''
    internal function to convert from wgs to utm

    param: coord, numpy array, lis, or tuple of (lon, lat)
    output: coord_utm, list of lon-lat in utm
    '''
    
    coord_point = Point(coord[0], coord[1])
    wgs84 = pyproj.CRS('EPSG:4326') #  CRS.WGS84 degree to meter for buffer
    utm_code = _find_utm_band(coord) # use the first coord to find epsg code
    utm = pyproj.CRS(f'EPSG:{utm_code}') # metre
    project = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform
    coord_utm = transform(project, coord_point)
    coord_utm = [coord_utm.x, coord_utm.y] # change to normal list
    return coord_utm

def _polygon_size_formula(polygon_coord_metre):
    '''
    internal function to calculate polygon size in square meter

    param: polygon_coord_metre, list of lists (or tuples, or numpy arrays) of coord_utm for a field polygon
    output: size, float of polygon size in square meter
    '''
    sum = 0
    for i in range(len(polygon_coord_metre)-1):
        sum += polygon_coord_metre[i][0] * polygon_coord_metre[i+1][1] - polygon_coord_metre[i][1] * polygon_coord_metre[i+1][0]
    size = np.abs(sum)/2 * 0.0001 # in hecter
    return size


def cal_field_size(polygon_coord):
    '''
    function for calculating area from polygon coordinates
    param: polygon_coord, list of lists (or tuples, or numpy arrays) coordinate in [[lon1,lat1], [lon2, lat2],...]
    return: size, float in hectare
    '''
    polygon_coord_metre = []
    for coord in polygon_coord:
        coord_utm = _convert_wgs_to_utm(coord)
        polygon_coord_metre.append(coord_utm)

    size  = _polygon_size_formula(polygon_coord_metre)
    return size


if __name__ == '__main__':
    pass