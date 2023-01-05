import pandas as pd
import numpy as np
import pyodbc as odbc
import math
import openrouteservice as ors
import logging
from k_means_constrained import KMeansConstrained
from multiprocessing.pool import ThreadPool
from ortools.graph.python import linear_sum_assignment



class Tech_Assignment_Optimizer:
    def __init__(self, regions):
        self._regions = regions
        self._df_raw_prop_data_all = pd.DataFrame()
        self._df_raw_tech_data_all = pd.DataFrame()
        self._df_results = pd.DataFrame()
        self._client = ors.Client(base_url='http://localhost:8080/ors')
        self._logger = self.setup_logging()


    def setup_logging(self):

        logger = logging.getLogger('Tech_Assignment_Optimizer')
        logger.setLevel(logging.DEBUG)

        # Create a file and console log handler and set the log levels to DEBUG and overwrite file
        fh = logging.FileHandler('log.log', 'w+')
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger



    def get_data(self, connection):

        query = """
                SELECT 
                    [PropertyID] as propertyid
                  , [BranchName] as branchname
                  , [CensusTractGeoId]
                  , [latitude] as property_lat
                  , [longitude] as property_long
                  , [SUBDIVISION_NAME] as subdivision_name
                  
               FROM [ModelTestBed].[dbo].[MSRportfolio]
               
               where [Still Under Mgmt]='Yes'
        """
        self._logger.info(": Pulling property data....")
        self._df_raw_prop_data_all = pd.io.sql.read_sql(query, connection)

        query = """
                select distinct 
                      ta.branchname
                    , ta.Worker as tech_name
                    , ta.[Email - Primary Work] as tech_email 
                    ,ta.TechZipCode as tech_zip
                    , ZipShp.LAT as tech_lat 
                    , ZipShp.lon as tech_long
                    
                from IRSPublish..tv_technicianAddress ta
                
                join thirdpartydata..Shp_ZipCode_dt ZipShp
                    on ta.TechZipCode = ZipShp.ZipCode
                    
                join modeltestbed..msrportfolio msr
                    on msr.branchname like ta.Branchname+'%'
                    and [Still Under Mgmt] = 'Yes'
                    
                join thirdpartydata..Shp_Blocks_dt blkShp
                    on blkShp.fips+blkShp.censusTract+blkshp.BlockGroup = msr.census_block
        """
        self._logger.info(": Pulling technician data....")
        self._df_raw_tech_data_all = pd.io.sql.read_sql(query, connection)


    def get_centers(self, mini_pod_centers, mini_pod_labels):

        df_centers = pd.DataFrame(mini_pod_centers)
        df_centers.rename(columns={0: 'miniPod_lat', 1: 'miniPod_long'}, inplace=True)

        df_labels = pd.DataFrame(set(mini_pod_labels))
        df_labels = df_labels.rename(columns={0: 'clusterLabel'})

        df_centers['miniPod_id'] = df_labels['clusterLabel']
        df_centers['miniPod_id'] = df_centers['miniPod_id']

        return df_centers


    def mini_pod_center_to_property(self, row):

        index, value = row
        try:
            coords = ((value["miniPod_long"], value["miniPod_lat"]), (value['property_long'], value['property_lat']))
            routes = self._client.directions(coords, optimize_waypoints=True, radiuses=15000)
            result = {
                'propertyid': value['propertyid'],
                'branchname': value['branchname'],
                'CensusTractGeoId': value['CensusTractGeoId'],
                'subdivision_name': value['subdivision_name'],
                'property_lat': value['property_lat'],
                'property_long': value['property_long'],
                'miniPod_id': value['miniPod_id'],
                'bigPod_id': value['bigPod_id'],
                'miniPod_lat': value['miniPod_lat'],
                'miniPod_long': value['miniPod_long'],
                'bigPod_lat': value['bigPod_lat'],
                'bigPod_long': value['bigPod_long'],
                'miniPodcenter2prop_driving_distance(meters)': routes['routes'][0]['summary']['distance'],
                'miniPodcenter2prop_driving_time(secs)': routes['routes'][0]['summary']['duration']
            }
        except:
            result = {'invalid_row': index}
        return result


    def big_pod_center_to_property(self, row):

        index, value = row
        try:
            coords = ((value["bigPod_long"], value["bigPod_lat"]), (value['property_long'], value['property_lat']))
            routes = self._client.directions(coords, optimize_waypoints=True, radiuses=50000)
            result = {
                'propertyid': value['propertyid'],
                'branchname': value['branchname'],
                'CensusTractGeoId': value['CensusTractGeoId'],
                'subdivision_name': value['subdivision_name'],
                'property_lat': value['property_lat'],
                'property_long': value['property_long'],
                'miniPod_id': value['miniPod_id'],
                'bigPod_id': value['bigPod_id'],
                'miniPod_lat': value['miniPod_lat'],
                'miniPod_long': value['miniPod_long'],
                'bigPod_lat': value['bigPod_lat'],
                'bigPod_long': value['bigPod_long'],
                'miniPodcenter2prop_driving_distance(meters)': value['miniPodcenter2prop_driving_distance(meters)'],
                'miniPodcenter2prop_driving_time(secs)': value['miniPodcenter2prop_driving_time(secs)'],
                'bigPodcenter2prop_driving_distance(meters)': routes['routes'][0]['summary']['distance'],
                'bigPodcenter2prop_driving_time(secs)': routes['routes'][0]['summary']['duration']
            }
        except:
            result = {'invalid_row': index}
        return result


    def tech_to_center(self, row):

        index, value = row
        try:
            coords = ((value["tech_long"], value["tech_lat"]), (value['miniPod_long'], value['miniPod_lat']))
            routes = self._client.directions(coords, optimize_waypoints=True, radiuses=15000)
            result = {
                'tech_zip': value['tech_zip'],
                'tech_name': value['tech_name'],
                'tech_email': value['tech_email'],
                'tech_lat': value['tech_lat'],
                'tech_long': value['tech_long'],
                'miniPod_lat': value['miniPod_lat'],
                'miniPod_long': value['miniPod_long'],
                'miniPod_id': value['miniPod_id'],
                'driving_distance': routes['routes'][0]['summary']['distance'],
                'driving_time': routes['routes'][0]['summary']['duration']
            }
        except:
            result = {'invalid_row': index}
        return result


    def tech_to_property(self, row):

        index, value = row
        try:
            coords = ((value["tech_long"], value["tech_lat"]), (value['property_long'], value['property_lat']))
            routes = self._client.directions(coords, optimize_waypoints=True, radiuses=50000)
            result = {
                'propertyid': value['propertyid'],
                'branchname': value['branchname'],
                'subdivision_name': value['subdivision_name'],
                'property_lat': value['property_lat'],
                'property_long': value['property_long'],
                'miniPod_id': value['miniPod_id'],
                'bigPod_id': value['bigPod_id'],
                'miniPod_lat': value['miniPod_lat'],
                'miniPod_long': value['miniPod_long'],
                'bigPod_lat': value['bigPod_lat'],
                'bigPod_long': value['bigPod_long'],
                'miniPodcenter2prop_driving_distance(meters)': value['miniPodcenter2prop_driving_distance(meters)'],
                'miniPodcenter2prop_driving_time(secs)': value['miniPodcenter2prop_driving_time(secs)'],
                'bigPodcenter2prop_driving_distance(meters)': value['bigPodcenter2prop_driving_distance(meters)'],
                'bigPodcenter2prop_driving_time(secs)': value['bigPodcenter2prop_driving_time(secs)'],
                'tech_name': value['tech_name'],
                'tech_zip': value['tech_zip'],
                'tech_email': value['tech_email'],
                'tech_lat': value['tech_lat'],
                'tech_long': value['tech_long'],
                'techZip2prop_driving_distance(meters)': routes['routes'][0]['summary']['distance'],
                'techZip2prop_driving_time(secs)': routes['routes'][0]['summary']['duration']
            }
        except:
            result = {'invalid_row': index}
        return result


    def techt_to_mini_pod(self, row):

        index, value = row
        try:
            coords = ((value["tech_long"], value["tech_lat"]), (value['property_long'], value['property_lat']))
            routes = self._client.directions(coords, optimize_waypoints=True, radiuses=50000)
            result = {
                'propertyid': value['propertyid'],
                'branchname': value['branchname'],
                'subdivision_name': value['subdivision_name'],
                'property_latitude': value['property_lat'],
                'property_longitude': value['property_long'],
                'miniPod_id': value['miniPod_id'],
                'bigPod_id': value['bigPod_id'],
                'miniPod_lat': value['miniPod_lat'],
                'miniPod_long': value['miniPod_long'],
                'bigPod_lat': value['bigPod_lat'],
                'bigPod_long': value['bigPod_long'],
                'miniPod2prop_driving_distance(meters)': value['miniPodcenter2prop_driving_distance(meters)'],
                'miniPod2prop_driving_time(secs)': value['miniPodcenter2prop_driving_time(secs)'],
                'bigPod2prop_driving_distance(meters)': value['bigPodcenter2prop_driving_distance(meters)'],
                'bigPod2prop_driving_time(secs)': value['bigPodcenter2prop_driving_time(secs)'],
                'tech_name': value['tech_name'],
                'tech_zip': value['tech_zip'],
                'tech_email': value['tech_email'],
                'tech_lat': value['tech_lat'],
                'tech_long': value['tech_long'],
                'techZip2prop_driving_distance(meters)': value['techZip2prop_driving_distance(meters)'],
                'techZip2prop_driving_time(secs)': value['techZip2prop_driving_time(secs)'],
                'techZip2miniPod_driving_distance(meters)': routes['routes'][0]['summary']['distance'],
                'techZip2miniPod_driving_time(secs)': routes['routes'][0]['summary']['duration']
            }
        except:
            result = {'invalid_row': index}
        return result


    def assign_all_regions(self):

        for region in self._regions:
            self._logger.info('\n')
            self._logger.info(': Running Region: ' + region + '.....')
            self.assign_by_region(region)


    def assign_by_region(self, region):

        #===================================
        # raw data by region
        #===================================

        df_raw_prop_data = self._df_raw_prop_data_all[self._df_raw_prop_data_all['branchname'] == region]
        df_raw_tech_data = self._df_raw_tech_data_all[self._df_raw_tech_data_all['branchname'] == region]

        #===================================
        # lat long data by region
        #===================================

        df_lat_long = df_raw_prop_data.copy()
        df_lat_long.drop(df_lat_long.loc[df_lat_long['property_long'].isnull()].index, inplace=True)
        df_lat_long.drop(df_lat_long.loc[df_lat_long['property_lat'].isnull()].index, inplace=True)

        #===================================
        # K-means
        #===================================
        df_tech = df_raw_tech_data.copy()

        mini_pod_kmeans = KMeansConstrained(n_clusters = len(df_tech), size_min = round(len(df_lat_long) / len(df_tech)) - 3)

        df_lat_long['miniPod_id'] = mini_pod_kmeans.fit_predict(df_lat_long[['property_lat', 'property_long']])
        df_lat_long['miniPod_id'] = df_lat_long['miniPod_id'] + 1

        mini_pod_centers = mini_pod_kmeans.cluster_centers_
        mini_pod_labels = mini_pod_kmeans.predict(df_lat_long[['property_lat', 'property_long']])
        mini_pod_labels = mini_pod_labels + 1

        number_of_clusters = math.ceil(len(df_lat_long) / 1500)

        big_pod_kmeans = KMeansConstrained(n_clusters = number_of_clusters, size_min=round(len(df_lat_long) / number_of_clusters) - 3)

        df_lat_long['bigPod_id'] = big_pod_kmeans.fit_predict(df_lat_long[['property_lat', 'property_long']])
        df_lat_long['bigPod_id'] = df_lat_long['bigPod_id'] + 1

        big_pod_centers = big_pod_kmeans.cluster_centers_
        big_pod_labels = big_pod_kmeans.predict(df_lat_long[['property_lat', 'property_long']])
        big_pod_labels = big_pod_labels + 1

        df_collected = df_lat_long.copy()

        #===================================
        # censustract split
        #===================================
        df_collected['fips'] = [x[0:5] for x in df_collected['CensusTractGeoId'].astype(str)]

        df_assign = df_collected.groupby(['fips', 'subdivision_name'])['miniPod_id'].nunique().to_frame().reset_index()
        df_assign = df_assign[df_assign['miniPod_id'] > 1]
        df_assign = df_assign.rename(columns={'miniPod_id': 'nMiniPods'})

        df_property_n = df_collected.groupby(['fips', 'subdivision_name', 'miniPod_id'])['propertyid'].count().to_frame().reset_index()
        df_property_n = df_property_n.rename(columns={'propertyid': 'nProp'})

        df_to_choose = df_property_n.merge(df_assign, left_on=['fips', 'subdivision_name'], right_on=['fips', 'subdivision_name'])
        df_to_choose['miniPod_rank'] = df_to_choose.groupby(['fips', 'subdivision_name'])['nProp'].rank(method='first', ascending=False)
        df_to_choose = df_to_choose[df_to_choose['miniPod_rank'] == 1][['fips', 'subdivision_name', 'miniPod_id']]

        # now we just merge back
        #==========================
        df_assigned = df_collected.merge(df_to_choose, how = 'left', left_on = ['fips', 'subdivision_name'], right_on=  ['fips', 'subdivision_name'])
        df_assigned = df_assigned.rename(columns = {'miniPod_id_x': 'prevMiniPodID', 'miniPod_id_y': 'miniPod_id'})

        df_new_assigned = df_assigned[df_assigned['miniPod_id'].notnull()].drop('prevMiniPodID', axis=1)

        property_list = list(df_new_assigned['propertyid'])

        df_collected.drop(df_collected.loc[df_collected['propertyid'].isin(property_list)].index, inplace=True)
        df_collected = pd.concat([df_collected, df_new_assigned])
        df_collected = df_collected.drop('fips', axis = 1)
        df_collected['propertyid'] = df_collected['propertyid'].astype(int)
        df_collected['miniPod_id'] = df_collected['miniPod_id'].astype(int)
        df_collected['bigPod_id'] = df_collected['bigPod_id'].astype(int)

        #===================================
        # get centers
        #===================================
        df_collected = df_collected.merge(self.get_centers(mini_pod_centers, mini_pod_labels), how='left', on='miniPod_id')

        #===================================
        # pod centers
        #===================================
        df_pod_centers = pd.DataFrame(big_pod_centers)
        df_pod_centers['bigPod_id'] = np.unique(big_pod_labels)
        df_pod_centers.rename(columns={0: 'bigPod_lat', 1: 'bigPod_long'}, inplace=True)

        df_collected = df_collected.merge(df_pod_centers, how='left', on='bigPod_id')

        #===================================
        # mini pod to center
        #===================================
        pool = ThreadPool(14)
        results = pool.map(self.mini_pod_center_to_property, df_collected.iterrows())
        pool.close()

        df_collected = pd.DataFrame([x_ for x_ in results if not x_.get('invalid_row')])

        #===================================
        # big pod to center
        #===================================
        pool = ThreadPool(14)
        results = pool.map(self.big_pod_center_to_property, df_collected.iterrows())
        pool.close()

        df_collected = pd.DataFrame([x_ for x_ in results if not x_.get('invalid_row')])

        #===================================
        # tech to center
        #===================================
        df_tech_to_center = df_tech.merge(self.get_centers(mini_pod_centers, mini_pod_labels), how='cross')

        pool = ThreadPool(14)
        results = pool.map(self.tech_to_center, df_tech_to_center.iterrows())
        pool.close()

        df_tech_to_center = pd.DataFrame([x_ for x_ in results if not x_.get('invalid_row')])
        df_tech_to_center_final = df_tech_to_center.groupby(['tech_email', 'miniPod_id'])['driving_time'].sum().to_frame().reset_index().sort_values(['tech_email', 'miniPod_id'])

        df_tech_assigned = df_tech_to_center_final.copy()

        df_tech_to_center_final = df_tech_to_center_final.set_index(['miniPod_id', 'tech_email'], drop=True).unstack('miniPod_id')
        df_tech_to_center_final.columns = df_tech_to_center_final.columns.droplevel(0)

        #===================================
        # drive time arrays
        #===================================
        driving_time_array = df_tech_to_center_final.to_numpy()

        end_nodes_unraveled_array, start_nodes_unraveled_array = np.meshgrid(np.arange(driving_time_array.shape[1]), np.arange(driving_time_array.shape[0]))
        start_nodes_array = start_nodes_unraveled_array.ravel()
        end_nodes_array = end_nodes_unraveled_array.ravel()
        arc_costs_array = driving_time_array.ravel()

        #===================================
        # status assignment
        #===================================
        assignment = linear_sum_assignment.SimpleLinearSumAssignment()
        assignment.add_arcs_with_cost(start_nodes_array, end_nodes_array, arc_costs_array)

        status = assignment.solve()
        if status == assignment.OPTIMAL:
            self._logger.info(f'Total driving time for all Techs: {assignment.optimal_cost()}', 'mins\n')

            for i in range(0, assignment.num_nodes()):
                self._logger.info(f'Technician {i} assigned to mini pod: {assignment.right_mate(i)}' + f'  (Driving Time(mins) = {assignment.assignment_cost(i)})')

        elif status == assignment.INFEASIBLE:
            self._logger.info('No assignment is possible.')

        elif status == assignment.POSSIBLE_OVERFLOW:
            self._logger.info('Some input costs are too large and may cause an integer overflow.')

        optimized_time_list = []
        cluster_list = []
        for i in range(0, assignment.num_nodes()):
            optimized_time_list.append(assignment.assignment_cost(i))
            cluster_list.append(assignment.right_mate(i))

        df_tech_assigned = df_tech_assigned.merge(df_tech_to_center[['tech_email','tech_lat', 'tech_long', 'miniPod_lat', 'miniPod_long']])

        lat_list = []
        long_list = []
        df = pd.DataFrame()
        for i in df_tech['tech_name']:
            lat = df_tech.loc[df_tech['tech_name'] == i, 'tech_lat'].iloc[0]
            long = df_tech.loc[df_tech['tech_name'] == i, 'tech_long'].iloc[0]
            lat_list.append(lat)
            long_list.append(long)

        df['tech_lat'] = lat_list
        df['tech_long'] = long_list

        df_optimized_time_tech_assignment = pd.DataFrame({
            'tech_email': list(df_tech_assigned['tech_email'].unique()),
            'tech_lat': list(df['tech_lat']),
            'tech_long': list(df['tech_long']),
            'miniPod_lat': list(df_tech_assigned['miniPod_lat'].unique()),
            'miniPod_long': list(df_tech_assigned['miniPod_long'].unique()),
            'miniPod_id': cluster_list,
            'optimizedDrivingTimes(mins)': optimized_time_list,
        })

        df_optimized_time_tech_assignment['miniPod_id'] = df_optimized_time_tech_assignment['miniPod_id'] + 1
        df_tech_assignment = df_optimized_time_tech_assignment.copy()
        df_tech_assignment = df_tech_assignment.merge(df_tech[['tech_name', 'tech_zip', 'tech_email']], how='left', on='tech_email')
        df_collected = df_collected.merge(df_tech_assignment[['miniPod_id', 'tech_name', 'tech_email', 'tech_zip', 'tech_lat', 'tech_long']], how='left', on='miniPod_id')


        #===================================
        # tech to property
        #=================================
        pool = ThreadPool(14)
        results = pool.map(self.tech_to_property, df_collected.iterrows())
        pool.close()

        df_collected = pd.DataFrame([x_ for x_ in results if not x_.get('invalid_row')])

        # ===================================
        # tech to mini pod
        # =================================
        pool = ThreadPool(14)
        results = pool.map(self.techt_to_mini_pod, df_collected.iterrows())
        pool.close()

        # ===================================
        # final df
        # =================================
        df_collected_final = pd.DataFrame([x_ for x_ in results if not x_.get('invalid_row')])
        df_collected_final = df_collected_final[['propertyid',
                                           'branchname',
                                           'subdivision_name',
                                           'property_latitude',
                                           'property_longitude',
                                           'tech_lat',
                                           'tech_long',
                                           'bigPod_lat',
                                           'bigPod_long',
                                           'miniPod_lat',
                                           'miniPod_long',
                                           'miniPod_id',
                                           'bigPod_id',
                                           'bigPod2prop_driving_distance(meters)',
                                           'bigPod2prop_driving_time(secs)',
                                           'miniPod2prop_driving_distance(meters)',
                                           'miniPod2prop_driving_time(secs)',
                                           'techZip2prop_driving_distance(meters)',
                                           'techZip2prop_driving_time(secs)',
                                           'techZip2miniPod_driving_distance(meters)',
                                           'techZip2miniPod_driving_time(secs)',
                                           'tech_zip',
                                           'tech_name',
                                           'tech_email'
                                           ]]

        df_collected_final['bigPod_id'] = df_collected_final['bigPod_id'].astype(int)

        try:
            self._df_results = pd.concat([self._df_results, df_collected_final])

        except:
            print('ERROR: ' + region + 'failed')




if __name__ == '__main__':

    connection = odbc.connect("Driver={SQL Server Native Client 11.0};"
                        "Server=DFILSQL02.insightlabs.amherst.com;"
                        "Trusted_Connection=yes;")


    regions = ['Dallas', 'Houston', 'San Antonio']
    Optimizer = Tech_Assignment_Optimizer(regions=regions)
    Optimizer.get_data(connection=connection)
    Optimizer.assign_all_regions()

    Optimizer._df_results.to_csv('output.csv')

    print('Done')