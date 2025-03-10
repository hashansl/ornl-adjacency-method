# Import libraries
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import gudhi
from tqdm import tqdm
from persim import PersistenceImager
import invr
import matplotlib as mpl

from PIL import Image
import io
from matplotlib.patches import Polygon

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Matplotlib default settings
mpl.rcParams.update(mpl.rcParamsDefault)

# Utility functions
def get_folders(location):
    """Get list of folders in a directory."""
    return [name for name in os.listdir(location) if os.path.isdir(os.path.join(location, name))]

def generate_adjacent_counties(dataframe, variable_name):
    """Generate adjacent counties based on given dataframe and variable."""
    filtered_df = dataframe
    adjacent_counties = gpd.sjoin(filtered_df, filtered_df, predicate='intersects', how='left')
    adjacent_counties = adjacent_counties.query('sortedID_left != sortedID_right')
    adjacent_counties = adjacent_counties.groupby('sortedID_left')['sortedID_right'].apply(list).reset_index()
    adjacent_counties.rename(columns={'sortedID_left': 'county', 'sortedID_right': 'adjacent'}, inplace=True)
    adjacencies_list = adjacent_counties['adjacent'].tolist()
    county_list = adjacent_counties['county'].tolist()
    merged_df = pd.merge(adjacent_counties, dataframe, left_on='county', right_on='sortedID', how='left')
    merged_df = gpd.GeoDataFrame(merged_df, geometry='geometry')
    return adjacencies_list, merged_df, county_list

def form_simplicial_complex(adjacent_county_list, county_list):
    """Form a simplicial complex based on adjacent counties."""
    max_dimension = 3
    V = invr.incremental_vr([], adjacent_county_list, max_dimension, county_list)
    return V

def create_variable_folders(base_path, variables):
    """Create folders for each variable."""
    for variable in variables:
        os.makedirs(os.path.join(base_path, variable), exist_ok=True)
    print('Done creating folders for each variable')


def fig2img(fig):
     #convert matplot fig to image and return it

     buf = io.BytesIO()
     fig.savefig(buf)
     buf.seek(0)
     img = Image.open(buf)
     return img

def plot_simplicial_complex(dataframe, simplices,variable, list_gif=[]):
    # Extract city centroids
    city_coordinates = {row['sortedID']: np.array((row['geometry'].centroid.x, row['geometry'].centroid.y)) for _, row in dataframe.iterrows()}

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_off() 

    # Plot the "dataframe" DataFrame
    dataframe.plot(ax=ax, edgecolor='black', linewidth=0.3, color="white")

    # Plot the centroid of the large square with values
    for _, row in dataframe.iterrows():
        centroid = row['geometry'].centroid
        text_to_display = f"{row[variable]:.3f}"
        plt.text(centroid.x, centroid.y, text_to_display, fontsize=10, ha='center', color="black")

    # Plot edges and triangles from simplices
    for edge_or_triangle in simplices:

        # color sub regions based on how it enter the simplcial complex in adjacency method
        if len(edge_or_triangle) == 1:

            vertex = edge_or_triangle[0]
            geometry = dataframe.loc[dataframe['sortedID'] == vertex, 'geometry'].values[0]
            ax.add_patch(Polygon(np.array(geometry.exterior.coords), closed=True, color='orange', alpha=0.3))
            img = fig2img(fig)
            list_gif.append(img)

        elif len(edge_or_triangle) == 2:
            # Plot an edge
            ax.plot(*zip(*[city_coordinates[vertex] for vertex in edge_or_triangle]), color='red', linewidth=2)
            img = fig2img(fig)
            list_gif.append(img)
        elif len(edge_or_triangle) == 3:
            # Plot a triangle
            ax.add_patch(plt.Polygon([city_coordinates[vertex] for vertex in edge_or_triangle], color='green', alpha=0.2))
            img = fig2img(fig)
            list_gif.append(img)

        #can change above code block
        plt.close(fig)

    return list_gif

def process_state(state, selected_variables, selected_variables_with_censusinfo, base_path, PERSISTENCE_IMAGE_PARAMS, INFINITY):
    """Process data for a given state."""
    svi_od_path = os.path.join(data_path, state, state + '.shp')
    svi_od = gpd.read_file(svi_od_path)
    # # for variable in selected_variables:
    #     # svi_od = svi_od[svi_od[variable] != -999]

        
    svi_od_filtered_state = svi_od[selected_variables_with_censusinfo].reset_index(drop=True)

    # Get the unique counties
    unique_county_stcnty = svi_od_filtered_state['STCNTY'].unique()

    for county_stcnty in unique_county_stcnty:
        # Filter the dataframe to include only the current county
        county_svi_df = svi_od_filtered_state[svi_od_filtered_state['STCNTY'] == county_stcnty]
    
        for variable_name in selected_variables:
            df_one_variable = county_svi_df[['STCNTY', variable_name, 'geometry']]
            df_one_variable = df_one_variable.sort_values(by=variable_name)
            df_one_variable['sortedID'] = range(len(df_one_variable))
            df_one_variable = gpd.GeoDataFrame(df_one_variable, geometry='geometry')
            df_one_variable.crs = "EPSG:3395"

            adjacencies_list, adjacent_counties_df, county_list = generate_adjacent_counties(df_one_variable, variable_name)
            adjacent_counties_dict = dict(zip(adjacent_counties_df['county'], adjacent_counties_df['adjacent']))
            county_list = adjacent_counties_df['county'].tolist()
            simplices = form_simplicial_complex(adjacent_counties_dict, county_list)

            st = gudhi.SimplexTree()
            st.set_dimension(2)

            for simplex in simplices:
                if len(simplex) == 1:
                    st.insert([simplex[0]], filtration=0.0)
            
            for simplex in simplices:
                if len(simplex) == 2:
                    last_simplex = simplex[-1]
                    filtration_value = df_one_variable.loc[df_one_variable['sortedID'] == last_simplex, variable_name].values[0]
                    st.insert(simplex, filtration=filtration_value)

            for simplex in simplices:
                if len(simplex) == 3:
                    last_simplex = simplex[-1]
                    filtration_value = df_one_variable.loc[df_one_variable['sortedID'] == last_simplex, variable_name].values[0]
                    st.insert(simplex, filtration=filtration_value)

            st.compute_persistence()
            persistence = st.persistence()

            intervals_dim0 = st.persistence_intervals_in_dimension(0)
            intervals_dim1 = st.persistence_intervals_in_dimension(1)
            pdgms = [[birth, death] for birth, death in intervals_dim1 if death < np.inf]

            # add interval dim 0  to the pdgms
            for birth, death in intervals_dim0:
                if death < np.inf:
                    pdgms.append([birth, death])

                # elif death == np.inf:
                    # pdgms.append([birth, INFINITY])
                

            save_path = os.path.join(base_path, variable_name, county_stcnty)

            if len(pdgms) > 0:
                
                # print(f'Processing {variable_name} for {county_stcnty}')
                # print(f'Number of persistence diagrams: {len(pdgms)}')
                # print(intervals_dim1)
                # for i in range(len(intervals_dim1)):
                #     if np.isinf(pdgms[i][1]):
                #         pdgms[i][1] = 1
                #     if np.isinf(pdgms[i][0]):
                #         pdgms[i][0] = 1

                pimgr = PersistenceImager(pixel_size=0.01)
                pimgr.fit(pdgms)

                pimgr.pixel_size = PERSISTENCE_IMAGE_PARAMS['pixel_size']
                pimgr.birth_range = PERSISTENCE_IMAGE_PARAMS['birth_range']
                pimgr.pers_range = PERSISTENCE_IMAGE_PARAMS['pers_range']
                pimgr.kernel_params = PERSISTENCE_IMAGE_PARAMS['kernel_params']

                pimgs = pimgr.transform(pdgms)
                pimgs = np.rot90(pimgs, k=1) 

                np.save(save_path, pimgs)

                # plt.figure(figsize=(2.4, 2.4))
                # plt.imshow(pimgs, cmap='viridis')  # Assuming 'viridis' colormap, change as needed
                # plt.axis('off')  # Turn off axis
                # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust subplot parameters to remove borders
                
                # # plt.savefig(f'{base_path}/{variable_name}/{county_stcnty}.png',dpi=300)
                # plt.close()

def process_single_county(state, county_code, variable_name, selected_variables_with_censusinfo, base_path, PERSISTENCE_IMAGE_PARAMS, INFINITY):
    """Process data for a given county."""

    svi_path = os.path.join(data_path, state, state + '.shp')
    svi_df = gpd.read_file(svi_path)

    svi_filtered_state = svi_df[selected_variables_with_censusinfo].reset_index(drop=True)

    # Filter the dataframe to include only the current county
    county_svi_df = svi_filtered_state[svi_filtered_state['STCNTY'] == county_code]

    list_gif = []

    df_one_variable = county_svi_df[['STCNTY', variable_name, 'geometry']]
    df_one_variable = df_one_variable.sort_values(by=variable_name)
    df_one_variable['sortedID'] = range(len(df_one_variable))
    df_one_variable = gpd.GeoDataFrame(df_one_variable, geometry='geometry')
    df_one_variable.crs = "EPSG:3395"
    adjacencies_list, adjacent_counties_df, county_list = generate_adjacent_counties(df_one_variable, variable_name)
    adjacent_counties_dict = dict(zip(adjacent_counties_df['county'], adjacent_counties_df['adjacent']))
    county_list = adjacent_counties_df['county'].tolist()
    simplices = form_simplicial_complex(adjacent_counties_dict, county_list)

    print(f'Done forming simplicial complex for {variable_name} in {county_code}')

    list_gif = plot_simplicial_complex(df_one_variable, simplices, variable_name,list_gif)
    list_gif[0].save(f'{base_path}/{variable_name}_{county_code}.gif', save_all=True,append_images=list_gif[1:],optimize=False,duration=600,loop=0) #png also works




# Define the main function
if __name__ == "__main__":
    # Main execution
    base_path = '/Users/h6x/ORNL/git/WORKSTAION GIT/ornl-adjacency-method/overdose_graphs/plots'
    data_path = '/home/h6x/git_projects/ornl-svi-data-processing/processed_data/SVI/SVI2018_MIN_MAX_SCALED_MISSING_REMOVED'

    # EP_PCI not included in the list of selected variables
    selected_variables = [
         'EP_POV','EP_UNEMP', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65', 'EP_AGE17', 'EP_DISABL', 
        'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 'EP_GROUPQ'
    ]
  
    selected_variables_with_censusinfo = ['FIPS', 'STCNTY'] + selected_variables + ['geometry']

    # PERSISTENCE_IMAGE_PARAMS = {
    #     'pixel_size': 0.001,
    #     'birth_range': (0.0, 1.00),
    #     'pers_range': (0.0, 0.40),
    #     'kernel_params': {'sigma': 0.0003}
    # }

    PERSISTENCE_IMAGE_PARAMS = {
        'pixel_size': 0.01,
        'birth_range': (0.0, 1.00),
        'pers_range': (0.0, 1.00),
        'kernel_params': {'sigma': 0.0003}
    }

    INF_DELTA = 0.1
    # INFINITY = (PERSISTENCE_IMAGE_PARAMS['birth_range'][1] - PERSISTENCE_IMAGE_PARAMS['birth_range'][0]) * INF_DELTA
    INFINITY = 1

    # state = 'WV'
    # county_code = '54037'
    # variable = 'EP_UNEMP'

    state = 'MI'
    county_code = '26147'
    variable = 'EP_POV'

    process_single_county(state, county_code, variable, selected_variables_with_censusinfo, base_path, PERSISTENCE_IMAGE_PARAMS, INFINITY)

    print(f'County {county_code} in state {state} processed.')

