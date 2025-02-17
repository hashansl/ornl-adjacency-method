import geopandas as gpd
import pandas as pd
import utills.invr as invr
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class AdjacencySimplex:
    """
    A class to process a GeoDataFrame, filter and sort it based on a variable, 
    compute adjacency relationships, and form a simplicial complex.
    """
    
    def __init__(self, gdf, variable, threshold=None, filter_method='up'):
        """
        Initialize with a GeoDataFrame.
        
        Parameters:
        - gdf: GeoDataFrame containing geographic and attribute data.
        - variable: Column name used for filtering and sorting.
        - threshold: Tuple (min, max) for filtering values within a range.
        - filter_method: Sorting method, either 'up' (descending) or 'down' (ascending).
        """
        self.gdf = gdf
        self.variable = variable
        self.filter_method = filter_method
        self.threshold = threshold
        self.filtered_df = None
        self.adjacent_counties_dict = None
        self.simplicial_complex = None

    def filter_sort_gdf(self):
        """
        Filter and sort the GeoDataFrame based on the specified variable and method.
        """
        gdf = self.gdf.copy()

        # Sort the DataFrame based on the specified method
        if self.filter_method == 'up':
            gdf = gdf.sort_values(by=self.variable, ascending=True)
        elif self.filter_method == 'down':
            # get the max value
            max_value = gdf[self.variable].max()
            # invert the values - Assuming negative values are not present
            gdf[self.variable] = max_value - gdf[self.variable]
            gdf = gdf.sort_values(by=self.variable, ascending=True)
        else:
            raise ValueError("Invalid filter method. Use 'up' or 'down'.")
        
        # this need to be done before filtering
        gdf['sortedID'] = range(len(gdf))

        # this for the below filter
        filtered_df = gdf.copy()

        # Apply threshold filtering if specified
        if self.threshold:
            filtered_df = filtered_df[(filtered_df[self.variable] >= self.threshold[0]) &
                                      (filtered_df[self.variable] <= self.threshold[1])]

        # Convert DataFrame to GeoDataFrame
        filtered_df = gpd.GeoDataFrame(filtered_df, geometry='geometry')

        # Set Coordinate Reference System (CRS)
        filtered_df.crs = "EPSG:4326"

        self.filtered_df = filtered_df

        # this reurns a filtered dataframe and the original dataframe with the sortedID
        return filtered_df, gdf

    def calculate_adjacent_countries(self):
        """
        Compute adjacency relationships between geographic entities.
        """
        # Ensure filter_sort_gdf() has been executed
        if not hasattr(self, 'filtered_df') or not isinstance(self.filtered_df, gpd.GeoDataFrame):
            raise ValueError("Run filter_sort_gdf() before calling this method.")

        # Perform spatial join to find adjacent entities
        adjacent_entities = gpd.sjoin(self.filtered_df, self.filtered_df, predicate='intersects', how='left')

        # Remove self-intersections
        adjacent_entities = adjacent_entities.query('sortedID_left != sortedID_right')

        # Group by entity and store adjacent entities in a list
        adjacent_entities = adjacent_entities.groupby('sortedID_left')['sortedID_right'].apply(list).reset_index()
        adjacent_entities.rename(columns={'sortedID_left': 'county', 'sortedID_right': 'adjacent'}, inplace=True)

        # Create adjacency dictionary
        adjacent_dict = dict(zip(adjacent_entities['county'], adjacent_entities['adjacent']))

        # Merge adjacency information with the original dataset
        merged_df = pd.merge(adjacent_entities, self.filtered_df, left_on='county', right_on='sortedID', how='left')
        
        # Convert to GeoDataFrame
        merged_df = gpd.GeoDataFrame(merged_df, geometry='geometry')
        merged_df.crs = "EPSG:4326"

        # Store results
        self.adjacent_counties_dict = adjacent_dict
        self.merged_df = merged_df

    def form_simplicial_complex(self):
        """
        Construct a simplicial complex using adjacency relationships.
        """
        if not hasattr(self, 'adjacent_counties_dict'):
            raise ValueError("Run calculate_adjacent_countries() before calling this method.")
        
        max_dimension = 3  # Define maximum dimension for the simplicial complex
        simplicial_complex = invr.incremental_vr([], self.adjacent_counties_dict, max_dimension, list(self.adjacent_counties_dict.keys()))
        
        self.simplicial_complex = simplicial_complex
        return simplicial_complex
    
    @staticmethod
    def fig2img(fig):
        #convert matplot fig to image and return it

        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        return img
    

    def plot_simplicial_complex(self, save_dir=None, list_gif=[]):
        """
        Plot the simplicial complex.
        """
        if not hasattr(self, 'simplicial_complex'):
            raise ValueError("Run form_simplicial_complex() before calling this method.")
        
        # Needed data: simplices, original_df, variable, list_gif = []
        # Extract city centroids

        # sortedID is in filtered_df
        city_coordinates = {row['sortedID']: np.array((row['geometry'].centroid.x, row['geometry'].centroid.y)) for _, row in self.filtered_df.iterrows()}

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_axis_off() 

        # Plot the original GeoDataFrame without any filtration
        self.gdf.plot(ax=ax, edgecolor='black', linewidth=0.3, color="white")

        # Plot the centroid of the large square with values
        for _, row in self.gdf.iterrows():
            centroid = row['geometry'].centroid
            text_to_display = f"{row[self.variable]:.2f}"
            plt.text(centroid.x, centroid.y, text_to_display, fontsize=7, ha='center', color="black")

        # Plot edges and triangles from simplices
        for edge_or_triangle in self.simplicial_complex:

            # color sub regions based on how it enter the simplcial complex in adjacency method
            if len(edge_or_triangle) == 1:

                vertex = edge_or_triangle[0]
                # geometry = self.filtered_df.iterrows().loc[self.filtered_df.iterrows()['sortedID'] == vertex, 'geometry'].values[0]
                geometry = self.filtered_df[self.filtered_df['sortedID'] == vertex]['geometry'].values[0]
                ax.add_patch(Polygon(np.array(geometry.exterior.coords), closed=True, color='orange', alpha=0.3))
                img = self.fig2img(fig)
                list_gif.append(img)

            elif len(edge_or_triangle) == 2:
                # Plot an edge
                ax.plot(*zip(*[city_coordinates[vertex] for vertex in edge_or_triangle]), color='red', linewidth=2)
                img = self.fig2img(fig)
                list_gif.append(img)
            elif len(edge_or_triangle) == 3:
                # Plot a triangle
                ax.add_patch(plt.Polygon([city_coordinates[vertex] for vertex in edge_or_triangle], color='green', alpha=0.2))
                img = self.fig2img(fig)
                list_gif.append(img)

            #can change above code block
            plt.close(fig)

        if save_dir is None:
            list_gif[0].save(f'./adj_simplex_{self.variable}_{self.filter_method}.gif', save_all=True,append_images=list_gif[1:],optimize=False,duration=600,loop=0)
        else:
            list_gif[0].save(f'{save_dir}/adj_simplex_{self.variable}_{self.filter_method}.gif', save_all=True,append_images=list_gif[1:],optimize=False,duration=600,loop=0)
            print("GIF created and saved in the current directory.")

