import numpy as np


def lower_neighbors(adjacencies, vertex):
    return [v for v in adjacencies[vertex] if v < vertex]


def incremental_vr(V, adjacencies, maxDimension,county_list):
    Vnew = list(V)
    # print("county list len",len(county_list))
    # print("adjacencies len",len(adjacencies))
    # for vertex in np.arange(len(adjacencies)):
    for vertex in county_list:
        # print("vertex",vertex)
        # print("adjacencies",adjacencies[vertex])
        N = sorted(lower_neighbors(adjacencies, vertex))
        add_cofaces(adjacencies, maxDimension, [vertex], N, Vnew)
    return Vnew


def add_cofaces(adjacencies, maxDimension, face, N, V):
    if sorted(face) not in V:
        V.append(sorted(face))
    if len(face) >= maxDimension:
        return
    else:
        for vertex in N:
            coface = list(face)
            coface.append(vertex)
            M = list(set(N) & set(lower_neighbors(adjacencies, vertex)))
            add_cofaces(adjacencies, maxDimension, coface, M, V)




# Pseudocode
# Given unique census tracts in a county. Variable value for each census tract and a single filtration for a county.
# Sort there census tracts based on the variable value and select the census tracts below the filtration value.
# For the selected census tracts, find the adjacent counties.
# FOR each selected census tract(lower to the filtration value)
#   Add the census tract to the simplicial complex.
#   IF there are lower(variable value) adjacent counties to the census tract in the simplicial complex:
#       from lower adjacent census tract to the highes adjacent census tract create faces(within max dimention)and add that to the simplicial complex.
#   END IF
# END FOR



# # Function: incremental_vr
# # Purpose: Build the Vietoris-Rips (VR) simplicial complex incrementally.
# Input:
#     - V: Current simplicial complex (initially empty)
#     - adjacencies: List of adjacent counties
#     - maxDimension: Maximum dimension of the complex
#     - county_list: List of counties
# Output:
#     - Updated simplicial complex Vnew

# Steps:
# 1. Initialize a copy of the simplicial complex (Vnew).
# 2. For each county (vertex) in the county list:
#     a. Find the lower neighbors of the vertex (counties that are adjacent and have a lower index).
#     b. Call the add_cofaces function to add cofaces of the current vertex to the complex.
# 3. Return the updated simplicial complex.


# # Function: add_cofaces
# # Purpose: Recursively add cofaces to the simplicial complex.
# Input:
#     - adjacencies: List of adjacent counties
#     - maxDimension: Maximum dimension of the complex
#     - face: Current face (subset of counties)
#     - N: List of lower neighbors for the current vertex
#     - V: Simplicial complex being constructed
# Output:
#     - Updated simplicial complex (V)

# Steps:
# 1. If the current face is not already in the complex, add it.
# 2. If the face size reaches the maximum dimension, return (base case for recursion).
# 3. For each vertex in the lower neighbors (N):
#     a. Form a coface by adding the vertex to the current face.
#     b. Find the new lower neighbors that can be combined with the vertex.
#     c. Recursively add cofaces for the new face.