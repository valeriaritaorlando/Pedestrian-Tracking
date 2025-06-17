from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
import numpy as np

    

def calculate_cosine_distance(features1, features2):
  
    # Normalizza i vettori di caratteristiche
    features1 = normalize(features1.reshape(1, -1))
    features2 = normalize(features2.reshape(1, -1))

    # Combina i due vettori in una matrice
    matrix = np.vstack([features1, features2])
    # Calcola la distanza coseno
    distance = cosine_distances(matrix)[0, 1]
    return distance


def calculate_Cost_Matrix_Features(track_features, detection_features, cosine_max):
    cost_matrix = np.zeros((len(track_features), len(detection_features)))

    for i, track_feature in enumerate(track_features):
        for j, detection_feature in enumerate(detection_features):
            distance = calculate_cosine_distance(track_feature, detection_feature)

            if distance < cosine_max:
                cost_matrix[i, j] = distance
            else:
                cost_matrix[i,j] = 1
    
    return cost_matrix

 