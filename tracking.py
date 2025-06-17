from calculate_IoU import calculate_Cost_Matrix_IoU
from calculate_distance_features import calculate_Cost_Matrix_Features
from scipy.optimize import linear_sum_assignment




def update_tracks(detections, tracks, w_IoU, w_features,IoU_min,cosine_max, max_frame_to_keep):
    track_ids = list(tracks.keys())
    track_features = [tracks[track_id]['feature'] for track_id in track_ids]
    track_boxs = [tracks[track_id]['box'] for track_id in track_ids]

    detection_features = [detections[detection_id]['feature'] for detection_id in detections]
    detection_boxs = [detections[detection_id]['box'] for detection_id in detections]

    
    cost_matrix_IoU = calculate_Cost_Matrix_IoU(track_boxs, detection_boxs,IoU_min)
    cost_matrix_features = calculate_Cost_Matrix_Features(track_features, detection_features,cosine_max)

    # UNIONE MATRICE DEI COSTI
    cost_matrix = w_IoU * cost_matrix_IoU + w_features * cost_matrix_features

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    new_tracks = {}

    nomi_righe = []
    for i, track_id in enumerate(track_ids):
        nomi_righe.append(track_id)
        # riconosce la persona con l'id associato
        if i in row_ind:
            detection_index = col_ind[list(row_ind).index(i)]
            new_tracks[track_id] = {'feature': detections[str(detection_index)]['feature'],
                                    'box': detections[str(detection_index)]['box'],
                                    'T_lost': 0}

        # gestisco occlusioni
        elif tracks[track_id]['T_lost']  <= max_frame_to_keep:  #frame_counter[track_id] <= max_frame_to_keep:
            new_tracks[track_id] = tracks[track_id]
            new_tracks[track_id]['T_lost'] += 1
  

    # Associamo nuovi ID alle detection che sono stare rilevale la prima volta
    for detection_index in range(len(detections)):
        if detection_index not in col_ind:
            max_id = max([int(id) for id in new_tracks.keys()]) if new_tracks else 0
            new_id = str(max_id + 1)
            new_tracks[new_id] = {'feature': detections[str(detection_index)]['feature'],
                                'box': detections[str(detection_index)]['box'],
                                    'T_lost': 0}

    return new_tracks