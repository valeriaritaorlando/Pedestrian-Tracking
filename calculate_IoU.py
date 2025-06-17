# Calcolo della IoU tra due bounding box
import numpy as np

def calculate_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / float(union_area)
    return iou

 
def calculate_Cost_Matrix_IoU(track_bboxes, detection_bboxes, IoU_min):
    cost_matrix = np.zeros((len(track_bboxes), len(detection_bboxes)))
    # Costruisco la matrice dei costi usando la IoU 
    for i, track_bbox in enumerate(track_bboxes):
        for j, detection_bbox in enumerate(detection_bboxes):
            iou = calculate_iou(track_bbox, detection_bbox)
            if iou < IoU_min : 
                cost_matrix[i, j] = 1
            else :
                cost_matrix[i, j] = 1 - iou  # Converti IoU in metrica di costo

    return cost_matrix

#Un valore di IoU alto indica un'alta corrispondenza. Ma la linear_sum_assignment prende i valori minori,motivo 
#per cui memorizzo 1-iou invece che iou 