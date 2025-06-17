import json
import numpy as np
from feature_extraction import strutturized_detection
from tracking import update_tracks
from file_utils import find_association_file
import os
import sys
import subprocess

#Funzioni utili per la validazione dei parametri 
#-------------- Parametri di Validation -------------------
max_frame_to_keep_list = [10, 15, 30]  
IoU_min_list = [0.6, 0.8, 0.9]  
cosine_max_list = [0.4, 0.2, 0.1]  
IoU_weight = [0.6, 0.7, 0.8,0.9]  # dentro update

#Dataset scelti per la validazione 
video_list = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09']
# Lista per memorizzare i migliori 5 tracker
top_5_trackers = []

# Percorsi settaggio dove trovo bounding box ed i video 
directory_video = 'C:/Users/valer/OneDrive/Desktop/progetto/videos'
sys.path.append(os.path.join(os.getcwd(), "TrackEval"))


# Carica Json ground truth 
def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_txt(txt_path):
    with open(txt_path, 'r') as f:
        data = f.read()
    return data



params = [
    {'w_IoU': w_IoU, 'IoU_min': IoU_min, 'cosine_max': cosine_max, 'max_frame_to_keep': max_frame_to_keep}
    for w_IoU in IoU_weight 
    for IoU_min in IoU_min_list for cosine_max in cosine_max_list
    for max_frame_to_keep in max_frame_to_keep_list
]


#---------------------------

def run_mot_challenge(trackers_to_eval):
    command = [
   
        'python', 'C:/Users/valer/OneDrive/Desktop/progetto/progetto/TrackEval/scripts/run_mot_challenge.py',
        '--BENCHMARK', 'MOT17',
        '--SPLIT_TO_EVAL','train', 
        '--METRICS', 'HOTA', 'CLEAR', 
        '--TRACKERS_TO_EVAL', trackers_to_eval
    ]

    
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("Errore nell'esecuzione dello script")
    else:
        print("Script eseguito correttamente")

#-----------------------------
def apply_model(video_list, params):
    global directory_video
    count_tracker = 0

    for param in params:
        nome_tracker = f'Track_{count_tracker}'

        for video in video_list:
            tracks = {}
            track_list = []
            
            directory = os.path.join(directory_video, video)

            bbox_path = find_association_file(directory, video, 'bounding_boxes')
            features_path = find_association_file(directory, video, 'features')

            # Carica bounding_boxes dal file JSON Bounding box
            with open(bbox_path, 'r') as f:
                bounding_boxes = json.load(f)
            print(f"Bounding boxes caricate da {bbox_path}")

            # Carica features_dict dal file JSON features
            with open(features_path, 'r') as f:
                features_dict = json.load(f)
            print(f"Features dictionary caricato da {features_path}")

            for frame_key in bounding_boxes:
                detections = strutturized_detection(features_dict[frame_key], bounding_boxes[frame_key])

                if frame_key == '0':
                    tracks = detections
                else:
                    w_features = 1 - param['w_IoU']
                    tracks = update_tracks(detections, tracks, param['w_IoU'], w_features, param['IoU_min'], param['cosine_max'], param['max_frame_to_keep'])
                
                for track_id, track_dict in tracks.items():
                    x1, y1, x2, y2 = track_dict['box']

                    if track_dict['T_lost'] == 0 :
                        track_list.append(f"{int(frame_key)+1},{round(float(track_id),1)},{x1},{y1},{x2-x1},{y2-y1},-1,-1,-1,-1") 
                        
            
            directory_data =f'C:/Users/valer/OneDrive/Desktop/progetto/progetto/TrackEval/data/trackers/mot_challenge/MOT17-train/{nome_tracker}/data'
            
            if not os.path.exists(directory_data):
                os.makedirs(directory_data)

            video_txt = video + '.txt'
            path_det = os.path.join(directory_data, video_txt)

            with open(path_det, 'w') as f:
                for elemento in track_list:
                    f.write(elemento + '\n')




        print(f"Eseguendo con TRACKERS_TO_EVAL={nome_tracker}")
        print(f"\n{nome_tracker} con parametri w_IoU: {param['w_IoU']}, Iou_min: {param['IoU_min']}, cosine_max: {param['cosine_max']}, max_frame: {param['max_frame_to_keep']}")
        run_mot_challenge(nome_tracker)
        
        
        with open(os.path.join(directory_data, 'params_track.txt'), 'w') as f:
            f.write(f"w_IoU: {param['w_IoU']}\n")
            f.write(f"IoU_min: {param['IoU_min']}\n")
            f.write(f"cosine_max: {param['cosine_max']}\n")
            f.write(f"max_frame: {param['max_frame_to_keep']}\n")

        count_tracker += 1

    


# Run il codice
apply_model(video_list, params)


#Funzioni per trovare i migliori tracker dopo il processo di validazione 
def update_top_trackers(hota, tracker_name):
    top_5_trackers.append((hota, tracker_name))
    top_5_trackers.sort(reverse=True, key=lambda x: x[0])
    if len(top_5_trackers) > 5:
        top_5_trackers.pop()

for i in range(108):
    file_name = f'C:/Users/valer/OneDrive/Desktop/progetto/progetto/TrackEval/data/trackers/mot_challenge/MOT17-train/Track_{i}/pedestrian_summary.txt'
    with open(file_name, 'r') as infile:
        lines = infile.readlines()
        
        second_line_values = lines[1].strip().split()
        hota = float(second_line_values[0])
        update_top_trackers(hota, f'Track_{i}')


print('Migliori 5 tracker:')
for hota, tracker_name in top_5_trackers:
    print(f'{tracker_name} con HOTA: {hota}')