import json
import numpy as np
from feature_extraction import strutturized_detection
from tracking import update_tracks
from file_utils import find_association_file
import os
import sys
import subprocess

#--------------  -------------------
#Funzioni utili per testare il modello dopo la fase di validazione 

#Video di test 
video_list = ['MOT17-10', 'MOT17-11', 'MOT17-13']


# Percorsi settaggio dove trovo i video, bounding box e features
directory_video = 'C:/Users/valer/OneDrive/Desktop/progetto/videos'
sys.path.append(os.path.join(os.getcwd(), "TrackEval"))


# Carica Json ground truth ground truth
def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_txt(txt_path):
    with open(txt_path, 'r') as f:
        data = f.read()
    return data


def run_mot_challenge(trackers_to_eval):
    command = [
   
        'python', 'C:/Users/valer/OneDrive/Desktop/progetto/progetto/TrackEval/scripts/run_mot_challenge.py',
        '--BENCHMARK', 'MOT17',
        '--SPLIT_TO_EVAL','test', 
        '--METRICS', 'HOTA', 'CLEAR','Identity',
        '--TRACKERS_TO_EVAL', trackers_to_eval
    ]

    
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("Errore nell'esecuzione dello script")
    else:
        print("Script eseguito correttamente")


def apply_model_test(video_list):
    global directory_video

    nome_tracker = 'Track_30'

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
            
                #Passo i valori migliori trovati nella validazione 
                tracks = update_tracks(detections, tracks, 0.7, 0.3, 0.6, 0.2, 10)
            
            for track_id, track_dict in tracks.items():
                x1, y1, x2, y2 = track_dict['box']

                if track_dict['T_lost'] == 0 :
                    track_list.append(f"{int(frame_key)+1},{round(float(track_id),1)},{x1},{y1},{x2-x1},{y2-y1},-1,-1,-1,-1") 

        
        directory_data =f'C:/Users/valer/OneDrive/Desktop/progetto/progetto/TrackEval/data/trackers/mot_challenge/MOT17-test/{nome_tracker}/data'
        
        if not os.path.exists(directory_data):
            os.makedirs(directory_data)

        video_txt = video + '.txt'
        path_det = os.path.join(directory_data, video_txt)

        with open(path_det, 'w') as f:
            for elemento in track_list:
                f.write(elemento + '\n')

    
    run_mot_challenge(nome_tracker)

# Run il codice per il test 
apply_model_test(video_list)


