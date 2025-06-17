import configparser
import cv2
import json
from feature_extraction import strutturized_detection
from tracking import update_tracks
from extract_and_save import extract_save_bbox, extract_save_features
from file_utils import find_video_path, find_association_file, create_output_video_path
import os

from tensorflow.keras.applications import ResNet50  # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
import torch


# Definizione dei modelli
# DETR
detr_model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)

# RESNET
resNet_model = ResNet50(weights='imagenet', include_top=False)
# Aggiungo Global Average Pooling dopo l'output del modello base
x = resNet_model.output
x = GlobalAveragePooling2D()(x)
model_resNet = Model(inputs=resNet_model.input, outputs=x)

# Definiamo il percorso del video su cui vogliamo effettuare il tracking
nome_video = 'MOT17-02'
directory = 'C:/Users/valer/OneDrive/Desktop/progetto/videos'
directory_video= os.path.join(directory,nome_video)

# trovo il percorso del video
video_path = find_video_path(directory_video, nome_video)

# creo i percorsi per i file .json delle bbox e features
bbox_path = find_association_file(directory_video, nome_video, 'bounding_boxes')
features_path = find_association_file(directory_video, nome_video, 'features')
output_video_path = create_output_video_path(directory_video, nome_video)

# Leggi i valori dal file seqinfo.ini
config = configparser.ConfigParser()
seqinfo_path = os.path.join(directory_video, 'seqinfo.ini')
config.read(seqinfo_path)
desired_width = int(config['Sequence']['imWidth'])
desired_height = int(config['Sequence']['imHeight'])
frame_rate = int(config['Sequence']['frameRate'])

# estraggo bbox e features
extract_save_bbox(video_path, bbox_path, detr_model, seqinfo_path)
extract_save_features(video_path, bbox_path, features_path, model_resNet, seqinfo_path)

# Carica bounding_boxes dal file JSON
with open(bbox_path, 'r') as f:
    bounding_boxes = json.load(f)
print(f"Bounding boxes caricate da {bbox_path}")

# Carica features_dict dal file JSON
with open(features_path, 'r') as f:
    features_dict = json.load(f)
print(f"Features dictionary caricato da {features_path}")


cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (desired_width, desired_height))


#--- Generazione video con visual tracking ---
frame_num = 0
tracks = {}
det_output = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize del fotogramma
    resized_frame = cv2.resize(frame, (desired_width, desired_height))
    
    # Seleziono le detections del frame corrente
    frame_key = str(frame_num)
    detections = strutturized_detection(features_dict[frame_key], bounding_boxes[frame_key])

    # Fase di assegnazione degli ID
    if frame_key == '0':
        tracks = detections
    else:
        tracks = update_tracks(detections, tracks, w_IoU=0.7, w_features=0.3,IoU_min=0.6,cosine_max=0.2,max_frame_to_keep=10)


    for track_id, track_dict in tracks.items():
        x1, y1, x2, y2 = track_dict['box']

        if track_dict['T_lost'] == 0 :
            cv2.rectangle(resized_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(resized_frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(resized_frame)
    frame_num += 1

cap.release()
out.release()
cv2.destroyAllWindows()