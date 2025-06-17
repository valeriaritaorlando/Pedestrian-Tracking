from tensorflow.keras.applications.resnet50 import preprocess_input  # type: ignore
import numpy as np
import cv2
import configparser

# Definizione della funzione di crop
def crop_bounding_boxes(frame, bounding_boxes):
    cropped_images = []

    for box in bounding_boxes:
        x, y, w, h = map(int, box)
        cropped_image = frame[y:y+h, x:x+w]
        cropped_images.append(cropped_image)

    return cropped_images

def process_video(video_path, bounding_boxes_dict, seqinfo_path):

    # Leggi i valori dal file seqinfo.ini
    config = configparser.ConfigParser()
    config.read(seqinfo_path)

    desired_width = int(config['Sequence']['imWidth'])
    desired_height = int(config['Sequence']['imHeight'])

    cap = cv2.VideoCapture(video_path)
    frame_dict = {}
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize del fotogramma
        resized_frame = cv2.resize(frame, (desired_width, desired_height))

        frame_key = str(frame_number)
        if frame_key in bounding_boxes_dict:
            boxes = bounding_boxes_dict[frame_key]
            cropped_images = crop_bounding_boxes(resized_frame, boxes)
            frame_dict[frame_key] = cropped_images

        frame_number += 1

    cap.release()
    return frame_dict

def extract_features(image_crops_dict, model, batch_size=32):

    features_dict = {}

    for frame_key, crops in image_crops_dict.items():
        num_crops = len(crops)
        frame_features = []

        for start in range(0, num_crops, batch_size):
            end = min(start + batch_size, num_crops)
            batch_crops = crops[start:end]

            # Preprocessa le immagini del batch
            batch_images = []
            for crop in batch_crops:
                if crop is None or crop.size == 0:
                    print(f"Immagine ritagliata vuota nel frame {frame_key}, ignorata.")
                    continue

                try:
                    # Ridimensionamento dell'immagine al formato richiesto da ResNet50
                    crop_resized = cv2.resize(crop, (224, 224))
                except cv2.error as e:
                    print(f"Errore nel ridimensionamento dell'immagine ritagliata nel frame {frame_key}: {e}")
                    continue

                batch_images.append(crop_resized)

            if not batch_images:
                continue

            # Converti batch_images in un array numpy
            batch_images = np.array(batch_images, dtype=np.float32)
            batch_images = preprocess_input(batch_images)  # Preprocessa le immagini

            # Estrai le caratteristiche
            batch_features = model.predict(batch_images)

            # Appiattisci le caratteristiche e aggiungile alla lista
            for features in batch_features:
                frame_features.append(features.flatten().tolist())

        features_dict[frame_key] = frame_features

    return features_dict

def strutturized_detection(features, boxs):
    return {f'{n}': {'feature': np.array(features[n]), 'box': boxs[n], 'T_lost': 0} for n in range(len(features))}
