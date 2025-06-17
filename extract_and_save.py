import json
import gc
from feature_extraction import extract_features, process_video
from detector.extract_bbox import get_bounding_boxes
import torch



def extract_save_bbox(video_path, bbox_path, model, seqinfo_path):
    # ---------- EXTRACT BBOX ----------
    bounding_boxes = get_bounding_boxes(video_path, model, seqinfo_path)

    # Salva bounding_boxes in un file JSON
    with open(bbox_path, 'w') as f:
        json.dump(bounding_boxes, f)
    print(f"Bounding boxes salvate in {bbox_path}")

    # Libera la memoria usata
    del model
    del bounding_boxes
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_save_features(video_path, bbox_path, features_path, model, seqinfo_path):
    # Carica bounding_boxes dal file JSON
    with open(bbox_path, 'r') as f:
        bounding_boxes = json.load(f)
    print(f"Bounding boxes caricate da {bbox_path}")

    # ---------- CROPPED  ----------
    cropped_images_dict = process_video(video_path, bounding_boxes, seqinfo_path)

    # ---------- EXTRACT FEATURES ----------
    features_dict = extract_features(cropped_images_dict, model)

    # Salva features_dict in un file JSON
    with open(features_path, 'w') as f:
        json.dump(features_dict, f)
    print(f"Features dictionary salvato in {features_path}")

    # Libera la memoria usata
    del model
    del features_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()