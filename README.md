Il progetto implementa un sistema di pedestrian tracking basato su un modello di rilevamento (DETR) e un algoritmo di tracciamento personalizzato.
Il tracciamento si basa sulla combinazione di IoU (Intersection over Union) e distanza coseno tra feature estratte, con assegnazione ID tramite l'algoritmo ungherese.
Tutti i parametri sono stati ottimizzati tramite validazione sui video del dataset MOT17.

File principali:

tracking.py	Funzione update_tracks, cuore del tracciamento
calculate_IoU.py	Calcolo della matrice IoU
calculate_distance_features.py	Calcolo della matrice delle distanze coseno
feature_extractions.py	Estrazione delle feature
extract_and_save.py	Salvataggio feature e bounding box
validation.py	Grid search sui parametri, selezione top tracker
test.py	Test finale con modello selezionato
generate_video_tracking.py	Genera il video di output con le tracce
