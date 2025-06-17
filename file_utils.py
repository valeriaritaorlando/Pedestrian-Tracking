import os


def find_video_path(directory, nome_video):
    # Lista di estensioni video comuni (inclusa .webm)
    estensioni_video = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm']
    
    # Itera sui file nella directory
    for file in os.listdir(directory):
        # Ottieni il nome del file e l'estensione
        nome_file, estensione_file = os.path.splitext(file)
        
        # Verifica se il nome del file corrisponde e l'estensione è valida
        if nome_file == nome_video and estensione_file.lower() in estensioni_video:
            return os.path.join(directory, file)
    
    # Se nessun file corrisponde, restituisci un messaggio di errore
    return f"File {nome_video} con estensione video non trovato in {directory}"


def find_association_file(directory, nome_video, tipo_file):

    # Costruisce il nome del file da cercare
    nome_file_cercato = f"{nome_video}_{tipo_file}.json"
    
    # Costruisce il percorso completo del file
    file_path = os.path.join(directory, nome_file_cercato)
    
    return file_path
    


def create_output_video_path(directory, nome_video):

    # Costruisce il percorso completo del file di output video
    output_video_path = os.path.join(directory, nome_video + '_output.mp4')
    return output_video_path
