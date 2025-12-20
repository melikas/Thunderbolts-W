import zipfile

def extract_zip(source_path, destination_path):
    with zipfile.ZipFile(source_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)

