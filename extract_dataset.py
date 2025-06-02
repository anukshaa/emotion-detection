import zipfile
import os
# Define the path to the ZIP file and the extraction directory
zip_path = 'fer2013.zip'
extract_dir = 'fer2013'
# Create the extraction directory if it doesn't exist
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)
# Extract the contents of the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print("Extraction completed.")
