import requests
import zipfile
import os

# URL của file ZIP
url = "https://www.conradsanderson.id.au/lfwcrop/lfwcrop_color.zip"
zip_path = "lfwcrop_color.zip"
extract_path = "lfwcrop_color"

# Tải file ZIP
print("Đang tải file...")
response = requests.get(url, stream=True)
with open(zip_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
print("Tải xong!")

# Giải nén file ZIP
print("Đang giải nén...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print(f"Đã giải nén vào thư mục '{extract_path}'")
