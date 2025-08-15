import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Khởi tạo API
api = KaggleApi()
api.authenticate()

# Tải dataset ví dụ: Titanic
# dataset = 'jessicali9530/lfw-dataset'
# download_path = 'data/lfw-dataset'

# dataset = 'jessicali9530/celeba-dataset'
# download_path = 'data/celeba-dataset'

# dataset = 'ntl0601/casia-webface'
# download_path = 'data/casia-webface'

dataset = 'chinafax/cfpw-dataset'
download_path = 'data/cfpw-dataset'

# Tải toàn bộ dataset
api.dataset_download_files(dataset, path=download_path, unzip=True)

print(f'Dataset đã được tải về thư mục: {download_path}')
