import shutil
import os
import h5py
import numpy as np
import argparse

from constants import DATA_DIR

def copy_folder(src_folder, dst_folder):
    if not os.path.exists(src_folder):
        raise FileNotFoundError(f"원본 폴더가 존재하지 않습니다: {src_folder}")
    if os.path.exists(dst_folder):
        raise FileExistsError(f"대상 폴더가 이미 존재합니다: {dst_folder}")
    
    shutil.copytree(src_folder, dst_folder)

def add_noise_to_images_in_hdf5(folder_path):
    # 폴더 내의 모든 .hdf5 파일을 순회
    for filename in os.listdir(folder_path):
        if filename.endswith('.hdf5') or filename.endswith('.h5'):
            file_path = os.path.join(folder_path, filename)
            print(f'Processing: {file_path}')
            try:
                with h5py.File(file_path, 'r+') as hdf5_file:
                    # TODO: 다른 cam으로도 확장하기
                    image_data = hdf5_file['observations']['images']['top']
                    shape = image_data.shape
                    dtype = image_data.dtype

                    noise = np.random.randint(0, 256, size=shape, dtype=dtype)

                    # 이미지 데이터셋에 덮어쓰기
                    image_data[...] = noise
                    print(f'Image in "{filename}" overwritten with noise.')
                    
            except Exception as e:
                print(f'Failed to process {file_path}: {e}')



parser = argparse.ArgumentParser()
parser.add_argument('--target_dataset_folder', action='store', type=str, help='target dataset folder', required=True)
args = parser.parse_args()

src = DATA_DIR + args['target_dataset_folder']
dst = src + '_noisy_img'
copy_folder(src, dst)
add_noise_to_images_in_hdf5(dst)
