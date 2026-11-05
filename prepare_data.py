import os
import numpy as np
from tqdm import tqdm
import soundfile as sf
import librosa
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

TARGET_DIR = '/mnt/windows_d/mimii_data'
device_map = {'fan': 1, 'pump': 2, 'slider': 3, 'valve': 4}

def get_filelist():
    file_list = []
    print(f"正在深度扫描: {TARGET_DIR}")
    for root, dirs, files in os.walk(TARGET_DIR):
        for file in files:
            if file.lower().endswith(('.wav', '.flac')):
                full_path = os.path.join(root, file)
                path_lower = full_path.lower()

                device = None
                for d in device_map:
                    if f'/{d}/' in path_lower: device = device_map[d]

                label = 0 if 'normal' in path_lower else (1 if 'abnormal' in path_lower else None)

                snr = 1
                if '-6' in path_lower: snr = 1
                elif '0' in path_lower: snr = 2
                elif '6' in path_lower: snr = 3

                if device and label is not None:
                    file_list.append((full_path, snr, device, label))
    return file_list

def process_file(file):
    try:
        audio_data, _ = sf.read(file[0], always_2d=True)
        audio = audio_data.transpose(1, 0)[0].astype(np.float32)
        mfcc = librosa.feature.mfcc(y=audio, n_mfcc=13, n_fft=2048, hop_length=512, n_mels=128)
        return mfcc.astype(np.float32), [file[1], file[2], file[3]]
    except Exception:
        return None, None

def extract_data(file_list, batch_size=128):
    total = len(file_list)
    if total == 0: return None, None
    mfcc_mmap = np.lib.format.open_memmap('mfcc_mmap.npy', dtype=np.float32, mode='w+', shape=(total,13,313))
    label_mmap = np.lib.format.open_memmap('label_mmap.npy', dtype=np.int8, mode='w+', shape=(total, 3))
    pbar = tqdm(total=total, desc="处理音频", unit="units")
    with ThreadPoolExecutor(max_workers=16) as executor:
        for i in range(0, total, batch_size):
            batch = file_list[i:i+batch_size]
            futures = [executor.submit(process_file, info) for info in batch]
            for j, future in enumerate(as_completed(futures)):
                mfcc, label = future.result()
                if mfcc is not None:
                    mfcc_mmap[i+j] = mfcc
                    label_mmap[i+j] = label
                pbar.update(1)
    pbar.close()
    mfcc_mmap.flush()
    label_mmap.flush()
    return mfcc_mmap, label_mmap

def generate_pkl(mfcc_data, label_data):
    if mfcc_data is None: return
    grouped = {1: [], 2: [], 3: []}
    for mfcc, label in zip(mfcc_data, label_data):
        grouped[label[0]].append(((mfcc, label[1]), label[2]))
    final_data = [grouped[1], grouped[2], grouped[3]]
    with open('data.pkl', 'wb') as f:
        pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Success] 总样本数：{sum(len(x) for x in final_data)}")

if __name__ == '__main__':
    files = get_filelist()
    print(f"找到 {len(files)} 个有效音频文件！")
    mfcc_data, label_data = extract_data(files)
    generate_pkl(mfcc_data, label_data)
    print('预处理完成，已保存为 data.pkl')
