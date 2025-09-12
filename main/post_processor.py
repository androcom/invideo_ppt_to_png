# post_processor.py

import os
import cv2
import numpy as np
import shutil
from sklearn.cluster import DBSCAN
from tqdm import tqdm

def get_image_histogram(image_path):
    """이미지 파일 경로를 받아 정규화된 히스토그램을 반환합니다."""
    img_array = np.fromfile(image_path, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def group_similar_images_in_folder(single_video_folder):
    """
    지정된 폴더 내의 PNG 파일들을 유사도에 따라 클러스터링하여
    'Group_X' 형태의 하위 폴더로 재분류합니다.
    """
    print(f"\n--- '{os.path.basename(single_video_folder)}' 폴더 내 이미지 분류 시작 ---")
    
    # 해당 폴더 내의 PNG 파일 경로 수집
    all_image_paths = []
    for file in os.listdir(single_video_folder):
        if file.lower().endswith('.png'):
            all_image_paths.append(os.path.join(single_video_folder, file))

    if len(all_image_paths) < 2:
        print("분류할 이미지가 충분하지 않습니다.")
        return

    # 히스토그램 계산
    features = []
    valid_image_paths = []
    for img_path in tqdm(all_image_paths, desc="이미지 특징 추출 중"):
        hist = get_image_histogram(img_path)
        if hist is not None:
            features.append(hist)
            valid_image_paths.append(img_path)

    # DBSCAN 클러스터링
    # eps = 히스토그램 간의 상관관계 거리 (상관도, 낮을수록 유사)
    db = DBSCAN(eps=0.02, min_samples=2, metric='correlation').fit(features)
    labels = db.labels_

    # 파일 이동
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"총 {num_clusters}개의 유사 이미지 그룹을 찾았습니다.")

    for i, label in enumerate(tqdm(labels, desc="파일 분류 중")):
        if label != -1:
            group_folder_name = f"Group_{label}"
            group_folder_path = os.path.join(single_video_folder, group_folder_name)
            
            if not os.path.exists(group_folder_path):
                os.makedirs(group_folder_path)

            shutil.move(valid_image_paths[i], group_folder_path)

    print(f"--- '{os.path.basename(single_video_folder)}' 폴더 내 이미지 분류 완료 ---")