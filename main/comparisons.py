# comparisons.py

import cv2
from skimage.metrics import structural_similarity as ssim

def compare_pixel_diff(frame1_gray, frame2_gray, threshold):
    """픽셀 차이를 계산하여 변화 여부를 반환합니다."""
    
    # 밝기 차이 이미지 생성 (절대 차이)
    # 밝기 차이가 클수록 밝음(흰색, 255에 가까워짐)
    diff = cv2.absdiff(frame1_gray, frame2_gray) 

    # Otsu 알고리즘을 사용하여 이진화
    # Otsu 임계값보다 크면 흰색(255), 작으면 검은색(0)
    otsu_threshold, thresholded_diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 흰색 픽셀 개수 = 변화가 감지된 픽셀 개수
    non_zero_count = cv2.countNonZero(thresholded_diff) 
    # 전체 픽셀 개수 (shape[0] = height, shape[1] = width)
    total_pixels = frame1_gray.shape[0] * frame1_gray.shape[1]
    # 변화가 감지된 비율
    score = non_zero_count / total_pixels
    
    change_detected = score > threshold
    log_message = f"PIXEL_DIFF: {score:.6f} / otsu: {otsu_threshold:.6f}"
    
    return change_detected, log_message

def compare_ssim_diff(frame1_gray, frame2_gray, threshold):
    """구조적 유사성(SSIM)의 차이를 계산하여 변화 여부를 반환합니다."""

    # SSIM: 구조적 유사성 지수 측정
    # https://en.wikipedia.org/wiki/Structural_similarity_index_measure
    # SSIM이 1에 가까울수록 두 이미지 유사
    score = 1 - ssim(frame1_gray, frame2_gray)
    
    change_detected = score > threshold
    log_message = f"SSIM_DIFF: {score:.6f}"
    
    return change_detected, log_message