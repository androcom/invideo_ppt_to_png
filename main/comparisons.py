# comparisons.py

import cv2
from skimage.metrics import structural_similarity as ssim

def compare_pixel_diff(frame1_gray, frame2_gray, threshold):
    """픽셀 차이를 계산하여 변화 여부를 반환합니다."""
    diff = cv2.absdiff(frame1_gray, frame2_gray)
    _, thresholded_diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    non_zero_count = cv2.countNonZero(thresholded_diff)
    total_pixels = frame1_gray.shape[0] * frame1_gray.shape[1]
    score = non_zero_count / total_pixels
    
    change_detected = score > threshold
    log_message = f"변경된 픽셀 비율: {score:.6f}"
    
    return change_detected, log_message

def compare_ssim(frame1_gray, frame2_gray, threshold):
    """구조적 유사성(SSIM)을 계산하여 변화 여부를 반환합니다."""
    h, w = frame1_gray.shape
    win_size = min(7, h, w)
    if win_size % 2 == 0:
        win_size -= 1
    
    score = ssim(frame1_gray, frame2_gray, win_size=win_size)
    
    change_detected = score < threshold
    log_message = f"유사도: {score:.6f}"
    
    return change_detected, log_message