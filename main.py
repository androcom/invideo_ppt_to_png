import cv2
import os
import numpy as np
from tqdm import tqdm

def extract_ppt_frames(video_path, output_folder, frame_interval=30, threshold_low=0.5, threshold_high=0.98):
    """
    MP4 영상에서 일정 프레임 간격마다 화면을 비교하여 슬라이드 변경을 감지하고 추출합니다.

    :param video_path: 원본 동영상 파일 경로
    :param output_folder: 이미지를 저장할 폴더 경로
    :param frame_interval: 프레임을 비교할 간격 (예: 30은 30프레임마다 1번 비교)
    :param threshold_low: 유사도 하한값
    :param threshold_high: 유사도 상한값
    """
    # 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("오류: 동영상을 열 수 없습니다.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"총 프레임 수: {total_frames}")

    prev_frame_hist = None
    saved_frame_count = 0
    frame_number = 0

    with tqdm(total=total_frames, desc="프레임 처리 중") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            pbar.update(1)

            # --- 핵심 변경 부분 ---
            # 설정된 간격(frame_interval)에 해당하는 프레임이 아니면 비교를 건너뜀
            if frame_number % frame_interval != 0:
                continue

            # --- 아래는 간격에 맞는 프레임일 때만 실행되는 로직 ---
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_frame_hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
            cv2.normalize(current_frame_hist, current_frame_hist, 0, 1, cv2.NORM_MINMAX)

            if prev_frame_hist is not None:
                similarity = cv2.compareHist(prev_frame_hist, current_frame_hist, cv2.HISTCMP_CORREL)

                if threshold_low < similarity < threshold_high:
                    saved_frame_count += 1
                    output_filename = os.path.join(output_folder, f"slide_{saved_frame_count:03d}.png")
                    cv2.imwrite(output_filename, frame)
                    print(f"\n[{frame_number} 프레임] 화면 전환 감지! {output_filename} 저장됨 (유사도: {similarity:.4f})")

            # 현재 비교한 프레임의 히스토그램을 '이전 히스토그램'으로 저장
            prev_frame_hist = current_frame_hist
            
    cap.release()
    print(f"\n총 {saved_frame_count}개의 슬라이드 이미지를 추출했습니다.")


# --- 여기부터 설정 부분 ---

VIDEO_FILE_PATH = "test.mp4" 
OUTPUT_FOLDER_NAME = "extracted_slides"

# ✨ 1. 프레임 비교 간격 설정
# 30fps 동영상 기준, 30으로 설정 시 약 1초에 1번 비교합니다.
# 15로 설정 시 약 0.5초에 1번 비교합니다.
FRAME_INTERVAL = 60

# 2. 유사도 상한값
THRESHOLD_HIGH_VALUE = 0.999

# 3. 유사도 하한값
THRESHOLD_LOW_VALUE = 0.6

# 함수 실행
extract_ppt_frames(
    VIDEO_FILE_PATH,
    OUTPUT_FOLDER_NAME,
    frame_interval=FRAME_INTERVAL,
    threshold_low=THRESHOLD_LOW_VALUE,
    threshold_high=THRESHOLD_HIGH_VALUE
)