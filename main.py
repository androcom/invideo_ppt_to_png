import cv2
import os
import numpy as np
from tqdm import tqdm

def extract_ppt_frames(video_path, output_folder, threshold=0.98):
    """
    MP4 영상에서 PPT 화면이 변경될 때마다 해당 프레임을 이미지로 추출합니다.

    :param video_path: 원본 동영상 파일 경로
    :param output_folder: 이미지를 저장할 폴더 경로
    :param threshold: 프레임 유사도 임계값 (값이 높을수록 미세한 차이에도 저장)
    """
    # 결과물을 저장할 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("오류: 동영상을 열 수 없습니다.")
        return

    # 동영상의 총 프레임 수 확인
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"총 프레임 수: {total_frames}")

    # 이전 프레임을 저장할 변수 초기화
    prev_frame_hist = None
    frame_count = 0
    saved_frame_count = 0

    # tqdm을 사용하여 진행 상황 표시
    with tqdm(total=total_frames, desc="프레임 처리 중") as pbar:
        while True:
            # 동영상에서 프레임 하나씩 읽기
            ret, frame = cap.read()
            if not ret:
                break # 동영상이 끝나면 반복 종료

            # 현재 프레임을 그레이스케일로 변환
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 히스토그램 계산 (프레임의 픽셀 밝기 분포)
            current_frame_hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
            cv2.normalize(current_frame_hist, current_frame_hist, 0, 1, cv2.NORM_MINMAX)

            if prev_frame_hist is not None:
                # 이전 프레임의 히스토그램과 현재 프레임의 히스토그램 유사도 비교
                similarity = cv2.compareHist(prev_frame_hist, current_frame_hist, cv2.HISTCMP_CORREL)

                # 유사도가 설정한 임계값보다 낮으면 (즉, 화면 변화가 크면)
                if similarity < threshold:
                    saved_frame_count += 1
                    # 파일명 형식: slide_001.png, slide_002.png ...
                    output_filename = os.path.join(output_folder, f"slide_{saved_frame_count:03d}.png")
                    cv2.imwrite(output_filename, frame)
                    print(f"\n화면 전환 감지! {output_filename} 저장됨 (유사도: {similarity:.4f})")

            # 현재 프레임의 히스토그램을 이전 프레임으로 저장
            prev_frame_hist = current_frame_hist
            pbar.update(1) # 진행 막대 업데이트

    # 작업 완료 후 리소스 해제
    cap.release()
    print(f"\n총 {saved_frame_count}개의 슬라이드 이미지를 추출했습니다.")


# --- 여기부터 설정 부분 ---

# 1. 동영상 파일 경로를 지정하세요.
# 예: "C:/Users/YourUser/Desktop/lecture.mp4"
VIDEO_FILE_PATH = "C:/test.mp4" 

# 2. 추출된 이미지를 저장할 폴더 이름을 지정하세요.
OUTPUT_FOLDER_NAME = "extracted_slides"

# 3. 임계값 설정 (옵션)
# 0.9 ~ 0.99 사이의 값을 권장합니다.
# 값이 낮을수록: 사소한 변화(마우스 커서 움직임 등)에도 프레임을 저장할 수 있음
# 값이 높을수록: 정말 화면이 크게 바뀌었을 때만 저장함
THRESHOLD_VALUE = 0.98

# 함수 실행
extract_ppt_frames(VIDEO_FILE_PATH, OUTPUT_FOLDER_NAME, THRESHOLD_VALUE)