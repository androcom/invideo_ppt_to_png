import cv2
import os
import numpy as np
from tqdm import tqdm

def extract_ppt_frames(video_path, output_folder, frame_interval=30, threshold_low=0.5, threshold_high=0.98):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 동영상 파일을 열 수 없습니다 - {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # tqdm 설명에 현재 처리 중인 파일 이름 추가
    video_filename = os.path.basename(video_path)
    pbar = tqdm(total=total_frames, desc=f"처리 중: {video_filename}")

    prev_frame_hist = None
    saved_frame_count = 0
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        pbar.update(1)

        if frame_number % frame_interval != 0:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_frame_hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        cv2.normalize(current_frame_hist, current_frame_hist, 0, 1, cv2.NORM_MINMAX)

        if prev_frame_hist is not None:
            similarity = cv2.compareHist(prev_frame_hist, current_frame_hist, cv2.HISTCMP_CORREL)

            if threshold_low < similarity < threshold_high:
                saved_frame_count += 1
                output_filename = os.path.join(output_folder, f"slide_{saved_frame_count:03d}.png")
                cv2.imwrite(output_filename, frame)

        prev_frame_hist = current_frame_hist
            
    pbar.close()
    cap.release()
    print(f"✅ 완료: '{video_filename}' 처리 완료. 총 {saved_frame_count}개의 슬라이드 추출.")


# ✨ 1. 영상들이 모여있는 폴더 경로를 지정하세요.
INPUT_FOLDER_PATH = "INPUT"

# ✨ 2. 모든 결과물을 저장할 최상위 폴더 이름을 지정하세요.
MASTER_OUTPUT_FOLDER = "extracted_slides"

# ✨ 3. 처리할 영상 파일의 확장자를 지정하세요. (소문자로)
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv')

# (아래는 처리 옵션 - 필요시 수정)
FRAME_INTERVAL = 60
THRESHOLD_HIGH_VALUE = 0.999
THRESHOLD_LOW_VALUE = 0.5

# --- ✨ 메인 실행 로직 ---
if __name__ == "__main__":
    if not os.path.isdir(INPUT_FOLDER_PATH):
        print(f"오류: 지정된 폴더를 찾을 수 없습니다 - {INPUT_FOLDER_PATH}")
    else:
        print(f"'{INPUT_FOLDER_PATH}' 폴더에서 영상 파일을 검색합니다...")
        
        # 지정된 폴더 내의 모든 파일 목록을 가져옴
        all_files = os.listdir(INPUT_FOLDER_PATH)
        
        # 확장자를 기준으로 영상 파일만 필터링
        video_files = [f for f in all_files if f.lower().endswith(VIDEO_EXTENSIONS)]

        if not video_files:
            print("처리할 영상 파일을 찾지 못했습니다.")
        else:
            print(f"총 {len(video_files)}개의 영상 파일을 처리합니다: {video_files}")
            
            # 영상 파일을 하나씩 순회하며 처리
            for video_file in video_files:
                # 전체 영상 파일 경로 생성
                full_video_path = os.path.join(INPUT_FOLDER_PATH, video_file)
                
                # 결과물을 저장할 하위 폴더 이름 생성 (예: 'lecture1_slides')
                video_name_without_ext = os.path.splitext(video_file)[0]
                output_subfolder_path = os.path.join(MASTER_OUTPUT_FOLDER, f"{video_name_without_ext}_slides")
                
                print(f"\n--- [{video_file}] 처리 시작 ---")
                
                # 메인 함수 호출
                extract_ppt_frames(
                    full_video_path,
                    output_subfolder_path,
                    frame_interval=FRAME_INTERVAL,
                    threshold_low=THRESHOLD_LOW_VALUE,
                    threshold_high=THRESHOLD_HIGH_VALUE
                )
            
            print("\n🎉 모든 작업이 완료되었습니다!")