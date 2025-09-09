import cv2
import os
import numpy as np
from tqdm import tqdm
import logging

def extract_ppt_frames(video_path, output_folder, frame_interval=30, threshold_low=0.5, threshold_high=0.98):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # ✨ --- 1. 로거(Logger) 설정 --- ✨
    # 각 비디오 파일 이름에 맞는 로그 파일 생성 (예: my_video.log)
    video_name_without_ext = os.path.splitext(os.path.basename(video_path))[0]
    log_file_path = os.path.join(output_folder, f"{video_name_without_ext}.log")
    
    # 로거 인스턴스 생성 및 설정
    logger = logging.getLogger(video_name_without_ext)
    logger.setLevel(logging.INFO)

    # 이전 실행에서 핸들러가 추가되었다면 초기화
    if logger.hasHandlers():
        logger.handlers.clear()

    # 파일 핸들러 설정 (파일을 새로 덮어쓰는 'w' 모드)
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(message)s') # 로그 메시지 형식 지정
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # ✨ --- 로거 설정 끝 --- ✨

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # 오류는 콘솔과 로그에 모두 남김
        error_msg = f"오류: 동영상 파일을 열 수 없습니다 - {video_path}"
        print(error_msg)
        logger.error(error_msg)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            hours = int(msec / (1000 * 60 * 60))
            minutes = int((msec / (1000 * 60)) % 60)
            seconds = int((msec / 1000) % 60)
            timestamp_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
            
            log_message = f"[{timestamp_str}] 유사도: {similarity:.6f}"

            if threshold_low < similarity < threshold_high:
                saved_frame_count += 1
                filename_ts_part = f"{hours:02d}h-{minutes:02d}m-{seconds:02d}s"
                output_filename = os.path.join(output_folder, f"{saved_frame_count:03d}_{filename_ts_part}.png")
                cv2.imwrite(output_filename, frame)
                
                # ✨ --- 2. print를 logger.info로 변경 --- ✨
                logger.info(f"{log_message} -> [저장 O] ({saved_frame_count:03d}_{filename_ts_part}.png)")
            else:
                logger.info(f"{log_message} -> [저장 X]")

        prev_frame_hist = current_frame_hist
            
    pbar.close()
    cap.release()
    
    # 최종 완료 메시지는 콘솔에 출력
    final_message = f"✅ 완료: '{video_filename}' 처리 완료. 총 {saved_frame_count}개의 슬라이드 추출."
    print(final_message)
    logger.info(f"\n--- 최종 결과 ---\n{final_message}")


# ✨ 1. 영상들이 모여있는 폴더 경로를 지정하세요.
INPUT_FOLDER_PATH = "INPUT"

# ✨ 2. 모든 결과물을 저장할 최상위 폴더 이름을 지정하세요.
MASTER_OUTPUT_FOLDER = "extracted_slides"

# ✨ 3. 처리할 영상 파일의 확장자를 지정하세요. (소문자로)
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv')

# (아래는 처리 옵션 - 필요시 수정)
FRAME_INTERVAL = 30
THRESHOLD_HIGH_VALUE = 0.999
THRESHOLD_LOW_VALUE = 0.6

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