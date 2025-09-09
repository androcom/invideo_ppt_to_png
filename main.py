import cv2
import os
import numpy as np
from tqdm import tqdm
import logging
from skimage.metrics import structural_similarity as ssim

# ✨ --- 비교 방식 선택 --- ✨
# 사용할 비교 방식을 아래 셋 중 하나로 선택하세요.
# 'METHOD_PIXEL_DIFF': 픽셀 차이 계산 (텍스트 변화 감지에 가장 효과적, 강력 추천!)
# 'METHOD_SSIM': 구조적 유사성 비교
# 'METHOD_ORB': 특징점 매칭 비교
COMPARISON_METHOD = ''

def extract_ppt_frames(video_path, output_folder, frame_interval=30):
    # 로거 설정
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    video_name_without_ext = os.path.splitext(os.path.basename(video_path))[0]
    log_file_path = os.path.join(output_folder, f"{video_name_without_ext}.log")
    
    logger = logging.getLogger(video_name_without_ext)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 동영상 처리 시작
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        error_msg = f"오류: 동영상 파일을 열 수 없습니다 - {video_path}"
        print(error_msg)
        logger.error(error_msg)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_filename = os.path.basename(video_path)
    pbar = tqdm(total=total_frames, desc=f"처리 중: {video_filename}")

    prev_gray_frame = None
    saved_frame_count = 0
    frame_number = 0
    
    # ORB 방식에 사용할 객체 및 이전 매칭 수 저장 변수
    if COMPARISON_METHOD == 'METHOD_ORB':
        orb = cv2.ORB_create(nfeatures=1000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # 안정적인 상태의 평균 매칭 수를 저장할 변수
        stable_match_count = -1

    # 메인 루프
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        pbar.update(1)

        if frame_number % frame_interval != 0:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray_frame is not None:
            score = 0
            change_detected = False
            log_message = ""

            # --- 1. 픽셀 차이 비교 로직 ---
            if COMPARISON_METHOD == 'METHOD_PIXEL_DIFF':
                PIXEL_DIFF_THRESHOLD = 0.01 # 임계값: 변경된 픽셀 비율 (1%)
                
                diff = cv2.absdiff(prev_gray_frame, gray_frame)
                _, thresholded_diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                non_zero_count = cv2.countNonZero(thresholded_diff)
                total_pixels = gray_frame.shape[0] * gray_frame.shape[1]
                score = non_zero_count / total_pixels
                
                if score > PIXEL_DIFF_THRESHOLD:
                    change_detected = True
                log_message = f"변경된 픽셀 비율: {score:.6f}"

            # --- 2. SSIM 비교 로직 ---
            elif COMPARISON_METHOD == 'METHOD_SSIM':
                SSIM_THRESHOLD = 0.98 # 임계값: 구조적 유사도 (1.0에 가까울수록 유사)
                
                # win_size는 이미지 크기보다 작아야 하며 홀수여야 함
                h, w = prev_gray_frame.shape
                win_size = min(7, h, w)
                if win_size % 2 == 0: win_size -= 1
                
                score = ssim(prev_gray_frame, gray_frame, win_size=win_size)
                if score < SSIM_THRESHOLD:
                    change_detected = True
                log_message = f"유사도: {score:.6f}"

            # --- 3. ORB 특징점 매칭 로직 ---
            elif COMPARISON_METHOD == 'METHOD_ORB':
                # 임계값: 안정 상태 대비 매칭 수가 몇 % 이하로 떨어졌을 때 변화로 볼 것인가
                # (예: 0.5 = 50% 이하)
                ORB_DROP_RATIO_THRESHOLD = 0.5 
                
                kp1, des1 = orb.detectAndCompute(prev_gray_frame, None)
                kp2, des2 = orb.detectAndCompute(gray_frame, None)

                score = 0
                if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                    matches = bf.match(des1, des2)
                    score = len(matches)

                # 안정 상태 매칭 수(stable_match_count)를 기준으로 변화 감지
                if stable_match_count == -1: # 아직 안정값이 없으면 현재 값을 기준으로 설정
                    if score > 500: # 초기 노이즈를 피하기 위해 일정 값 이상일 때만 안정값으로 인정
                         stable_match_count = score
                # 안정값이 있고, 현재 매칭 수가 안정값의 N% 이하로 떨어졌다면 변화로 감지
                elif score < stable_match_count * ORB_DROP_RATIO_THRESHOLD:
                    change_detected = True
                    # 변화 감지 후, 현재 값을 새로운 안정 상태 기준으로 업데이트
                    stable_match_count = score
                # 변화가 없다면, 안정 상태 매칭 수를 현재 값에 가깝게 점진적으로 업데이트 (평균 필터링 효과)
                else:
                    stable_match_count = (stable_match_count * 0.9) + (score * 0.1)

                log_message = f"매칭 수: {int(score)} (기준: {int(stable_match_count)})"

            # --- 로깅 및 저장 처리 ---
            msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            hours = int(msec / (1000 * 60 * 60))
            minutes = int((msec / (1000 * 60)) % 60)
            seconds = int((msec / 1000) % 60)
            timestamp_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"
            
            full_log_message = f"[{timestamp_str}] {log_message}"

            if change_detected:
                saved_frame_count += 1
                filename_ts_part = f"{hours:02d}h-{minutes:02d}m-{seconds:02d}s"
                output_filename = os.path.join(output_folder, f"{saved_frame_count:03d}_{filename_ts_part}.png")
                cv2.imwrite(output_filename, frame)
                logger.info(f"{full_log_message} -> [저장 O] - {saved_frame_count:03d}_{filename_ts_part}.png")
            else:
                logger.info(f"{full_log_message} -> [저장 X]")

        prev_gray_frame = gray_frame
            
    pbar.close()
    cap.release()
    
    final_message = f"✅ 완료: '{video_filename}' 처리 완료. 총 {saved_frame_count}개의 슬라이드 추출."
    print(final_message)
    logger.info(f"\n--- 최종 결과 ---\n{final_message}")


# ✨ 1. 영상들이 모여있는 폴더 경로를 지정하세요.
INPUT_FOLDER_PATH = "INPUT"

# ✨ 2. 모든 결과물을 저장할 최상위 폴더 이름을 지정하세요.
MASTER_OUTPUT_FOLDER = "EXTRACTED_SLIDES"

# ✨ 3. 처리할 영상 파일의 확장자를 지정하세요. (소문자로)
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv')

# (아래는 처리 옵션 - 필요시 수정)
FRAME_INTERVAL = 30

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
                    frame_interval=FRAME_INTERVAL
                )
            
            print("\n🎉 모든 작업이 완료되었습니다!")