# video_processor.py

import cv2
import os
from tqdm import tqdm
import logging
import comparisons

def extract_ppt_frames(video_path, output_folder, config):
    # --- 로거 설정 ---
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

    # --- 동영상 처리 시작 ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        error_msg = f"오류: 동영상 파일을 열 수 없습니다. - {video_path}"
        print(error_msg)
        logger.error(error_msg)
        return
    
    # --- 해상도 확인 ---
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # SSIM 계산에 필요한 최소 크기보다 작은 경우 오류 처리
    if width <= 7 or height <= 7:
        error_msg = f"오류: 영상 해상도({width}x{height})가 너무 낮아 처리가 불가능합니다. - {video_path}"
        print(error_msg)
        logger.error(error_msg)
        cap.release()
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_filename = os.path.basename(video_path)
    pbar = tqdm(total=total_frames, desc=f"처리 중: {video_filename}")

    prev_gray_frame = None
    saved_frame_count = 0
    frame_number = 0

    # --- 메인 루프 ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        pbar.update(1)

        # YAML에서 읽어온 설정값 사용
        if frame_number % config['frame_interval'] != 0:
            continue

        # 연산 속도 향상을 위해 그레이스케일로 변환
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray_frame is not None:
            change_detected = False
            log_message = ""

            if config['comparison_method'] == 'METHOD_PIXEL_DIFF':
                change_detected, log_message = comparisons.compare_pixel_diff(
                    prev_gray_frame, gray_frame, config['pixel_diff_threshold']
                )
            elif config['comparison_method'] == 'METHOD_SSIM_DIFF':
                change_detected, log_message = comparisons.compare_ssim_diff(
                    prev_gray_frame, gray_frame, config['ssim_diff_threshold']
                )

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
                logger.info(f"{full_log_message} -> [저장 O] ({saved_frame_count:03d}_{filename_ts_part}.png)")
            else:
                logger.info(f"{full_log_message} -> [저장 X]")

        prev_gray_frame = gray_frame
            
    pbar.close()
    cap.release()
    
    final_message = f"✅ 완료: '{video_filename}' 처리 완료. 총 {saved_frame_count}개의 슬라이드 추출."
    print(final_message)
    logger.info(f"\n--- 최종 결과 ---\n{final_message}")