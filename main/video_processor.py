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

    fps = cap.get(cv2.CAP_PROP_FPS)
    # FPS 정보를 읽지 못하는 경우 기본값 30으로 설정
    if fps == 0:
        logger.warning("경고: 영상의 FPS 정보를 읽을 수 없어 기본값(30)을 사용합니다.")
        fps = 30
    
    interval_in_seconds = config['frame_interval_sec']
    # '초' 단위 간격을 실제 프레임 수 간격으로 변환
    frame_check_interval = int(round(interval_in_seconds * fps))
    # 계산된 간격이 최소 1 이상이 되도록 보정
    if frame_check_interval < 1:
        frame_check_interval = 1
    
    logger.info(f"영상 정보: {width}x{height}, {fps:.2f} FPS")
    logger.info(f"설정된 간격: {interval_in_seconds}초 -> 적용 간격: {frame_check_interval}FPS")

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

        if frame_number % frame_check_interval != 0:
            continue
        
        # 연산 속도 향상을 위해 그레이스케일로 변환
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray_frame is not None:
            change_detected = False
            log_message = ""

            # PIXEL_DIFF 1차 비교
            pixel_change_detected, pixel_log = comparisons.compare_pixel_diff(
                prev_gray_frame, gray_frame, config['pixel_diff_threshold']
            )
            
            if pixel_change_detected:
                change_detected = True
                log_message = pixel_log
                
            else:
                # SSIM_DIFF 2차 비교
                ssim_change_detected, ssim_log = comparisons.compare_ssim_diff(
                    prev_gray_frame, gray_frame, config['ssim_diff_threshold']
                )
                
                if ssim_change_detected:
                    change_detected = True
                    log_message = f"{pixel_log} -> {ssim_log}"
                else:
                    change_detected = False
                    log_message = f"{pixel_log} -> {ssim_log}"

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