# main.py

import os
import yaml
from video_processor import extract_ppt_frames
from post_processor import group_similar_images_in_folder

def load_config(config_path='config.yaml'):
    """YAML 설정 파일을 읽어 딕셔너리로 반환합니다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"오류: 설정 파일({config_path})을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"오류: 설정 파일을 읽는 중 문제가 발생했습니다 - {e}")
        return None

def run_process():
    """메인 실행 함수"""
    # 설정 파일 로드
    config = load_config()
    if config is None:
        return

    input_folder = config['input_folder_path']
    master_output_folder = config['master_output_folder']
    video_extensions = tuple(config['video_extensions']) # endswith를 위해 튜플로 변환

    if not os.path.isdir(input_folder):
        print(f"오류: 지정된 폴더를 찾을 수 없습니다 - {input_folder}")
        return

    print(f"'{input_folder}' 폴더에서 영상 파일을 검색합니다...")
    
    all_files = os.listdir(input_folder)
    video_files = [f for f in all_files if f.lower().endswith(video_extensions)]

    if not video_files:
        print("처리할 영상 파일을 찾지 못했습니다.")
        return

    print(f"총 {len(video_files)}개의 영상 파일을 처리합니다: {video_files}")
    
    for video_file in video_files:
        full_video_path = os.path.join(input_folder, video_file)
        
        video_name_without_ext = os.path.splitext(video_file)[0]
        output_subfolder_path = os.path.join(master_output_folder, f"{video_name_without_ext}_slides")
        
        print(f"\n--- [{video_file}] 처리 시작 ---")
        
        # 영상에서 슬라이드 추출
        extract_ppt_frames(full_video_path, output_subfolder_path, config)
        # 추출된 슬라이드 그룹화
        group_similar_images_in_folder(output_subfolder_path)
    
    print("\n🎉 모든 작업이 완료되었습니다!")

if __name__ == "__main__":
    run_process()