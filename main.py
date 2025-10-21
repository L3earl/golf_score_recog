"""
골프 스코어카드 인식 메인 실행 파일

의도: 전체 처리 파이프라인을 순차적으로 실행
case1 → case2 → case3 → Claude API 순서로 예외 처리
"""

import sys
import os
import logging
from datetime import datetime
from modules.preprocessing import ImagePreprocessor
from modules.ocr_converter import OCRConverter
from modules.postprocessing import PostProcessor
from modules.claude_converter import ClaudeConverter
from config import RAW_IMG_FOLDER, IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

def _resolve_image_path(base_name):
    """raw_img 폴더에서 주어진 베이스 파일명의 실제 파일 경로를 찾습니다.

    확장자는 `IMAGE_EXTENSIONS` 목록을 순서대로 확인하여 존재하는 첫 번째 파일을 반환합니다.
    해당 파일이 없으면 None을 반환합니다.
    """
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(RAW_IMG_FOLDER, f"{base_name}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None

def _get_image_size(image_path):
    """이미지의 (width, height)를 반환합니다. 실패 시 None 반환.

    PIL(Pillow)을 선호하여 사용하고, 미설치/오류 시 None을 반환합니다.
    """
    try:
        from PIL import Image  # 지연 임포트로 의존성 문제 최소화
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        logger.error(f"이미지 크기 로드 실패: {image_path} - {e}")
        return None

def _is_ratio_3_2(width, height):
    """비율이 정확히 3:2 인지 확인 (허용 오차 없음)."""
    # 정확 정수비율 판정: 3/2 == width/height → 2*width == 3*height
    return (2 * width) == (3 * height)

def _filter_case1_eligible(image_names):
    """case1 대상: 크기 1800x1200 이거나 비율이 정확히 3:2 인 이미지의 베이스명 리스트 반환"""
    eligible = []
    ineligible = []
    for base in image_names:
        path = _resolve_image_path(base)
        if not path:
            ineligible.append(base)
            continue
        size = _get_image_size(path)
        if not size:
            ineligible.append(base)
            continue
        w, h = size
        if (w == 1800 and h == 1200) or _is_ratio_3_2(w, h):
            eligible.append(base)
        else:
            ineligible.append(base)
    return eligible, ineligible

def _filter_case2_eligible(image_names):
    """case2 대상: 크기 정확히 909x920 인 이미지의 베이스명 리스트 반환"""
    eligible = []
    ineligible = []
    for base in image_names:
        path = _resolve_image_path(base)
        if not path:
            ineligible.append(base)
            continue
        size = _get_image_size(path)
        if not size:
            ineligible.append(base)
            continue
        w, h = size
        if (w == 909 and h == 920):
            eligible.append(base)
        else:
            ineligible.append(base)
    return eligible, ineligible

def _filter_case3_eligible(image_names):
    """case3 대상: 세로가 가로보다 긴 이미지(h > w)의 베이스명 리스트 반환"""
    eligible = []
    ineligible = []
    for base in image_names:
        path = _resolve_image_path(base)
        if not path:
            ineligible.append(base)
            continue
        size = _get_image_size(path)
        if not size:
            ineligible.append(base)
            continue
        w, h = size
        if h > w:
            eligible.append(base)
        else:
            ineligible.append(base)
    return eligible, ineligible

def extract_image_names_from_raw_img():
    """raw_img 폴더에서 이미지 파일명 추출
    
    의도: 처리할 전체 이미지 목록 확인
    
    Returns:
        이미지 파일명 리스트 (확장자 제외)
    """
    try:
        image_files = [f for f in os.listdir(RAW_IMG_FOLDER) 
                      if f.lower().endswith(tuple(IMAGE_EXTENSIONS))]
        logger.info(f"총 {len(image_files)}개 이미지 파일 발견")
        return [os.path.splitext(f)[0] for f in image_files]
    except Exception as e:
        logger.error(f"이미지 파일명 추출 실패: {e}")
        return []

def process_case(case, target_files=None):
    """단일 케이스 처리 (전처리 → OCR 변환 → 후처리)
    
    의도: 지정된 케이스에 대해 전체 처리 파이프라인을 순차적으로 실행
    
    Args:
        case: 처리 케이스 ('case1', 'case2', 'case3')
        target_files: 처리할 파일명 리스트 (None이면 전체)
    
    Returns:
        처리 성공 여부 또는 실패한 파일 리스트
    """
    logger.info(f"{case.upper()} 처리 시작")
    if target_files:
        logger.info(f"대상 파일: {len(target_files)}개")
    
    try:
        # 1. 전처리
        logger.info(f"[1/3] {case} 전처리 중...")
        # case3일 때는 자동으로 OCR 크롭 사용
        use_ocr_crop = (case == "case3")
        preprocessor = ImagePreprocessor(case=case, use_ocr_crop=use_ocr_crop)
        preprocess_result = preprocessor.process_all_images(target_files=target_files)
        
        # 전처리 실패 처리
        if preprocess_result is False or preprocess_result is None:
            logger.error(f"{case} 전처리 실패")
            return None
        
        # case3 OCR 크롭 실패 시 실패한 파일 리스트 반환
        if isinstance(preprocess_result, list):
            logger.error(f"{case} OCR 크롭 실패")
            return preprocess_result
        
        # 2. OCR 변환
        logger.info(f"[2/3] {case} OCR 변환 중...")
        converter = OCRConverter(case=case)
        if not converter.convert_all_folders(target_files=target_files):
            logger.error(f"{case} OCR 변환 실패")
            return None
        
        # 3. 후처리
        logger.info(f"[3/3] {case} 후처리 중...")
        processor = PostProcessor(case=case)
        results = processor.process_all_files(target_files=target_files)
        if not results:
            logger.error(f"{case} 후처리 실패")
            return None
        
        # 예외 파일 수집
        exception_files = [r['folder'] for r in results 
                          if 'exceptions' in r and len(r['exceptions']) > 0]
        
        if exception_files:
            logger.warning(f"{case} 예외 발견: {len(exception_files)}개 파일")
        else:
            logger.info(f"{case} 예외 없음")
        
        logger.info(f"{case.upper()} 처리 완료!")
        return exception_files
    except Exception as e:
        logger.error(f"{case} 처리 중 오류 발생: {e}")
        return None

def main():
    """메인 실행 함수
    
    의도: 전체 골프 스코어카드 인식 파이프라인을 순차적으로 실행
    case1 → case2 → case3 → Claude API 순서로 예외 처리
    """
    start_time = datetime.now()
    logger.info("골프 스코어카드 인식 시작")
    
    try:
        # 0단계: raw_img 파일 목록 확인
        all_image_names = extract_image_names_from_raw_img()
        if not all_image_names:
            logger.error("raw_img 폴더에 이미지 없음")
            return False
        
        logger.info(f"총 이미지 파일: {len(all_image_names)}개")
        
        # 1단계: case1 - (w,h) == (1800,1200) 또는 비율 정확 3:2
        logger.info("[1/4] case1 대상 선별")
        case1_targets, not_case1 = _filter_case1_eligible(all_image_names)
        logger.info(f"case1 대상: {len(case1_targets)}개, 다음 단계로 전달: {len(not_case1)}개")

        exception_files = []
        if case1_targets:
            logger.info("[1/4] case1 처리")
            exception_files_case1 = process_case("case1", target_files=case1_targets)
            if exception_files_case1 is None:
                logger.error("case1 처리 실패")
                exception_files_case1 = case1_targets  # 보수적으로 모두 예외 처리
            exception_files.extend(exception_files_case1)
        else:
            logger.info("case1 대상 없음, 건너뜀")

        next_candidates = list(set(not_case1 + exception_files))
        if not next_candidates:
            logger.info("모든 파일 처리 완료")
            return True

        # 2단계: case2 - (w,h) == (909,920) 정확 일치
        logger.info("[2/4] case2 대상 선별")
        case2_targets, not_case2 = _filter_case2_eligible(next_candidates)
        logger.info(f"case2 대상: {len(case2_targets)}개, 다음 단계로 전달: {len(not_case2)}개")

        exception_files = []
        if case2_targets:
            logger.info("[2/4] case2 처리")
            exception_files_case2 = process_case("case2", target_files=case2_targets)
            if exception_files_case2 is None:
                logger.error("case2 처리 실패")
                exception_files_case2 = case2_targets
            exception_files.extend(exception_files_case2)
        else:
            logger.info("case2 대상 없음, 건너뜀")

        next_candidates = list(set(not_case2 + exception_files))
        if not next_candidates:
            logger.info("모든 예외 해결")
            return True

        # 3단계: case3 - 세로가 가로보다 긴 이미지 (h > w)
        logger.info("[3/4] case3 대상 선별")
        case3_targets, not_case3 = _filter_case3_eligible(next_candidates)
        logger.info(f"case3 대상: {len(case3_targets)}개, 다음 단계로 전달: {len(not_case3)}개")

        exception_files = []
        if case3_targets:
            logger.info("[3/4] case3 처리")
            exception_files_case3 = process_case("case3", target_files=case3_targets)
            if exception_files_case3 is None:
                logger.error("case3 처리 실패")
                exception_files_case3 = case3_targets
            exception_files.extend(exception_files_case3)
        else:
            logger.info("case3 대상 없음, 건너뜀")

        next_candidates = list(set(not_case3 + exception_files))
        if not next_candidates:
            logger.info("모든 예외 해결")
            return True
        
        # 4단계: Claude API 처리
        logger.info(f"[4/4] Claude API 처리 ({len(next_candidates)}개 파일)")
        logger.warning(f"예상 비용: {len(next_candidates) * 76:.2f}원")
        user_input = input("계속 진행하시겠습니까? (y/n): ").lower().strip()
        
        if user_input == 'y':
            claude_converter = ClaudeConverter()  # 기본 모델 사용
            claude_converter.convert_specific_images(next_candidates)
        else:
            logger.warning("Claude 변환 건너뜀")
        
        # 최종 결과
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"총 작업 시간: {elapsed_time:.2f}초")
        
        if next_candidates and user_input != 'y':
            logger.warning(f"처리되지 못한 케이스: {len(next_candidates)}개")
        else:
            logger.info("모든 데이터가 성공적으로 처리되었습니다!")
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("사용자에 의해 중단됨")
        return False
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)