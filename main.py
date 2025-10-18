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
from modules.simple_ocr_crop import process_case3_ocr_crop
from config import CASES, RAW_IMG_FOLDER, IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

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

def process_case(case, target_files=None, use_template_matching=False):
    """단일 케이스 처리 (템플릿 매칭 → 전처리 → OCR → 후처리)
    
    의도: 지정된 케이스에 대해 전체 처리 파이프라인을 순차적으로 실행
    
    Args:
        case: 처리 케이스 ('case1', 'case2', 'case3')
        target_files: 처리할 파일명 리스트 (None이면 전체)
        use_template_matching: case3에서 템플릿 매칭 사용 여부
    
    Returns:
        처리 성공 여부 또는 실패한 파일 리스트
    """
    logger.info(f"{case.upper()} 처리 시작")
    if target_files:
        logger.info(f"대상 파일: {len(target_files)}개")
    
    try:
        # OCR 크롭 단계 (case3에서만 사용)
        ocr_crop_failed_files = []
        if use_template_matching:
            logger.info(f"[1/4] {case} OCR 크롭 중...")
            success_files = process_case3_ocr_crop(target_files)
            
            if not success_files:
                logger.error(f"{case} OCR 크롭 실패")
                # OCR 크롭이 완전히 실패해도 실패한 파일들을 예외로 처리
                ocr_crop_failed_files = target_files
                logger.warning(f"모든 파일이 OCR 크롭 실패: {len(ocr_crop_failed_files)}개 파일")
                return ocr_crop_failed_files
            
            # OCR 크롭 실패한 파일들 추적
            ocr_crop_failed_files = [f for f in target_files if f not in success_files]
            
            logger.info(f"OCR 크롭 성공: {len(success_files)}개 파일")
            if ocr_crop_failed_files:
                logger.warning(f"OCR 크롭 실패: {len(ocr_crop_failed_files)}개 파일")
            
            target_files = success_files
            step_prefix = "[2/4]"
            step_suffix = "[3/4]"
            step_final = "[4/4]"
        else:
            step_prefix = "[1/3]"
            step_suffix = "[2/3]"
            step_final = "[3/3]"
    
        # 1. 전처리
        logger.info(f"{step_prefix} {case} 전처리 중...")
        preprocessor = ImagePreprocessor(case=case)
        if not preprocessor.process_all_images(target_files=target_files):
            logger.error(f"{case} 전처리 실패")
            return None
        
        # 2. OCR 변환
        logger.info(f"{step_suffix} {case} OCR 변환 중...")
        converter = OCRConverter(case=case)
        if not converter.convert_all_folders(target_files=target_files):
            logger.error(f"{case} OCR 변환 실패")
            return None
        
        # 3. 후처리
        logger.info(f"{step_final} {case} 후처리 중...")
        processor = PostProcessor(case=case)
        results = processor.process_all_files(target_files=target_files)
        if not results:
            logger.error(f"{case} 후처리 실패")
            return None
        
        # 예외 파일 수집
        exception_files = [r['folder'] for r in results 
                          if 'exceptions' in r and len(r['exceptions']) > 0]
        
        # OCR 크롭 실패한 파일들도 예외 파일에 추가
        if ocr_crop_failed_files:
            exception_files.extend(ocr_crop_failed_files)
            logger.warning(f"OCR 크롭 실패 파일 추가: {len(ocr_crop_failed_files)}개")
    
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
        
        # 1단계: case1으로 전체 처리
        logger.info("[1/4] case1 처리")
        exception_files = process_case("case1")
        
        if not exception_files:
            logger.info("모든 파일 처리 완료")
            return True
        
        # 2단계: case1 예외를 case2로 처리
        logger.info(f"[2/4] case2 재처리 ({len(exception_files)}개 파일)")
        exception_files = process_case("case2", target_files=exception_files)
        
        if not exception_files:
            logger.info("모든 예외 해결")
            return True
        
        # 3단계: case2 예외를 case3로 처리
        logger.info(f"[3/4] case3 재처리 ({len(exception_files)}개 파일)")
        exception_files = process_case("case3", target_files=exception_files, use_template_matching=True)
        
        if not exception_files:
            logger.info("모든 예외 해결")
            return True
        
        # 4단계: Claude API 처리
        logger.info(f"[4/4] Claude API 처리 ({len(exception_files)}개 파일)")
        logger.warning(f"예상 비용: {len(exception_files) * 76:.2f}원")
        user_input = input("계속 진행하시겠습니까? (y/n): ").lower().strip()
        
        if user_input == 'y':
            claude_converter = ClaudeConverter(case="case1")  # 기본 case
            claude_converter.convert_specific_images(exception_files)
        else:
            logger.warning("Claude 변환 건너뜀")
        
        # 최종 결과
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"총 작업 시간: {elapsed_time:.2f}초")
        
        if exception_files and user_input != 'y':
            logger.warning(f"처리되지 못한 케이스: {len(exception_files)}개")
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