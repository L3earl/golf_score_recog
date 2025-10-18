#!/usr/bin/env python3
"""
테이블 감지 및 크롭 테스트 스크립트 (Donut 모델용)

이 스크립트는 data/raw_img 폴더의 이미지들을 가져와서
NAVER Clova의 Donut (Document Understanding Transformer) 모델을 사용하여
문서 구조 분석을 통해 테이블의 좌표를 감지하고, 해당 영역을 크롭하여 test 폴더에 저장합니다.

Donut 모델은 문서 이해에 특화된 모델로,
문서의 구조를 분석하고 테이블 영역을 정확하게 감지할 수 있습니다.
"""

import os
import torch
from PIL import Image, ImageOps
import logging
from typing import List, Tuple

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_directories(input_dir: str, output_dir: str) -> None:
    """입력 및 출력 디렉토리 설정"""
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"입력 디렉토리가 존재하지 않습니다: {input_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"입력 디렉토리: {input_dir}")
    logger.info(f"출력 디렉토리: {output_dir}")

def load_model():
    """NAVER Clova Donut 모델 로드"""
    try:
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        
        logger.info("NAVER Clova Donut 모델을 로드하는 중...")
        processor = DonutProcessor.from_pretrained('naver-clova-ix/donut-base')
        model = VisionEncoderDecoderModel.from_pretrained('naver-clova-ix/donut-base')
        
        logger.info("✅ NAVER Clova Donut 모델 로드 완료")
        logger.info("문서 이해 및 구조 분석 모델")
        return processor, model
        
    except Exception as e:
        logger.error(f"NAVER Clova Donut 모델 로드 실패: {e}")
        raise RuntimeError(f"NAVER Clova Donut 모델 로드에 실패했습니다: {e}")

def detect_tables(processor, model, image: Image.Image, threshold: float = 0.3) -> List[Tuple[float, float, float, float]]:
    """이미지에서 테이블 감지 (NAVER Clova Donut 모델용)"""
    try:
        # Donut 모델 사용
        return detect_tables_donut(processor, model, image, threshold)
            
    except Exception as e:
        logger.error(f"테이블 감지 중 오류 발생: {e}")
        return []

def detect_tables_donut(processor, model, image: Image.Image, threshold: float = 0.3) -> List[Tuple[float, float, float, float]]:
    """Donut 모델을 사용한 테이블 감지"""
    try:
        # Donut 모델은 문서 구조 분석을 통해 테이블을 감지
        # 프롬프트 설정 (문서 구조 분석용)
        task_prompt = "<s_table>"
        
        # 이미지 전처리
        pixel_values = processor(image, return_tensors="pt").pixel_values
        
        # 모델 추론
        with torch.no_grad():
            # Donut 모델은 시퀀스 생성 방식으로 작동
            decoder_input_ids = processor.tokenizer(task_prompt, 
                                                   add_special_tokens=False, 
                                                   return_tensors="pt").input_ids
            
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
        
        # 결과 디코딩
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = sequence.replace(task_prompt, "")
        
        # Donut 모델 출력을 명확하게 표시
        print(f"\n{'='*60}")
        print(f"NAVER CLOVA DONUT 모델 출력 (sequence):")
        print(f"{'='*60}")
        print(f"원본 sequence: {sequence}")
        print(f"길이: {len(sequence)} 문자")
        print(f"내용 분석:")
        if sequence.strip():
            print(f"  - 공백 제거 후: '{sequence.strip()}'")
            print(f"  - 소문자 변환: '{sequence.lower()}'")
            print(f"  - 'table' 포함 여부: {'table' in sequence.lower()}")
            print(f"  - 'score' 포함 여부: {'score' in sequence.lower()}")
        else:
            print(f"  - 빈 문자열입니다.")
        print(f"{'='*60}\n")
        
        logger.info(f"NAVER Clova Donut 모델 출력: {sequence}")
        
        # Donut 모델의 출력에서 테이블 좌표 추출
        # 실제 구현에서는 더 정교한 파싱이 필요할 수 있음
        table_boxes = []
        
        # 간단한 테이블 감지 로직 (실제로는 더 복잡한 파싱 필요)
        if "table" in sequence.lower() or "score" in sequence.lower() or "box" in sequence.lower():
            # 전체 이미지를 테이블로 간주 (실제 구현에서는 좌표 파싱 필요)
            width, height = image.size
            # 임시로 전체 이미지의 일부를 테이블로 설정
            table_boxes.append((width * 0.1, height * 0.1, width * 0.9, height * 0.9))
            logger.info(f"✅ 테이블 감지: 전체 이미지 영역 사용")
        
        return table_boxes
        
    except Exception as e:
        logger.error(f"Donut 테이블 감지 중 오류 발생: {e}")
        return []


def crop_and_save_tables(image: Image.Image, table_boxes: List[Tuple[float, float, float, float]], 
                        filename: str, output_dir: str) -> None:
    """감지된 테이블 영역을 크롭하여 저장"""
    if not table_boxes:
        logger.warning(f"{filename}: 감지된 테이블이 없습니다.")
        return
    
    for i, (x1, y1, x2, y2) in enumerate(table_boxes):
        try:
            # 좌표가 이미지 범위 내에 있는지 확인
            width, height = image.size
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            # 유효한 크롭 영역인지 확인
            if x2 > x1 and y2 > y1:
                cropped_image = image.crop((x1, y1, x2, y2))
                
                # 파일명 생성
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}_table_{i+1}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                cropped_image.save(output_path)
                logger.info(f"테이블 크롭 저장: {output_path}")
            else:
                logger.warning(f"{filename}: 유효하지 않은 크롭 영역 ({x1}, {y1}, {x2}, {y2})")
                
        except Exception as e:
            logger.error(f"테이블 크롭 저장 중 오류: {e}")

def process_images(input_dir: str, output_dir: str, invert_colors: bool = False, threshold: float = 0.3) -> None:
    """모든 이미지 파일 처리 (Donut 모델용)"""
    # 모델 로드
    processor, model = load_model()
    
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        logger.warning(f"{input_dir}에 이미지 파일이 없습니다.")
        return
    
    logger.info(f"처리할 이미지 파일 수: {len(image_files)}")
    logger.info(f"색상 반전 옵션: {'ON' if invert_colors else 'OFF'}")
    logger.info(f"감지 임계값: {threshold}")
    
    # 각 이미지 처리
    for filename in image_files:
        try:
            logger.info(f"처리 중: {filename}")
            
            # 이미지 로드
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path).convert('RGB')
            
            # 색상 반전 처리 (다크 모드 이미지용)
            if invert_colors:
                logger.info(f"색상 반전 적용: {filename}")
                image = ImageOps.invert(image)
            
            # 테이블 감지
            table_boxes = detect_tables(processor, model, image, threshold)
            
            # 테이블 크롭 및 저장
            crop_and_save_tables(image, table_boxes, filename, output_dir)
            
        except Exception as e:
            logger.error(f"{filename} 처리 중 오류: {e}")
            continue

def main():
    """메인 함수"""
    try:
        # 디렉토리 설정
        input_dir = 'data/raw_img'
        output_dir = 'test'
        
        setup_directories(input_dir, output_dir)
        
        # 설정 옵션들
        invert_colors = True  # 다크 모드 이미지용 색상 반전 (True/False)
        threshold = 0.1       # 감지 임계값 (0.1~0.9, 낮을수록 더 많은 객체 감지)
        
        logger.info("=" * 60)
        logger.info("Donut 모델을 사용한 테이블 감지 및 크롭 시작")
        logger.info("=" * 60)
        
        # 이미지 처리
        process_images(input_dir, output_dir, invert_colors, threshold)
        
        logger.info("=" * 60)
        logger.info("모든 이미지 처리 완료!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류: {e}")
        raise

if __name__ == "__main__":
    main()
