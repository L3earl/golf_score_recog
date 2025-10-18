#!/usr/bin/env python3
"""
테이블 감지 및 크롭 테스트 스크립트 (PubLayNet 모델용)

이 스크립트는 data/raw_img 폴더의 이미지들을 가져와서
Facebook의 DETR ResNet-101 PubLayNet 모델을 사용하여
테이블의 좌표를 감지하고, 해당 영역을 크롭하여 test 폴더에 저장합니다.

PubLayNet 모델은 문서 레이아웃 분석에 특화된 모델로,
Text, Title, List, Table, Figure 등의 클래스를 감지할 수 있습니다.
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
    """Facebook DETR ResNet-101 PubLayNet 모델 로드"""
    try:
        from transformers import DetrImageProcessor, DetrForObjectDetection
        
        logger.info("Facebook DETR ResNet-101 PubLayNet 모델을 로드하는 중...")
        processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-101-publaynet')
        model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101-publaynet')
        
        logger.info("✅ Facebook DETR ResNet-101 PubLayNet 모델 로드 완료")
        logger.info(f"모델 라벨 정보: {list(model.config.id2label.values())}")
        return processor, model
        
    except Exception as e:
        logger.error(f"Facebook DETR ResNet-101 PubLayNet 모델 로드 실패: {e}")
        raise RuntimeError(f"Facebook DETR ResNet-101 PubLayNet 모델 로드에 실패했습니다: {e}")

def detect_tables(processor, model, image: Image.Image, threshold: float = 0.3) -> List[Tuple[float, float, float, float]]:
    """이미지에서 테이블 감지 (PubLayNet 모델용)"""
    try:
        # 이미지 전처리
        inputs = processor(images=image, return_tensors='pt')
        
        # 모델 추론
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 결과 후처리 (더 낮은 threshold로 시도)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]
        
        # 테이블 박스 추출
        table_boxes = []
        detected_objects = []
        
        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            # 모델의 라벨 ID를 라벨 이름으로 변환
            label_name = model.config.id2label.get(label.item(), f"label_{label.item()}")
            detected_objects.append((label_name, score.item()))
            
            # ❗️❗️ PubLayNet 모델용 테이블 감지 로직 ❗️❗️
            # PubLayNet은 'Table' 클래스를 가질 수 있음
            if (label_name.lower() == 'table' or label_name.lower() == 'table_caption') and score > threshold:
                box_coords = [round(coord.item(), 2) for coord in box]
                table_boxes.append(tuple(box_coords))
                logger.info(f"✅ 'Table' 감지: 신뢰도: {score:.3f}, 좌표: {box_coords}")
            elif score > threshold:
                # 테이블이 아닌 다른 객체가 감지된 경우 (디버깅용)
                logger.info(f"🔍 감지했으나 무시함: {label_name}, 신뢰도: {score:.3f}")
        
        # 감지된 모든 객체 요약
        if detected_objects:
            logger.info(f"📊 감지된 모든 객체 (threshold={threshold}): {detected_objects}")
        
        return table_boxes
        
    except Exception as e:
        logger.error(f"테이블 감지 중 오류 발생: {e}")
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
    """모든 이미지 파일 처리 (PubLayNet 모델용)"""
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
        threshold = 0.3       # 감지 임계값 (0.1~0.9, 낮을수록 더 많은 객체 감지)
        
        logger.info("=" * 60)
        logger.info("PubLayNet 모델을 사용한 테이블 감지 및 크롭 시작")
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
