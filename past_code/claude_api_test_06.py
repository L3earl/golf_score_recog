"""
Claude API 테스트 스크립트 v06

의도: tst_img 폴더의 이미지들을 허깅페이스 SwinIR으로 업스케일링한 후 개선된 Claude API로 테스트
- 허깅페이스 SwinIR 모델을 사용하여 이미지를 4배 업스케일링하여 선명하게 만듦
- 개선된 Prompt와 유연한 Tool 스키마 적용
- tool_choice 제거하여 모델이 자율적으로 판단
- 각 이미지 처리 전에 사용자에게 처리 여부를 물어봄
- 결과를 콘솔에만 출력 (저장 없음)
- 최대한 독립적으로 작동
"""

import os
import base64
import json
import pandas as pd
from anthropic import Anthropic
from datetime import datetime
from dotenv import load_dotenv
import logging
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 프로젝트 루트 디렉토리
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 테스트 이미지 폴더 경로
TST_IMG_FOLDER = os.path.join(PROJECT_ROOT, "tst_img")

# 업스케일링된 이미지 저장 폴더 경로
TST_IMG2_FOLDER = os.path.join(PROJECT_ROOT, "tst_img2")

# 변환 결과 저장 폴더 경로
TST_CONVERT_FOLDER = os.path.join(PROJECT_ROOT, "tst_convert")

# 지원하는 이미지 확장자
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

# Claude 모델 설정
DEFAULT_CLAUDE_MODEL = "claude-opus-4-1-20250805"
CLAUDE_MAX_TOKENS = 4000

# SwinIR 모델 (전역 변수)
swinir_model = None
swinir_processor = None


def get_improved_prompt():
    """개선된 골프 스코어카드 분석 프롬프트
    
    의도: 모델이 스스로 데이터를 정제하도록 명확하고 구체적인 지침 제공
    
    Returns:
        개선된 프롬프트 문자열
    """
    return """당신은 골프 스코어카드 분석 전문가입니다. 이미지에서 각 홀(1-18)의 'PAR' 정보와 플레이어의 'SCORE' 정보만을 추출해 주세요.

[필수 준수 규칙]

'PAR' 행과 'SCORE' (또는 '점수', 'Rnd' 등) 행만 추출합니다.

무시할 행: 'YARDS'(거리), 'PUTT'(퍼트수), 'Status', 'Points', '센서영상' 등 스코어와 직접 관련 없는 숫자 행은 반드시 무시하세요.

무시할 열: 'OUT', 'IN', 'TOTAL', 'T', '합계' 등 요약/합계 열은 반드시 무시하세요. 오직 1번부터 18번 홀까지의 데이터만 순서대로 추출합니다.

레이아웃 처리: 스코어카드가 전반(1-9)과 후반(10-18)으로 나뉘어 있어도, 항상 1번부터 18번 홀까지 순서대로 결합하여 제공해야 합니다.

스코어 유형: 스코어는 4, 5, 3처럼 절대 타수일 수도 있고, -1, 0, 1처럼 PAR 기준 상대 타수일 수도 있습니다. 테이블에 보이는 그대로 추출해 주세요.

다중 플레이어: 여러 명의 플레이어가 있다면 모두 감지하여 각각의 스코어를 추출합니다.

플레이어 이름: 이름이 명확히 보이면 그대로 사용하고, 이름이 없으면 'Player 1', 'Player 2' 등으로 지정해 주세요.

제공된 tool을 사용하여 구조화된 JSON 형식으로 응답해 주세요."""


def get_flexible_tool_schema():
    """유연한 골프 스코어카드 추출 Tool 스키마
    
    의도: 플레이어 수에 제한이 없는 유연한 구조로 변경
    
    Returns:
        Tool 스키마 딕셔너리
    """
    return {
        "name": "extract_golf_scorecard",
        "description": "골프 스코어카드에서 PAR 및 모든 플레이어의 스코어를 추출합니다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "par": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "홀 1~18의 PAR 값 배열 (총 18개)"
                },
                "players": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "플레이어 이름 (e.g., 'Tiger Woods', '신재운'). 이름이 없으면 'Player 1' 등으로 지정."
                            },
                            "scores": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "해당 플레이어의 홀 1~18 스코어 배열 (총 18개). 테이블에 보이는 값 그대로(절대 또는 상대 스코어)."
                            }
                        },
                        "required": ["name", "scores"]
                    },
                    "description": "스코어카드에 있는 모든 플레이어의 목록"
                }
            },
            "required": ["par", "players"]
        }
    }


def load_swinir_model():
    """허깅페이스 SwinIR 모델 로드
    
    의도: 허깅페이스에서 제공하는 SwinIR 모델을 로드하여 이미지 업스케일링에 사용
    
    Returns:
        SwinIR 모델과 프로세서 튜플 또는 None (실패 시)
    """
    global swinir_model, swinir_processor
    
    try:
        # GPU 사용 가능 여부 확인
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"사용할 디바이스: {device}")
        
        # SwinIR 모델과 프로세서 로드 (4배 업스케일링)
        model_name = "caidas/swin2SR-classical-sr-x4-64"
        logger.info(f"SwinIR 모델 로딩 중: {model_name}")
        
        processor = Swin2SRImageProcessor.from_pretrained(model_name)
        model = Swin2SRForImageSuperResolution.from_pretrained(model_name)
        
        # 디바이스로 이동
        model = model.to(device)
        
        swinir_model = model
        swinir_processor = processor
        
        logger.info("SwinIR 모델 로드 완료!")
        return model, processor
        
    except Exception as e:
        logger.warning(f"SwinIR 모델 로드 실패: {e}")
        logger.info("SwinIR 없이 진행합니다.")
        return None, None


def resize_image_if_needed(image, max_size=2000):
    """이미지가 지정된 크기를 초과하면 리사이즈
    
    의도: 이미지의 가로세로가 2000px을 초과하면 비율을 유지하면서 리사이즈
    
    Args:
        image: OpenCV 이미지 배열
        max_size: 최대 크기 (기본값: 2000px)
    
    Returns:
        리사이즈된 이미지 배열
    """
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        # 비율 유지하면서 리사이즈
        if height > width:
            new_height = max_size
            new_width = int(width * max_size / height)
        else:
            new_width = max_size
            new_height = int(height * max_size / width)
        
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        logger.info(f"이미지 리사이즈: {width}x{height} → {new_width}x{new_height}")
    
    return image


def upscale_image(image_path):
    """이미지를 SwinIR으로 업스케일링하고 tst_img2 폴더에 저장
    
    의도: 입력 이미지를 SwinIR으로 4배 업스케일링하여 선명하게 만들고 파일로 저장
    
    Args:
        image_path: 업스케일링할 이미지 파일 경로
    
    Returns:
        업스케일링된 이미지 배열 또는 None (실패 시)
    """
    global swinir_model, swinir_processor
    
    try:
        # 이미지 로드 (OpenCV로 로드 후 PIL로 변환)
        img_cv = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_cv is None:
            logger.warning(f"이미지 로드 실패: {image_path}")
            return None
        
        # BGR을 RGB로 변환
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # SwinIR 모델이 있으면 사용
        if swinir_model is not None and swinir_processor is not None:
            try:
                # GPU 사용 가능 여부 확인
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # 이미지 전처리
                inputs = swinir_processor(img_pil, return_tensors="pt").to(device)
                
                # 모델 추론
                with torch.no_grad():
                    outputs = swinir_model(**inputs)
                
                # 결과 이미지 추출 및 후처리
                output_image = outputs.reconstruction.data.squeeze().cpu().clamp_(0, 1)
                output_image = torch.permute(output_image, (1, 2, 0)).numpy()
                output_image = (output_image * 255).astype(np.uint8)
                
                # RGB를 BGR로 변환 (OpenCV 형식)
                output = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
                
                logger.info(f"SwinIR 업스케일링 완료: {image_path}")
                
            except Exception as e:
                logger.warning(f"SwinIR 추론 실패: {e}")
                # SwinIR 실패시 간단한 업스케일링으로 fallback
                height, width = img_cv.shape[:2]
                output = cv2.resize(img_cv, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)
                logger.info(f"간단한 업스케일링으로 fallback: {image_path}")
        else:
            # SwinIR 모델이 없으면 간단한 업스케일링
            height, width = img_cv.shape[:2]
            output = cv2.resize(img_cv, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)
            logger.info(f"간단한 업스케일링 완료: {image_path}")
        
        # 이미지 크기 제한 (2000px 이하)
        output = resize_image_if_needed(output, max_size=2000)
        
        # tst_img2 폴더 생성 (없으면)
        os.makedirs(TST_IMG2_FOLDER, exist_ok=True)
        
        # 업스케일링된 이미지를 tst_img2 폴더에 저장
        image_name = os.path.basename(image_path)
        image_name_no_ext = os.path.splitext(image_name)[0]
        output_path = os.path.join(TST_IMG2_FOLDER, f"{image_name_no_ext}.png")
        
        success = cv2.imwrite(output_path, output)
        if success:
            logger.info(f"업스케일링된 이미지 저장 완료: {output_path}")
        else:
            logger.warning(f"업스케일링된 이미지 저장 실패: {output_path}")
        
        return output
            
    except Exception as e:
        logger.warning(f"이미지 업스케일링 실패 ({image_path}): {e}")
        return None


def get_claude_client():
    """Claude API 클라이언트 생성
    
    의도: 환경변수에서 API 키를 읽어 Claude 클라이언트 생성
    
    Returns:
        Anthropic 클라이언트 인스턴스
    
    Raises:
        ValueError: API 키가 설정되지 않은 경우
    """
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("ANTHROPIC_API_KEY가 .env 파일에 설정되지 않았습니다")
        raise ValueError("❌ ANTHROPIC_API_KEY가 .env 파일에 설정되지 않았습니다.")

    logger.info(f"Claude API 클라이언트 생성 완료! (모델: {DEFAULT_CLAUDE_MODEL})")
    return Anthropic(api_key=api_key)


def get_media_type(image_path):
    """파일 확장자에 따른 적절한 media_type 반환
    
    의도: 파일 확장자에 따라 Claude API가 요구하는 올바른 media_type 설정
    
    Args:
        image_path: 이미지 파일 경로
    
    Returns:
        적절한 media_type 문자열
    """
    ext = os.path.splitext(image_path)[1].lower()
    media_type_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff'
    }
    return media_type_map.get(ext, 'image/png')  # 기본값은 png


def compress_image_for_api(image, max_size_mb=4.5):
    """API 전송을 위해 이미지 압축
    
    의도: 이미지 파일 크기가 5MB 제한을 초과하지 않도록 압축
    
    Args:
        image: OpenCV 이미지 배열
        max_size_mb: 최대 크기 (MB, 기본값: 4.5MB)
    
    Returns:
        압축된 이미지 배열
    """
    # PNG로 인코딩하여 크기 확인
    _, buffer = cv2.imencode('.png', image)
    size_mb = len(buffer) / (1024 * 1024)
    
    if size_mb <= max_size_mb:
        return image
    
    logger.info(f"이미지 크기: {size_mb:.2f}MB, 압축 필요")
    
    # JPEG로 변환하여 압축
    quality = 95
    while quality > 30:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', image, encode_param)
        size_mb = len(buffer) / (1024 * 1024)
        
        if size_mb <= max_size_mb:
            logger.info(f"JPEG 압축 완료: {size_mb:.2f}MB (품질: {quality})")
            return image
        
        quality -= 10
    
    # 여전히 크면 이미지 크기 자체를 줄이기
    logger.warning("품질 압축으로도 크기 초과, 이미지 크기 축소")
    height, width = image.shape[:2]
    scale_factor = 0.8
    while scale_factor > 0.3:
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        size_mb = len(buffer) / (1024 * 1024)
        
        if size_mb <= max_size_mb:
            logger.info(f"이미지 크기 축소 완료: {size_mb:.2f}MB ({new_width}x{new_height})")
            return resized
        
        scale_factor -= 0.1
    
    logger.error("이미지 압축 실패")
    return image


def process_image(image_path, client, model_name, upscaled_img=None):
    """Claude API로 이미지 처리
    
    의도: 업스케일링된 이미지를 base64로 인코딩하여 Claude API에 전송
    
    Args:
        image_path: 처리할 이미지 파일 경로
        client: Claude API 클라이언트
        model_name: 사용할 모델명
        upscaled_img: 업스케일링된 이미지 배열 (선택사항)
    
    Returns:
        Claude API 응답 메시지 또는 None (실패 시)
    """
    try:
        if upscaled_img is not None:
            # 이미지 압축 적용
            compressed_img = compress_image_for_api(upscaled_img)
            
            # 압축된 이미지를 JPEG로 인코딩 (PNG보다 작음)
            _, buffer = cv2.imencode('.jpg', compressed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            image_data = base64.b64encode(buffer).decode('utf-8')
            media_type = 'image/jpeg'
            logger.info(f"업스케일링된 이미지 사용: {image_path}")
        else:
            # 업스케일링된 이미지가 없으면 원본 이미지 사용
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            media_type = get_media_type(image_path)
            logger.info(f"원본 이미지 사용: {image_path}")
        
        # 개선된 프롬프트와 유연한 스키마 사용
        prompt = get_improved_prompt()
        tools = [get_flexible_tool_schema()]

        message = client.messages.create(
            model=model_name,
            max_tokens=CLAUDE_MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            tools=tools
            # tool_choice 제거 - 모델이 자율적으로 판단
        )

        logger.debug(f"Claude API 호출 성공: {image_path} (media_type: {media_type})")
        return message
    except Exception as e:
        logger.error(f"Claude API 호출 실패 ({image_path}): {e}")
        return None


def process_image_with_retry(image_path, client, model_name, upscaled_img=None, max_retries=3):
    """재시도 로직이 포함된 Claude API 이미지 처리
    
    의도: API 호출 실패시 최대 3번까지 재시도
    
    Args:
        image_path: 처리할 이미지 파일 경로
        client: Claude API 클라이언트
        model_name: 사용할 모델명
        upscaled_img: 업스케일링된 이미지 배열 (선택사항)
        max_retries: 최대 재시도 횟수 (기본값: 3)
    
    Returns:
        Claude API 응답 메시지 또는 None (실패 시)
    """
    for attempt in range(max_retries):
        try:
            response = process_image(image_path, client, model_name, upscaled_img)
            if response is not None:
                if attempt > 0:
                    logger.info(f"재시도 성공 (시도 {attempt + 1})")
                return response
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 지수 백오프: 2초, 4초, 8초
                logger.warning(f"시도 {attempt + 1} 실패, {wait_time}초 후 재시도 중... ({e})")
                time.sleep(wait_time)
            else:
                logger.error(f"모든 재시도 실패: {e}")
    
    return None


def convert_to_csv_format(df, image_name):
    """DataFrame을 CSV 저장용 형태로 변환
    
    의도: case1_02.csv 형태로 변환 (Transpose + total1, total2, sum 추가)
    
    Args:
        df: 원본 DataFrame (홀, PAR, 플레이어들)
        image_name: 이미지 파일명 (확장자 제외)
    
    Returns:
        변환된 DataFrame 또는 None (실패 시)
    """
    try:
        # PAR 행과 플레이어 행들만 추출
        par_row = df['PAR'].values
        player_rows = []
        
        # 플레이어 컬럼들 추출 (PAR 제외)
        for col in df.columns:
            if col not in ['홀', 'PAR']:
                player_rows.append(df[col].values)
        
        if not player_rows:
            logger.warning(f"'{image_name}'에서 플레이어 데이터가 없습니다.")
            return None
        
        # 새로운 DataFrame 생성 (Transpose)
        data = {}
        
        # 홀별 컬럼 추가 (1홀, 2홀, ..., 18홀)
        for i in range(len(par_row)):
            data[f"{i+1}홀"] = [par_row[i]] + [player[i] for player in player_rows]
        
        # total1, total2, sum 초기화
        data['total1'] = []
        data['total2'] = []
        data['sum'] = []
        
        # PAR 행의 total1, total2, sum 계산
        par_total1 = sum(par_row[:9])
        par_total2 = sum(par_row[9:18])
        par_sum = par_total1 + par_total2
        
        data['total1'].append(par_total1)
        data['total2'].append(par_total2)
        data['sum'].append(par_sum)
        
        # 각 플레이어의 total1, total2, sum 계산
        for player_scores in player_rows:
            total1 = sum(player_scores[:9])  # 전반 9홀
            total2 = sum(player_scores[9:18])  # 후반 9홀
            total_sum = total1 + total2
            
            data['total1'].append(total1)
            data['total2'].append(total2)
            data['sum'].append(total_sum)
        
        result_df = pd.DataFrame(data)
        
        # CSV 파일로 저장
        os.makedirs(TST_CONVERT_FOLDER, exist_ok=True)
        csv_path = os.path.join(TST_CONVERT_FOLDER, f"{image_name}.csv")
        result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"CSV 저장 완료: {csv_path}")
        return result_df
        
    except Exception as e:
        logger.error(f"CSV 변환 오류 ({image_name}): {e}")
        return None


def parse_response_to_dataframe(response, image_name):
    """Claude API 응답을 DataFrame으로 변환
    
    의도: Claude API의 tool_use 응답에서 JSON 데이터를 추출하여 DataFrame 생성
    
    Args:
        response: Claude API 응답 메시지
        image_name: 이미지 파일명
    
    Returns:
        변환된 DataFrame 또는 None (실패 시)
    """
    try:
        # tool_use 응답인지 확인
        if response.stop_reason != "tool_use":
            logger.warning(f"'{image_name}'에서 tool_use 응답이 아닙니다. (Stop Reason: {response.stop_reason})")
            return None
        
        # tool_use 블록 찾기
        tool_use_block = next(
            (block for block in response.content if block.type == "tool_use"),
            None
        )
        
        if not tool_use_block:
            logger.warning(f"'{image_name}'에서 tool_use 블록을 찾을 수 없습니다.")
            return None
        
        # JSON 데이터 추출
        json_data = tool_use_block.input
        
        # 디버깅: Claude에서 받은 JSON 데이터 출력
        print(f"  🔍 Claude JSON 응답:")
        print(f"  {json.dumps(json_data, indent=2, ensure_ascii=False)}")
        
        # 필수 필드 확인
        if "par" not in json_data or "players" not in json_data:
            logger.warning(f"'{image_name}'에서 필수 필드 'par' 또는 'players'가 누락되었습니다.")
            return None
        
        par_data = json_data["par"]
        players_data = json_data["players"]
        
        # par 배열 길이 검증 (18개 이상)
        min_expected_length = 18
        actual_length = len(par_data)
        
        if actual_length < min_expected_length:
            logger.warning(f"par 배열 길이가 너무 짧습니다. 최소: {min_expected_length}, 실제: {actual_length}")
            return None
        
        # 플레이어가 있는지 확인
        if not players_data:
            logger.warning(f"'{image_name}'에서 플레이어 데이터가 없습니다.")
            return None
        
        # 모든 플레이어의 스코어 배열 길이 검증
        for i, player in enumerate(players_data):
            if "scores" not in player:
                logger.warning(f"플레이어 {i+1}에서 'scores' 필드가 누락되었습니다.")
                return None
            
            scores = player["scores"]
            if len(scores) != actual_length:
                player_name = player.get("name", f"Player {i+1}")
                logger.warning(f"'{player_name}' 스코어 배열 길이가 다릅니다. 예상: {actual_length}, 실제: {len(scores)}")
                return None
        
        print(f"  ✅ 모든 배열 길이 일치: {actual_length}개 홀, {len(players_data)}명 플레이어")
        
        # DataFrame 생성
        data = {
            "홀": list(range(1, actual_length + 1)),
            "PAR": par_data
        }
        
        # 각 플레이어의 스코어 추가
        for player in players_data:
            player_name = player.get("name", "Unknown Player")
            data[player_name] = player["scores"]
        
        df = pd.DataFrame(data)
        
        # 빈 DataFrame 체크
        if df.empty:
            logger.warning(f"'{image_name}'에서 빈 DataFrame이 생성되었습니다")
            return None
        
        return df
        
    except Exception as e:
        logger.error(f"JSON 파싱 오류 ({image_name}): {e}")
        logger.debug(f"API Raw Response - Stop Reason: {response.stop_reason}")
        return None


def get_image_files(folder_path):
    """폴더에서 이미지 파일 목록 가져오기
    
    의도: 지정된 폴더에서 지원하는 확장자의 이미지 파일들을 찾아 반환
    
    Args:
        folder_path: 검색할 폴더 경로
    
    Returns:
        이미지 파일 경로 리스트
    """
    image_files = []
    if not os.path.exists(folder_path):
        logger.error(f"폴더가 존재하지 않습니다: {folder_path}")
        return image_files
    
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext.lower()) for ext in IMAGE_EXTENSIONS):
            image_files.append(os.path.join(folder_path, filename))
    
    return sorted(image_files)


def main():
    """메인 실행 함수
    
    의도: tst_img 폴더의 모든 이미지를 SwinIR으로 업스케일링한 후 Claude API로 처리
    """
    print("=" * 60)
    print("Claude API 테스트 스크립트 v06 (SwinIR 업스케일링)")
    print("=" * 60)
    print(f"처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"테스트 폴더: {TST_IMG_FOLDER}")
    print(f"업스케일링 저장 폴더: {TST_IMG2_FOLDER}")
    print(f"CSV 저장 폴더: {TST_CONVERT_FOLDER}")
    print(f"모델: {DEFAULT_CLAUDE_MODEL}")
    print("-" * 60)
    
    # 환경변수 로드
    load_dotenv()
    
    try:
        # SwinIR 모델 로드
        print("SwinIR 모델 로딩 중...")
        load_swinir_model()
        
        # Claude API 클라이언트 생성
        client = get_claude_client()
        
        # 이미지 파일 목록 가져오기
        image_files = get_image_files(TST_IMG_FOLDER)
        
        if not image_files:
            logger.error(f"테스트 이미지 파일을 찾을 수 없습니다: {TST_IMG_FOLDER}")
            return
        
        logger.info(f"처리할 이미지 수: {len(image_files)}")
        
        # 처리 결과 통계
        processed_count = 0
        failed_count = 0
        
        # 먼저 모든 이미지 업스케일링 진행 (자동)
        print(f"\n🔄 모든 이미지 업스케일링 시작...")
        upscaled_images = {}
        
        for i, image_path in enumerate(image_files, 1):
            image_name = os.path.basename(image_path)
            print(f"[{i}/{len(image_files)}] 업스케일링 중: {image_name}")
            
            upscaled_img = upscale_image(image_path)
            if upscaled_img is not None:
                upscaled_images[image_path] = upscaled_img
                print(f"  ✅ 업스케일링 완료!")
            else:
                print(f"  ❌ 업스케일링 실패")
        
        print(f"\n✅ 모든 이미지 업스케일링 완료! ({len(upscaled_images)}개 성공)")
        
        # 사용자에게 전체 API 호출 여부 물어보기
        while True:
            user_input = input(f"\n모든 이미지에 대해 Claude API를 호출하시겠습니까? (y/n): ").lower().strip()
            if user_input in ['y', 'yes']:
                print("🚀 모든 이미지에 대해 Claude API 호출을 시작합니다...")
                break
            elif user_input in ['n', 'no']:
                print("⏭️ 모든 이미지 API 호출을 건너뜁니다.")
                return
            else:
                print("잘못된 입력입니다. 'y' (전체 호출) 또는 'n' (전체 건너뛰기)를 입력하세요.")
        
        # 각 이미지에 대해 Claude API 호출
        for i, image_path in enumerate(image_files, 1):
            image_name = os.path.basename(image_path)
            print(f"\n[{i}/{len(image_files)}] API 처리 중: {image_name}")
            
            if image_path not in upscaled_images:
                print(f"  ⏭️ 업스케일링 실패로 건너뛰기: {image_name}")
                continue
            
            try:
                # Claude API 호출
                response = process_image_with_retry(image_path, client, DEFAULT_CLAUDE_MODEL, upscaled_images[image_path])
                
                if response is None:
                    failed_count += 1
                    print(f"  ❌ 실패: Claude API 호출 실패")
                    continue
                
                # JSON을 DataFrame으로 변환
                result_df = parse_response_to_dataframe(response, image_name)
                
                if result_df is not None:
                    processed_count += 1
                    print(f"  ✅ 성공: {result_df.shape[0]}개 홀, {result_df.shape[1]-2}명 플레이어")
                    
                    # CSV 형태로 변환하여 저장
                    image_name_no_ext = os.path.splitext(image_name)[0]
                    csv_df = convert_to_csv_format(result_df, image_name_no_ext)
                    
                    if csv_df is not None:
                        print(f"  💾 CSV 저장 완료: {csv_df.shape[0]}행, {csv_df.shape[1]}열")
                        print(f"  📊 결과 미리보기:")
                        print(csv_df.to_string(index=False))
                    else:
                        print(f"  ⚠️ CSV 변환 실패")
                else:
                    failed_count += 1
                    print(f"  ❌ 실패: DataFrame 변환 실패")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"오류 발생 ({image_name}): {e}")
                print(f"  ❌ 실패: {e}")
                continue
        
        # 최종 결과 출력
        print("\n" + "=" * 60)
        print("처리 완료!")
        print(f"성공: {processed_count}개")
        print(f"실패: {failed_count}개")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"전체 처리 중 오류 발생: {e}")
        import traceback
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    main()
