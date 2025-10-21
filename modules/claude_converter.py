"""
Claude API 변환 모듈

의도: 이미지를 Claude API로 전송하여 골프 스코어카드 데이터 추출
- 이미지를 base64로 인코딩하여 API 호출
- 구조화된 JSON 응답을 DataFrame으로 변환
- CSV 파일로 결과 저장
"""

import os
import base64
import json
import pandas as pd
from anthropic import Anthropic
from datetime import datetime
from dotenv import load_dotenv
import logging
import sys
import cv2
import numpy as np
import urllib.request
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.utils import get_image_files, ensure_directory
from config import (
    DEFAULT_CLAUDE_MODEL, 
    RAW_IMG_FOLDER, 
    RAW_IMG_UPSCALE_FOLDER,
    RESULT_CONVERT_NUM_FOLDER, 
    CSV_ENCODING,
    IMAGE_EXTENSIONS,
    CLAUDE_GOLF_PROMPT,
    CLAUDE_EXTRACT_SCORECARD_TOOL,
    CLAUDE_MAX_TOKENS,
    MODELS_FOLDER,
    EDSR_MODEL_URL,
    UPSCALE_MAX_SIZE,
    CLAUDE_MAX_RETRIES,
    IMAGE_MAX_SIZE_MB,
    JPEG_INITIAL_QUALITY,
    JPEG_MIN_QUALITY
)

logger = logging.getLogger(__name__)

class ClaudeConverter:
    """Claude API 변환 클래스
    
    의도: 이미지를 Claude API로 전송하여 골프 스코어카드 데이터를 구조화된 형태로 추출
    """
    
    def __init__(self, model_name=None):
        """Claude 변환기 초기화
        
        의도: 설정값을 로드하고 Claude API 클라이언트를 생성
        
        Args:
            model_name: 사용할 Claude 모델명 (None이면 기본값 사용)
        """
        self.model_name = model_name or DEFAULT_CLAUDE_MODEL
        self.prompt = CLAUDE_GOLF_PROMPT
        self.input_folder = RAW_IMG_FOLDER
        self.upscale_folder = RAW_IMG_UPSCALE_FOLDER  # 업스케일링된 이미지 저장 폴더
        self.output_folder = RESULT_CONVERT_NUM_FOLDER
        self.csv_encoding = CSV_ENCODING
        self.image_extensions = IMAGE_EXTENSIONS
        self.models_folder = MODELS_FOLDER

        # 환경변수 로드
        load_dotenv()

        # Claude API 클라이언트 생성
        self.client = self._get_claude_client()
        
        # Super Resolution 모델 로드
        self.superres_model = self._load_superres_model()
        logger.debug(f"ClaudeConverter 초기화 완료 (모델: {self.model_name})")
    
    def _get_claude_client(self):
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

        logger.info(f"Claude API 클라이언트 생성 완료! (모델: {self.model_name})")
        return Anthropic(api_key=api_key)
    
    def _download_model_file(self):
        """EDSR 모델 파일 다운로드
        
        의도: EDSR_x4.pb 모델 파일을 자동으로 다운로드
        
        Returns:
            모델 파일 경로 또는 None (실패 시)
        """
        try:
            # models 폴더 생성
            os.makedirs(self.models_folder, exist_ok=True)
            
            model_path = os.path.join(self.models_folder, "EDSR_x4.pb")
            
            # 이미 다운로드된 파일이 있으면 스킵
            if os.path.exists(model_path):
                logger.info(f"모델 파일이 이미 존재합니다: {model_path}")
                return model_path
            
            # 모델 파일 다운로드
            logger.info("EDSR 모델 파일 다운로드 중...")
            
            urllib.request.urlretrieve(EDSR_MODEL_URL, model_path)
            logger.info(f"모델 파일 다운로드 완료: {model_path}")
            
            return model_path
            
        except Exception as e:
            logger.warning(f"모델 파일 다운로드 실패: {e}")
            return None

    def _load_superres_model(self):
        """OpenCV DNN Super Resolution 모델 로드
        
        의도: cv2.dnn_superres를 사용하여 슈퍼 해상도 모델 로드
        
        Returns:
            슈퍼 해상도 모델 인스턴스 또는 None (실패 시)
        """
        try:
            # 모델 파일 다운로드
            model_path = self._download_model_file()
            if not model_path:
                logger.warning("모델 파일을 다운로드할 수 없습니다.")
                return None
            
            # 슈퍼 해상도 모델 생성
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(model_path)
            sr.setModel("edsr", 4)  # EDSR 모델, 4배 업스케일링
            
            logger.info("OpenCV DNN Super Resolution 모델 로드 완료!")
            return sr
            
        except Exception as e:
            logger.warning(f"슈퍼 해상도 모델 로드 실패: {e}")
            logger.info("슈퍼 해상도 없이 진행합니다.")
            return None

    def _resize_image_if_needed(self, image, max_size=None):
        """이미지가 지정된 크기를 초과하면 리사이즈
        
        의도: 이미지의 가로세로가 지정된 크기를 초과하면 비율을 유지하면서 리사이즈
        
        Args:
            image: OpenCV 이미지 배열
            max_size: 최대 크기 (기본값: UPSCALE_MAX_SIZE)
        
        Returns:
            리사이즈된 이미지 배열
        """
        if max_size is None:
            max_size = UPSCALE_MAX_SIZE
            
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

    def _upscale_image(self, image_path):
        """이미지를 OpenCV DNN Super Resolution으로 업스케일링하고 파일로 저장
        
        의도: 입력 이미지를 cv2.dnn_superres로 4배 업스케일링하여 선명하게 만들고 파일로 저장
        
        Args:
            image_path: 업스케일링할 이미지 파일 경로
        
        Returns:
            업스케일링된 이미지 배열 또는 None (실패 시)
        """
        try:
            # 이미지 로드
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"이미지 로드 실패: {image_path}")
                return None
            
            # 업스케일링된 이미지 생성
            if self.superres_model is not None:
                output = self.superres_model.upsample(img)
                logger.info(f"OpenCV DNN Super Resolution 업스케일링 완료: {image_path}")
            else:
                # 슈퍼 해상도 모델이 없으면 간단한 업스케일링
                height, width = img.shape[:2]
                output = cv2.resize(img, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)
                logger.info(f"간단한 업스케일링 완료: {image_path}")
            
            # 이미지 크기 제한
            output = self._resize_image_if_needed(output)
            
            # 업스케일링된 이미지를 파일로 저장
            image_name = os.path.basename(image_path)
            image_name_no_ext = os.path.splitext(image_name)[0]
            upscale_output_path = os.path.join(self.upscale_folder, f"{image_name_no_ext}.png")
            
            # 업스케일링 폴더 생성
            ensure_directory(self.upscale_folder)
            
            # PNG로 저장 (품질 손실 없음)
            success = cv2.imwrite(upscale_output_path, output)
            if success:
                logger.info(f"업스케일링된 이미지 저장 완료: {upscale_output_path}")
            else:
                logger.warning(f"업스케일링된 이미지 저장 실패: {upscale_output_path}")
            
            return output
                
        except Exception as e:
            logger.warning(f"이미지 업스케일링 실패 ({image_path}): {e}")
            return None

    def _get_media_type(self, image_path):
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

    def _compress_image_for_api(self, image, max_size_mb=None):
        """API 전송을 위해 이미지 압축
        
        의도: 이미지 파일 크기가 제한을 초과하지 않도록 압축
        
        Args:
            image: OpenCV 이미지 배열
            max_size_mb: 최대 크기 (MB, 기본값: IMAGE_MAX_SIZE_MB)
        
        Returns:
            압축된 이미지 배열
        """
        if max_size_mb is None:
            max_size_mb = IMAGE_MAX_SIZE_MB
            
        # PNG로 인코딩하여 크기 확인
        _, buffer = cv2.imencode('.png', image)
        size_mb = len(buffer) / (1024 * 1024)
        
        if size_mb <= max_size_mb:
            return image
        
        logger.info(f"이미지 크기: {size_mb:.2f}MB, 압축 필요")
        
        # JPEG로 변환하여 압축
        quality = JPEG_INITIAL_QUALITY
        while quality > JPEG_MIN_QUALITY:
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
    
    def _process_image_with_retry(self, image_path, upscaled_img=None, max_retries=None):
        """재시도 로직이 포함된 Claude API 이미지 처리
        
        의도: API 호출 실패시 최대 재시도 횟수까지 재시도
        
        Args:
            image_path: 처리할 이미지 파일 경로
            upscaled_img: 업스케일링된 이미지 배열 (선택사항)
            max_retries: 최대 재시도 횟수 (기본값: CLAUDE_MAX_RETRIES)
        
        Returns:
            Claude API 응답 메시지 또는 None (실패 시)
        """
        if max_retries is None:
            max_retries = CLAUDE_MAX_RETRIES
            
        for attempt in range(max_retries):
            try:
                response = self._process_image(image_path, upscaled_img)
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

    def _process_image(self, image_path, upscaled_img=None):
        """Claude API로 이미지 처리
        
        의도: 이미지를 base64로 인코딩하여 Claude API에 전송하고 구조화된 데이터 추출
        
        Args:
            image_path: 처리할 이미지 파일 경로
            upscaled_img: 업스케일링된 이미지 배열 (선택사항)
        
        Returns:
            Claude API 응답 메시지 또는 None (실패 시)
        """
        try:
            if upscaled_img is not None:
                # 이미지 압축 적용
                compressed_img = self._compress_image_for_api(upscaled_img)
                
                # 압축된 이미지를 JPEG로 인코딩 (PNG보다 작음)
                _, buffer = cv2.imencode('.jpg', compressed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                image_data = base64.b64encode(buffer).decode('utf-8')
                media_type = 'image/jpeg'
                logger.info(f"업스케일링된 이미지 사용: {image_path}")
            else:
                # 업스케일링된 이미지가 없으면 원본 이미지 사용
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                media_type = self._get_media_type(image_path)
                logger.info(f"원본 이미지 사용: {image_path}")
            
            # 개선된 프롬프트와 유연한 스키마 사용
            tools = [CLAUDE_EXTRACT_SCORECARD_TOOL]

            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=CLAUDE_MAX_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                            {"type": "text", "text": self.prompt}
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
    
    def _parse_json_to_dataframe(self, message_response, image_name):
        """Claude API 응답을 DataFrame으로 변환
        
        의도: Claude API의 tool_use 응답에서 JSON 데이터를 추출하여 DataFrame 생성
        
        Args:
            message_response: Claude API 응답 메시지
            image_name: 이미지 파일명
        
        Returns:
            변환된 DataFrame 또는 None (실패 시)
        """
        try:
            # tool_use 응답인지 확인
            if message_response.stop_reason != "tool_use":
                logger.warning(f"'{image_name}'에서 tool_use 응답이 아닙니다. (Stop Reason: {message_response.stop_reason})")
                return None
            
            # tool_use 블록 찾기
            tool_use_block = next(
                (block for block in message_response.content if block.type == "tool_use"),
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
            logger.debug(f"API Raw Response - Stop Reason: {message_response.stop_reason}")
            return None
    
    def _save_dataframe_to_csv(self, df, output_file):
        """DataFrame을 CSV 파일로 저장
        
        의도: 처리된 DataFrame을 지정된 경로에 CSV 파일로 저장
        
        Args:
            df: 저장할 DataFrame
            output_file: 출력 파일 경로
        """
        ensure_directory(os.path.dirname(output_file))
        df.to_csv(output_file, index=False, header=True, encoding=self.csv_encoding)

    def _convert_to_csv_format(self, df, image_name):
        """DataFrame을 CSV 저장용 형태로 변환
        
        의도: case1_02.csv 형태로 변환 (Transpose + total1, total2, sum 추가)
        추가로 절대점수로 표시된 플레이어 데이터를 상대점수로 변환하는 후처리 적용
        
        후처리 로직:
        1. 각 플레이어의 diff_sum 계산 (|스코어 - PAR|의 합)
        2. diff_sum < 36인 플레이어는 절대점수로 판단하여 상대점수로 변환
        3. 모든 플레이어의 total1, total2, sum을 절대점수 기준으로 재계산
        
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
            player_names = []
            
            # PAR 데이터 길이 확인 (18홀이어야 함)
            if len(par_row) != 18:
                logger.warning(f"'{image_name}'에서 PAR 데이터 길이가 예상과 다릅니다. 예상: 18, 실제: {len(par_row)}")
                return None
            
            # 플레이어 컬럼들 추출 (PAR 제외)
            for col in df.columns:
                if col not in ['홀', 'PAR']:
                    player_rows.append(df[col].values)
                    player_names.append(col)
            
            if not player_rows:
                logger.warning(f"'{image_name}'에서 플레이어 데이터가 없습니다.")
                return None
            
            # 후처리: 절대점수 → 상대점수 변환
            processed_player_rows = []
            for i, (player_name, player_scores) in enumerate(zip(player_names, player_rows)):
                # 1단계: 0 이하 숫자 확인
                has_zero_or_negative = any(score <= 0 for score in player_scores)
                
                # 2단계: 평균값 기반 판단
                par_average = sum(par_row) / len(par_row)  # PAR 평균
                player_average = sum(player_scores) / len(player_scores)  # 플레이어 평균
                threshold = par_average - 0.5  # 임계값
                
                logger.info(f"플레이어 '{player_name}' PAR 평균: {par_average:.2f}, 플레이어 평균: {player_average:.2f}, 임계값: {threshold:.2f}")
                
                if has_zero_or_negative:
                    # 0 이하 숫자가 있으면 상대점수로 판단 (변환 안함)
                    logger.info(f"플레이어 '{player_name}' 상대점수로 판단 (0 이하 숫자 포함)")
                    processed_player_rows.append(player_scores)
                elif player_average > threshold:
                    # 플레이어 평균 > (PAR 평균 - 0.5)이면 절대점수로 판단하여 상대점수로 변환
                    logger.info(f"플레이어 '{player_name}' 절대점수 → 상대점수 변환 (플레이어 평균: {player_average:.2f} > 임계값: {threshold:.2f})")
                    relative_scores = [score - par for score, par in zip(player_scores, par_row)]
                    processed_player_rows.append(relative_scores)
                else:
                    # 그 외는 상대점수로 판단하여 변환 안함
                    logger.info(f"플레이어 '{player_name}' 상대점수 유지 (플레이어 평균: {player_average:.2f} <= 임계값: {threshold:.2f})")
                    processed_player_rows.append(player_scores)
            
            # 새로운 DataFrame 생성 (Transpose)
            data = {}
            
            # 홀별 컬럼 추가 (1홀, 2홀, ..., 18홀)
            for i in range(len(par_row)):
                data[f"{i+1}홀"] = [par_row[i]] + [player[i] for player in processed_player_rows]
            
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
            
            # 3단계: 각 플레이어의 total1, total2, sum 재계산 (절대점수 기준)
            for player_scores in processed_player_rows:
                # 플레이어의 1~9홀, 10~18홀 합계 계산
                player_total1 = sum(player_scores[:9])   # 전반 9홀 합계
                player_total2 = sum(player_scores[9:18]) # 후반 9홀 합계
                
                # 절대점수 기준으로 total 계산
                total1 = par_total1 + player_total1  # PAR의 total1 + 플레이어의 1~9홀 합계
                total2 = par_total2 + player_total2  # PAR의 total2 + 플레이어의 10~18홀 합계
                total_sum = total1 + total2
                
                data['total1'].append(total1)
                data['total2'].append(total2)
                data['sum'].append(total_sum)
                
                logger.info(f"플레이어 total 재계산: total1={total1}, total2={total2}, sum={total_sum}")
            
            result_df = pd.DataFrame(data)
            
            return result_df
            
        except Exception as e:
            logger.error(f"CSV 변환 오류 ({image_name}): {e}")
            return None
    
    def convert_specific_images(self, image_names):
        """특정 이미지들만 Claude API로 변환
        
        의도: 지정된 이미지 파일명들을 Claude API로 처리하여 개별 CSV 파일 생성
        흐름: data/raw_img → data/raw_img_upscale → Claude API → data/result_convert/case99
        
        Args:
            image_names: 처리할 이미지 파일명 리스트 (확장자 제외)
        
        Returns:
            변환 성공 여부
        """
        logger.info("Claude API 변환 시작...")
        logger.info(f"처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"입력 폴더: {self.input_folder}")
        logger.info(f"업스케일링 저장 폴더: {self.upscale_folder}")
        logger.info(f"출력 폴더: {self.output_folder}")
        logger.info(f"모델: {self.model_name}")
        logger.info("-" * 60)
        
        try:
            # 특정 이미지 파일들만 처리
            image_files = []
            for image_name in image_names:
                # 확장자 추가
                for ext in self.image_extensions:
                    image_path = os.path.join(self.input_folder, f"{image_name}{ext}")
                    if os.path.exists(image_path):
                        image_files.append(image_path)
                        break
            
            if not image_files:
                logger.error(f"지정된 이미지 파일들을 찾을 수 없습니다: {image_names}")
                return False
            
            logger.info(f"처리할 이미지 수: {len(image_files)}")
            
            # 처리 결과 통계
            processed_count = 0
            failed_count = 0
            
            # 먼저 모든 이미지 업스케일링 진행
            logger.info("모든 이미지 업스케일링 시작...")
            upscaled_images = {}
            
            for i, image_path in enumerate(image_files, 1):
                image_name = os.path.basename(image_path)
                logger.info(f"[{i}/{len(image_files)}] 업스케일링 중: {image_name}")
                
                upscaled_img = self._upscale_image(image_path)
                if upscaled_img is not None:
                    upscaled_images[image_path] = upscaled_img
                    logger.info(f"업스케일링 완료!")
                else:
                    logger.warning(f"업스케일링 실패")
            
            logger.info(f"모든 이미지 업스케일링 완료! ({len(upscaled_images)}개 성공)")
            
            # case99 전용 출력 폴더 설정
            case99_output_folder = os.path.join(self.output_folder, "case99")
            ensure_directory(case99_output_folder)
            
            # 각 이미지 처리
            for i, image_path in enumerate(image_files, 1):
                image_name = os.path.basename(image_path)
                image_name_no_ext = os.path.splitext(image_name)[0]
                logger.info(f"[{i}/{len(image_files)}] API 처리 중: {image_name}")
                
                if image_path not in upscaled_images:
                    logger.warning(f"업스케일링 실패로 건너뛰기: {image_name}")
                    failed_count += 1
                    continue
                
                try:
                    # Claude API 호출
                    response = self._process_image_with_retry(image_path, upscaled_images[image_path])
                    
                    if response is None:
                        failed_count += 1
                        logger.warning(f"실패: Claude API 호출 실패")
                        continue
                    
                    # JSON을 DataFrame으로 변환
                    result_df = self._parse_json_to_dataframe(response, image_name)
                    
                    if result_df is not None:
                        logger.info(f"성공: {result_df.shape[0]}개 홀, {result_df.shape[1]-2}명 플레이어")
                        
                        # CSV 형태로 변환하여 저장
                        csv_df = self._convert_to_csv_format(result_df, image_name_no_ext)
                        
                        if csv_df is not None:
                            # case99 폴더에 개별 CSV 파일로 저장
                            output_file = os.path.join(case99_output_folder, f"{image_name_no_ext}.csv")
                            self._save_dataframe_to_csv(csv_df, output_file)
                            
                            processed_count += 1
                            logger.info(f"CSV 저장 완료: {csv_df.shape[0]}행, {csv_df.shape[1]}열")
                            logger.info(f"저장 위치: {output_file}")
                        else:
                            failed_count += 1
                            logger.warning(f"실패: CSV 변환 실패")
                    else:
                        failed_count += 1
                        logger.warning(f"실패: DataFrame 변환 실패")
                        
                except Exception as e:
                    failed_count += 1
                    logger.error(f"오류 발생 ({image_name}): {e}")
                    continue
            
            # 최종 결과
            logger.info("Claude API 변환 완료!")
            logger.info(f"성공: {processed_count}개 이미지")
            logger.info(f"실패: {failed_count}개 이미지")
            logger.info(f"업스케일링된 이미지 저장 위치: {self.upscale_folder}")
            logger.info(f"결과 CSV 저장 위치: {case99_output_folder}")
            
            return processed_count > 0
            
        except Exception as e:
            logger.error(f"전체 처리 중 오류 발생: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def convert_all_images(self):
        """모든 이미지를 Claude API로 변환
        
        의도: 입력 폴더의 모든 이미지를 Claude API로 처리하여 CSV 파일 생성
        
        Returns:
            변환 성공 여부
        """
        image_files = get_image_files(self.input_folder, self.image_extensions)
        image_names = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
        return self.convert_specific_images(image_names)

