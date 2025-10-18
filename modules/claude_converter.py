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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.utils import get_image_files, ensure_directory
from config import (
    DEFAULT_CLAUDE_MODEL, 
    RAW_IMG_FOLDER, 
    RESULT_CLAUDE_FOLDER, 
    CSV_ENCODING,
    IMAGE_EXTENSIONS,
    CLAUDE_GOLF_PROMPT,
    CLAUDE_EXTRACT_SCORECARD_TOOL,
    CLAUDE_MAX_TOKENS
)

logger = logging.getLogger(__name__)

class ClaudeConverter:
    """Claude API 변환 클래스
    
    의도: 이미지를 Claude API로 전송하여 골프 스코어카드 데이터를 구조화된 형태로 추출
    """
    
    def __init__(self, model_name=None, case="case1"):
        """Claude 변환기 초기화
        
        의도: 설정값을 로드하고 Claude API 클라이언트를 생성
        
        Args:
            model_name: 사용할 Claude 모델명 (None이면 기본값 사용)
            case: 처리 케이스 ('case1', 'case2', 'case3')
        """
        self.case = case
        self.model_name = model_name or DEFAULT_CLAUDE_MODEL
        self.prompt = CLAUDE_GOLF_PROMPT  # config에서 가져오기
        self.input_folder = RAW_IMG_FOLDER
        self.output_folder = RESULT_CLAUDE_FOLDER
        self.csv_encoding = CSV_ENCODING
        self.image_extensions = IMAGE_EXTENSIONS

        # 환경변수 로드
        load_dotenv()

        # Claude API 클라이언트 생성
        self.client = self._get_claude_client()
        logger.debug(f"ClaudeConverter 초기화 완료 (모델: {self.model_name}, 케이스: {self.case})")
    
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
    
    def _process_image_with_claude(self, image_path):
        """Claude API로 이미지 처리
        
        의도: 이미지를 base64로 인코딩하여 Claude API에 전송하고 구조화된 데이터 추출
        
        Args:
            image_path: 처리할 이미지 파일 경로
        
        Returns:
            Claude API 응답 메시지 또는 None (실패 시)
        """
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')

            # config에서 tools 가져오기
            tools = [CLAUDE_EXTRACT_SCORECARD_TOOL]

            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=CLAUDE_MAX_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
                            {"type": "text", "text": self.prompt}
                        ]
                    }
                ],
                tools=tools,
                tool_choice={"type": "tool", "name": "extract_golf_scorecard"}
            )

            logger.debug(f"Claude API 호출 성공: {image_path}")
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
                print(f"  ⚠️ '{image_name}'에서 tool_use 블록을 찾을 수 없습니다.")
                return None
            
            # JSON 데이터 추출
            json_data = tool_use_block.input
            
            # 디버깅: Claude에서 받은 JSON 데이터 출력
            print(f"  🔍 Claude JSON 응답:")
            print(f"  {json.dumps(json_data, indent=2, ensure_ascii=False)}")
            
            # 필수 필드 확인
            required_fields = ["par", "player1_score"]
            for field in required_fields:
                if field not in json_data:
                    print(f"  ⚠️ '{image_name}'에서 필수 필드 '{field}'가 누락되었습니다.")
                    return None
            
            # 배열 길이 검증 (18개 이상, 모든 배열이 같은 길이여야 함)
            min_expected_length = 18
            actual_length = len(json_data["par"])
            
            # 최소 길이 확인
            if actual_length < min_expected_length:
                print(f"  ❌ par 배열 길이가 너무 짧습니다. 최소: {min_expected_length}, 실제: {actual_length}")
                print(f"  데이터: {json_data['par']}")
                return None
            
            # 모든 배열이 같은 길이인지 확인
            all_arrays = [json_data["par"], json_data["player1_score"]]
            for i in range(2, 5):
                player_key = f"player{i}_score"
                if player_key in json_data:
                    all_arrays.append(json_data[player_key])
            
            for i, array in enumerate(all_arrays):
                if len(array) != actual_length:
                    array_name = ["par", "player1_score", "player2_score", "player3_score", "player4_score"][i]
                    print(f"  ❌ '{array_name}' 배열 길이가 다릅니다. 예상: {actual_length}, 실제: {len(array)}")
                    print(f"  데이터: {array}")
                    return None
            
            print(f"  ✅ 모든 배열 길이 일치: {actual_length}개 홀")
            
            # DataFrame 생성 (동적 홀 수)
            data = {
                "홀": list(range(1, actual_length + 1)),
                "PAR": json_data["par"],
                "플레이어1": json_data["player1_score"]
            }
            
            # 선택적 플레이어 스코어 추가
            for i in range(2, 5):
                player_key = f"player{i}_score"
                if player_key in json_data:
                    data[f"플레이어{i}"] = json_data[player_key]
            
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
        ensure_directory(self.output_folder)
        df.to_csv(output_file, index=False, header=True, encoding=self.csv_encoding)
    
    def convert_specific_images(self, image_names):
        """특정 이미지들만 Claude API로 변환
        
        의도: 지정된 이미지 파일명들을 Claude API로 처리하여 CSV 파일 생성
        
        Args:
            image_names: 처리할 이미지 파일명 리스트 (확장자 제외)
        
        Returns:
            변환 성공 여부
        """
        logger.info("Claude API 변환 시작...")
        logger.info(f"처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"입력 폴더: {self.input_folder}")
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
            
            # 모든 이미지 처리 결과를 저장할 리스트
            all_dataframes = []
            processed_count = 0
            failed_count = 0
            
            # 각 이미지 처리
            for i, image_path in enumerate(image_files, 1):
                image_name = os.path.basename(image_path)
                logger.info(f"[{i}/{len(image_files)}] 처리 중: {image_name}")
                
                try:
                    # Claude API 호출
                    message_response = self._process_image_with_claude(image_path)
                    
                    if message_response is None:
                        failed_count += 1
                        logger.warning(f"실패: Claude API 호출 실패")
                        continue
                    
                    # JSON을 DataFrame으로 변환
                    result_df = self._parse_json_to_dataframe(message_response, image_name)
                    
                    if result_df is not None:
                        all_dataframes.append(result_df)
                        processed_count += 1
                        logger.info(f"성공: {result_df.shape[0]}개 행 추출")
                    else:
                        failed_count += 1
                        logger.warning(f"실패: DataFrame 변환 실패")
                        
                except Exception as e:
                    failed_count += 1
                    logger.error(f"오류 발생 ({image_name}): {e}")
                    continue
            
            # 모든 DataFrame을 하나로 합치기
            if all_dataframes:
                logger.info("데이터 통합 중...")
                combined_df = pd.concat(all_dataframes, axis=0, ignore_index=False)
                
                # 결과 저장
                output_file = os.path.join(self.output_folder, f"{self.model_name.replace('-', '_')}.csv")
                self._save_dataframe_to_csv(combined_df, output_file)
                
                logger.info("Claude API 변환 완료!")
                logger.info(f"성공: {processed_count}개 이미지")
                logger.info(f"실패: {failed_count}개 이미지")
                logger.info(f"총 데이터: {combined_df.shape[0]}개 행, {combined_df.shape[1]}개 열")
                logger.info(f"저장 위치: {output_file}")
                
                return True
            else:
                logger.warning("처리된 데이터가 없습니다")
                return False
            
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

