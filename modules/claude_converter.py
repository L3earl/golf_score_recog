"""
Claude API 변환 모듈
- Claude API를 사용한 이미지 텍스트 추출
- CSV 응답을 DataFrame으로 변환
- CSV 파일로 저장
"""

import os
import base64
import json
import pandas as pd
from anthropic import Anthropic
from datetime import datetime
from dotenv import load_dotenv
from io import StringIO
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEFAULT_CLAUDE_MODEL, 
    CLAUDE_PROMPT, 
    RAW_IMG_FOLDER, 
    RESULT_CLAUDE_FOLDER, 
    CSV_ENCODING,
    IMAGE_EXTENSIONS
)

class ClaudeConverter:
    """Claude API 변환 클래스"""
    
    def __init__(self, model_name=None):
        """Claude 변환기 초기화"""
        self.model_name = model_name or DEFAULT_CLAUDE_MODEL
        self.prompt = CLAUDE_PROMPT
        self.input_folder = RAW_IMG_FOLDER
        self.output_folder = RESULT_CLAUDE_FOLDER
        self.csv_encoding = CSV_ENCODING
        self.image_extensions = IMAGE_EXTENSIONS
        
        # 환경변수 로드
        load_dotenv()
        
        # Claude API 클라이언트 생성
        self.client = self._get_claude_client()
    
    def _get_claude_client(self):
        """Claude API 클라이언트를 생성합니다."""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("❌ ANTHROPIC_API_KEY가 .env 파일에 설정되지 않았습니다.")
        
        print(f"✅ Claude API 클라이언트 생성 완료! (모델: {self.model_name})")
        return Anthropic(api_key=api_key)
    
    def _process_image_with_claude(self, image_path):
        """Claude API를 사용하여 이미지에서 텍스트를 추출합니다."""
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
                            {"type": "text", "text": self.prompt}
                        ]
                    }
                ]
            )
            
            return message.content[0].text
        except Exception as e:
            print(f"  ❌ Claude API 호출 실패 {image_path}: {e}")
            return None
    
    def _parse_csv_to_dataframe(self, csv_string, image_name):
        """Claude가 반환한 CSV 형식의 문자열을 Pandas DataFrame으로 변환합니다."""
        try:
            # CSV 문자열 정리 (코드 블록 마크 제거)
            if csv_string.strip().startswith("```csv"):
                csv_string = csv_string.strip()[6:-3].strip()
            elif csv_string.strip().startswith("```"):
                csv_string = csv_string.strip()[3:-3].strip()
            
            # 빈 문자열 체크
            if not csv_string.strip():
                print(f"  ⚠️ '{image_name}'에서 빈 CSV 데이터를 받았습니다.")
                return None
            
            # StringIO를 사용하여 CSV 문자열을 DataFrame으로 변환
            csv_buffer = StringIO(csv_string)
            df = pd.read_csv(csv_buffer)
            
            # 빈 DataFrame 체크
            if df.empty:
                print(f"  ⚠️ '{image_name}'에서 빈 DataFrame이 생성되었습니다.")
                return None
            
            # 이미지명을 인덱스에 추가
            df.index = [f"{image_name}_{idx}" for idx in df.index]
            
            return df
            
        except Exception as e:
            print(f"  ❌ CSV 파싱 오류 ({image_name}): {e}")
            print("  --- API Raw Response ---")
            print("  " + (csv_string[:200] + "..." if len(csv_string) > 200 else csv_string))
            print("  ------------------------")
            return None
    
    def _get_image_files(self):
        """raw_img 폴더에서 이미지 파일 목록을 가져옵니다."""
        image_files = []
        
        if not os.path.exists(self.input_folder):
            print(f"❌ 입력 폴더가 존재하지 않습니다: {self.input_folder}")
            return []
        
        for file in os.listdir(self.input_folder):
            if any(file.lower().endswith(ext) for ext in self.image_extensions):
                image_files.append(os.path.join(self.input_folder, file))
        
        return sorted(image_files)
    
    def _save_dataframe_to_csv(self, df, output_file):
        """결과 DataFrame을 CSV 파일로 저장합니다."""
        os.makedirs(self.output_folder, exist_ok=True)
        df.to_csv(output_file, index=True, encoding=self.csv_encoding)
    
    def convert_specific_images(self, image_names):
        """특정 이미지들만 Claude API로 변환"""
        print("🚀 Claude API 변환 시작...")
        print(f"📅 처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 입력 폴더: {self.input_folder}")
        print(f"📁 출력 폴더: {self.output_folder}")
        print(f"🤖 모델: {self.model_name}")
        print("-" * 60)
        
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
                print(f"❌ 지정된 이미지 파일들을 찾을 수 없습니다: {image_names}")
                return False
            
            print(f"📷 처리할 이미지 수: {len(image_files)}")
            
            # 모든 이미지 처리 결과를 저장할 리스트
            all_dataframes = []
            processed_count = 0
            failed_count = 0
            
            # 각 이미지 처리
            for i, image_path in enumerate(image_files, 1):
                image_name = os.path.basename(image_path)
                print(f"\n🔄 [{i}/{len(image_files)}] 처리 중: {image_name}")
                
                try:
                    # Claude API 호출
                    json_response = self._process_image_with_claude(image_path)
                    
                    if json_response is None:
                        failed_count += 1
                        print(f"  ❌ 실패: Claude API 호출 실패")
                        continue
                    
                    # CSV를 DataFrame으로 변환
                    result_df = self._parse_csv_to_dataframe(json_response, image_name)
                    
                    if result_df is not None:
                        all_dataframes.append(result_df)
                        processed_count += 1
                        print(f"  ✅ 성공: {result_df.shape[0]}개 행 추출")
                    else:
                        failed_count += 1
                        print(f"  ❌ 실패: DataFrame 변환 실패")
                        
                except Exception as e:
                    failed_count += 1
                    print(f"  ❌ 오류 발생 ({image_name}): {e}")
                    continue
            
            # 모든 DataFrame을 하나로 합치기
            if all_dataframes:
                print(f"\n📊 데이터 통합 중...")
                combined_df = pd.concat(all_dataframes, axis=0, ignore_index=False)
                
                # 결과 저장
                output_file = os.path.join(self.output_folder, f"{self.model_name.replace('-', '_')}.csv")
                self._save_dataframe_to_csv(combined_df, output_file)
                
                print(f"\n🎉 Claude API 변환 완료!")
                print(f"✅ 성공: {processed_count}개 이미지")
                print(f"❌ 실패: {failed_count}개 이미지")
                print(f"📄 총 데이터: {combined_df.shape[0]}개 행, {combined_df.shape[1]}개 열")
                print(f"💾 저장 위치: {output_file}")
                
                return True
            else:
                print("❌ 처리된 데이터가 없습니다.")
                return False
            
        except Exception as e:
            print(f"❌ 전체 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return False

    def convert_all_images(self):
        """모든 이미지를 Claude API로 변환 (기존 메서드 유지)"""
        # 모든 이미지 파일 목록 가져오기
        image_files = self._get_image_files()
        image_names = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
        return self.convert_specific_images(image_names)

