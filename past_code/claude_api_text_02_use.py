"""
Claude API를 이용한 이미지 텍스트 추출 및 Pandas DataFrame 변환 스크립트
raw_img 폴더의 모든 이미지를 Claude API로 처리하여 하나의 CSV 파일로 저장
"""

# ==================== 사용자 설정 변수 ====================
# Claude API 설정
# MODEL_NAME = "claude-opus-4-1-20250805"
# MODEL_NAME = "claude-sonnet-4-5-20250929"
MODEL_NAME = "claude-3-5-haiku-20241022"

PROMPT = """
첨부된 골프 스코어카드 이미지에서 숫자를 테이블 형태로 뽑아줘
결과는 "scores"라는 단일 키를 가진 JSON 객체 형식으로만 출력해줘.
모든 값은 숫자로 정확하게 인식하고, 특히 음수 값에 유의해줘.
JSON 객체 앞뒤로 어떤 설명 텍스트도 절대 포함하지 마.
"""

# 폴더 경로 설정
INPUT_FOLDER = "raw_img"
OUTPUT_FOLDER = "result_claude"

# ==================== 라이브러리 임포트 ====================
import os
import base64
import json
import pandas as pd
from anthropic import Anthropic
from datetime import datetime
from dotenv import load_dotenv

# ==================== 환경변수 로드 ====================
load_dotenv()

# ==================== Claude API 클라이언트 설정 ====================
def get_claude_client():
    """Claude API 클라이언트를 생성합니다."""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY가 .env 파일에 설정되지 않았습니다.")
    
    return Anthropic(api_key=api_key)

# ==================== 이미지 처리 함수 ====================
def process_image_with_claude(image_path, client):
    """Claude API를 사용하여 이미지에서 텍스트를 추출합니다."""
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4000,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
                    {"type": "text", "text": PROMPT}
                ]
            }
        ]
    )
    
    return message.content[0].text

# ==================== JSON 파싱 및 DataFrame 변환 함수 ====================
def parse_to_dataframe(json_string: str, image_name: str) -> pd.DataFrame:
    """Claude가 반환한 JSON 형식의 문자열을 Pandas DataFrame으로 변환합니다."""
    try:
        # JSON 문자열 정리
        if json_string.strip().startswith("```json"):
            json_string = json_string.strip()[7:-3].strip()
        elif json_string.strip().startswith("```"):
            json_string = json_string.strip()[3:-3].strip()
            
        data = json.loads(json_string)
        scores_data = data.get('scores', {})
        
        if not scores_data:
            print(f"경고: '{image_name}'에서 'scores' 키가 없거나 비어있습니다.")
            return None
        
        # 데이터 구조 분석 및 정규화
        normalized_data = {}
        
        for course_name, course_data in scores_data.items():
            if isinstance(course_data, dict):
                normalized_data[course_name] = {}
                for player_name, scores in course_data.items():
                    if isinstance(scores, list):
                        # 리스트를 문자열로 변환하여 일관성 유지
                        normalized_data[course_name][player_name] = str(scores)
                    else:
                        # 단일 값을 문자열로 변환
                        normalized_data[course_name][player_name] = str(scores)
            else:
                # 단일 값인 경우
                normalized_data[course_name] = str(course_data)
        
        # DataFrame 생성 (모든 값을 문자열로 통일)
        df = pd.DataFrame(normalized_data)
        
        # 이미지명을 인덱스에 추가
        df.index = [f"{image_name}_{idx}" for idx in df.index]
        
        return df
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"JSON 파싱 오류 ({image_name}): {e}")
        print("--- API Raw Response ---")
        print(json_string[:200] + "..." if len(json_string) > 200 else json_string)
        print("------------------------")
        return None

# ==================== 이미지 파일 목록 가져오기 ====================
def get_image_files():
    """raw_img 폴더에서 이미지 파일 목록을 가져옵니다."""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"입력 폴더가 존재하지 않습니다: {INPUT_FOLDER}")
        return []
    
    for file in os.listdir(INPUT_FOLDER):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(INPUT_FOLDER, file))
    
    return sorted(image_files)

# ==================== DataFrame을 CSV로 저장하는 함수 ====================
def save_dataframe_to_csv(df, output_file):
    """결과 DataFrame을 CSV 파일로 저장합니다."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    df.to_csv(output_file, index=True, encoding='utf-8-sig')

# ==================== 메인 실행 함수 ====================
def main():
    """메인 실행 함수"""
    print("🚀 Claude API 다중 이미지 텍스트 추출 및 DataFrame 변환 시작...")
    print(f"📅 처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Claude API 클라이언트 생성
        client = get_claude_client()
        print(f"✅ Claude API 클라이언트 생성 완료! (모델: {MODEL_NAME})")
        
        # 이미지 파일 목록 가져오기
        image_files = get_image_files()
        
        if not image_files:
            print(f"❌ {INPUT_FOLDER} 폴더에 이미지 파일이 없습니다.")
            return
        
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
                json_response = process_image_with_claude(image_path, client)
                
                # JSON을 DataFrame으로 변환
                result_df = parse_to_dataframe(json_response, image_name)
                
                if result_df is not None:
                    all_dataframes.append(result_df)
                    processed_count += 1
                    print(f"✅ 성공: {result_df.shape[0]}개 행 추출")
                else:
                    failed_count += 1
                    print(f"❌ 실패: DataFrame 변환 실패")
                    
            except Exception as e:
                failed_count += 1
                print(f"❌ 오류 발생 ({image_name}): {e}")
                continue
        
        # 모든 DataFrame을 하나로 합치기
        if all_dataframes:
            print(f"\n📊 데이터 통합 중...")
            combined_df = pd.concat(all_dataframes, axis=0, ignore_index=False)
            
            # 결과 저장
            output_file = os.path.join(OUTPUT_FOLDER, f"{MODEL_NAME.replace('-', '_')}.csv")
            save_dataframe_to_csv(combined_df, output_file)
            
            print(f"\n🎉 처리 완료!")
            print(f"✅ 성공: {processed_count}개 이미지")
            print(f"❌ 실패: {failed_count}개 이미지")
            print(f"📄 총 데이터: {combined_df.shape[0]}개 행, {combined_df.shape[1]}개 열")
            print(f"💾 저장 위치: {output_file}")
            
            # 데이터 미리보기
            print(f"\n📋 데이터 미리보기 (상위 5개 행):")
            print("-" * 60)
            print(combined_df.head())
            
        else:
            print("❌ 처리된 데이터가 없습니다.")
        
    except Exception as e:
        print(f"❌ 전체 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()