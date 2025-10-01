"""
Claude API를 이용한 이미지 텍스트 추출 및 동적 테이블 변환 스크립트
raw_img 폴더의 이미지에서 Claude API로 텍스트를 추출하여 다양한 형태의 테이블로 변환
"""

# ==================== 사용자 설정 변수 ====================
# Claude API 설정
MODEL_NAME = "claude-opus-4-1-20250805"
IMAGE_NAME = "KakaoTalk_20250930_112251022.png"  # 처리할 이미지 파일명

# 프롬프트 설정
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

# ==================== 동적 JSON 파싱 및 테이블 변환 함수 ====================
def parse_json_to_tables(json_string: str):
    """Claude가 반환한 JSON을 다양한 형태의 테이블로 변환합니다."""
    try:
        # JSON 문자열 정리
        if json_string.strip().startswith("```json"):
            json_string = json_string.strip()[7:-3].strip()
        elif json_string.strip().startswith("```"):
            json_string = json_string.strip()[3:-3].strip()
            
        data = json.loads(json_string)
        scores_data = data.get('scores', {})
        
        if not scores_data:
            print("경고: 'scores' 키가 없거나 비어있습니다.")
            return None, None, None
        
        # 1. 원본 구조 그대로 DataFrame 생성
        original_df = pd.DataFrame(scores_data)
        
        # 2. 플랫한 구조로 변환 (모든 레벨을 하나의 테이블로)
        flattened_data = []
        for course_name, course_data in scores_data.items():
            if isinstance(course_data, dict):
                for player_name, scores in course_data.items():
                    if isinstance(scores, list):
                        # 각 홀별 점수를 개별 행으로 변환
                        for hole_num, score in enumerate(scores, 1):
                            flattened_data.append({
                                'Course': course_name,
                                'Player': player_name,
                                'Hole': hole_num,
                                'Score': score
                            })
                    else:
                        # 단일 값인 경우
                        flattened_data.append({
                            'Course': course_name,
                            'Player': player_name,
                            'Hole': 'Total',
                            'Score': scores
                        })
        
        flattened_df = pd.DataFrame(flattened_data)
        
        # 3. 피벗 테이블 형태로 변환 (플레이어별 홀 점수)
        if not flattened_df.empty:
            pivot_df = flattened_df.pivot_table(
                index=['Course', 'Player'], 
                columns='Hole', 
                values='Score', 
                fill_value='-'
            ).reset_index()
        else:
            pivot_df = pd.DataFrame()
        
        return original_df, flattened_df, pivot_df
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"JSON 파싱 오류: {e}")
        print("--- API Raw Response ---")
        print(json_string)
        print("------------------------")
        return None, None, None

# ==================== 테이블 저장 함수 ====================
def save_tables_to_files(original_df, flattened_df, pivot_df, base_filename):
    """다양한 형태의 테이블을 파일로 저장합니다."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    saved_files = []
    
    # 1. 원본 구조 CSV
    if original_df is not None and not original_df.empty:
        original_file = os.path.join(OUTPUT_FOLDER, f"{base_filename}_original.csv")
        original_df.to_csv(original_file, index=True, encoding='utf-8-sig')
        saved_files.append(original_file)
        print(f"✅ 원본 구조 테이블 저장: {original_file}")
    
    # 2. 플랫한 구조 CSV
    if flattened_df is not None and not flattened_df.empty:
        flattened_file = os.path.join(OUTPUT_FOLDER, f"{base_filename}_flattened.csv")
        flattened_df.to_csv(flattened_file, index=False, encoding='utf-8-sig')
        saved_files.append(flattened_file)
        print(f"✅ 플랫한 구조 테이블 저장: {flattened_file}")
    
    # 3. 피벗 테이블 CSV
    if pivot_df is not None and not pivot_df.empty:
        pivot_file = os.path.join(OUTPUT_FOLDER, f"{base_filename}_pivot.csv")
        pivot_df.to_csv(pivot_file, index=False, encoding='utf-8-sig')
        saved_files.append(pivot_file)
        print(f"✅ 피벗 테이블 저장: {pivot_file}")
    
    # 4. JSON 원본 저장
    json_file = os.path.join(OUTPUT_FOLDER, f"{base_filename}_raw.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(original_df.to_dict() if original_df is not None else {}, f, ensure_ascii=False, indent=2)
    saved_files.append(json_file)
    print(f"✅ JSON 원본 저장: {json_file}")
    
    return saved_files

# ==================== 테이블 미리보기 함수 ====================
def preview_tables(original_df, flattened_df, pivot_df):
    """테이블들을 콘솔에 미리보기로 출력합니다."""
    print("\n" + "="*60)
    print("📊 테이블 미리보기")
    print("="*60)
    
    # 원본 구조 미리보기
    if original_df is not None and not original_df.empty:
        print("\n1️⃣ 원본 구조 테이블:")
        print("-" * 40)
        print(original_df.head(10))
        print(f"Shape: {original_df.shape}")
    
    # 플랫한 구조 미리보기
    if flattened_df is not None and not flattened_df.empty:
        print("\n2️⃣ 플랫한 구조 테이블 (상위 10개 행):")
        print("-" * 40)
        print(flattened_df.head(10))
        print(f"Shape: {flattened_df.shape}")
    
    # 피벗 테이블 미리보기
    if pivot_df is not None and not pivot_df.empty:
        print("\n3️⃣ 피벗 테이블 (플레이어별 홀 점수):")
        print("-" * 40)
        print(pivot_df.head(10))
        print(f"Shape: {pivot_df.shape}")

# ==================== 메인 실행 함수 ====================
def main():
    """메인 실행 함수"""
    print("🚀 Claude API 텍스트 추출 및 동적 테이블 변환 시작...")
    print(f"📅 처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Claude API 클라이언트 생성
        client = get_claude_client()
        print(f"✅ Claude API 클라이언트 생성 완료! (모델: {MODEL_NAME})")
        
        # 이미지 파일 경로 확인
        image_path = os.path.join(INPUT_FOLDER, IMAGE_NAME)
        if not os.path.exists(image_path):
            print(f"❌ 이미지 파일이 존재하지 않습니다: {image_path}")
            return
        
        print(f"📷 처리할 이미지: {IMAGE_NAME}")
        print("🔄 Claude API 호출 중...")
        
        # 1. Claude API에서 JSON 형식의 텍스트 추출
        json_response = process_image_with_claude(image_path, client)
        
        # 2. JSON을 다양한 형태의 테이블로 변환
        original_df, flattened_df, pivot_df = parse_json_to_tables(json_response)
        
        if original_df is not None:
            # 3. 테이블 미리보기
            preview_tables(original_df, flattened_df, pivot_df)
            
            # 4. 테이블들을 파일로 저장
            base_filename = f"claude_{IMAGE_NAME.split('.')[0]}"
            saved_files = save_tables_to_files(original_df, flattened_df, pivot_df, base_filename)
            
            print(f"\n🎉 처리 완료! 총 {len(saved_files)}개 파일이 저장되었습니다:")
            for file in saved_files:
                print(f"   📄 {file}")
        else:
            print("❌ JSON 파싱에 실패했습니다.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()