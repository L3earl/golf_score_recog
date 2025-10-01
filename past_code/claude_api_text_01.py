"""
Claude API를 이용한 이미지 텍스트 추출 스크립트
raw_img 폴더의 이미지에서 Claude API로 텍스트를 추출
"""

# ==================== 사용자 설정 변수 ====================
# Claude API 설정
MODEL_NAME = "claude-opus-4-1-20250805"  # Claude 모델명
IMAGE_NAME = "KakaoTalk_20250930_112251022.png"  # 처리할 이미지 파일명
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
    # 이미지를 base64로 인코딩
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Claude API 호출
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": PROMPT
                    }
                ]
            }
        ]
    )
    
    return message.content[0].text

# ==================== 결과 저장 함수 ====================
def save_result(text, output_file):
    """결과를 파일에 저장합니다."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Claude API 텍스트 추출 결과\n")
        f.write(f"모델: {MODEL_NAME}\n")
        f.write(f"이미지: {IMAGE_NAME}\n")
        f.write(f"처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        f.write(text)

# ==================== 메인 실행 함수 ====================
def main():
    """메인 실행 함수"""
    print("Claude API 텍스트 추출 시작...")
    
    try:
        # Claude API 클라이언트 생성
        client = get_claude_client()
        print("Claude API 클라이언트 생성 완료!")
        
        # 이미지 파일 경로 확인
        image_path = os.path.join(INPUT_FOLDER, IMAGE_NAME)
        if not os.path.exists(image_path):
            print(f"이미지 파일이 존재하지 않습니다: {image_path}")
            return
        
        print(f"처리할 이미지: {IMAGE_NAME}")
        print("Claude API 호출 중...")
        
        # 이미지 처리
        extracted_text = process_image_with_claude(image_path, client)
        
        # 결과 저장
        output_file = os.path.join(OUTPUT_FOLDER, f"claude_{IMAGE_NAME.split('.')[0]}.txt")
        save_result(extracted_text, output_file)
        
        print(f"처리 완료! 결과가 저장되었습니다: {output_file}")
        print("추출된 텍스트 미리보기:")
        print("-" * 30)
        print(extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text)
        
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()
