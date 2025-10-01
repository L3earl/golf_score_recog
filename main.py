"""
골프 스코어카드 인식 메인 실행 파일
"""

import sys
from datetime import datetime
from modules.preprocessing import ImagePreprocessor
from modules.ocr_converter import OCRConverter
from modules.postprocessing import PostProcessor
from modules.claude_converter import ClaudeConverter

def main():
    """메인 실행 함수"""
    start_time = datetime.now()
    print("🏌️ 골프 스코어카드 인식 시작...\n")
    
    try:
        # 1단계: 전처리
        print("🔄 [1/6] 전처리 중...")
        preprocessor = ImagePreprocessor()
        if not preprocessor.process_all_images():
            print("❌ 전처리 실패")
            return False
        
        # 2단계: OCR 변환
        print("\n🔄 [2/6] OCR 변환 중...")
        converter = OCRConverter()
        if not converter.convert_all_folders():
            print("❌ OCR 변환 실패")
            return False
        
        # 3단계: 후처리
        print("\n🔄 [3/6] 후처리 중...")
        processor = PostProcessor()
        results = processor.process_all_files()
        if not results:
            print("❌ 후처리 실패")
            return False
        
        # 4단계: 예외 확인
        print("\n🔄 [4/6] 예외 확인 중...")
        exception_files = []
        total_exceptions = 0
        
        for result in results:
            if 'exceptions' in result and len(result['exceptions']) > 0:
                exception_files.append(result)
                total_exceptions += len(result['exceptions'])
        
        if exception_files:
            print(f"⚠️  예외 발견: {len(exception_files)}개 파일, {total_exceptions}개 예외")
        else:
            print("✅ 예외 없음")
        
        # 5단계: Claude 변환 (선택적)
        if exception_files:
            print(f"\n🔄 [5/6] Claude API 변환 ({len(exception_files)}개 파일)")
            print(f"⚠️  예상 비용: {len(exception_files) * 76:.2f}원")
            user_input = input("계속 진행하시겠습니까? (y/n): ").lower().strip()
            
            if user_input == 'y':
                claude_converter = ClaudeConverter()
                # 예외가 발생한 파일들만 처리
                exception_image_names = [result['folder'] for result in exception_files]
                claude_converter.convert_specific_images(exception_image_names)
            else:
                print("⚠️  Claude 변환 건너뜀")
        else:
            print("\n⚠️  [5/6] 예외 없음 - Claude 변환 건너뜀")
        
        # 6단계: 완료
        print("\n✅ [6/6] 처리 완료")
        
        # 최종 결과
        elapsed_time = (datetime.now() - start_time).total_seconds()
        print(f"\n{'='*50}")
        print(f"⏱️  총 작업 시간: {elapsed_time:.2f}초")
        
        if exception_files and user_input != 'y':
            print(f"⚠️  처리되지 못한 케이스: {len(exception_files)}개")
        else:
            print("✅ 모든 데이터가 성공적으로 처리되었습니다!")
        
        print(f"{'='*50}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  사용자에 의해 중단됨")
        return False
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
