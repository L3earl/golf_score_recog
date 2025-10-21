## 전체 프로세스 흐름

시스템은 아래와 같은 순서로 작동합니다. 이미지는 **case1, case2, case3** 중 하나의 워크플로우를 따르거나, 어느 케이스에도 해당하지 않거나 이전 단계에서 예외가 발생한 경우 **Claude API** 워크플로우로 처리됩니다.

---

**[이미지 입력]** (`RAW_IMG_FOLDER`)
    |
    ▼
**[0. 케이스 분류 (`main.py`)]**: 이미지 크기와 비율에 따라 처리 경로 결정
    * **Case 1**: 1800x1200 또는 정확한 3:2 비율 이미지
    * **Case 2**: 정확히 909x920 크기 이미지
    * **Case 3**: 세로가 가로보다 긴 이미지 (h > w)
    * **Case 99 (Claude API)**: 위 케이스에 해당하지 않거나, 이전 케이스 처리 중 예외 발생 시
    |
    ├─> **Case 1, 2 워크플로우** ───────────────────────────────────────────────┐
    │       |                                                                   │
    │       ▼                                                                   │
    │   **[1-1. 전처리 (`preprocessing.py`)]** │
    │       * **좌표 기반 크롭 (`image_cropper.py`)**:                           │
    │           정의된 좌표(`crop_coordinates.py`)로 이미지 분할 (`RAW_CROP_NUM_FOLDER`) │
    │       * **이미지 정리 (`image_cleaner.py`)**:                             │
    │           * Case 1: 검은색 외 색상 제거                                   │
    │           * Case 2: 숫자/기호 모드 분리 처리                             │
    │           (`RAW_CLEAN_NUM_FOLDER`)                                       │
    │       |                                                                   │
    │       ▼                                                                   │
    │   **[1-2. OCR 변환 (`ocr_converter.py`)]** │
    │       * **TrOCR**: 정리된 이미지를 텍스트로 변환                         │
    │       * **Case 2 기호 검출 (`symbol_detector.py`)**: '-' 또는 '.' 검출     │
    │       * **데이터 구조화**: 케이스별 규칙에 따라 테이블 형태 구성             │
    │       * **결과 저장**: `RESULT_CONVERT_NUM_FOLDER/case<N>/`에 CSV 저장    │
    │       |                                                                   │
    │       ▼                                                                   │
    │   **[1-3. 후처리 및 검증 (`postprocessing.py`)]** │
    │       * 빈 행 제거                                                         │
    │       * 'sum' 컬럼 계산 (total1 + total2)                               │
    │       * 데이터 검증 (숫자 형식, 합계 일치 여부 등)                        │
    │       * **예외 파일 식별**: 검증 실패 시 파일명을 다음 단계로 전달          │
    │       * **결과 업데이트**: 검증 통과 시 CSV 파일 덮어쓰기                   │
    │       |                                                                   │
    ├─> **Case 3 워크플로우** ───────────────────────────────────────────────┤
    │       |                                                                   │
    │       ▼                                                                   │
    │   **[2-1. 전처리 (`preprocessing.py`)]** │
    │       * **OCR 기반 동적 크롭 (`simple_ocr_crop.py`)**:                    │
    │           EasyOCR로 'HOLE', 'T' 감지 후 테이블 영역 동적 크롭, 3분할 저장 │
    │           (`RAW_CROP_NUM_FOLDER/case3/<파일명>/`)                          │
    │       * **이미지 정리 (`image_cleaner.py`)**: Case 1과 동일 로직 적용      │
    │           (`RAW_CLEAN_NUM_FOLDER/case3/`)                                  │
    │       |                                                                   │
    │       ▼                                                                   │
    │   **[2-2. OCR 변환 (`ocr_converter.py`)]** │
    │       * **TrOCR (Enhanced)**: 분할된 이미지를 텍스트 변환 후 병합         │
    │       * **데이터 구조화**: PAR 행과 플레이어 행(상대 점수) 구성, 절대 점수 totals 계산 │
    │       * **결과 저장**: `RESULT_CONVERT_NUM_FOLDER/case3/`에 CSV 저장      │
    │       |                                                                   │
    │       ▼                                                                   │
    │   **[2-3. 후처리 및 검증 (`postprocessing.py`)]**: Case 1, 2와 동일         │
    │       * **예외 파일 식별**: 검증 실패 시 파일명을 Claude API 단계로 전달  │
    │       |                                                                   │
    ├─> **Case 99 (Claude API) 워크플로우** ───────────────────────────────┤
    │       |                                                                   │
    │       ▼                                                                   │
    │   **[3-1. 업스케일링 & 압축 (`claude_converter.py`)]** │
    │       * **EDSR 업스케일링**: 이미지 해상도 향상 (`RAW_IMG_UPSCALE_FOLDER`) │
    │       * **압축**: API 제한 크기에 맞게 이미지 압축 (필요시 JPEG 변환)       │
    │       |                                                                   │
    │       ▼                                                                   │
    │   **[3-2. Claude API 변환 (`claude_converter.py`)]** │
    │       * **API 호출**: 업스케일/압축된 이미지를 Claude API로 전송         │
    │       * **Tool 사용**: 정의된 스키마(`CLAUDE_EXTRACT_SCORECARD_TOOL`)로 JSON 데이터 추출 │
    │       |                                                                   │
    │       ▼                                                                   │
    │   **[3-3. 데이터 변환 및 저장 (`claude_converter.py`)]** │
    │       * **JSON → DataFrame**: API 응답 파싱                             │
    │       * **형식 변환 및 후처리**: 표준 CSV 형식(Transpose)으로 변환, totals 계산, 상대점수 변환 로직 적용 │
    │       * **결과 저장**: `RESULT_CONVERT_NUM_FOLDER/case99/`에 CSV 저장    │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘
    |
    ▼
**[최종 결과]** (`RESULT_CONVERT_NUM_FOLDER/case<N>/` 또는 `RESULT_CONVERT_NUM_FOLDER/case99/`)
    * 각 케이스별 폴더 또는 case99 폴더에 처리된 스코어카드 CSV 파일 저장

## 실행 전 준비 사항
프로젝트 루트 폴더에 .env 파일을 만들고 ANTHROPIC_API_KEY = YOUR_API_KEY 형식으로 Claude API 키를 입력해야 합니다. (선택: 구글 시트 연동을 위해 GCP_SERVICE_ACCOUNT_JSON 추가 가능)

pyproject.toml 또는 requirements.txt에 명시된 라이브러리 (특히 torch, transformers, opencv-python, easyocr, anthropic 등)를 설치해야 합니다. GPU 가속을 위해서는 CUDA 설치 및 PyTorch GPU 버전 설치가 필요합니다. [제가 한 방법](https://earls.notion.site/119abb83012043159fee15b3c73235cc?pvs=74) 들어가셔서 환경설정> 윈도우 자체에 CUDA 설치 > 1번 부터 참고하시면 됩니다 (window일시)

## 그외 정보

scoring.py를 이용하면 검증이 쉽습니다. 구글 시트 정답지의 정답을 가져와서 결과물과 비교한 후, 구글 시트 중 result_detail에 정확도와 비교 결과를 올려줍니다


## 주요 이슈
- GPU가 필요합니다. CPU로는 속도가 나지 않음

현재 매 처리마다 모델을 GPU 메모리에 로드하고 있습니다. 이 시간이 전체 시간의 50% 정도를 차지합니다. 실제 서비스에서는 모델을 한 번만 GPU 메모리에 올린 후, 여러 요청을 메모리 상에서 계속 처리하도록 아키텍처를 변경해야 합니다.

- 다양한 케이스 테스트가 필요합니다.

- 코드를 직접 짠게 아니라 AI를 이용했기 때문에, 아주 디테일하게 이해하고 있지 않습니다. 의도와 약간 다른 코드들이 있을 수 있으므로 리펙토링할때 너무 고민하지 않아야 함
    



