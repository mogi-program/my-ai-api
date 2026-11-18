
# main.py · AI 푸드 도우미 API
# 사진을 전달받아 Gemini가 음식 이름, 칼로리, 영양 성분, 가격 등을 분석합니다.

import base64
import binascii
import os
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

MODEL_NAME = "gemini-3.5-flash-lite"

MAX_B64 = 4_000_000

app = FastAPI(
    title="AI 푸드 도우미 API",
    description="음식 사진을 분석하여 음식 정보와 영양 성분을 제공합니다."
)


# --------------------------------------------------
# 요청 모델
# --------------------------------------------------

class FoodRequest(BaseModel):
    image_b64: str = Field(
        min_length=1,
        description="음식 사진을 Base64로 변환한 문자열"
    )

    question: str = Field(
        default="음식의 이름, 칼로리, 영양 성분 및 가격을 분석해 줘.",
        max_length=300
    )


# --------------------------------------------------
# 응답 모델
# --------------------------------------------------

class Nutrition(BaseModel):
    carbohydrates_g: float | None = None
    protein_g: float | None = None
    fat_g: float | None = None
    sodium_mg: float | None = None


class FoodAnalysis(BaseModel):
    is_food: bool

    food_name: str | None = None
    category: str | None = None
    description: str = ""

    ingredients: list[str] = Field(default_factory=list)

    calories_kcal: int | None = None

    nutrition: Nutrition = Field(default_factory=Nutrition)

    price_min: int | None = None
    price_max: int | None = None

    confidence: Literal["높음", "보통", "낮음", "판단 어려움"] = "판단 어려움"

    notes: str = ""


# --------------------------------------------------
# 접근 코드 검사
# --------------------------------------------------

def check_access(access_code):
    """접근 코드가 일치하는지 확인"""

    expected = os.environ.get("APP_ACCESS_CODE")

    if expected and access_code != expected:
        raise HTTPException(
            status_code=401,
            detail="접근 코드가 틀렸습니다."
        )


# --------------------------------------------------
# Base64 검사
# --------------------------------------------------

def check_b64(text):
    """Base64 형식과 이미지 크기 검사"""

    if len(text) > MAX_B64:
        raise HTTPException(
            status_code=413,
            detail="이미지가 너무 큽니다. 사진 크기를 줄여 주세요."
        )

    try:
        image_data = base64.b64decode(text, validate=True)

    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=422,
            detail="올바른 Base64 이미지가 아닙니다."
        )

    if not image_data:
        raise HTTPException(
            status_code=422,
            detail="이미지 데이터가 비어 있습니다."
        )

    return image_data


# --------------------------------------------------
# Gemini 클라이언트 생성
# --------------------------------------------------

def get_client():
    """환경 변수에서 Gemini API 키를 가져옴"""

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="서버에 GEMINI_API_KEY가 없습니다."
        )

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=120_000,
            retry_options=types.HttpRetryOptions(
                http_status_codes=[500, 503]
            )
        )
    )


# --------------------------------------------------
# 음식 분석 프롬프트
# --------------------------------------------------

FOOD_PROMPT = """
너는 음식 이미지 분석 전문 AI야.

사용자가 제공한 사진을 분석하고 음식에 관한 정보를 추출해 줘.

다음 항목을 분석해.

1. 음식인지 여부
2. 음식 이름
3. 음식 종류 (한식, 중식, 일식, 양식 등)
4. 음식에 대한 설명
5. 주요 재료
6. 예상 칼로리
7. 탄수화물, 단백질, 지방, 나트륨 함량
8. 대한민국 기준 예상 판매 가격

[분석 규칙]

- 사진에서 확인할 수 있는 정보를 우선적으로 활용해.
- 영양 성분은 일반적인 1인분을 기준으로 추정해.
- 사진만으로 정확하게 판단하기 어려운 수치는 추정치임을 명시해.
- 음식의 양을 정확하게 알 수 없다면 그 한계를 notes에 작성해.
- 영양 성분을 신뢰할 수 없으면 null로 반환해.
- 가격은 대한민국 일반적인 외식 가격을 참고한 대략적인 추정치로 제공해.
- 실제 판매 가격을 확인한 것처럼 표현하지 마.
- 사진에 없는 재료를 확정적으로 단정하지 마.
- 음식이 아닌 사진이면 is_food를 false로 설정해.
- 음식이 아닌 경우 음식 이름, 칼로리, 가격 등은 null로 반환해.
- 분석 결과는 요청된 JSON 스키마에 맞춰 반환해.

사용자가 추가 질문을 입력했다면 분석에 반영하되,
사진에서 확인할 수 없는 사실을 만들어 내지는 마.
"""


# --------------------------------------------------
# Gemini 음식 분석
# --------------------------------------------------

def analyze_food(image_data, question):
    """Gemini에 사진을 전달하고 음식 분석 결과를 반환"""

    client = get_client()

    prompt = FOOD_PROMPT + f"\n\n사용자의 추가 요청: {question}"

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,

            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image_data,
                    mime_type="image/jpeg"
                )
            ],

            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FoodAnalysis,
                temperature=0.2
            )
        )

        if not response.text:
            raise HTTPException(
                status_code=502,
                detail="Gemini가 분석 결과를 반환하지 않았습니다."
            )

        result = FoodAnalysis.model_validate_json(response.text)

        return result

    except HTTPException:
        raise

    except Exception as error:
        code = getattr(error, "status_code", None)

        if code == 429:
            raise HTTPException(
                status_code=429,
                detail="Gemini API 사용 한도를 초과했습니다. 잠시 후 다시 시도하세요."
            )

        raise HTTPException(
            status_code=502,
            detail=f"Gemini 음식 분석 실패 (상태 번호: {code})"
        )


# --------------------------------------------------
# 기본 경로
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "AI 푸드 도우미 API입니다.",
        "endpoint": "/analyze-food",
        "method": "POST"
    }


# --------------------------------------------------
# 음식 분석 API
# --------------------------------------------------

@app.post("/analyze-food", response_model=FoodAnalysis)
def analyze_food_api(
    request: FoodRequest,
    x_access_code: str | None = Header(default=None)
):
    # ① 접근 코드 검사
    check_access(x_access_code)

    # ② Base64 및 이미지 데이터 검사
    image_data = check_b64(request.image_b64)

    # ③ Gemini 음식 분석
    result = analyze_food(
        image_data,
        request.question
    )

    # ④ 분석 결과 반환
    return result
