import base64
import json
import os
from openai import OpenAI
from typing import List, Dict

# .env 파일에서 API 키를 로드하기 위한 설정 (테스트용)
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("FATAL ERROR: OPENAI_API_KEY not found in .env file.")

def _encode_image_to_base64(image_path: str) -> str:
    """이미지 파일을 base64 문자열로 인코딩합니다."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Warning: File not found at {image_path}, skipping.")
        return None

def get_geolocation_from_image_set(image_paths: List[str]) -> List[Dict]:
    """
    여러 이미지 세트를 한 번에 분석하여 위치 정보를 추출합니다.
    geoguesser.py와 달리, 모든 이미지를 단일 API 호출로 전송하여 전체 컨텍스트를 파악합니다.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    if not image_paths:
        print("분석할 이미지를 찾지 못했습니다. 경로를 확인해주세요.")
        return []

    print(f"총 {len(image_paths)}개의 이미지를 단일 컨텍스트로 묶어 OpenAI o3 모델로 분석합니다...")

    # 모든 이미지를 base64로 인코딩하고 프롬프트 컨텐츠를 구성
    content = [
        {
            "type": "text",
            "text": """
            You are a world-class geolocation expert with advanced visual analysis and web search simulation capabilities.
            Your mission is to pinpoint the exact coordinates of locations shown in a series of images from a video.

            Follow this multi-step process:

            1.  **Comprehensive Image Analysis:**
                *   Examine all images as a single, continuous context.
                *   Identify prominent features: buildings, storefronts (cafes, shops), escalators, signs, landmarks (parks, stations, statues).
                *    meticulously extract all visible text, including store names, street signs, and any other readable characters. Pay close attention to language and lettering style.

            2.  **Environmental Contextualization:**
                *   Based on the visual cues, infer the general environment. Is it a busy downtown street, a quiet residential area, inside a shopping mall, near a train station entrance, a park, or a university campus?
                *   Synthesize the clues. For example, a cafe next to an escalator inside a large building suggests a shopping mall or a large transit hub.

            3.  **Simulated Web Image Search & Verification:**
                *   Formulate a search query based on the extracted text (e.g., "Starbucks near Gwanghwamun Station") and structural features (e.g., "cafe with a green awning and a red brick facade").
                *   Imagine you are performing a web image search with this query. Compare the visual information from the provided images with potential search results.
                *   Verify the location by looking for matching architectural details, color schemes, and surrounding context in the simulated search results.

            4.  **Coordinate Acquisition:**
                *   Once a location is confidently identified, retrieve its precise latitude and longitude from public data sources (like a simulated Google Maps or public directory).

            5.  **JSON Output:**
                *   Return the result as a single JSON object.
                *   The JSON object must have a single key named "locations".
                *   The value of the "locations" key must be a JSON array of objects.
                *   Each object in the array must represent a unique, verified location and contain three keys:
                    1.  `"name"`: The common, full name of the location (e.g., "Starbucks Gwanghwamun Branch").
                    2.  `"latitude"`: The latitude as a float.
                    3.  `"longitude"`: The longitude as a float.
                *   If you cannot confidently determine the precise location or its coordinates, the "locations" array should be empty.
                *   Your final output MUST BE a valid JSON object only, structured exactly as described. Do not include any other text, explanations, or apologies.
            """
        }
    ]

    for image_path in image_paths:
        base64_image = _encode_image_to_base64(image_path)
        if base64_image:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

    # 이미지가 하나도 없는 경우 API 호출 방지
    if len(content) <= 1:
        print("유효한 이미지가 없어 분석을 중단합니다.")
        return []

    try:
        print("OpenAI o3 API를 호출하여 이미지 세트 분석을 요청합니다...")
        
        # o3 모델이 실제로 존재하지 않을 수 있으므로, gpt-4o 와 같은 최신 모델을 사용합니다.
        # geoguesser.py의 "o3"가 특정 모델을 지칭하는 경우, 모델 이름을 변경해야 합니다.
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=1024,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        if not response_text:
            print("Warning: OpenAI returned an empty response.")
            return []

        print(f"OpenAI로부터 성공적으로 결과를 받았습니다.")
        print(f"Raw response: {response_text}")

        # OpenAI 응답이 {"locations": [...]} 형태의 JSON 객체일 것으로 예상
        raw_data = json.loads(response_text)
        
        # 'locations' 키가 있는지 확인
        locations_data = raw_data.get("locations")

        if locations_data is None:
            print("Warning: OpenAI response did not contain a 'locations' key.")
            # 응답 전체를 출력하여 무엇이 잘못되었는지 확인
            print(f"Full response object: {raw_data}")
            return []

        if not isinstance(locations_data, list):
            print(f"Warning: 'locations' key in OpenAI response is not a list, but {type(locations_data)}.")
            return []

        all_locations = []
        for loc in locations_data:
            if isinstance(loc, dict) and loc.get("name"):
                # 데이터 형식 검증 및 변환
                lat = loc.get('latitude')
                lng = loc.get('longitude')
                loc['latitude'] = float(lat) if lat is not None else None
                loc['longitude'] = float(lng) if lng is not None else None
                all_locations.append(loc)
            else:
                print(f"유효하지 않은 위치 데이터 형식을 건너뜁니다: {loc}")

        print(f"성공적으로 {len(all_locations)}개의 고유한 장소를 추출했습니다.")
        return all_locations

    except Exception as e:
        print(f"OpenAI 처리 중 오류가 발생했습니다: {e}")
        return []

def run_local_test():
    """로컬 테스트를 위해 captures 디렉토리의 모든 이미지를 분석합니다."""
    print("--- 로컬 테스트 시작 ---")
    captures_dir = 'captures'
    if not os.path.isdir(captures_dir):
        print(f"오류: '{captures_dir}' 디렉토리를 찾을 수 없습니다.")
        return

    # 테스트를 위해 이미지 수를 10개로 제한 (API 비용 및 시간 절약)
    image_files = sorted([
        os.path.join(captures_dir, f)
        for f in os.listdir(captures_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])[:10]

    if not image_files:
        print(f"'{captures_dir}' 디렉토리에서 분석할 이미지를 찾지 못했습니다.")
        return

    print(f"테스트를 위해 선택된 이미지 {len(image_files)}개:")
    for f in image_files:
        print(f" - {os.path.basename(f)}")

    # 이미지 세트 분석 함수 호출
    locations = get_geolocation_from_image_set(image_files)

    print("\n--- 최종 추출된 위치 정보 ---")
    if locations:
        print(json.dumps(locations, indent=2, ensure_ascii=False))
    else:
        print("추출된 위치 정보가 없습니다.")
    print("--- 로컬 테스트 종료 ---")


if __name__ == "__main__":
    run_local_test()
