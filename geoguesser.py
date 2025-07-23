import base64
import json
import os
from openai import OpenAI
from typing import List, Dict

# 테스트용 세팅

from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn

app = FastAPI()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("FATAL ERROR: OPENAI_API_KEY not found in .env file.")





def _encode_image_to_base64(image_path: str) -> str:
    """Encodes a single image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_geolocation_from_images(image_paths: List[str]) -> List[Dict]:
    """
    Analyzes a list of image files using OpenAI's o3 model to find geolocation data.
    """
    client = OpenAI()
    if not image_paths:
        print("장소를 분석할 이미지를 찾지 못했습니다. 경로를 다시 확인해주세요.")
        return []

    print(f"{len(image_paths)}개의 이미지를 OpenAI o3 모델로 분석 중입니다...")
    all_locations = []

    for image_path in image_paths:
        try:
            print(f"분석 중인 이미지 : {os.path.basename(image_path)}")
            base64_image = _encode_image_to_base64(image_path)

            # The input structure for the o3 model
            o3_input = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": """
                            Analyze this image from a YouTube vlog. Identify the specific name of the location (restaurant, cafe, store, etc.) shown.
                            Return the result as a JSON object with three keys:
                            1. \"name\": The common name of the location (e.g., \"Fengmi Bunsik\").
                            2. \"latitude\": The latitude as a float.
                            3. \"longitude\": The longitude as a float.
                            If you cannot determine the precise location or coordinates, return null for the values. Your output must be ONLY the JSON object.
                            """
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ]

            print("OpenAI o3 API를 불러오는 중...")
            # Using the specific client.responses.create method for the o3 model
            response = client.responses.create(
                model="o3",
                reasoning={"effort": "medium"},
                input=o3_input,
                max_output_tokens=300,
            )

            # Assuming the response object has a .text attribute for the content
            response_text = response.text
            print(f"OpenAI로부터 성공적으로 결과를 가져왔습니다. : {response_text}")

            if not response_text:
                print(f"  - Warning: {os.path.basename(image_path)}에서 OpenAI가 아무런 결과를 보내지 못했습니다.")
                continue
            
            location_data = json.loads(response_text)

            lat = location_data.get('latitude')
            lng = location_data.get('longitude')
            location_data['latitude'] = float(lat) if lat is not None else None
            location_data['longitude'] = float(lng) if lng is not None else None

            if location_data.get("name") and location_data["name"] is not None:
                all_locations.append(location_data)
            else:
                print(f"{os.path.basename(image_path)}로부터 장소 정보를 추출하지 못했습니다.")

        except Exception as e:
            print(f"{os.path.basename(image_path)}를 OpenAI로 처리 중 에러가 발생했습니다. : {e}")
            continue
        # finally:
        #     if os.path.exists(image_path):
        #         os.remove(image_path)
    
    print(f"성공적으로 {len(all_locations)}개의 장소를 추출했습니다.")
    return all_locations


captures_dir = 'captures'
image_files = [os.path.join(captures_dir, f) for f in os.listdir(captures_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
get_geolocation_from_images(image_files)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000, reload=True)