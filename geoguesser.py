import base64
import json
import os
from openai import OpenAI
from typing import List, Dict

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
        print("No images provided for geolocation analysis.")
        return []

    print(f"Analyzing {len(image_paths)} images with OpenAI o3 model...")
    all_locations = []

    for image_path in image_paths:
        try:
            print(f"Processing image: {os.path.basename(image_path)}")
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
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            print("Calling OpenAI o3 API...")
            # Using the specific client.responses.create method for the o3 model
            response = client.responses.create(
                model="o3",
                reasoning={"effort": "medium"},
                input=o3_input,
                max_output_tokens=300,
            )

            # Assuming the response object has a .text attribute for the content
            response_text = response.text
            print(f"Received response from OpenAI: {response_text}")

            if not response_text:
                print(f"  - Warning: Received an empty response from OpenAI for {os.path.basename(image_path)}.")
                continue
            
            location_data = json.loads(response_text)

            lat = location_data.get('latitude')
            lng = location_data.get('longitude')
            location_data['latitude'] = float(lat) if lat is not None else None
            location_data['longitude'] = float(lng) if lng is not None else None

            if location_data.get("name") and location_data["name"] is not None:
                all_locations.append(location_data)
            else:
                print(f"No valid location name found for image {os.path.basename(image_path)}")

        except Exception as e:
            print(f"An error occurred while processing {os.path.basename(image_path)} with OpenAI: {e}")
            continue
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)
    
    print(f"Successfully extracted {len(all_locations)} locations.")
    return all_locations
