

import os
from dotenv import load_dotenv

# --- Step 1: Load Environment Variables FIRST ---
# This is the most critical step. Load variables before any other code runs.
print("Loading environment variables from .env file...")
load_dotenv()

# --- Step 2: Configure API Clients ---
# Now that variables are loaded, import and configure the API libraries.
import google.generativeai as genai

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("FATAL ERROR: GOOGLE_API_KEY not found in .env file.")
if not OPENAI_API_KEY:
    raise ValueError("FATAL ERROR: OPENAI_API_KEY not found in .env file.")

print("Configuring Google Gemini API client...")
genai.configure(api_key=GOOGLE_API_KEY)
# Note: OpenAI client is initialized inside the geoguesser function to be more robust.
print("API client configuration complete.")


# --- Step 3: Import Remaining Modules ---
# Now it's safe to import our own modules and FastAPI components.
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from youtube_utils import analyze_and_capture_locations
from geoguesser import get_geolocation_from_images


# --- Step 4: FastAPI App Initialization ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    youtube_url: str

# --- Step 5: API Endpoint ---
@app.post("/extract-ylocations")
def extract_ylocations(request: VideoRequest):
    """
    Processes a YouTube URL through a two-step AI pipeline:
    1. Gemini analyzes the transcript to find and capture location screenshots.
    2. OpenAI GPT-4o analyzes the screenshots to extract geolocation data.
    """
    print(f"\n--- Received request for URL: {request.youtube_url} ---")

    # Step 1: Use Gemini to find timestamps and capture screenshots
    print("--- Step 1: Analyzing video with Gemini to find and capture locations ---")
    captured_image_paths = analyze_and_capture_locations(request.youtube_url)

    if not captured_image_paths:
        print("Pipeline ended: No locations were captured from the video.")
        raise HTTPException(
            status_code=404,
            detail="No locations could be identified and captured from the video."
        )
    print(f"--- Step 1 Complete: Captured {len(captured_image_paths)} images ---")

    # Step 2: Use OpenAI to get geolocation from the captured images
    print("\n--- Step 2: Analyzing captured images with OpenAI GPT-4o ---")
    locations = get_geolocation_from_images(captured_image_paths)

    if not locations:
        print("Pipeline ended: Could not extract geolocation data from the captured images.")
        raise HTTPException(
            status_code=404,
            detail="Location images were captured, but could not be identified."
        )
    print(f"--- Step 2 Complete: Extracted {len(locations)} locations ---")

    # Step 3: Return the final data
    print("\n--- Pipeline Complete: Returning final data ---")
    return {"locations": locations}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000, reload=True)
