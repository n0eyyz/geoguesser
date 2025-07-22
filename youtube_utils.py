
import os
import subprocess
import json
import cv2
import google.generativeai as genai
from typing import List

def _get_location_timestamps_with_gemini_vision(video_path: str) -> List[int]:
    """
    Analyzes a video file directly using the Gemini 1.5 Pro vision model
    to identify timestamps where significant locations are shown.

    Args:
        video_path: The local path to the video file.

    Returns:
        A list of timestamps (in seconds) pinpointing location scenes.
    """
    print("[Step 1a] Uploading video to Gemini for analysis...")
    video_file = genai.upload_file(path=video_path)
    print(f"Video uploaded successfully. Waiting for processing...")

    while video_file.state.name == "PROCESSING":
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise Exception("Gemini video processing failed.")

    print("[Step 1b] Analyzing video with Gemini 1.5 Pro to find location timestamps...")
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        prompt = """
        You are an expert AI "Director" analyzing a YouTube travel or food vlog.
        Your task is to watch this entire video and identify the precise moments (in seconds) when a new, significant location is visually presented.

        **What to look for:**
        - A clear shot of a building's exterior (e.g., a restaurant, cafe, shop, landmark).
        - A clear shot of a sign with the location's name.
        - A panoramic or establishing shot of a well-known public place (e.g., a famous square, park, or monument).

        **Instructions:**
        1.  Analyze the entire video provided.
        2.  For each moment a new significant location is clearly shown, note the timestamp in **total seconds**.
        3.  Do not include timestamps for general scenery, inside a car, or when the location is not clearly identifiable.
        4.  Return the results as a single JSON array of unique integer timestamps, sorted in ascending order.
        5.  If no such locations are found, you MUST return an empty array `[]`.
        6.  Your final output must be ONLY the JSON array, with no other text, explanations, or markdown.

        **Example Output:**
        [45, 182, 350, 512]
        """
        response = model.generate_content([video_file, prompt])
        response_text = response.text.strip().replace('```json', '').replace('```', '').strip()

        print(f"Received response from Gemini: {response_text}")
        timestamps = json.loads(response_text)

        if not isinstance(timestamps, list) or not all(isinstance(t, int) for t in timestamps):
            raise ValueError("Gemini did not return a valid list of integer timestamps.")

        print(f"Found {len(timestamps)} potential location timestamps: {timestamps}")
        return sorted(list(set(timestamps)))

    except Exception as e:
        print(f"An error occurred while calling Gemini for video analysis: {e}")
        return []
    finally:
        print(f"Deleting uploaded file from Gemini server: {video_file.name}")
        genai.delete_file(video_file.name)

def analyze_and_capture_locations(video_url: str, output_dir: str = "captures") -> List[str]:
    """
    Analyzes a YouTube video using the "AI Director" pipeline.
    1. Downloads the video file.
    2. Gemini Vision analyzes the video to find precise location timestamps.
    3. Captures screenshots at only those precise moments.

    Args:
        video_url: The URL of the YouTube video.
        output_dir: The directory to save the captured images.

    Returns:
        A list of file paths for the captured, high-value images.
    """
    print(f"Starting AI Director analysis for URL: {video_url}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_id = "temp_video_for_analysis"
    downloaded_video_path = os.path.join(output_dir, f"{video_id}.mp4")

    try:
        print(f"[Step 1] Downloading video to {downloaded_video_path}...")
        subprocess.run(
            ["yt-dlp", "-f", "best[ext=mp4][height<=720]", "-o", downloaded_video_path, video_url],
            check=True, capture_output=True, text=True
        )
        print("Video download complete.")

        timestamps = _get_location_timestamps_with_gemini_vision(downloaded_video_path)
        if not timestamps:
            print("No location timestamps found by Gemini. Aborting.")
            return []

        print(f"[Step 2] Capturing frames at {len(timestamps)} precise moments...")
        cap = cv2.VideoCapture(downloaded_video_path)
        if not cap.isOpened():
            raise Exception("Could not open downloaded video file.")

        captured_image_paths = []
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ret, frame = cap.read()
            if ret:
                image_filename = f"{video_id}_location_at_{ts}s.jpg"
                image_path = os.path.join(output_dir, image_filename)
                cv2.imwrite(image_path, frame)
                captured_image_paths.append(image_path)
                print(f"  - Captured frame for timestamp {ts}s to {image_path}")
            else:
                print(f"  - Failed to capture frame at {ts}s.")
        cap.release()
        return captured_image_paths

    except subprocess.CalledProcessError as e:
        print(f"Failed to download video with yt-dlp: {e.stderr}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during the pipeline: {e}")
        return []
    finally:
        if os.path.exists(downloaded_video_path):
            print(f"Cleaning up downloaded video file: {downloaded_video_path}")
            os.remove(downloaded_video_path)
