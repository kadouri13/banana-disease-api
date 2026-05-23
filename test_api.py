import requests
import sys

# 1. Put your actual Render URL here
url = "https://banana-disease-api.onrender.com/predict" 

# 2. Put the path to your image here
image_path = "test_image.png"

print(f"Sending image to {url}...")

try:
    # This is exactly how you must send it (as a 'file' in multipart/form-data)
    with open(image_path, "rb") as image_file:
        files = {"file": image_file}
        response = requests.post(url, files=files)
    
    # Print the result
    if response.status_code == 200:
        print("\n✅ SUCCESS! Here is the prediction:")
        import json
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\n❌ FAILED! Status Code: {response.status_code}")
        print(response.text)

except FileNotFoundError:
    print(f"\n❌ ERROR: I could not find the image '{image_path}'.")
    print("Make sure you put an image named 'test_image.png' in the exact same folder as this script!")
