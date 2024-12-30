import requests
import time

image1_path = "medium_image.png"
image2_path = "medium_image.png"

data = {"kernel_size": 11, "segment_size": 50, "brightness_threshold": 0}

files = {"image1": open(image1_path, "rb"), "image2": open(image2_path, "rb")}

def send_request():
    response = requests.post("http://localhost:8000/process", data=data, files=files)
    return response

start_time = time.time()

response = send_request()

end_time = time.time()
elapsed_time = end_time - start_time

if response.status_code == 200:
    print("Request succeeded")
else:
    print("Error:", response.status_code, response.text)
print(f"Request response time: {elapsed_time:.2f} seconds.")

files["image1"].close()
files["image2"].close()