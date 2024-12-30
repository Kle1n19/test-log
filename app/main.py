from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import cv2
import numpy as np
from PIL import Image, ImageDraw
import multiprocessing
import os
import time
import threading
import uvicorn

app = FastAPI()

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/process")
def process_images(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    kernel_size: int = Form(...),
    segment_size: int = Form(...),
    brightness_threshold: int = Form(...),
):
    num_threads_per_process = os.cpu_count()//2
    image1_data = image1.file.read()
    image2_data = image2.file.read()

    images_args = [(image1_data, kernel_size, segment_size, brightness_threshold, num_threads_per_process),(image2_data, kernel_size, segment_size, brightness_threshold, num_threads_per_process)]
    start_time = time.time()
    with multiprocessing.Pool(processes=2) as pool:
        processed_images = pool.map(process_image_in_processes, images_args)
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Total processing time for both images: {processing_time:.2f} seconds")

    start_time = time.time()
    collage_width = max(img.width for img in processed_images)
    collage_height = sum(img.height for img in processed_images)
    collage = Image.new("RGB", (collage_width, collage_height))
    y_offset = 0
    for img in processed_images:
        collage.paste(img, (0, y_offset))
        y_offset += img.height

    collage_path = os.path.join(output_dir, "final_collage.png")
    collage.save(collage_path)
    end_time = time.time()
    saving_time = end_time - start_time
    print(f"Saving time for the final collage: {saving_time:.2f} seconds")
    return FileResponse(collage_path, media_type="application/octet-stream")


def process_image_in_processes(args):
    image_data, kernel_size, segment_size, brightness_threshold, num_threads = args
    
    img_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    h, w = blurred.shape
    segments = []
    segments_lock = threading.Lock()
    threads = []

    start_time = time.time()

    for i in range(num_threads):
        thread = threading.Thread(target=process_image_segment, args=(i, num_threads, blurred, segment_size, brightness_threshold, segments, segments_lock))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processing time for image with {num_threads} threads: {processing_time:.2f} seconds")
    start_time = time.time()    
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    for x1, y1, x2, y2 in segments:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    end_time = time.time()
    drawing_time = end_time - start_time
    print(f"Drawing time for image with {num_threads} threads: {drawing_time:.2f} seconds")   

    return pil_img


def process_image_segment(process_id, num_threads, blurred, segment_size, brightness_threshold, segments, segments_lock):
    h, w = blurred.shape
    rows_per_thread = h // num_threads
    start_row = process_id * rows_per_thread
    if process_id < num_threads - 1:
        end_row = (process_id + 1) * rows_per_thread  
    else:
        end_row = h

    local_segments = []
    for y in range(start_row, end_row, segment_size):
        for x in range(0, w, segment_size):
            x_end = min(x + segment_size, w)
            y_end = min(y + segment_size, h)
            segment = blurred[y:y_end, x:x_end]
            avg_brightness = np.mean(segment)
            if avg_brightness > brightness_threshold:
                local_segments.append((x, y, x_end, y_end))
    with segments_lock:
        segments.extend(local_segments)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
