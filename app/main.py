from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import cv2
import numpy as np
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
    num_threads_per_process = os.cpu_count() // 2
    image1_data = cv2.imdecode(np.frombuffer(image1.file.read(), np.uint8), cv2.IMREAD_COLOR)
    image2_data = cv2.imdecode(np.frombuffer(image2.file.read(), np.uint8), cv2.IMREAD_COLOR)

    images_args = [
        (image1_data, kernel_size, segment_size, brightness_threshold, num_threads_per_process),
        (image2_data, kernel_size, segment_size, brightness_threshold, num_threads_per_process),
    ]

    start_time = time.time()
    with multiprocessing.Pool(processes=2) as pool:
        processed_images = pool.map(process_image_in_processes, images_args)
    end_time = time.time()
    print(f"Total processing time for both images: {end_time - start_time:.2f} seconds")

    collage = create_collage(processed_images)
    collage_path = os.path.join(output_dir, "final_collage.png")
    cv2.imwrite(collage_path, collage)
    return FileResponse(collage_path, media_type="application/octet-stream")


def process_image_in_processes(args):
    image, kernel_size, segment_size, brightness_threshold, num_threads = args

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    h, w = blurred.shape
    segments = []
    segment_lock = multiprocessing.Lock()
    collage = image.copy()

    def worker(process_id):
        nonlocal collage
        num_of_rows = h // segment_size
        rows_per_thread = num_of_rows // num_threads
        start_row = process_id * rows_per_thread
        end_row = (process_id + 1) * rows_per_thread if process_id < num_threads - 1 else h

        local_segments = []
        for y in range(start_row, end_row, segment_size):
            for x in range(0, w, segment_size):
                x_end = min(x + segment_size, w)
                y_end = min(y + segment_size, h)
                segment = blurred[y:y_end, x:x_end]
                if np.mean(segment) > brightness_threshold:
                    local_segments.append((x, y, x_end, y_end))

        with segment_lock:
            segments.extend(local_segments)
            for x1, y1, x2, y2 in local_segments:
                cv2.rectangle(collage, (x1, y1), (x2, y2), (0, 0, 255), 3)

    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return collage


def create_collage(images):
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]

    total_height = sum(heights)
    max_width = max(widths)

    collage = np.zeros((total_height, max_width, 3), dtype=np.uint8)
    offsets = [0] + list(np.cumsum(heights[:-1]))

    row_map = []
    for img_index, (offset, height) in enumerate(zip(offsets, heights)):
        for row in range(offset, offset + height):
            row_map.append((row, img_index, row - offset))

    def process_rows(thread_id):
        total_rows = len(row_map)
        for i in range(thread_id, total_rows, os.cpu_count()):
            collage_row, img_index, img_row = row_map[i]
            collage[collage_row, :widths[img_index]] = images[img_index][img_row, :]

    threads = []
    for thread_id in range(os.cpu_count()):
        thread = threading.Thread(target=process_rows, args=(thread_id,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return collage


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
