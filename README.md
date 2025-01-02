# FastAPI Image Processing Application
Read full task in Test_assignment.pdf file.

## Requirements

- Python 3.9+
- FastAPI
- Uvicorn
- OpenCV
- NumPy
- Pillow
- Python Multipart(exactly 0.0.12 version as with other versions are some issues)

## Setup

### Virtual Environment

1. Navigate to your project directory:
    ```sh
    cd /Users/username/Desktop/test-log
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment(I used MacOS):
        ```sh
        source venv/bin/activate
        ```
4. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Docker

1. Navigate to your project directory:
    ```sh
    cd /Users/username/Desktop/test-log
    ```

2. Build the Docker image:
    ```sh
    docker build -t fastapi-app .
    ```

3. Run the Docker container:
    ```sh
    docker run -p 8000:8000 fastapi-app
    ```

## Usage

### Endpoints

- **Health Check**: `GET /health`
    - Returns the status of the application.

- **Process Images**: `POST /process`
    - Upload two images and provide the following form data:
        - `kernel_size`: Integer
        - `segment_size`: Integer
        - `brightness_threshold`: Integer
    - Returns the processed collage image.

### Example Request

```sh
curl -X POST "http://localhost:8000/process" \
  -F "image1=@path/to/image1.jpg" \
  -F "image2=@path/to/image2.jpg" \
  -F "kernel_size=5" \
  -F "segment_size=50" \
  -F "brightness_threshold=100"
```
