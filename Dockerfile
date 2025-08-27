FROM python:3.11-slim

# 1. Set working directory
WORKDIR /app

# 2. Install OS dependencies for OpenCV and building wheels
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenblas-dev \
    gfortran \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy requirements
COPY requirements.txt .

# 4. Upgrade pip
RUN pip install --upgrade pip

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your backend code
COPY . .

# 7. Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0

# 8. Set default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
