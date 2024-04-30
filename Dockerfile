# Use an NGC base image with CUDA and cuDNN compatible with PyTorch 2.2.2
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set the working directory
WORKDIR /workspace

# Install any additional dependencies if necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    curl \
    ca-certificates -y && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file from the host to the image
COPY requirements.txt /workspace/

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from opencv_fixer import AutoFix; AutoFix()"

# Copy your application files (from your host to your image)
COPY . /workspace