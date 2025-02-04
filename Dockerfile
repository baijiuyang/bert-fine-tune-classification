# Use the official PyTorch image with CUDA 11.7
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter and ipykernel
RUN pip install jupyter ipykernel

# Register ipykernel with a custom name
RUN python -m ipykernel install --user --name=bert-fine-tune --display-name "bert-fine-tune"

# Copy source code and notebooks
COPY src/ src/
COPY notebook.ipynb .

# Expose Jupyter Notebook port
EXPOSE 8888

# Verify GPU availability inside the container
RUN python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# Run Jupyter Notebook with GPU-enabled PyTorch
CMD jupyter notebook --NotebookApp.kernel_name="bert-fine-tune" --ip=0.0.0.0 --port=8888 --no-browser --allow-root