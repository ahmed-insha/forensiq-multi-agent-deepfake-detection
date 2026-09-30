FROM python:3.11-slim

WORKDIR /app

# System dependencies needed by opencv and git (for cloning MVSS-Net/UnivFD at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0