FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch — much smaller image, no GPU required for the demo
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Remaining dependencies
RUN pip install --no-cache-dir \
    streamlit \
    pyvis \
    pandas \
    numpy \
    scikit-learn \
    networkx \
    snorkel \
    transformers \
    peft \
    sentence-transformers \
    openai \
    datasketch \
    pyarrow

# Pre-download the sentence-transformer model used by DeepER
# so the image works offline at runtime
ENV HF_HOME=/app/.hf_cache
RUN python - <<'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
EOF

# Streamlit config (sets address=0.0.0.0, port=8501, headless=true)
COPY .streamlit .streamlit

# Application source and data
COPY showcase showcase

EXPOSE 8501

CMD ["streamlit", "run", "showcase/app/main.py"]
