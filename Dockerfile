FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by OpenCV headless and albumentations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Patch Streamlit's index.html to permanently reserve the scrollbar gutter.
# Streamlit's own CSS has no overflow rule on any layout container, so the
# page scrollbar lives on <html> (browser default). st.markdown CSS is
# managed by React and briefly absent during reconciliation, causing the
# 10px width jump on every rerun. Patching index.html puts the fix before
# React loads — it can never be removed.
RUN python -c "\
import streamlit, pathlib; \
idx = pathlib.Path(streamlit.__file__).parent / 'static' / 'index.html'; \
css = '<style>html{overflow-y:scroll!important}*,*::before,*::after{scrollbar-gutter:stable}</style>'; \
content = idx.read_text(); \
idx.write_text(content.replace('</head>', css + '</head>', 1))"

# HuggingFace Spaces requires apps to listen on port 7860
EXPOSE 7860

CMD ["streamlit", "run", "apps/omyfish_web/main.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
