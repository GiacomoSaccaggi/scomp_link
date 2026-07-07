FROM python:3.12-slim

# HF Spaces requirements
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

WORKDIR /app

# Install scomp-link with MCP support
RUN pip install --no-cache-dir "scomp-link[mcp]>=1.2.8" "mcp>=1.6.0"

# Copy the SSE entrypoint
COPY app.py .

# HF Spaces expects port 7860
EXPOSE 7860

# Non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user

CMD ["python", "app.py"]
