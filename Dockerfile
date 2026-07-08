FROM python:3.12-slim

LABEL org.opencontainers.image.title="scomp-link"
LABEL org.opencontainers.image.description="End-to-end ML toolkit with 25 CLI commands and MCP server"
LABEL org.opencontainers.image.source="https://github.com/GiacomoSaccaggi/scomp_link"
LABEL org.opencontainers.image.version="1.2.9"

WORKDIR /app

# Install scomp-link with all extras
RUN pip install --no-cache-dir scomp-link[mcp,serve]

# Default: serve mode (override with docker run args)
EXPOSE 8080
ENTRYPOINT ["scomp-link"]
CMD ["--help"]

# Usage examples:
# docker build -t scomp-link .
# docker run scomp-link describe --data /data/train.csv
# docker run -v ./models:/models scomp-link serve --artifact /models/model.scomp --port 8080
# docker run scomp-link mcp
