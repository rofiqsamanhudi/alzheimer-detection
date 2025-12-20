# Alzheimer Detection Dashboard - Docker Setup

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

Access the dashboard at: http://localhost:8501

### Using Docker Commands

```bash
# Build the image
docker build -t alzheimer-dashboard .

# Run with volume mounting
docker run -d \
  --name alzheimer-dashboard \
  -p 8501:8501 \
  -v $(pwd):/app \
  alzheimer-dashboard

# View logs
docker logs -f alzheimer-dashboard

# Stop and remove
docker stop alzheimer-dashboard
docker rm alzheimer-dashboard
```

## Volume Mounts

The Dashboard folder is mounted to `/app` in the container, allowing:

- Live code changes without rebuilding
- Model files to be accessed from `src/*/model/` directories
- Easy debugging and development

## Notes

- Models must be trained before running the dashboard
- Ensure model files exist in `src/CNN/model/`, `src/Transformer/model/`, and `src/classical/model/`
- The container uses GPU if available (requires nvidia-docker)
