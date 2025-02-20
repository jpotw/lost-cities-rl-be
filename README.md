# lost-cities-rl-be

## Configuration

```yaml
sdk: docker
```

## Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
uvicorn api:app --reload
```

## Deployment

The service is configured to be deployed using Docker. The Dockerfile is set up to use multi-stage builds for optimized image size and build time.

### Using Docker with WSL

If you have Docker Desktop installed in Windows and are using WSL:

1. Open Docker Desktop
2. Go to Settings -> Resources -> WSL Integration
3. Enable integration for your WSL distro (Ubuntu)
4. Click "Apply & Restart"

### Building and Running

To build and run locally:
```bash
docker build -t lost-cities-rl-be .
docker run -p 7860:7860 lost-cities-rl-be
``` 