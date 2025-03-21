---
title: Lost Cities RL Backend
emoji: 🎮
sdk: docker
app_file: api.py
app_port: 8080
pinned: false
---


# Lost Cities RL Backend

This is the backend service for the Lost Cities Reinforcement Learning project.

## Development

1. Make a virtual environment(Optional but recommended) in the root directory:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For development, install additional tooling:
```bash
pip install -r requirements-dev.txt
```

4. Set up pre-commit hooks:
```bash
pre-commit install
```

5. Run the server:
```bash
uvicorn api:app --reload --port 8080
```

### Testing

To run the tests:
```bash
pytest tests
```


## Development Tooling

The project uses pre-commit hooks to ensure code quality. These include:
- black (code formatting)
- isort (import sorting)
- flake8 (linting)
- mypy (type checking)
- Various file checks (trailing whitespace, merge conflicts, etc.)

To manually run all checks on all files:
```bash
pre-commit run --all-files
```

## Docker Deployment

The service is configured to be deployed using Docker with multi-stage builds for optimized image size and build time.

### Using Docker with WSL

If you have Docker Desktop installed in Windows and are using WSL:

1. Open Docker Desktop
2. Go to Settings -> Resources -> WSL Integration
3. Enable integration for your WSL distro (Ubuntu)
4. Click "Apply & Restart"

### Building and Running

To run locally:
```bash
uvicorn api:app --reload --port 8080
```

or

to build and run in a container:

```bash
docker build -t lost-cities-rl-be .
docker run -p 8080:8080 lost-cities-rl-be
```


Also, check the [frontend repository](https://github.com/jpotw/lost-cities-rl-fe) to run with the frontend.
