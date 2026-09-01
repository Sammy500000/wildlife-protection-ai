# Setup

Python 3.11+ is recommended. Create a virtual environment and install the root requirements plus API requirements. On NVIDIA systems, install a PyTorch wheel matching the installed driver/CUDA support first.

Use the official CVAT Docker Compose deployment for annotation. Keep raw datasets and videos outside Git.

Run the API with: uvicorn apps.api.main:app --reload --port 8000

First ML milestone: place a test clip at data/raw/demo.mp4 and run the pipeline entrypoint after detector/tracker/species adapters are configured.
