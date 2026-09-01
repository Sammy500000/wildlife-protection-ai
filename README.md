# Wildlife Protection AI

AI-based wildlife video surveillance: detection → tracking → species → behaviour → conservation risk → ranger alerting.

## Architecture

Camera/video → MegaDetector → ByteTrack → SpeciesNet → ResNet+LSTM → Risk Engine → Alert Engine → FastAPI/PostgreSQL/Object Storage → Next.js/Vercel Ranger Dashboard.

The ML worker runs separately from Vercel because GPU/video inference is not a suitable serverless workload.

## Development status

Repository bootstrap in progress. The implementation prioritizes reuse of mature open-source components and reproducible research experiments.
