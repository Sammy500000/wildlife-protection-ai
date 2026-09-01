# Wildlife Protection AI

AI-based wildlife video surveillance: detection → tracking → species → behaviour → conservation risk → ranger alerting.

## Architecture

Camera/video → MegaDetector V6 → ByteTrack → SpeciesNet + VideoMAE behaviour → Risk Engine → Alert API → Next.js/Vercel Ranger Dashboard.

The ML worker runs separately from Vercel because video inference and model execution are not appropriate serverless workloads.

## Implemented

- MegaDetector V6 + ByteTrack video pipeline
- SpeciesNet integration
- Animal Kingdom annotation adapter
- deterministic 8-frame sequence extraction and train/val/test split by clip
- CPU ResNet18 + LSTM behaviour baseline (legacy)
- VideoMAE cattle-behaviour checkpoint adapter for V1
- per-track temporal crop → VideoMAE behaviour inference
- stable wildlife behaviour ontology mapping
- behaviour evaluation with macro-F1/confusion matrix
- transparent risk scoring
- event/alert orchestration and deduplication
- FastAPI MVP
- Next.js/Vercel ranger dashboard MVP
- deployment scaffolding and reproducibility documentation

## Current execution boundary

The software architecture is now sufficiently complete for the first end-to-end research run. The remaining work that cannot be performed from the repository is execution against the actual local dataset/model environment:

1. Download/obtain the Animal Kingdom action-recognition data under its current access/licensing terms.
2. Run the sequence preparation script.
3. Train the CPU ResNet18+LSTM model.
4. Evaluate it on the frozen test manifest.
5. Run the integrated video pipeline with the trained checkpoint.
6. Inspect failures and refine the source-to-project behaviour mapping.
7. Run the final research ablations and target-domain evaluation.
8. Deploy the web dashboard to Vercel and host the inference/API worker separately.

## Behaviour dataset

See \`docs/BEHAVIOR_DATASET_ANIMAL_KINGDOM.md\`.

## First execution

From the activated project virtual environment:

\`\`\`powershell
python -m scripts.prepare_animal_kingdom --annotations <PATH_TO_ACTION_CSV> --video-root <PATH_TO_ACTION_VIDEOS> --output data/processed/animal_kingdom/manifest.json --sequence-length 8
\`\`\`

Then inspect:

\`\`\`powershell
type data\\processed\\animal_kingdom\\manifest_summary.json
\`\`\`

Train:

\`\`\`powershell
python -m scripts.train_behavior_lstm --train data/processed/animal_kingdom/train.json --val data/processed/animal_kingdom/val.json --output data/models/behavior/resnet18_lstm.pt --epochs 5 --batch-size 2
\`\`\`

Evaluate:

\`\`\`powershell
python -m scripts.evaluate_behavior --manifest data/processed/animal_kingdom/test.json --checkpoint data/models/behavior/resnet18_lstm.pt
\`\`\`

Run the complete surveillance pipeline:

\`\`\`powershell
python -m scripts.run_pipeline --input data\\raw\\test_video\\test.mp4 --output-dir data\\outputs\\full_pipeline --behavior-checkpoint models\\behavior\\videomae\\videomae_combined_v1.pt
\`\`\`

## Important research limitation

Animal Kingdom is not an India-specific conservation-risk dataset. Its behaviour labels are multi-label and are mapped into the project's smaller ontology for the baseline. Animal Kingdom performance must not be presented as Indian-wildlife performance. Target-domain validation and ranger-reviewed scenarios are required before making conservation claims.

## Deployment decision

GitHub is the source of truth. The Next.js dashboard is intended for Vercel deployment. The Python inference/ML worker is intentionally separate and should run on a machine/server capable of sustained video inference. Vercel should serve the UI/API edge layer, not the heavy video model runtime.
