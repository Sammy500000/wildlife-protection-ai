# Behaviour Dataset: Animal Kingdom

## Decision
Animal Kingdom is the primary external dataset for the first behaviour-modeling stage. The official repository is https://github.com/sutdcv/Animal-Kingdom. The CVPR 2022 paper reports 30K video sequences for fine-grained multi-label action recognition, 50 hours of behaviour-grounded video and 33K pose frames across 850 species and six major animal classes.

## Important modeling decision
Animal Kingdom uses multi-label action recognition. Our production ontology is intentionally smaller: RESTING, NORMAL_MOVEMENT, RUNNING, CHASING, AGGRESSIVE_ABNORMAL, UNKNOWN. Therefore the training adapter must preserve the original Animal Kingdom action labels and provide an explicit mapping layer rather than pretending the source labels are identical to our conservation ontology.

## Lowest-effort path
1. Download only the action-recognition portion initially; do not download the full ~80GB distribution until required.
2. Parse the official Charades-format CSV annotations (clip_id, clip_number, frame_number, clip_path, action_labels).
3. Build a local manifest containing sequence frame paths, original multi-label actions, source split and source clip ID.
4. Train a multi-label source-domain baseline first, then create the project ontology mapping for transfer learning.
5. Keep source-domain test clips separate from Indian/target-domain evaluation clips.
6. Fine-tune the ResNet18+LSTM model on any permitted Indian-labelled clips later; do not claim Animal Kingdom performance as Indian-wildlife performance.

## Reproducibility
Record dataset version/download date, source URL, annotation checksum, generated manifest checksum and exact action-label mapping in every experiment.

## Citation
Ng, Xun Long, et al. “Animal Kingdom: A Large and Diverse Dataset for Animal Behavior Understanding.” CVPR 2022, pp. 19023–19034.
