# TIPT-ViTPose Hugging Face Prototype

Questo progetto implementa la prima versione richiesta dalla specifica PDF:

- wrapper `TiptVitPoseForPoseEstimation` basato su `transformers.VitPoseForPoseEstimation`;
- stream strutturale Sobel -> patch embedding -> 4 Transformer block;
- fusione shape-guided cross-attention con `alpha` iniziale a `0.1`;
- dataset COCO2017 top-down con GT bbox, crop `256 x 192` e target heatmap gaussiane `64 x 48`;
- loss MSE pesata dalla visibility mask, training loop, checkpoint, summary run e COCO keypoint AP.

## Setup

```bash
pip install -r requirements.txt
```

Scarica COCO2017 keypoints e aggiorna i path in `configs/tipt_vitpose_hf_coco.yaml`:

```text
coco/
  train2017/
  val2017/
  annotations/
    person_keypoints_train2017.json
    person_keypoints_val2017.json
```

## Training

```bash
python train.py --config configs/tipt_vitpose_hf_coco.yaml
```

TIPT-v2 shape-first con canali strutturali, edge CNN stem, gate dinamico e fusion residua multi-livello:

```bash
python train.py --config configs/tipt_vitpose_v2_hf_coco.yaml
```

TIPT-v3 aggiunge training two-view e invarianza esplicita tra shape token blur/pixelation:

```bash
python train.py --config configs/tipt_vitpose_v3_hf_coco.yaml
```

Sweep blur baseline ViTPose-B vs TIPT-v3:

```bash
python scripts/eval_blur_sweep.py \
  --baseline-config configs/vitpose_b_baseline_hf_coco.yaml \
  --min-kernel 3 \
  --max-kernel 17
```

Lo script salva metriche e predizioni in `runs/blur_sweeps/<timestamp>/`, più `summary.csv` e `summary.json`.

La configurazione di default fa warm-up congelando ViTPose per 2 epoch, poi sblocca backbone/head mantenendo due learning rate group:

- nuovi moduli TIPT: `1e-4`;
- pesi pretrained ViTPose: `1e-5`.

Ogni run salva:

- `runs/tipt_vitpose/<run_id>/latest.pt`
- `runs/tipt_vitpose/<run_id>/best.pt`
- `runs/tipt_vitpose/<run_id>/summary.json`
- `runs/tipt_vitpose/<run_id>/coco_keypoints_val.json` se `training.final_coco_eval: true`
- `runs/tipt_vitpose/latest_run.txt` con il path dell'ultima run

## Valutazione COCO

```bash
RUN_DIR=$(cat runs/tipt_vitpose/latest_run.txt)
python eval_coco.py --config configs/tipt_vitpose_hf_coco.yaml --checkpoint "$RUN_DIR/best.pt"
```

Test senza creare dataset offuscati su disco:

```bash
python eval_coco.py \
  --config configs/tipt_vitpose_hf_coco.yaml \
  --checkpoint "$RUN_DIR/best.pt" \
  --obfuscation blur \
  --blur-kernel-size 11 \
  --metrics-json "$RUN_DIR/eval_blur.json"

python eval_coco.py \
  --config configs/tipt_vitpose_hf_coco.yaml \
  --checkpoint "$RUN_DIR/best.pt" \
  --obfuscation pixelate \
  --pixel-size 8 \
  --metrics-json "$RUN_DIR/eval_pixelate.json"
```

Le metriche COCO sono calcolate in modalità top-down con GT bbox COCO, quindi confrontano il pose estimator senza includere un detector.

Nei range `blur_kernel_size`, vengono campionati solo valori dispari. Per esempio `[3, 17]` produce `3, 5, 7, ..., 17`.

## Controllo Visuale

```bash
python visualize_predictions.py \
  --config configs/tipt_vitpose_hf_coco.yaml \
  --checkpoint "$RUN_DIR/best.pt" \
  --output-dir "$RUN_DIR/visuals" \
  --num-samples 16
```

I PNG salvati mostrano predizioni in rosso e ground truth in verde sui crop persona del validation set.

Per le ablation richieste, modifica `model.variant` e `model.fusion`:

- baseline HF: `variant: baseline`;
- TIPT fusione semplice: `variant: tipt_simple`, `fusion: simple`;
- TIPT cross-attention: `variant: tipt_cross_attention`, `fusion: cross_attention`;
- shape only o texture only: `fusion: shape_only` oppure `fusion: texture_only`.
- Sobel vs grayscale: `structural_view: sobel` oppure `structural_view: grayscale`.
