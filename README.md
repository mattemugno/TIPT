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

Sweep clean/blur/pixelation baseline ViTPose-B vs TIPT-v3:

```bash
python scripts/eval_obfuscation_sweep.py \
  --baseline-config configs/vitpose_b_baseline_hf_coco.yaml \
  --blur-min-kernel 3 \
  --blur-max-kernel 17 \
  --pixel-sizes 4 6 8 10 12 16 \
  --plot
```

Lo script salva metriche, predizioni, `summary.csv`, `summary.json` e, con `--plot`, i grafici in `runs/obfuscation_sweeps/<timestamp>/plots/`.

Il vecchio comando resta valido se vuoi solo il blur:

```bash
python scripts/eval_blur_sweep.py \
  --baseline-config configs/vitpose_b_baseline_hf_coco.yaml \
  --min-kernel 3 \
  --max-kernel 17
```

Per generare o rigenerare grafici da una sweep gia' eseguita:

```bash
python scripts/plot_sweep.py runs/obfuscation_sweeps/<timestamp>/summary.json --metrics AP AP50 AP75
```

Sweep del peso della loss di invarianza shape:

```bash
python scripts/train_invariance_weight_sweep.py \
  --weights 0.02 0.05 0.10 \
  --blur-min-kernel 3 \
  --blur-max-kernel 17 \
  --pixel-sizes 4 6 8 10 12 16 \
  --plot
```

Questo script crea config temporanei in `runs/invariance_weight_sweeps/<timestamp>/configs/`, addestra un checkpoint per ogni peso e poi valuta ogni checkpoint su clean, blur e pixelation.

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

L'evaluator usa di default il protocollo piu' vicino a ViTPose/Hugging Face:

- `data.crop_method: vitpose` per crop affine top-down da bbox COCO;
- `eval.decode: hf` per decoding DARK/unbiased delle heatmap.

Per riprodurre vecchi risultati con argmax grezzo e crop PIL puoi usare `--decode simple` e `crop_method: pil`, ma i confronti baseline/TIPT vanno fatti tutti con lo stesso protocollo.

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
