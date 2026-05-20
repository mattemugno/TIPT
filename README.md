# TIPT-ViTPose Hugging Face Prototype

Questo progetto implementa la prima versione richiesta dalla specifica PDF:

- wrapper `TiptVitPoseForPoseEstimation` basato su `transformers.VitPoseForPoseEstimation`;
- stream strutturale Sobel -> patch embedding -> 4 Transformer block;
- fusione shape-guided cross-attention con `alpha` iniziale a `0.1`;
- dataset COCO2017 top-down con GT bbox, crop `256 x 192` e target heatmap gaussiane `64 x 48`;
- loss MSE pesata dalla visibility mask, training loop, checkpoint e metrica proxy PCK.

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

La configurazione di default fa warm-up congelando ViTPose per 2 epoch, poi sblocca backbone/head mantenendo due learning rate group:

- nuovi moduli TIPT: `1e-4`;
- pesi pretrained ViTPose: `1e-5`.

## Valutazione Proxy

```bash
python eval_coco.py --config configs/tipt_vitpose_hf_coco.yaml --checkpoint runs/tipt_vitpose/best.pt
```

Per le ablation richieste, modifica `model.variant` e `model.fusion`:

- baseline HF: `variant: baseline`;
- TIPT fusione semplice: `variant: tipt_simple`, `fusion: simple`;
- TIPT cross-attention: `variant: tipt_cross_attention`, `fusion: cross_attention`;
- shape only o texture only: `fusion: shape_only` oppure `fusion: texture_only`.
- Sobel vs grayscale: `structural_view: sobel` oppure `structural_view: grayscale`.
