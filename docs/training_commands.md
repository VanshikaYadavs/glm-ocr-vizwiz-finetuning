# Training Commands

## Train LoRA

```bash
llamafactory-cli train vizwiz_lora_m4.yaml
```

## Export / Merge LoRA

```bash
llamafactory-cli export export_glmocr.yaml
```

## Test Merged Model

```bash
python test_vizwiz.py
```

## Evaluate LoRA Model

```bash
python evaluate_merged_vizwiz.py
```

## Evaluate Base GLM-OCR

```bash
python3 evaluate_vizwiz.py \
  --dataset ../dataset/vizwiz/val_glmocr_full.json \
  --image-root ../dataset/vizwiz \
  --api-url http://127.0.0.1:8080/chat/completions \
  --model mlx-community/GLM-OCR-bf16 \
  --limit 100
```

