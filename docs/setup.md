# Setup Guide

## Environment

* macOS
* Python 3.11
* GLM-OCR
* LLaMA-Factory

## Clone Dependencies

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
git clone https://github.com/zai-org/GLM-OCR.git
```

## Create Python Environment

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
```

## Install LLaMA-Factory

```bash
cd LLaMA-Factory
pip install -e .
```

## Dataset

VizWiz dataset was converted into GLM-OCR training format and used for LoRA fine-tuning.

```
```

