# CWE-Specific Vulnerability Detection

## Introduction
This repository contains the code and models for our paper **"From Generalist to Specialist: Exploring CWE-Specific Vulnerability Detection"**.

## Repository Structure
- data: Can be accessed here: https://drive.google.com/drive/folders/1olK4RwMA4xSmXY8rkkL4_ZlafEb_cBGi?usp=sharing
- finetuned models: Can be accessed here: https://lu.box.com/s/r84lieu71j1w93g6s9gywaqregczy27g
- `RQ1/`: Scripts for building m_all and m_CWE in RQ1.
- `RQ2/`: Scripts for building m_binary and m_multiclass in RQ2.

## Prerequisites
- Python 3.8 or higher
- PyTorch
- Transformers library (Hugging Face)
- scikit-learn
- pandas
- numpy

## Installation
To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Running the Scripts

Each script in the RQ1 and RQ2 directories is designed to be run on a computer with an Nvidia A100 (80GB) GPU. This hardware is required to build the models as described in our paper.
