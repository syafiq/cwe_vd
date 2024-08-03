# CWE-Specific Vulnerability Detection

## Introduction
This repository contains the code and models for our paper **"From Generalist to Specialist: Exploring CWE-Specific Vulnerability Detection"**. The goal of this research is to improve vulnerability detection by leveraging CWE-specific classifiers to address the heterogeneity of vulnerability types. Our results demonstrate that CWE-specific classifiers outperform a single binary classifier trained on all vulnerabilities.

## Repository Structure
- `data/`: Can be accessed here : https://drive.google.com/drive/folders/1olK4RwMA4xSmXY8rkkL4_ZlafEb_cBGi?usp=sharing
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
