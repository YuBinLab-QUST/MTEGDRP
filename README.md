# MTEGDRP

Interpretable molecular self-attention transformer and equivariant graph neural network based on multi-omics for drug response prediction in cancer cell lines

## Data

Available data files (some data is available in Releases due to size restrictions):

- `Cell_line_RMA_proc_basalExp.csv` - Gene expression data used for model training
- `Cell_line_RMA_proc_basalExp.txt` - Gene expression data used for model training
- `Cell_list.csv` - List of cancer cell line data information
- `drug_smiles.csv` - Contains information about all drug smiles
- `Druglist.csv` - All drugs involved in the training of the model
- `METH_CELLLINES_BEMs_PANCAN.csv` - DNA methylation data used for model training
- `PANCANCER_Genetic_feature.csv` - Genomic mutation data used for model training
- `PANCANCER_IC.csv` - Drug response data for known cancer cell lines in the GDSC2 database
- `pychem_cid.csv` - pychem cid information for model training drugs
- `small_molecule.csv` - Small molecule information for model training drugs
- `unknow_drug_by_pychem.csv` - No drugs listed for pychem cid

## Source Code

- `Data_encoding.py`: Encodes drug data and cancer cell line data into pytorch tensor format for model training. Also handles partitioning into training, test and validation sets.
- `Model_training.py`: Contains the overall model framework for drug response prediction using drug data and cancer cell line data.
- `Model_utils.py`: Provides function call support for data encoding, model training and model validation.

## Requirements

- torch==1.10.2+cu11
- python==3.8.20
- rdkit==2024.3.5
- pandas==2.0.3
- numpy==1.24.3
- scipy==1.10.1
- torch-cluster==1.5.9
- torch-geometric==2.0.4
- torch-scatter==2.0.9
- torch-sparse==0.6.12
- torch-spline-conv==1.2.1
- torchaudio==0.10.2+cu113
- torchvision==0.11.3+cu113

## Step-by-step Running

1. Create data in pytorch tensor format:
```bash
python Data_encoding.py
```

2.Train a MTEGDRP model:
```
python Model_training.py
