# AMP Target Predictor

This repository contains the code to predict whether an antimicrobial peptide (AMP) targets the bacterial membrane (membrane disruptive) or an intracellular target (membrane penetrating). 

The tool extracts amino acid and dipeptide compositions from protein sequences and uses pre-trained machine learning models to classify them. It also calculates the distance to the training data's Applicability Domain (AD) using a 1-Nearest Neighbor (1NN) approach to give you a confidence metric for the predictions.

## Features
* **Two Classification Modes:**
  * **Ternary Model (Default):** Classifies peptides into Membrane Disruptive (MDP), DNA/RNA targeting (DNA), or Cytoplasmic Protein (CP) targeting.
  * **Binary Model:** Classifies peptides strictly into Membrane Disruptive (MDP) or Membrane Penetrating Peptides (MPP).
* **Input Flexibility:** Accepts standard `.fasta` files or `.csv` files containing `ID` and `SEQUENCE` columns.
* **Applicability Domain (AD):** Flags whether a given sequence falls within the reliable prediction space of the models (distance < 0.41).

## Prerequisites

To run the script, you will need Python 3 installed along with the following libraries:

```bash
pip install pandas numpy tqdm joblib scikit-learn
```

## Usage

Once installed, test the installation by executing:

```bash
python predict.py -i example/example.fasta
```
This will create a CSV file named `predictions.csv` with the following columns:
* ID: peptide identifier.
* SEQUENCE: peptide sequence.
* pred label: predicted label (0-1-2 for ternary and 0-1 for binary model).
* pred class: predicted class (MDP-DNA-CP for ternary and MDP-MPP for binary model).
* pred proba: the corresponding predicted probabilities.
* distance: cosine distance from training dataset.
* inside AD: whether the peptide is inside or outside the AD.

To see all the flags run:
```bash
python predict.py -h
```


