import argparse
import pandas as pd
import sys
import os
from tqdm import tqdm
import joblib
import numpy as np

def parse_fasta(file_path):
    """Parses a FASTA file and returns a list of dictionaries with ID and SEQUENCE."""
    records = []
    current_id = None
    current_seq = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue # Skip empty lines
                
            if line.startswith(">"):
                # Save the previous record if it exists
                if current_id is not None:
                    records.append({
                        "ID": current_id, 
                        "SEQUENCE": "".join(current_seq)
                    })
                
                # Extract the new ID (usually the first word after '>')
                current_id = line[1:].split()[0] 
                current_seq = []
            else:
                current_seq.append(line)
                
        # Don't forget to append the very last record
        if current_id is not None:
            records.append({
                "ID": current_id, 
                "SEQUENCE": "".join(current_seq)
            })
            
    return records

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
DIPEPTIDES = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]
tqdm.pandas()

def aa_percentages(seq):
    """Return % composition of each AA (including X)."""
    seq = seq.upper()
    length = len(seq)
    counts = {aa: 0 for aa in AMINO_ACIDS}
    for aa in seq:
        if aa in counts:
            counts[aa] += 1
    return {aa: (counts[aa] / length) * 100 if length > 0 else 0
            for aa in AMINO_ACIDS}

def dipeptide_percentages(seq):
    """Return % composition of each dipeptide (AA, AC, ..., XX)."""
    seq = seq.upper()
    n = len(seq)
    counts = {dp: 0 for dp in DIPEPTIDES}

    # Count dipeptides
    for i in range(n - 1):
        dp = seq[i:i+2]
        if dp in counts:
            counts[dp] += 1

    # Frequency = count / total possible dipeptides
    denom = max(n - 1, 1)  # avoid division by zero
    return {dp: (counts[dp] / denom) * 100 for dp in DIPEPTIDES}

def featurize_aa_count(df):
    composition_df = df["SEQUENCE"].apply(aa_percentages).apply(pd.Series)
    return composition_df

def featurize_aa_and_dipeptides(df):
    """Return a DataFrame with AA % and dipeptide % features."""
    aa_df = df["SEQUENCE"].progress_apply(aa_percentages).apply(pd.Series)
    di_df = df["SEQUENCE"].progress_apply(dipeptide_percentages).apply(pd.Series)
    return pd.concat([aa_df, di_df], axis=1)

def featurize_aa_and_dipeptides_fast(df):
    # Calculate AA percentages
    print("Calculating Amino Acid composition...")
    aa_data = [aa_percentages(seq) for seq in tqdm(df["SEQUENCE"])]
    aa_df = pd.DataFrame(aa_data)

    # Calculate Dipeptide percentages
    print("Calculating Dipeptide composition...")
    di_data = [dipeptide_percentages(seq) for seq in tqdm(df["SEQUENCE"])]
    di_df = pd.DataFrame(di_data)

    return pd.concat([aa_df, di_df], axis=1)

def main():
    # Setup argparse for helper functionality and command line inputs
    parser = argparse.ArgumentParser(
        description="Process a FASTA or CSV file containing protein sequences into a standardized pandas DataFrame."
    )
    
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Path to the input file (.fasta, .fa, or .csv)"
    )
    
    parser.add_argument(
        "-o", "--output", 
        required=False, 
        help="Optional: Path to save the resulting DataFrame as a CSV file."
    )
    
    parser.add_argument(
    "-b", "--binary",
    action="store_true",
    help="Include this flag to switch from ternary to binary model."
    )

    args = parser.parse_args()
    input_file = args.input
    current_path = os.getcwd()
    ad_threshold = 0.41

    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        sys.exit(1)

    # Determine file type and process accordingly
    file_ext = os.path.splitext(input_file)[1].lower()

    if file_ext in ['.csv']:
        print("Detected CSV file. Loading...")
        try:
            df = pd.read_csv(input_file)
            # Capitalize all column names to standard "ID" and "SEQUENCE"
            df.columns = [str(col).upper() for col in df.columns]
            
            # Validate that the required columns exist
            if 'ID' not in df.columns or 'SEQUENCE' not in df.columns:
                print("Error: The CSV must contain an 'ID' (or 'id') and 'SEQUENCE' (or 'sequence') column.")
                sys.exit(1)
                
            # Filter to just keep our target columns
            df = df[['ID', 'SEQUENCE']]
            
        except Exception as e:
            print(f"Failed to parse CSV: {e}")
            sys.exit(1)

    elif file_ext in ['.fasta', '.fa', '.txt']:
        print("Detected FASTA file. Parsing...")
        try:
            records = parse_fasta(input_file)
            df = pd.DataFrame(records)
        except Exception as e:
            print(f"Failed to parse FASTA: {e}")
            sys.exit(1)
            
    else:
        print(f"Error: Unsupported file extension '{file_ext}'. Please provide a .csv or .fasta file.")
        sys.exit(1)

    # Compute AA_composition features
    X = featurize_aa_and_dipeptides_fast(df)
    print('Predicting class ...')
    # Compute predictions
    if args.binary:
        model = joblib.load(os.path.join(current_path, 'src', 'binary_model.joblib'))
        df['pred label'] = model.predict(X)
        df['pred class'] = df['pred label'].map({0: 'MDP', 1: 'MPP'})
        df['pred proba'] = model.predict_proba(X)[:, -1]
    else:
        model = joblib.load(os.path.join(current_path, 'src', 'ternary_model.joblib'))
        df['pred label'] = model.predict(X)
        df['pred class'] = df['pred label'].map({0: 'MDP', 1: 'DNA', 2:'CP'})
        probas = model.predict_proba(X)
        df['MDP proba'] = np.round(probas[:, 0], 3)
        df['DNA proba'] = np.round(probas[:, 1], 3)
        df['CP proba'] = np.round(probas[:, -1], 3)
    
    nn = joblib.load(os.path.join(current_path, 'src', '1NN.joblib'))
    distances, _ = nn.kneighbors(X)
    distances = distances.flatten()
    df['distance'] = np.round(distances, 3)
    df['inside AD'] = np.round(distances, 3) < ad_threshold
    # Save to a new CSV if the user requested it
    if args.output:
        df.to_csv(args.output, index=False)
    else:
        df.to_csv(os.path.join(current_path, 'predictions.csv'), index=False)

if __name__ == "__main__":
    main()