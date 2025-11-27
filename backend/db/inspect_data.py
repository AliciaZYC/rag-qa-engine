"""
Script to inspect the parquet data structure
"""
import pandas as pd
import sys

def inspect_data():
    try:
        df = pd.read_parquet('/app/data/train.parquet')
        
        print("=" * 60)
        print("DATA INSPECTION")
        print("=" * 60)
        print(f"\nShape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nColumn Types:")
        print(df.dtypes)
        print(f"\nFirst 3 rows:")
        print(df.head(3).to_string())
        print("\n" + "=" * 60)
        
        return df
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    inspect_data()

