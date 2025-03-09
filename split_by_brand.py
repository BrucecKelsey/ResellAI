import pandas as pd
import os

def split_by_brand(input_file="resell_data.csv", output_dir="C:\\Users\\truec\\OneDrive\\Desktop\\Code\\ResellAI\\CsvData"):
    """
    Split resell_data.csv into separate CSV files by brand.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_dir (str): Directory to save the output CSV files.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Read the CSV file
        print(f"Reading {input_file}...")
        df = pd.read_csv(input_file, dtype=str)  # Load as strings to preserve data
        print(f"Loaded {len(df)} rows.")
        
        # Check if 'brand' column exists
        if 'brand' not in df.columns:
            print("Error: 'brand' column not found in the CSV file.")
            return
        
        # Group by brand
        grouped = df.groupby('brand')
        
        # Save each brand's data to a separate CSV
        for brand, brand_data in grouped:
            # Clean brand name for filename (remove invalid characters)
            safe_brand = "".join(c for c in brand if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
            output_file = os.path.join(output_dir, f"{safe_brand}.csv")
            
            # Save to CSV
            print(f"Saving {len(brand_data)} rows for brand '{brand}' to {output_file}...")
            brand_data.to_csv(output_file, index=False)
        
        print("Splitting complete!")
        
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
    except PermissionError:
        print(f"Error: Permission denied when writing to {output_dir}. Ensure no files are open and you have write access.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Run the function
    split_by_brand()