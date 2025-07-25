import camelot
import pandas as pd

def extract_full_table_to_csv(pdf_path, output_csv):
    tables = camelot.read_pdf(pdf_path, flavor='stream', pages='all')
    pdf_data = pd.concat([table.df for table in tables], ignore_index=True)
    pdf_data.to_csv(output_csv, index=False)
    print(f"Full table extracted and saved to: {output_csv}")
    print(f"Rows: {len(pdf_data)}, Columns: {list(pdf_data.columns)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python test_pdf_extractor.py <pdf_path> <output_csv>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    output_csv = sys.argv[2]
    extract_full_table_to_csv(pdf_path, output_csv)