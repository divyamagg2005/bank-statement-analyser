import streamlit as st
import os
import pandas as pd
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import from util
sys.path.append(str(Path(__file__).parent.parent))
from util.pdf_parser import extract_full_table_to_csv

# Set page config
st.set_page_config(
    page_title="Bank Statement Analyzer",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("Bank Statement Analyzer")
    st.write("Upload your bank statement PDF to analyze transactions")
    
    # Create a file uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    output_file = "output/statement.csv"
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file = "temp_uploaded.pdf"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the PDF when the button is clicked
        if st.button("Process PDF"):
            with st.spinner('Processing PDF...'):
                try:
                    # Extract data from PDF
                    extract_full_table_to_csv(temp_file, output_file)
                    
                    # Read the CSV data
                    df = pd.read_csv(output_file)
                    
                    # Replace None/NaN with empty strings for display
                    df_display = df.fillna('')
                    
                    # Display success message
                    st.success("PDF processed successfully!")
                    
                    # Show the data
                    st.subheader("Extracted Transactions")
                    st.dataframe(df_display, use_container_width=True)
                    
                    # Add download button for the CSV
                    with open(output_file, 'rb') as f:
                        st.download_button(
                            label="Download CSV",
                            data=f,
                            file_name="bank_statement.csv",
                            mime="text/csv"
                        )
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
            
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    main()