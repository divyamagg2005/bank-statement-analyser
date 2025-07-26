import pandas as pd
import numpy as np
from datetime import datetime
import re
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

class FinancialStatementAnalyzer:
    def __init__(self, csv_file_path: str):
        """
        Initialize the analyzer with processed CSV file.
        
        Args:
            csv_file_path: Path to the processed CSV file
        """
        self.df = pd.read_csv(csv_file_path)
        self.column_mapping = self.detect_column_mapping()
        self.prepare_data()
    
    def consolidate_multirow_transactions(self) -> pd.DataFrame:
        """
        Enhanced consolidation of multi-row transactions where description spans multiple rows.
        This handles cases where banks split transaction details across multiple rows
        but only the first row contains the actual transaction data.
        """
        df_copy = self.df.copy()
        
        # Identify potential amount columns (including debit/credit columns with +/- signs)
        amount_cols = []
        for col in df_copy.columns:
            if any(keyword in col.lower() for keyword in ['amount', 'debit', 'credit', 'balance']):
                amount_cols.append(col)
        
        if not amount_cols:
            return df_copy  # No amount columns found, return as is
        
        # Find rows that are likely continuation rows (empty/null in key fields)
        consolidated_rows = []
        i = 0
        
        while i < len(df_copy):
            current_row = df_copy.iloc[i].copy()
            
            # Check if this row has substantial data (not a continuation row)
            has_amount = False
            has_date = False
            has_reference = False
            
            # Check for amount data (including +/- signed amounts)
            for col in amount_cols:
                if pd.notna(current_row[col]) and str(current_row[col]).strip() != '':
                    amount_str = str(current_row[col]).strip().replace(',', '').replace('"', '')
                    if amount_str and amount_str not in ['0', '0.0', '0.00', '']:
                        try:
                            # Handle +/- signs
                            if amount_str.startswith(('+', '-')):
                                float(amount_str)
                                has_amount = True
                                break
                            else:
                                float(amount_str)
                                if float(amount_str) != 0:
                                    has_amount = True
                                    break
                        except:
                            continue
            
            # Check for date data
            for col in df_copy.columns:
                if 'date' in col.lower() and pd.notna(current_row[col]) and str(current_row[col]).strip() != '':
                    date_str = str(current_row[col]).strip()
                    if date_str and len(date_str) > 5:  # Basic date length check
                        try:
                            pd.to_datetime(current_row[col], errors='raise')
                            has_date = True
                            break
                        except:
                            continue
            
            # Check for reference number or transaction ID
            for col in df_copy.columns:
                if any(keyword in col.lower() for keyword in ['ref', 'chq', 'transaction', 'txn']) and pd.notna(current_row[col]):
                    ref_str = str(current_row[col]).strip()
                    if ref_str and ref_str != '' and len(ref_str) > 3:
                        has_reference = True
                        break
            
            # If this row has substantial data, it's a main transaction row
            if has_amount or has_date or has_reference:
                # Look ahead for continuation rows
                description_parts = []
                
                # Get description from current row
                for col in df_copy.columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in ['description', 'particulars', 'narration', 'details', 'credit']) and 'amount' not in col_lower:
                        if pd.notna(current_row[col]) and str(current_row[col]).strip() != '':
                            desc_text = str(current_row[col]).strip()
                            if desc_text not in description_parts and len(desc_text) > 2:
                                description_parts.append(desc_text)
                
                # Check next rows for continuation
                j = i + 1
                while j < len(df_copy):
                    next_row = df_copy.iloc[j]
                    
                    # Check if next row is a continuation (no amount, no date, no reference)
                    next_has_amount = False
                    next_has_date = False
                    next_has_reference = False
                    
                    # Check for amount in next row
                    for col in amount_cols:
                        if pd.notna(next_row[col]) and str(next_row[col]).strip() != '':
                            amount_str = str(next_row[col]).strip().replace(',', '').replace('"', '')
                            if amount_str and amount_str not in ['0', '0.0', '0.00', '']:
                                try:
                                    if amount_str.startswith(('+', '-')):
                                        float(amount_str)
                                        next_has_amount = True
                                        break
                                    else:
                                        if float(amount_str) != 0:
                                            next_has_amount = True
                                            break
                                except:
                                    continue
                    
                    # Check for date in next row
                    for col in df_copy.columns:
                        if 'date' in col.lower() and pd.notna(next_row[col]) and str(next_row[col]).strip() != '':
                            date_str = str(next_row[col]).strip()
                            if date_str and len(date_str) > 5:
                                try:
                                    pd.to_datetime(next_row[col], errors='raise')
                                    next_has_date = True
                                    break
                                except:
                                    continue
                    
                    # Check for reference in next row
                    for col in df_copy.columns:
                        if any(keyword in col.lower() for keyword in ['ref', 'chq', 'transaction', 'txn']) and pd.notna(next_row[col]):
                            ref_str = str(next_row[col]).strip()
                            if ref_str and ref_str != '' and len(ref_str) > 3:
                                next_has_reference = True
                                break
                    
                    # If next row has no substantial data, it's likely a continuation
                    if not next_has_amount and not next_has_date and not next_has_reference:
                        # Extract description from continuation row
                        for col in df_copy.columns:
                            if pd.notna(next_row[col]) and str(next_row[col]).strip() != '':
                                desc_text = str(next_row[col]).strip()
                                # Skip if it's just numbers, commas, or very short text
                                if (desc_text and len(desc_text) > 2 and 
                                    desc_text not in description_parts and
                                    not desc_text.replace(',', '').replace('.', '').replace(' ', '').isdigit()):
                                    description_parts.append(desc_text)
                        j += 1
                    else:
                        break
                
                # Combine all description parts
                if description_parts:
                    combined_description = ' '.join(description_parts)
                    # Update the description in current row - find the best column for description
                    desc_col_found = False
                    for col in df_copy.columns:
                        col_lower = col.lower()
                        if any(keyword in col_lower for keyword in ['description', 'particulars', 'narration', 'details']) and 'amount' not in col_lower:
                            current_row[col] = combined_description
                            desc_col_found = True
                            break
                    
                    # If no dedicated description column found, use the credit column if it contains text
                    if not desc_col_found:
                        for col in df_copy.columns:
                            if 'credit' in col.lower() and 'amount' not in col.lower():
                                current_row[col] = combined_description
                                break
                
                consolidated_rows.append(current_row)
                i = j  # Skip the continuation rows
            else:
                # This might be a continuation row that wasn't caught, skip it
                i += 1
        
        if consolidated_rows:
            result_df = pd.DataFrame(consolidated_rows).reset_index(drop=True)
            print(f"Consolidated {len(df_copy)} rows into {len(result_df)} transactions")
            return result_df
        else:
            return df_copy

    def detect_column_mapping(self) -> Dict[str, str]:
        """
        Enhanced column mapping detection for different bank statement formats.
        Returns a mapping of standard column names to actual column names.
        """
        columns = [col.lower().strip() for col in self.df.columns]
        mapping = {}
        
        print(f"Available columns: {list(self.df.columns)}")
        
        # Date column detection
        date_patterns = ['date', 'transaction_date', 'txn_date', 'value_date', 'posting_date']
        for pattern in date_patterns:
            matches = [col for col in columns if pattern in col]
            if matches:
                mapping['date'] = self.df.columns[columns.index(matches[0])]
                break
        
        # Enhanced amount column detection - look for columns with +/- signs
        amount_patterns = ['amount', 'transaction_amount', 'txn_amount', 'debit_amount', 'credit_amount', 'debit/credit']
        for pattern in amount_patterns:
            matches = [col for col in columns if pattern in col]
            if matches:
                mapping['amount'] = self.df.columns[columns.index(matches[0])]
                break
        
        # Special handling for columns that might contain both debit and credit with +/- signs
        combined_patterns = ['debit/credit', 'amount', 'debit_credit', 'transaction_amount', 'debit', 'credit']
        for pattern in combined_patterns:
            matches = [col for col in columns if pattern in col and 'balance' not in col]
            if matches:
                col_name = self.df.columns[columns.index(matches[0])]
                # Check if this column contains +/- signs indicating it's a combined column
                sample_data = self.df[col_name].astype(str).head(20)
                plus_count = sum(1 for val in sample_data if '+' in str(val))
                minus_count = sum(1 for val in sample_data if '-' in str(val))
                
                if plus_count > 0 or minus_count > 0:
                    mapping['amount'] = col_name
                    print(f"Detected combined debit/credit column with +/- signs: {mapping['amount']}")
                    break
        
        # If no single amount column, look for separate debit/credit columns
        if 'amount' not in mapping:
            debit_patterns = ['debit', 'debit_amount', 'dr', 'withdrawal']
            credit_patterns = ['credit', 'credit_amount', 'cr', 'deposit']
            
            for pattern in debit_patterns:
                matches = [col for col in columns if pattern in col and 'balance' not in col]
                if matches:
                    mapping['debit'] = self.df.columns[columns.index(matches[0])]
                    break
            
            for pattern in credit_patterns:
                matches = [col for col in columns if pattern in col and 'balance' not in col]
                if matches:
                    mapping['credit'] = self.df.columns[columns.index(matches[0])]
                    break
        
        # Description column detection - enhanced to handle various formats
        desc_patterns = ['description', 'particulars', 'narration', 'details', 'transaction_details', 'remark']
        for pattern in desc_patterns:
            matches = [col for col in columns if pattern in col]
            if matches:
                mapping['description'] = self.df.columns[columns.index(matches[0])]
                break
        
        # If no explicit description column, check if credit column contains text descriptions
        if 'description' not in mapping:
            for col in columns:
                if 'credit' in col and 'amount' not in col:
                    col_name = self.df.columns[columns.index(col)]
                    # Check if this column contains text (not just numbers)
                    sample_data = self.df[col_name].dropna().astype(str).head(10)
                    text_count = sum(1 for val in sample_data if not val.replace(',', '').replace('.', '').replace('+', '').replace('-', '').replace(' ', '').isdigit())
                    if text_count > len(sample_data) * 0.5:  # More than 50% are text
                        mapping['description'] = col_name
                        print(f"Using column '{col_name}' as description column")
                        break
        
        # Balance column detection
        balance_patterns = ['balance', 'closing_balance', 'running_balance', 'available_balance']
        for pattern in balance_patterns:
            matches = [col for col in columns if pattern in col]
            if matches:
                mapping['balance'] = self.df.columns[columns.index(matches[0])]
                break
        
        # Reference/Cheque number detection
        ref_patterns = ['ref', 'chq', 'cheque', 'reference', 'transaction_id', 'txn_id']
        for pattern in ref_patterns:
            matches = [col for col in columns if pattern in col]
            if matches:
                mapping['reference'] = self.df.columns[columns.index(matches[0])]
                break
        
        print(f"Detected column mapping: {mapping}")
        return mapping
    
    def prepare_data(self):
        """Enhanced data preparation and cleaning."""
        try:
            # First, clean up multi-row transactions
            self.df = self.consolidate_multirow_transactions()
            
            # Handle date column
            if 'date' in self.column_mapping:
                date_col = self.column_mapping['date']
                self.df['date'] = pd.to_datetime(self.df[date_col], errors='coerce', dayfirst=True)
            else:
                # Try to find any date-like column
                for col in self.df.columns:
                    if self.df[col].dtype == 'object':
                        try:
                            sample_val = self.df[col].dropna().iloc[0] if not self.df[col].dropna().empty else ""
                            pd.to_datetime(sample_val, errors='raise')
                            self.df['date'] = pd.to_datetime(self.df[col], errors='coerce', dayfirst=True)
                            break
                        except:
                            continue
            
            # Enhanced amount column handling with proper +/- sign interpretation
            if 'amount' in self.column_mapping:
                # Single amount column exists (likely with +/- signs)
                amount_col = self.column_mapping['amount']
                # Clean amount data - preserve +/- signs but remove commas and quotes
                self.df['amount_raw'] = self.df[amount_col].astype(str).str.replace(',', '').str.replace('"', '').str.strip()
                
                # Enhanced parsing for +/- signs
                def parse_amount_with_sign(amount_str):
                    if pd.isna(amount_str) or amount_str == '' or amount_str == 'nan' or amount_str == 'nan':
                        return 0.0
                    amount_str = str(amount_str).strip()
                    
                    # Handle cases with explicit + sign (credits - money coming in)
                    if amount_str.startswith('+'):
                        return float(amount_str[1:])  # Remove + and keep positive (credit)
                    # Handle cases with explicit - sign (debits - money going out)
                    elif amount_str.startswith('-'):
                        return float(amount_str)  # Keep negative (debit)
                    # Handle cases with no sign - assume positive (credit) unless context suggests otherwise
                    else:
                        try:
                            return float(amount_str)
                        except:
                            return 0.0
                
                self.df['amount'] = self.df['amount_raw'].apply(parse_amount_with_sign)
                
                # Create debit and credit columns based on amount sign
                # Positive amounts = Credits (money in), Negative amounts = Debits (money out)
                self.df['debit'] = self.df['amount'].apply(lambda x: abs(x) if x < 0 else 0)
                self.df['credit_amount'] = self.df['amount'].apply(lambda x: x if x > 0 else 0)
                self.df['dr_cr_indicator'] = self.df['amount'].apply(lambda x: 'DR' if x < 0 else 'CR')
                
                print(f"Processed amounts with +/- signs. Credits: {len(self.df[self.df['amount'] > 0])}, Debits: {len(self.df[self.df['amount'] < 0])}")
            
            elif 'debit' in self.column_mapping and 'credit' in self.column_mapping:
                # Separate debit and credit columns
                debit_col = self.column_mapping['debit']
                credit_col = self.column_mapping['credit']
                
                # Enhanced parsing for separate columns that might still have +/- signs
                def parse_amount_column(col_data):
                    cleaned = col_data.astype(str).str.replace(',', '').str.replace('"', '').str.strip()
                    
                    def parse_single_amount(amount_str):
                        if pd.isna(amount_str) or amount_str == '' or amount_str == 'nan':
                            return 0.0
                        amount_str = str(amount_str).strip()
                        
                        if amount_str.startswith('+'):
                            return float(amount_str[1:])
                        elif amount_str.startswith('-'):
                            return abs(float(amount_str))  # Make positive for debit/credit columns
                        else:
                            try:
                                return abs(float(amount_str))
                            except:
                                return 0.0
                    
                    return cleaned.apply(parse_single_amount)
                
                self.df['debit'] = parse_amount_column(self.df[debit_col])
                self.df['credit_amount'] = parse_amount_column(self.df[credit_col])
                
                # Create amount column and dr_cr_indicator
                self.df['amount'] = self.df.apply(
                    lambda row: -row['debit'] if row['debit'] > 0 else row['credit_amount'], axis=1
                )
                self.df['dr_cr_indicator'] = self.df.apply(
                    lambda row: 'DR' if row['debit'] > 0 else 'CR', axis=1
                )
            
            else:
                # Try to find amount-like columns by examining data with +/- signs
                for col in self.df.columns:
                    try:
                        # Clean the data first but preserve signs
                        cleaned_data = self.df[col].astype(str).str.replace(',', '').str.replace('"', '').str.strip()
                        
                        # Check if this column contains +/- signs (indicating it's an amount column)
                        plus_count = sum(1 for val in cleaned_data if '+' in str(val))
                        minus_count = sum(1 for val in cleaned_data if '-' in str(val))
                        
                        if plus_count > 0 or minus_count > 0:
                            def parse_amount_with_sign(amount_str):
                                if pd.isna(amount_str) or amount_str == '' or amount_str == 'nan':
                                    return 0.0
                                amount_str = str(amount_str).strip()
                                
                                if amount_str.startswith('+'):
                                    return float(amount_str[1:])  # Remove + and keep positive
                                elif amount_str.startswith('-'):
                                    return float(amount_str)  # Keep negative
                                else:
                                    try:
                                        return float(amount_str)
                                    except:
                                        return 0.0
                            
                            numeric_data = cleaned_data.apply(parse_amount_with_sign)
                            
                            if not numeric_data.isna().all() and numeric_data.abs().max() > 100:  # Likely monetary values
                                self.df['amount'] = numeric_data.fillna(0)
                                self.df['debit'] = numeric_data.apply(lambda x: abs(x) if x < 0 else 0)
                                self.df['credit_amount'] = numeric_data.apply(lambda x: x if x > 0 else 0)
                                self.df['dr_cr_indicator'] = numeric_data.apply(lambda x: 'DR' if x < 0 else 'CR')
                                print(f"Using column '{col}' as amount column with +/- signs")
                                break
                    except:
                        continue
            
            # Handle description column
            if 'description' in self.column_mapping:
                desc_col = self.column_mapping['description']
                self.df['description'] = self.df[desc_col].fillna('')
            else:
                # Find the most likely description column (text column with varied content)
                text_cols = [col for col in self.df.columns if self.df[col].dtype == 'object']
                if text_cols:
                    # Use the first text column that's not date-related and contains meaningful text
                    for col in text_cols:
                        if ('date' not in col.lower() and 'balance' not in col.lower() and 
                            'amount' not in col.lower()):
                            # Check if it contains meaningful text descriptions
                            sample_data = self.df[col].dropna().astype(str).head(10)
                            if len(sample_data) > 0:
                                avg_length = sum(len(text) for text in sample_data) / len(sample_data)
                                if avg_length > 10:  # Meaningful descriptions are usually longer
                                    self.df['description'] = self.df[col].fillna('')
                                    print(f"Using column '{col}' as description column")
                                    break
                    else:
                        self.df['description'] = self.df[text_cols[0]].fillna('')
                else:
                    self.df['description'] = ''
            
            # Handle balance column with proper comma removal
            if 'balance' in self.column_mapping:
                balance_col = self.column_mapping['balance']
                # Enhanced balance cleaning - remove commas and quotes, preserve signs
                self.df['balance_clean'] = (self.df[balance_col].astype(str)
                                          .str.replace(',', '')
                                          .str.replace('"', '')
                                          .str.strip())
                self.df['balance'] = pd.to_numeric(self.df['balance_clean'], errors='coerce')
            
            # Ensure required columns exist
            required_cols = ['amount', 'debit', 'credit_amount', 'dr_cr_indicator', 'description']
            for col in required_cols:
                if col not in self.df.columns:
                    if col == 'amount':
                        self.df[col] = 0
                    elif col in ['debit', 'credit_amount']:
                        self.df[col] = 0
                    elif col == 'dr_cr_indicator':
                        self.df[col] = 'DR'
                    elif col == 'description':
                        self.df[col] = ''
            
            # Add month-year column for monthly analysis
            if 'date' in self.df.columns:
                self.df['month_year'] = self.df['date'].dt.to_period('M')
                # Sort by date and remove invalid dates
                self.df = self.df.sort_values('date').reset_index(drop=True)
                self.df = self.df.dropna(subset=['date'])
                
                # Remove rows where all financial columns are zero/null (likely continuation rows that weren't caught)
                financial_cols = ['amount', 'debit', 'credit_amount']
                existing_financial_cols = [col for col in financial_cols if col in self.df.columns]
                if existing_financial_cols:
                    # Keep rows where at least one financial column has a non-zero value
                    mask = (self.df[existing_financial_cols] != 0).any(axis=1)
                    self.df = self.df[mask].reset_index(drop=True)
            
            print(f"Data prepared successfully. Total transactions: {len(self.df)}")
            print(f"Columns available: {list(self.df.columns)}")
            
            # Print sample of processed data for verification
            if len(self.df) > 0:
                print("\nSample processed transactions:")
                sample_cols = ['date', 'description', 'amount', 'debit', 'credit_amount', 'dr_cr_indicator']
                available_cols = [col for col in sample_cols if col in self.df.columns]
                print(self.df[available_cols].head(3).to_string())
            
        except Exception as e:
            print(f"Error in data preparation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def get_cr_dr_counts(self) -> Dict[str, int]:
        """Get count of credit and debit transactions."""
        cr_count = len(self.df[self.df['dr_cr_indicator'] == 'CR'])
        dr_count = len(self.df[self.df['dr_cr_indicator'] == 'DR'])
        
        return {
            'credit_transactions': cr_count,
            'debit_transactions': dr_count,
            'total_transactions': len(self.df)
        }
    
    def get_eod_balance(self) -> float:
        """Get End of Day (EOD) balance - the last available balance."""
        if 'balance' in self.df.columns and not self.df['balance'].isna().all():
            # Get the last non-null balance
            last_balance = self.df['balance'].dropna().iloc[-1]
            return float(last_balance)
        else:
            # Calculate from transactions if balance column is not available
            # For +/- system: sum all amounts (positive credits, negative debits)
            return float(self.df['amount'].sum())
    
    def get_monthly_expenses(self) -> Dict[str, float]:
        """Get monthly expenses (debits) breakdown."""
        monthly_expenses = {}
        
        if 'month_year' in self.df.columns:
            for month_year, group in self.df.groupby('month_year'):
                month_debits = group[group['dr_cr_indicator'] == 'DR']['debit'].sum()
                monthly_expenses[str(month_year)] = float(month_debits)
        else:
            # If no date info, calculate total expenses
            total_debits = self.df[self.df['dr_cr_indicator'] == 'DR']['debit'].sum()
            monthly_expenses['Total'] = float(total_debits)
        
        return monthly_expenses
    
    def categorize_expenses(self) -> Dict[str, float]:
        """Categorize expenses based on transaction descriptions."""
        categories = {
            'EMI & Loans': ['emi', 'loan', 'installment', 'equated', 'repayment'],
            'Bills & Utilities': [
                'electricity', 'water', 'gas', 'bill', 'utility', 'pvvnlpostp', 
                'indraprast', 'airtel', 'mobile', 'recharge'
            ],
            'Bank Charges': ['charge', 'gst', 'consolidated', 'fee'],
            'Business Payments': [
                'neft', 'rtgs', 'imps', 'transfer', 'garment', 'chemical', 'dyes',
                'trading', 'industries', 'cetp', 'upsidc'
            ],
            'Cash Withdrawal': ['cash', 'atm', 'cwdr'],
            'Online Payments': ['payu', 'paytm', 'upi', 'nbsm', 'techprocess', 'blinkit'],
            'Cheque Payments': ['clg', 'cheque', 'dd issued', 'con'],
            'Others': []
        }
        
        category_totals = {category: 0.0 for category in categories}
        
        # Analyze debit transactions
        debit_transactions = self.df[self.df['dr_cr_indicator'] == 'DR']
        
        for _, row in debit_transactions.iterrows():
            description = str(row['description']).lower()
            amount = float(row['debit'])  # Use debit column instead of amount
            categorized = False
            
            for category, keywords in categories.items():
                if category == 'Others':
                    continue
                if any(keyword in description for keyword in keywords):
                    category_totals[category] += amount
                    categorized = True
                    break
            
            if not categorized:
                category_totals['Others'] += amount
        
        # Remove categories with zero amount
        return {k: v for k, v in category_totals.items() if v > 0}
    
    def get_most_expense_category(self) -> Dict[str, Any]:
        """Get the category with highest expenses."""
        categories = self.categorize_expenses()
        if not categories:
            return {'category': 'None', 'amount': 0}
        
        max_category = max(categories.items(), key=lambda x: x[1])
        return {
            'category': max_category[0],
            'amount': max_category[1],
            'percentage': (max_category[1] / sum(categories.values())) * 100
        }
    
    def get_monthly_surplus(self) -> Dict[str, Dict[str, float]]:
        """Calculate monthly surplus (income - expenses)."""
        monthly_analysis = {}
        
        if 'month_year' in self.df.columns:
            for month_year, group in self.df.groupby('month_year'):
                credits = group[group['dr_cr_indicator'] == 'CR']['credit_amount'].sum()
                debits = group[group['dr_cr_indicator'] == 'DR']['debit'].sum()
                surplus = credits - debits
                
                monthly_analysis[str(month_year)] = {
                    'income': float(credits),
                    'expenses': float(debits),
                    'surplus': float(surplus),
                    'savings_rate': (surplus / credits * 100) if credits > 0 else 0
                }
        else:
            # If no date info, calculate total
            credits = self.df[self.df['dr_cr_indicator'] == 'CR']['credit_amount'].sum()
            debits = self.df[self.df['dr_cr_indicator'] == 'DR']['debit'].sum()
            surplus = credits - debits
            
            monthly_analysis['Total'] = {
                'income': float(credits),
                'expenses': float(debits),
                'surplus': float(surplus),
                'savings_rate': (surplus / credits * 100) if credits > 0 else 0
            }
        
        return monthly_analysis
    
    def detect_emi_payments(self) -> List[Dict[str, Any]]:
        """Detect EMI payments based on recurring amounts and descriptions."""
        # Filter potential EMI transactions
        debit_transactions = self.df[self.df['dr_cr_indicator'] == 'DR'].copy()
        
        # Group by similar amounts (within 5% tolerance)
        potential_emis = []
        processed_amounts = set()
        
        for _, row in debit_transactions.iterrows():
            amount = float(row['debit'])
            description = str(row['description']).lower()
            
            # Skip if amount already processed
            if any(abs(amount - pa) / max(pa, 1) <= 0.05 for pa in processed_amounts):
                continue
            
            # Find similar amounts
            similar_transactions = debit_transactions[
                abs(debit_transactions['debit'] - amount) / amount <= 0.05
            ]
            
            # Check if it occurs multiple times and looks like EMI
            if len(similar_transactions) >= 2:
                is_likely_emi = any(keyword in description for keyword in 
                                  ['emi', 'loan', 'installment', 'equated'])
                
                # Check for regular pattern (monthly) if date info is available
                if 'date' in self.df.columns and len(similar_transactions) >= 3:
                    dates = similar_transactions['date'].sort_values()
                    date_diffs = dates.diff().dt.days.dropna()
                    avg_diff = date_diffs.mean()
                    is_regular = 25 <= avg_diff <= 35  # Roughly monthly
                else:
                    is_regular = False
                
                # If it's a significant recurring amount, consider it potential EMI
                if amount >= 1000 and (is_likely_emi or is_regular or len(similar_transactions) >= 3):
                    emi_info = {
                        'amount': amount,
                        'description': row['description'],
                        'frequency': len(similar_transactions),
                        'likely_emi': is_likely_emi,
                        'regular_pattern': is_regular,
                        'confidence': 'High' if (is_likely_emi and is_regular) else 'Medium',
                        'total_paid': amount * len(similar_transactions)
                    }
                    
                    if 'date' in self.df.columns:
                        emi_info.update({
                            'first_date': similar_transactions['date'].min().strftime('%Y-%m-%d'),
                            'last_date': similar_transactions['date'].max().strftime('%Y-%m-%d')
                        })
                    
                    potential_emis.append(emi_info)
                    processed_amounts.add(amount)
        
        return sorted(potential_emis, key=lambda x: x['amount'], reverse=True)
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive financial report."""
        report = {}
        
        # Basic counts
        report['transaction_counts'] = self.get_cr_dr_counts()
        
        # EOD Balance
        report['eod_balance'] = self.get_eod_balance()
        
        # Monthly expenses
        report['monthly_expenses'] = self.get_monthly_expenses()
        
        # Expense categories
        report['expense_categories'] = self.categorize_expenses()
        
        # Most expense category
        report['most_expense_category'] = self.get_most_expense_category()
        
        # Monthly surplus
        report['monthly_surplus'] = self.get_monthly_surplus()
        
        # EMI detection
        report['emi_analysis'] = self.detect_emi_payments()
        
        # Additional insights
        total_income = sum(data['income'] for data in report['monthly_surplus'].values())
        total_expenses = sum(data['expenses'] for data in report['monthly_surplus'].values())
        
        report['summary'] = {
            'total_income': total_income,
            'total_expenses': total_expenses,
            'net_surplus': total_income - total_expenses,
            'overall_savings_rate': (total_income - total_expenses) / total_income * 100 if total_income > 0 else 0,
            'average_monthly_income': total_income / len(report['monthly_surplus']) if report['monthly_surplus'] else 0,
            'average_monthly_expenses': total_expenses / len(report['monthly_surplus']) if report['monthly_surplus'] else 0
        }
        
        return report
    
    def print_report(self):
        """Print a formatted comprehensive report."""
        report = self.generate_comprehensive_report()
        
        print("=" * 80)
        print("                    COMPREHENSIVE FINANCIAL ANALYSIS")
        print("=" * 80)
        
        # Transaction Counts
        print("\n📊 TRANSACTION SUMMARY")
        print("-" * 40)
        counts = report['transaction_counts']
        print(f"Credit Transactions: {counts['credit_transactions']}")
        print(f"Debit Transactions:  {counts['debit_transactions']}")
        print(f"Total Transactions:  {counts['total_transactions']}")
        
        # EOD Balance
        print(f"\n💰 End of Day Balance: ₹{report['eod_balance']:,.2f}")
        
        # Monthly Expenses
        print("\n📈 MONTHLY EXPENSES")
        print("-" * 40)
        for month, amount in report['monthly_expenses'].items():
            print(f"{month}: ₹{amount:,.2f}")
        
        # Expense Categories
        print("\n🏷️  EXPENSE CATEGORIES")
        print("-" * 40)
        for category, amount in sorted(report['expense_categories'].items(), 
                                     key=lambda x: x[1], reverse=True):
            percentage = (amount / sum(report['expense_categories'].values())) * 100
            print(f"{category:<25}: ₹{amount:>10,.2f} ({percentage:5.1f}%)")
        
        # Most Expense Category
        print("\n🎯 HIGHEST EXPENSE CATEGORY")
        print("-" * 40)
        most_exp = report['most_expense_category']
        print(f"Category: {most_exp['category']}")
        print(f"Amount: ₹{most_exp['amount']:,.2f}")
        print(f"Percentage: {most_exp['percentage']:.1f}% of total expenses")
        
        # Monthly Surplus
        print("\n💸 MONTHLY SURPLUS ANALYSIS")
        print("-" * 80)
        headers = ['Month', 'Income', 'Expenses', 'Surplus', 'Savings Rate']
        table_data = []
        for month, data in report['monthly_surplus'].items():
            table_data.append([
                month,
                f"₹{data['income']:,.2f}",
                f"₹{data['expenses']:,.2f}",
                f"₹{data['surplus']:,.2f}",
                f"{data['savings_rate']:.1f}%"
            ])
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # EMI Analysis
        print("\n🏦 EMI/RECURRING PAYMENTS ANALYSIS")
        print("-" * 80)
        emis = report['emi_analysis']
        if emis:
            emi_headers = ['Amount', 'Description', 'Frequency', 'Confidence', 'Total Paid']
            emi_data = []
            total_emi_amount = 0
            for emi in emis:
                emi_data.append([
                    f"₹{emi['amount']:,.2f}",
                    emi['description'][:50] + "..." if len(emi['description']) > 50 else emi['description'],
                    f"{emi['frequency']} times",
                    emi['confidence'],
                    f"₹{emi['total_paid']:,.2f}"
                ])
                total_emi_amount += emi['amount']
            
            print(tabulate(emi_data, headers=emi_headers, tablefmt="grid"))
            print(f"\nTotal Monthly EMI/Recurring: ₹{total_emi_amount:,.2f}")
        else:
            print("No recurring EMI patterns detected.")
        
        # Summary
        print("\n📋 OVERALL SUMMARY")
        print("-" * 40)
        summary = report['summary']
        print(f"Total Income:           ₹{summary['total_income']:,.2f}")
        print(f"Total Expenses:         ₹{summary['total_expenses']:,.2f}")
        print(f"Net Surplus:            ₹{summary['net_surplus']:,.2f}")
        print(f"Overall Savings Rate:   {summary['overall_savings_rate']:.1f}%")
        print(f"Avg Monthly Income:     ₹{summary['average_monthly_income']:,.2f}")
        print(f"Avg Monthly Expenses:   ₹{summary['average_monthly_expenses']:,.2f}")
        
        print("\n" + "=" * 80)
    
    def save_report_to_file(self, filename: str = "financial_analysis_report.txt"):
        """Save the report to a text file."""
        import sys
        from io import StringIO
        
        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()
        
        self.print_report()
        
        # Get the output and restore stdout
        output = buffer.getvalue()
        sys.stdout = old_stdout
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(output)
        
        print(f"Report saved to {filename}")


# Usage example
def analyze_financial_statement(csv_file_path: str):
    """
    Main function to analyze the processed CSV file.
    
    Args:
        csv_file_path: Path to the processed CSV file
    """
    try:
        analyzer = FinancialStatementAnalyzer(csv_file_path)
        analyzer.print_report()
        
        # Optionally save to file
        analyzer.save_report_to_file("financial_analysis_report.txt")
        
        return analyzer.generate_comprehensive_report()
        
    except Exception as e:
        print(f"Error analyzing financial statement: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file_path = "processed_statement.csv"
    
    # Run the analysis
    report = analyze_financial_statement(csv_file_path)
    
    if report:
        print("\nAnalysis completed successfully!")
        print("Check 'financial_analysis_report.txt' for detailed report.")