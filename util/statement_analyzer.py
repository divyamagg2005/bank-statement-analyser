import pandas as pd
from datetime import datetime
import re
from typing import Dict, Tuple, List, Any, Optional
import logging
import numpy as np
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedBankStatementAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer with a pandas DataFrame containing bank statement data.
        
        Args:
            df: DataFrame containing the bank statement data
        """
        try:
            self.original_df = df.copy() if df is not None else pd.DataFrame()
            self.df = None
            self.account_info = {}
            self.opening_balance = None
            self.closing_balance = None
            
            # Handle duplicate column names first
            self._fix_duplicate_columns()
            
            # Process the data with error handling
            self._extract_metadata()
            self._find_transaction_table()
            self._clean_and_process_data()
            
        except Exception as e:
            logger.error(f"Error initializing analyzer: {str(e)}")
            # Set fallback values
            self.original_df = df.copy() if df is not None else pd.DataFrame()
            self.df = df.copy() if df is not None else pd.DataFrame()
            self.account_info = {}
            self.opening_balance = None
            self.closing_balance = None
            
            # Try to fix duplicate columns as a last resort
            try:
                self._fix_duplicate_columns()
            except:
                pass
                
            raise Exception(f"Failed to initialize bank statement analyzer: {str(e)}")
    
    def _fix_duplicate_columns(self) -> None:
        """Fix duplicate column names by adding suffixes."""
        if self.original_df is None or self.original_df.empty:
            return
            
        try:
            # Get column names and handle duplicates
            columns = self.original_df.columns.tolist()
            new_columns = []
            column_counts = {}
            
            for col in columns:
                col_str = str(col)
                if col_str in column_counts:
                    column_counts[col_str] += 1
                    new_col = f"{col_str}_{column_counts[col_str]}"
                else:
                    column_counts[col_str] = 0
                    new_col = col_str
                new_columns.append(new_col)
            
            # Assign new column names
            self.original_df.columns = new_columns
            
        except Exception as e:
            logger.warning(f"Error fixing duplicate columns: {str(e)}")
            # Fallback: create generic column names
            try:
                num_cols = len(self.original_df.columns)
                self.original_df.columns = [f'col_{i}' for i in range(num_cols)]
            except Exception as inner_e:
                logger.error(f"Failed to create fallback column names: {str(inner_e)}")
    
    def _extract_metadata(self) -> None:
        """Extract account information and metadata from the statement."""
        metadata = {}
        
        # Convert all data to string for searching - handle potential errors
        all_text = []
        try:
            for col in self.original_df.columns:
                col_values = self.original_df[col].fillna('').astype(str).tolist()
                all_text.extend(col_values)
        except Exception as e:
            logger.warning(f"Error extracting metadata: {str(e)}")
            # Fallback to simpler extraction
            try:
                all_text = self.original_df.fillna('').astype(str).values.flatten().tolist()
            except Exception:
                all_text = []
        
        if not all_text:
            self.account_info = {}
            return
        
        full_text = ' '.join(str(text) for text in all_text).upper()
        
        # Extract account number
        account_patterns = [
            r'ACCOUNT\s*NO\s*:?\s*(\d+)',
            r'A/C\s*NO\s*:?\s*(\d+)',
            r'ACCOUNT\s*NUMBER\s*:?\s*(\d+)',
            r'(\d{10,18})',  # Generic long number pattern
        ]
        
        for pattern in account_patterns:
            match = re.search(pattern, full_text)
            if match:
                metadata['account_number'] = match.group(1)
                break
        
        # Extract IFSC code
        ifsc_match = re.search(r'IFSC\s*CODE?\s*:?\s*([A-Z]{4}\d{7})', full_text)
        if ifsc_match:
            metadata['ifsc_code'] = ifsc_match.group(1)
        
        # Extract period
        period_patterns = [
            r'FROM\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s*TO\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'PERIOD\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s*TO\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        ]
        
        for pattern in period_patterns:
            match = re.search(pattern, full_text)
            if match:
                metadata['period_from'] = match.group(1)
                metadata['period_to'] = match.group(2)
                break
        
        # Extract customer name
        name_patterns = [
            r'CUSTOMER\s*NAME\s*:?\s*([A-Z\s]+)',
            r'NAME\s*:?\s*([A-Z\s]+)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, full_text)
            if match:
                name = match.group(1).strip()
                if len(name) > 3 and not any(x in name for x in ['AXIS', 'BANK', 'LTD', 'ADDRESS']):
                    metadata['customer_name'] = name
                    break
        
        self.account_info = metadata
    
    def _find_transaction_table(self) -> None:
        """Find the actual transaction table in the CSV."""
        # Common header patterns to identify transaction table
        transaction_headers = [
            'date', 'tran date', 'transaction date', 'value date', 'posting date',
            'particulars', 'description', 'transaction particulars', 'narration',
            'amount', 'debit', 'credit', 'withdrawal', 'deposit', 'dr', 'cr',
            'balance', 'closing balance', 'running balance'
        ]
        
        # Find the row that contains the most transaction-related headers
        best_header_row = -1
        best_score = 0
        
        for idx, row in self.original_df.iterrows():
            try:
                row_text = ' '.join(str(val) for val in row.fillna('').values).lower()
                score = sum(1 for header in transaction_headers if header in row_text)
                
                if score > best_score and score >= 3:  # At least 3 matching headers
                    best_score = score
                    best_header_row = idx
            except Exception as e:
                logger.warning(f"Error processing row {idx} for header detection: {str(e)}")
                continue
        
        if best_header_row == -1:
            # Fallback: look for rows with common patterns
            for idx, row in self.original_df.iterrows():
                try:
                    row_text = ' '.join(str(val) for val in row.fillna('').values).lower()
                    if any(pattern in row_text for pattern in ['date', 'amount', 'balance']):
                        best_header_row = idx
                        break
                except Exception:
                    continue
        
        if best_header_row != -1:
            try:
                # Set this row as headers and take data from next row onwards
                headers = self.original_df.iloc[best_header_row].fillna('').astype(str).tolist()
                data_start_row = best_header_row + 1
                
                # Clean headers
                headers = [str(h).strip().lower() for h in headers if str(h).strip() and str(h).strip().lower() != 'nan']
                
                # Create new dataframe with proper headers
                transaction_data = self.original_df.iloc[data_start_row:].copy()
                
                # Handle column count mismatch
                if len(headers) <= len(transaction_data.columns):
                    # Ensure unique column names
                    unique_headers = []
                    header_counts = {}
                    
                    for header in headers:
                        if header in header_counts:
                            header_counts[header] += 1
                            unique_header = f"{header}_{header_counts[header]}"
                        else:
                            header_counts[header] = 0
                            unique_header = header
                        unique_headers.append(unique_header)
                    
                    # Add generic names for remaining columns
                    remaining_cols = len(transaction_data.columns) - len(unique_headers)
                    for i in range(remaining_cols):
                        unique_headers.append(f'col_{len(unique_headers) + i}')
                    
                    transaction_data.columns = unique_headers
                else:
                    # More headers than columns - truncate headers
                    transaction_data.columns = headers[:len(transaction_data.columns)]
                
                self.df = transaction_data
                
            except Exception as e:
                logger.warning(f"Error setting up transaction table: {str(e)}")
                # Use original dataframe as fallback
                self.df = self.original_df.copy()
                # Fix column names
                try:
                    self.df.columns = [str(col).strip().lower() for col in self.df.columns]
                except:
                    self.df.columns = [f'col_{i}' for i in range(len(self.df.columns))]
        else:
            # Use original dataframe if no clear header found
            self.df = self.original_df.copy()
            try:
                self.df.columns = [str(col).strip().lower() for col in self.df.columns]
            except:
                self.df.columns = [f'col_{i}' for i in range(len(self.df.columns))]
    
    def _clean_and_process_data(self) -> None:
        """Clean and process the transaction data."""
        if self.df is None or self.df.empty:
            return
        
        try:
            # Remove empty rows
            self.df = self.df.dropna(how='all')
            
            # Standardize column names
            self.df.columns = [self._standardize_column_name(col) for col in self.df.columns]
            
            # Remove rows that contain metadata or summaries
            self.df = self._remove_non_transaction_rows()
            
            # Identify and map columns
            self._map_columns()
            
            # Clean data types
            self._clean_data_types()
            
            # Extract opening and closing balance
            self._extract_balances()
            
            # Process transactions
            self._process_transactions()
            
        except Exception as e:
            logger.error(f"Error in clean and process data: {str(e)}")
            # Minimal processing as fallback
            if self.df is not None:
                self.df = self.df.dropna(how='all')
    
    def _standardize_column_name(self, col_name: str) -> str:
        """Standardize column names to common format."""
        col_name = str(col_name).strip().lower()
        
        # Mapping dictionary for common variations
        mappings = {
            # Date columns
            'tran date': 'date',
            'transaction date': 'date',
            'value date': 'date',
            'posting date': 'date',
            'txn date': 'date',
            
            # Description columns
            'particulars': 'description',
            'transaction particulars': 'description',
            'narration': 'description',
            'transaction details': 'description',
            'details': 'description',
            
            # Amount columns
            'amount(inr)': 'amount',
            'amount (inr)': 'amount',
            'withdrawal amt': 'debit',
            'deposit amt': 'credit',
            'withdrawal': 'debit',
            'deposit': 'credit',
            'dr': 'debit',
            'cr': 'credit',
            'dr/cr': 'dr_cr_indicator',
            
            # Balance columns
            'balance(inr)': 'balance',
            'balance (inr)': 'balance',
            'closing balance': 'balance',
            'running balance': 'balance',
            'balance amt': 'balance',
            
            # Other columns
            'chq no': 'cheque_no',
            'cheque no': 'cheque_no',
            'ref no': 'reference_no',
            'reference no': 'reference_no',
            'branch name': 'branch',
        }
        
        return mappings.get(col_name, col_name)
    
    def _remove_non_transaction_rows(self) -> pd.DataFrame:
        """Remove rows that are not actual transactions."""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        df_clean = self.df.copy()
        
        # Patterns that indicate non-transaction rows
        exclude_patterns = [
            r'opening balance',
            r'closing balance',
            r'total.*dr.*cr',
            r'transaction total',
            r'legends?',
            r'registered office',
            r'branch address',
            r'in compliance with',
            r'system generated',
            r'this is a system',
            r'^$',  # Empty rows
            r'^\s*$',  # Whitespace only
        ]
        
        # Check each row
        rows_to_keep = []
        for idx, row in df_clean.iterrows():
            try:
                # Safely convert row to string
                row_text = ' '.join(str(val) for val in row.fillna('').values).lower()
                
                # Skip if matches exclude patterns
                should_exclude = any(re.search(pattern, row_text) for pattern in exclude_patterns)
                
                # Skip if row has too few non-null values
                non_null_count = row.count()
                if non_null_count < 2:
                    should_exclude = True
                
                if not should_exclude:
                    rows_to_keep.append(idx)
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {str(e)}")
                # If there's an error, skip this row
                continue
        
        return df_clean.loc[rows_to_keep].reset_index(drop=True) if rows_to_keep else df_clean
    
    def _map_columns(self) -> None:
        """Map columns to standard names based on content analysis."""
        if self.df is None or self.df.empty:
            return
            
        # Analyze column content to identify their purpose
        column_mapping = {}
        
        for col in self.df.columns:
            try:
                # Safely convert to string and get lowercase values
                col_data = self.df[col].fillna('').astype(str)
                
                # Check for date patterns
                date_patterns = [
                    r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
                    r'\d{2,4}[-/]\d{1,2}[-/]\d{1,2}',
                ]
                
                date_matches = 0
                for text in col_data:
                    text_str = str(text).strip()
                    if text_str and text_str.lower() not in ['nan', 'none', '']:
                        for pattern in date_patterns:
                            if re.search(pattern, text_str):
                                date_matches += 1
                                break
                
                if date_matches > len(self.df) * 0.5:  # More than 50% are dates
                    if 'date' not in column_mapping.values():
                        column_mapping[col] = 'date'
                    continue
                
                # Check for amount patterns
                amount_pattern = r'[\d,]+\.?\d*'
                amount_matches = 0
                
                for text in col_data:
                    text_str = str(text).strip()
                    if text_str and text_str.lower() not in ['nan', 'none', '']:
                        if re.search(amount_pattern, text_str):
                            amount_matches += 1
                
                if amount_matches > len(self.df) * 0.3:  # More than 30% are amounts
                    col_lower = col.lower()
                    # Determine if it's debit, credit, or balance
                    if any(indicator in col_lower for indicator in ['debit', 'withdrawal', 'dr']):
                        column_mapping[col] = 'debit'
                    elif any(indicator in col_lower for indicator in ['credit', 'deposit', 'cr']):
                        column_mapping[col] = 'credit'
                    elif 'balance' in col_lower:
                        column_mapping[col] = 'balance'
                    elif 'amount' in col_lower and 'amount' not in column_mapping.values():
                        column_mapping[col] = 'amount'
            
            except Exception as e:
                logger.warning(f"Error analyzing column '{col}': {str(e)}")
                continue
        
        # Apply mapping
        try:
            self.df = self.df.rename(columns=column_mapping)
        except Exception as e:
            logger.warning(f"Error applying column mapping: {str(e)}")
    
    def _clean_data_types(self) -> None:
        """Clean and convert data types."""
        if self.df is None or self.df.empty:
            return
            
        try:
            # Clean date column
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce', dayfirst=True)
            
            # Clean amount columns
            amount_columns = ['debit', 'credit', 'balance', 'amount']
            for col in amount_columns:
                if col in self.df.columns:
                    self.df[col] = self._clean_amount_column(self.df[col])
            
            # Handle DR/CR indicator column
            if 'dr_cr_indicator' in self.df.columns and 'amount' in self.df.columns:
                self._split_amount_by_indicator()
                
        except Exception as e:
            logger.warning(f"Error cleaning data types: {str(e)}")
    
    def _clean_amount_column(self, series: pd.Series) -> pd.Series:
        """Clean amount column by removing non-numeric characters."""
        def clean_amount(value):
            if pd.isna(value) or value == '':
                return 0.0
            
            # Convert to string and clean
            value_str = str(value).strip()
            
            # Remove common non-numeric characters but keep decimal point and minus
            cleaned = re.sub(r'[^\d.-]', '', value_str)
            
            # Handle empty string after cleaning
            if not cleaned or cleaned == '-' or cleaned == '.':
                return 0.0
            
            try:
                return float(cleaned)
            except ValueError:
                return 0.0
        
        try:
            return series.apply(clean_amount)
        except Exception as e:
            logger.warning(f"Error cleaning amount column: {str(e)}")
            return series.fillna(0.0)
    
    def _split_amount_by_indicator(self) -> None:
        """Split amount column based on DR/CR indicator."""
        if 'dr_cr_indicator' not in self.df.columns or 'amount' not in self.df.columns:
            return
        
        try:
            # Create debit and credit columns
            self.df['debit'] = self.df.apply(
                lambda row: row['amount'] if str(row['dr_cr_indicator']).upper() in ['DR', 'DEBIT'] 
                else 0.0, axis=1
            )
            
            self.df['credit'] = self.df.apply(
                lambda row: row['amount'] if str(row['dr_cr_indicator']).upper() in ['CR', 'CREDIT'] 
                else 0.0, axis=1
            )
        except Exception as e:
            logger.warning(f"Error splitting amount by indicator: {str(e)}")
    
    def _extract_balances(self) -> None:
        """Extract opening and closing balances."""
        # Look for opening balance in original data
        try:
            all_text = ' '.join(self.original_df.fillna('').astype(str).values.flatten()).upper()
            
            opening_match = re.search(r'OPENING\s*BALANCE[:\s]*([0-9,.-]+)', all_text)
            if opening_match:
                self.opening_balance = self._clean_amount_value(opening_match.group(1))
            
            closing_match = re.search(r'CLOSING\s*BALANCE[:\s]*([0-9,.-]+)', all_text)
            if closing_match:
                self.closing_balance = self._clean_amount_value(closing_match.group(1))
        except Exception as e:
            logger.warning(f"Error extracting balances from text: {str(e)}")
        
        # If not found in text, try to get from balance column
        if self.df is not None and 'balance' in self.df.columns and not self.df['balance'].isna().all():
            try:
                balance_col = self.df['balance'].dropna()
                if not balance_col.empty:
                    if self.opening_balance is None:
                        self.opening_balance = balance_col.iloc[0]
                    if self.closing_balance is None:
                        self.closing_balance = balance_col.iloc[-1]
            except Exception as e:
                logger.warning(f"Error extracting balances from balance column: {str(e)}")
    
    def _clean_amount_value(self, value: str) -> float:
        """Clean a single amount value."""
        if not value:
            return 0.0
        
        # Remove commas and other non-numeric characters except decimal point and minus
        cleaned = re.sub(r'[^\d.-]', '', str(value))
        
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    
    def _process_transactions(self) -> None:
        """Process transactions to identify transaction type and categorize them."""
        if self.df is None or self.df.empty:
            return
        
        try:
            # Create transaction type column
            if 'debit' in self.df.columns and 'credit' in self.df.columns:
                self.df['type'] = self.df.apply(
                    lambda row: 'debit' if (pd.notna(row['debit']) and row['debit'] > 0) 
                               else 'credit' if (pd.notna(row['credit']) and row['credit'] > 0) 
                               else 'unknown', 
                    axis=1
                )
            elif 'amount' in self.df.columns and 'dr_cr_indicator' in self.df.columns:
                self.df['type'] = self.df['dr_cr_indicator'].apply(
                    lambda x: 'debit' if str(x).upper() in ['DR', 'DEBIT'] 
                             else 'credit' if str(x).upper() in ['CR', 'CREDIT'] 
                             else 'unknown'
                )
            
            # Extract month and year for monthly analysis
            if 'date' in self.df.columns:
                self.df['month_year'] = self.df['date'].dt.to_period('M')
            
            # Add transaction amount column (unified amount regardless of type)
            if 'debit' in self.df.columns and 'credit' in self.df.columns:
                self.df['transaction_amount'] = self.df['debit'].fillna(0) + self.df['credit'].fillna(0)
            elif 'amount' in self.df.columns:
                self.df['transaction_amount'] = self.df['amount']
                
        except Exception as e:
            logger.warning(f"Error processing transactions: {str(e)}")
    
    def get_transaction_summary(self) -> Dict[str, Any]:
        """Get a summary of all transactions."""
        if self.df is None or self.df.empty:
            return {'error': 'No transaction data found'}
        
        try:
            summary = {
                'total_transactions': len(self.df),
                'account_info': self.account_info,
            }
            
            if 'date' in self.df.columns:
                date_series = self.df['date'].dropna()
                if not date_series.empty:
                    summary.update({
                        'start_date': date_series.min().strftime('%Y-%m-%d'),
                        'end_date': date_series.max().strftime('%Y-%m-%d'),
                    })
            
            if 'month_year' in self.df.columns:
                summary['unique_months'] = len(self.df['month_year'].dropna().unique())
            
            # Count credit and debit transactions
            if 'type' in self.df.columns:
                type_counts = self.df['type'].value_counts().to_dict()
                summary.update({
                    'credit_count': type_counts.get('credit', 0),
                    'debit_count': type_counts.get('debit', 0),
                })
            
            # Calculate total credit and debit amounts
            if 'credit' in self.df.columns:
                summary['total_credit'] = self.df['credit'].fillna(0).sum()
            if 'debit' in self.df.columns:
                summary['total_debit'] = self.df['debit'].fillna(0).sum()
            
            # Add opening and closing balance
            summary['opening_balance'] = self.opening_balance or 0
            summary['closing_balance'] = self.closing_balance or 0
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting transaction summary: {str(e)}")
            return {'error': f'Failed to generate transaction summary: {str(e)}'}
    
    def get_monthly_summary(self) -> Dict[str, Any]:
        """Get a summary of transactions by month."""
        if self.df is None or 'month_year' not in self.df.columns:
            return {}
        
        try:
            monthly_data = {}
            
            for month_year, month_df in self.df.groupby('month_year'):
                month_str = str(month_year)
                
                credit_sum = month_df['credit'].fillna(0).sum() if 'credit' in month_df.columns else 0
                debit_sum = month_df['debit'].fillna(0).sum() if 'debit' in month_df.columns else 0
                
                monthly_data[month_str] = {
                    'credit_count': (month_df['type'] == 'credit').sum() if 'type' in month_df.columns else 0,
                    'debit_count': (month_df['type'] == 'debit').sum() if 'type' in month_df.columns else 0,
                    'total_credit': credit_sum,
                    'total_debit': debit_sum,
                    'net_flow': credit_sum - debit_sum,
                }
                
                # Add monthly balance info if available
                if 'balance' in month_df.columns and 'date' in month_df.columns:
                    sorted_month = month_df.sort_values('date')
                    balance_series = sorted_month['balance'].dropna()
                    if not balance_series.empty:
                        monthly_data[month_str]['opening_balance'] = balance_series.iloc[0]
                        monthly_data[month_str]['closing_balance'] = balance_series.iloc[-1]
            
            return monthly_data
            
        except Exception as e:
            logger.error(f"Error getting monthly summary: {str(e)}")
            return {}
    
    def detect_emis(self, min_amount: float = 1000, min_occurrences: int = 3, 
                   amount_tolerance: float = 0.1) -> List[Dict]:
        """
        Detect potential EMI transactions based on recurring amounts.
        """
        if self.df is None or 'debit' not in self.df.columns:
            return []
        
        try:
            # Get description column name
            desc_col = None
            for col in ['description', 'particulars', 'narration', 'transaction details']:
                if col in self.df.columns:
                    desc_col = col
                    break
            
            if desc_col is None:
                return []
            
            # Filter potential EMI transactions
            potential_emis = self.df[
                (self.df['debit'] >= min_amount) & 
                (self.df['debit'].notna()) &
                (self.df['type'] == 'debit')
            ].copy()
            
            if potential_emis.empty:
                return []
            
            # Group similar amounts (within tolerance)
            emis = []
            processed_amounts = set()
            
            for _, row in potential_emis.iterrows():
                amount = row['debit']
                
                # Skip if already processed
                if any(abs(amount - pa) / pa <= amount_tolerance for pa in processed_amounts if pa > 0):
                    continue
                
                # Find similar amounts
                similar_amounts = potential_emis[
                    abs(potential_emis['debit'] - amount) / amount <= amount_tolerance
                ]
                
                if len(similar_amounts) >= min_occurrences:
                    # Check if description suggests EMI
                    desc_text = str(row[desc_col]).lower()
                    is_likely_emi = any(keyword in desc_text for keyword in 
                                      ['emi', 'loan', 'installment', 'repayment', 'equated'])
                    
                    # Also check for regular monthly pattern
                    if 'date' in similar_amounts.columns:
                        dates = similar_amounts['date'].dropna()
                        if len(dates) >= 2:
                            # Check if transactions are roughly monthly
                            date_diffs = dates.sort_values().diff().dt.days.dropna()
                            avg_diff = date_diffs.mean()
                            is_monthly = 25 <= avg_diff <= 35  # Roughly monthly
                        else:
                            is_monthly = False
                    else:
                        is_monthly = False
                    
                    if is_likely_emi or is_monthly:
                        emi = {
                            'amount': amount,
                            'description': row[desc_col],
                            'frequency': 'monthly' if is_monthly else 'unknown',
                            'occurrences': len(similar_amounts),
                            'confidence': 'high' if (is_likely_emi and is_monthly) else 'medium',
                        }
                        
                        if 'date' in similar_amounts.columns:
                            date_series = similar_amounts['date'].dropna()
                            if not date_series.empty:
                                emi['first_date'] = date_series.min().strftime('%Y-%m-%d')
                                emi['last_date'] = date_series.max().strftime('%Y-%m-%d')
                        
                        emis.append(emi)
                        processed_amounts.add(amount)
            
            return sorted(emis, key=lambda x: x['amount'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error detecting EMIs: {str(e)}")
            return []
    
    def get_expense_categories(self) -> Dict[str, float]:
        """Categorize expenses based on description keywords with enhanced patterns."""
        if self.df is None:
            return {}
        
        try:
            desc_col = None
            for col in ['description', 'particulars', 'narration', 'transaction details']:
                if col in self.df.columns:
                    desc_col = col
                    break
            
            if desc_col is None or 'debit' not in self.df.columns:
                return {}
            
            # Enhanced expense categories with more keywords
            categories = {
                'Food & Dining': [
                    'restaurant', 'cafe', 'food', 'dining', 'zomato', 'swiggy', 'eatsure',
                    'mcdonalds', 'dominos', 'pizza', 'kfc', 'subway', 'starbucks',
                    'hotel', 'dhaba', 'canteen', 'mess'
                ],
                'Shopping': [
                    'amazon', 'flipkart', 'myntra', 'ajio', 'shopping', 'store', 'retail',
                    'mall', 'market', 'bazaar', 'nykaa', 'limeroad', 'snapdeal',
                    'clothing', 'apparel', 'fashion'
                ],
                'Transportation': [
                    'uber', 'ola', 'rapido', 'auto', 'taxi', 'cab',
                    'petrol', 'fuel', 'diesel', 'gas station', 'hp', 'bharat petroleum',
                    'metro', 'railway', 'train', 'bus', 'flight', 'airlines',
                    'parking', 'toll', 'transport'
                ],
                'Bills & Utilities': [
                    'electricity', 'water', 'gas', 'bill', 'utility',
                    'mobile', 'recharge', 'phone', 'broadband', 'internet', 'wifi',
                    'dth', 'cable', 'tv', 'airtel', 'jio', 'bsnl', 'vodafone',
                    'electricity board', 'mseb', 'bescom'
                ],
                'Entertainment': [
                    'netflix', 'prime', 'hotstar', 'spotify', 'youtube', 'disney',
                    'cinema', 'movie', 'theatre', 'multiplex', 'pvr', 'inox',
                    'gaming', 'games', 'entertainment', 'amusement', 'club'
                ],
                'EMI & Loans': [
                    'emi', 'loan', 'repayment', 'installment', 'equated monthly',
                    'home loan', 'car loan', 'personal loan', 'education loan',
                    'credit card', 'hdfc', 'icici', 'sbi', 'axis', 'kotak'
                ],
                'Groceries': [
                    'grocery', 'supermarket', 'dmart', 'bigbasket', 'grofers', 'fresh',
                    'vegetables', 'fruits', 'milk', 'bread', 'rice', 'dal',
                    'reliance fresh', 'more', 'spencer', 'foodworld', 'market'
                ],
                'Travel': [
                    'flight', 'hotel', 'booking', 'makemytrip', 'goibibo', 'yatra',
                    'cleartrip', 'expedia', 'oyo', 'airbnb', 'resort',
                    'travel', 'tour', 'trip', 'vacation', 'holiday'
                ],
                'Health': [
                    'hospital', 'clinic', 'pharmacy', 'medical', 'medicine', 'drug',
                    'doctor', 'lab', 'diagnostic', 'pathology', 'apollo', 'fortis',
                    'health', 'dental', 'eye care', 'checkup', 'consultation'
                ],
                'Education': [
                    'school', 'college', 'university', 'course', 'education', 'tuition',
                    'fees', 'admission', 'exam', 'coaching', 'training', 'institute',
                    'academy', 'learning', 'study', 'books', 'stationery'
                ],
                'Investments': [
                    'mutual fund', 'mf', 'sip', 'stocks', 'shares', 'equity',
                    'investment', 'gold', 'silver', 'bond', 'fd', 'fixed deposit',
                    'recurring deposit', 'rd', 'ppf', 'nsc', 'elss'
                ],
                'Cash Withdrawal': [
                    'atm', 'cash', 'withdrawal', 'cwdr', 'cash out'
                ],
                'Transfers': [
                    'transfer', 'neft', 'rtgs', 'imps', 'upi', 'paytm', 'phonepe',
                    'gpay', 'amazon pay', 'mobikwik', 'freecharge'
                ]
            }
            
            # Initialize category amounts
            category_totals = {category: 0.0 for category in categories}
            uncategorized = 0.0
            
            # Process each debit transaction
            debit_transactions = self.df[self.df['type'] == 'debit'] if 'type' in self.df.columns else self.df[self.df['debit'] > 0]
            
            for _, row in debit_transactions.iterrows():
                description = str(row[desc_col]).lower()
                amount = row['debit'] if 'debit' in row else 0
                
                if amount <= 0:
                    continue
                
                categorized = False
                
                # Check each category
                for category, keywords in categories.items():
                    if any(keyword in description for keyword in keywords):
                        category_totals[category] += amount
                        categorized = True
                        break
                
                if not categorized:
                    uncategorized += amount
            
            # Add uncategorized amount
            if uncategorized > 0:
                category_totals['Uncategorized'] = uncategorized
            
            # Remove categories with zero amount
            category_totals = {k: v for k, v in category_totals.items() if v > 0}
            
            return dict(sorted(category_totals.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error categorizing expenses: {str(e)}")
            return {}
    
    def get_spending_patterns(self) -> Dict[str, Any]:
        """Analyze spending patterns and trends."""
        if self.df is None:
            return {}
        
        try:
            patterns = {}
            
            # Monthly spending trends
            if 'month_year' in self.df.columns and 'debit' in self.df.columns:
                monthly_spending = self.df.groupby('month_year')['debit'].sum().fillna(0)
                patterns['monthly_spending'] = monthly_spending.to_dict()
                
                # Calculate average monthly spending
                patterns['average_monthly_spending'] = monthly_spending.mean()
                
                # Find highest and lowest spending months
                if not monthly_spending.empty:
                    patterns['highest_spending_month'] = {
                        'month': str(monthly_spending.idxmax()),
                        'amount': monthly_spending.max()
                    }
                    patterns['lowest_spending_month'] = {
                        'month': str(monthly_spending.idxmin()),
                        'amount': monthly_spending.min()
                    }
            
            # Day of week patterns (if date available)
            if 'date' in self.df.columns and 'debit' in self.df.columns:
                self.df['day_of_week'] = self.df['date'].dt.day_name()
                day_spending = self.df.groupby('day_of_week')['debit'].sum().fillna(0)
                patterns['day_of_week_spending'] = day_spending.to_dict()
            
            # Large transactions (top 10% by amount)
            if 'debit' in self.df.columns:
                debit_amounts = self.df['debit'].dropna()
                if not debit_amounts.empty:
                    threshold = debit_amounts.quantile(0.9)  # Top 10%
                    large_transactions = self.df[self.df['debit'] >= threshold]
                    patterns['large_transactions'] = {
                        'threshold': threshold,
                        'count': len(large_transactions),
                        'total_amount': large_transactions['debit'].sum()
                    }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing spending patterns: {str(e)}")
            return {}
    
    def detect_unusual_transactions(self, std_multiplier: float = 2.5) -> List[Dict]:
        """Detect unusual transactions based on statistical analysis."""
        if self.df is None or 'debit' not in self.df.columns:
            return []
        
        try:
            debit_amounts = self.df['debit'].dropna()
            if debit_amounts.empty:
                return []
            
            # Calculate statistical measures
            mean_amount = debit_amounts.mean()
            std_amount = debit_amounts.std()
            threshold = mean_amount + (std_multiplier * std_amount)
            
            # Find unusual transactions
            unusual = self.df[self.df['debit'] > threshold].copy()
            
            unusual_transactions = []
            desc_col = None
            for col in ['description', 'particulars', 'narration', 'transaction details']:
                if col in self.df.columns:
                    desc_col = col
                    break
            
            for _, row in unusual.iterrows():
                transaction = {
                    'amount': row['debit'],
                    'deviation_factor': (row['debit'] - mean_amount) / std_amount if std_amount > 0 else 0,
                }
                
                if desc_col:
                    transaction['description'] = row[desc_col]
                
                if 'date' in row and pd.notna(row['date']):
                    transaction['date'] = row['date'].strftime('%Y-%m-%d')
                
                unusual_transactions.append(transaction)
            
            return sorted(unusual_transactions, key=lambda x: x['amount'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error detecting unusual transactions: {str(e)}")
            return []
    
    def get_income_analysis(self) -> Dict[str, Any]:
        """Analyze income patterns and sources."""
        if self.df is None or 'credit' not in self.df.columns:
            return {}
        
        try:
            income_analysis = {}
            
            # Total income
            total_income = self.df['credit'].fillna(0).sum()
            income_analysis['total_income'] = total_income
            
            # Monthly income
            if 'month_year' in self.df.columns:
                monthly_income = self.df.groupby('month_year')['credit'].sum().fillna(0)
                income_analysis['monthly_income'] = monthly_income.to_dict()
                income_analysis['average_monthly_income'] = monthly_income.mean()
            
            # Income sources categorization
            desc_col = None
            for col in ['description', 'particulars', 'narration', 'transaction details']:
                if col in self.df.columns:
                    desc_col = col
                    break
            
            if desc_col:
                income_sources = {
                    'Salary': ['salary', 'sal', 'payroll', 'wages', 'stipend'],
                    'Business Income': ['business', 'revenue', 'sales', 'invoice', 'payment'],
                    'Interest': ['interest', 'int', 'fd interest', 'sb interest'],
                    'Dividends': ['dividend', 'div'],
                    'Refunds': ['refund', 'refund', 'cashback', 'reversal'],
                    'Transfers': ['transfer', 'neft', 'rtgs', 'imps', 'upi'],
                    'Other Income': []
                }
                
                source_totals = {source: 0.0 for source in income_sources}
                
                credit_transactions = self.df[self.df['credit'] > 0]
                
                for _, row in credit_transactions.iterrows():
                    description = str(row[desc_col]).lower()
                    amount = row['credit']
                    categorized = False
                    
                    for source, keywords in income_sources.items():
                        if source == 'Other Income':
                            continue
                        if any(keyword in description for keyword in keywords):
                            source_totals[source] += amount
                            categorized = True
                            break
                    
                    if not categorized:
                        source_totals['Other Income'] += amount
                
                # Remove zero amounts
                income_analysis['income_sources'] = {k: v for k, v in source_totals.items() if v > 0}
            
            return income_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing income: {str(e)}")
            return {}
    
    def get_comprehensive_insights(self) -> Dict[str, Any]:
        """Generate comprehensive insights from the bank statement."""
        try:
            # Get all analysis components
            transaction_summary = self.get_transaction_summary()
            monthly_summary = self.get_monthly_summary()
            emis = self.detect_emis()
            expense_categories = self.get_expense_categories()
            spending_patterns = self.get_spending_patterns()
            unusual_transactions = self.detect_unusual_transactions()
            income_analysis = self.get_income_analysis()
            
            # Calculate financial health metrics
            financial_metrics = {}
            
            if 'total_credit' in transaction_summary and 'total_debit' in transaction_summary:
                total_income = transaction_summary['total_credit']
                total_expenses = transaction_summary['total_debit']
                
                financial_metrics['net_cashflow'] = total_income - total_expenses
                financial_metrics['expense_ratio'] = (total_expenses / total_income * 100) if total_income > 0 else 0
                financial_metrics['savings_rate'] = ((total_income - total_expenses) / total_income * 100) if total_income > 0 else 0
            
            # EMI to income ratio
            total_emi = sum(emi['amount'] for emi in emis)
            if 'average_monthly_income' in income_analysis and income_analysis['average_monthly_income'] > 0:
                financial_metrics['emi_to_income_ratio'] = (total_emi / income_analysis['average_monthly_income'] * 100)
            
            # Top spending categories
            top_expenses = dict(list(expense_categories.items())[:5]) if expense_categories else {}
            
            # Monthly surplus/deficit analysis
            monthly_surplus = {}
            for month, data in monthly_summary.items():
                income = data.get('total_credit', 0)
                expenses = data.get('total_debit', 0)
                monthly_surplus[month] = {
                    'income': income,
                    'expenses': expenses,
                    'surplus': income - expenses,
                    'opening_balance': data.get('opening_balance', 0),
                    'closing_balance': data.get('closing_balance', 0)
                }
            
            # Compile comprehensive insights
            insights = {
                'account_summary': {
                    'account_info': self.account_info,
                    'opening_balance': self.opening_balance,
                    'closing_balance': self.closing_balance,
                },
                'transaction_summary': transaction_summary,
                'financial_metrics': financial_metrics,
                'monthly_analysis': {
                    'monthly_summary': monthly_summary,
                    'monthly_surplus': monthly_surplus,
                },
                'spending_analysis': {
                    'expense_categories': expense_categories,
                    'top_expenses': top_expenses,
                    'spending_patterns': spending_patterns,
                },
                'income_analysis': income_analysis,
                'emi_analysis': {
                    'emis': emis,
                    'total_emi': total_emi,
                    'emi_count': len(emis)
                },
                'alerts': {
                    'unusual_transactions': unusual_transactions[:10],  # Top 10
                    'unusual_count': len(unusual_transactions)
                }
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return {'error': f'Failed to generate insights: {str(e)}'}
    
    def export_processed_data(self) -> pd.DataFrame:
        """Export the cleaned and processed transaction data."""
        return self.df.copy() if self.df is not None else pd.DataFrame()
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Generate a report on data quality and completeness."""
        if self.df is None:
            return {'error': 'No data to analyze'}
        
        try:
            report = {
                'total_rows': len(self.df),
                'columns_found': list(self.df.columns),
                'missing_data': {},
                'data_types': {},
                'quality_score': 0
            }
            
            # Check for missing data
            for col in self.df.columns:
                missing_count = self.df[col].isna().sum()
                missing_pct = (missing_count / len(self.df)) * 100
                report['missing_data'][col] = {
                    'count': missing_count,
                    'percentage': missing_pct
                }
            
            # Check data types
            for col in self.df.columns:
                report['data_types'][col] = str(self.df[col].dtype)
            
            # Calculate quality score
            essential_columns = ['date', 'description', 'amount', 'debit', 'credit']
            found_essential = sum(1 for col in essential_columns if col in self.df.columns)
            
            # Score based on essential columns found and data completeness
            column_score = (found_essential / len(essential_columns)) * 50
            
            # Data completeness score
            avg_completeness = 100 - (sum(info['percentage'] for info in report['missing_data'].values()) / len(report['missing_data']))
            completeness_score = (avg_completeness / 100) * 50
            
            report['quality_score'] = column_score + completeness_score
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating data quality report: {str(e)}")
            return {'error': f'Failed to generate data quality report: {str(e)}'}


# Example usage function
def analyze_bank_statement(csv_file_path: str) -> Dict[str, Any]:
    """
    Main function to analyze a bank statement CSV file.
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        Dictionary containing comprehensive analysis
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        
        # Initialize analyzer
        analyzer = EnhancedBankStatementAnalyzer(df)
        
        # Generate comprehensive insights
        insights = analyzer.get_comprehensive_insights()
        
        # Add data quality report
        insights['data_quality'] = analyzer.get_data_quality_report()
        
        return insights
        
    except Exception as e:
        logger.error(f"Error analyzing bank statement: {str(e)}")
        return {'error': f'Failed to analyze bank statement: {str(e)}'}


# Utility function for direct DataFrame analysis
def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze a bank statement DataFrame directly.
    
    Args:
        df: Pandas DataFrame containing bank statement data
        
    Returns:
        Dictionary containing comprehensive analysis
    """
    try:
        analyzer = EnhancedBankStatementAnalyzer(df)
        insights = analyzer.get_comprehensive_insights()
        insights['data_quality'] = analyzer.get_data_quality_report()
        return insights
    except Exception as e:
        logger.error(f"Error analyzing DataFrame: {str(e)}")
        return {'error': f'Failed to analyze DataFrame: {str(e)}'}