import streamlit as st
import os
import pandas as pd
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any
import json

# Add the parent directory to the path so we can import from util
sys.path.append(str(Path(__file__).parent.parent))
from util.pdf_parser import extract_full_table_to_csv
from util.statement_analyzer import EnhancedBankStatementAnalyzer
from util.finance_insights import FinancialStatementAnalyzer

# Set page config
st.set_page_config(
    page_title="Bank Statement Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #2d3748;
        color: #f7fafc;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        text-align: center;
        padding: 1.2rem 1rem;
        border-radius: 0.5rem;
        background-color: #2d3748;
        color: #f7fafc;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
    }
    
    .metric-card h3 {
        color: #fff;
        margin: 0.5rem 0;
        font-size: 1.5rem;
    }
    
    .metric-card div:first-child {
        color: #a0aec0;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .positive {
        color: #68d391;
        font-weight: bold;
    }
    .negative {
        color: #fc8181;
        font-weight: bold;
    }
    .section-title {
        color: #1f77b4;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

def format_currency(amount):
    """Format amount as currency."""
    if pd.isna(amount):
        return "₹0.00"
    return f"₹{amount:,.2f}"

def display_financial_insights(csv_file_path: str):
    """Display financial insights from the processed CSV file."""
    try:
        with st.spinner('Generating financial insights...'):
            # Initialize the financial analyzer
            analyzer = FinancialStatementAnalyzer(csv_file_path)
            
            # Generate the comprehensive report
            report = analyzer.generate_comprehensive_report()
            
            # Display the report in a structured way
            st.markdown("## 💰 Financial Insights")
            
            # 1. Transaction Summary
            with st.expander("📊 Transaction Summary"):
                counts = report['transaction_counts']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Transactions", counts['total_transactions'])
                with col2:
                    st.metric("Credit Transactions", counts['credit_transactions'])
                with col3:
                    st.metric("Debit Transactions", counts['debit_transactions'])
            
            # 2. Balance Overview
            with st.expander("💵 Balance Overview"):
                st.metric("End of Day Balance", f"₹{report['eod_balance']:,.2f}")
            
            # 3. Monthly Expenses
            with st.expander("📈 Monthly Expenses"):
                months = list(report['monthly_expenses'].keys())
                expenses = list(report['monthly_expenses'].values())
                
                fig = px.bar(
                    x=months,
                    y=expenses,
                    labels={'x': 'Month', 'y': 'Amount (₹)'},
                    title='Monthly Expenses',
                    color=expenses,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 4. Expense Categories
            with st.expander("🏷️ Expense Categories"):
                categories = list(report['expense_categories'].keys())
                amounts = list(report['expense_categories'].values())
                
                if categories:
                    fig = px.pie(
                        names=categories,
                        values=amounts,
                        title='Expense Distribution by Category',
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No expense categories found.")
            
            # 5. Monthly Surplus Analysis
            with st.expander("💸 Monthly Surplus Analysis"):
                months = list(report['monthly_surplus'].keys())
                income = [data['income'] for data in report['monthly_surplus'].values()]
                expenses = [data['expenses'] for data in report['monthly_surplus'].values()]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=months,
                    y=income,
                    name='Income',
                    marker_color='#2ecc71'
                ))
                fig.add_trace(go.Bar(
                    x=months,
                    y=expenses,
                    name='Expenses',
                    marker_color='#e74c3c'
                ))
                
                fig.update_layout(
                    barmode='group',
                    title='Monthly Income vs Expenses',
                    xaxis_title='Month',
                    yaxis_title='Amount (₹)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 6. Recurring Payments (EMIs)
            if report['emi_analysis']:
                with st.expander("🏦 Recurring Payments (EMIs)"):
                    emi_df = pd.DataFrame(report['emi_analysis'])
                    st.dataframe(
                        emi_df[['amount', 'description', 'frequency', 'confidence', 'total_paid']],
                        column_config={
                            'amount': 'Amount (₹)',
                            'description': 'Description',
                            'frequency': 'Frequency',
                            'confidence': 'Confidence',
                            'total_paid': 'Total Paid (₹)'
                        },
                        use_container_width=True
                    )
            
            # 7. Overall Summary
            with st.expander("📋 Overall Summary"):
                summary = report['summary']
                
                # Display metrics in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Income", f"₹{summary['total_income']:,.2f}")
                    st.metric("Total Expenses", f"₹{summary['total_expenses']:,.2f}")
                    st.metric("Net Surplus", f"₹{summary['net_surplus']:,.2f}")
                
                with col2:
                    st.metric("Avg Monthly Income", f"₹{summary['average_monthly_income']:,.2f}")
                    st.metric("Avg Monthly Expenses", f"₹{summary['average_monthly_expenses']:,.2f}")
                    st.metric("Savings Rate", f"{summary['overall_savings_rate']:.1f}%")
                
                # Add calculation explanations
                st.markdown("---")
                st.markdown("### How These Are Calculated:")
                st.markdown("""
                - **Total Income**: Sum of all credit transactions (where dr_cr_indicator is 'CR')
                - **Total Expenses**: Sum of all debit transactions (where dr_cr_indicator is 'DR')
                - **Net Surplus**: Total Income - Total Expenses (negative indicates a deficit)
                - **Avg Monthly**: Total amount divided by number of months in the statement period
                - **Savings Rate**: (Net Surplus / Total Income) × 100% (negative means spending exceeds income)
                
                *Note: These calculations are based on the transaction data available in your bank statement.*
                """)
    
    except Exception as e:
        st.error(f"Error generating financial insights: {str(e)}")
        st.exception(e)

def display_analysis(analyzer: EnhancedBankStatementAnalyzer) -> None:
    """Display comprehensive analysis of the bank statement."""
    try:
        # Get comprehensive insights from the analyzer
        insights = analyzer.get_comprehensive_insights()
        
        # Display account information if available
        if hasattr(analyzer, 'account_info') and analyzer.account_info:
            st.markdown("<h3 class='section-title'>Account Information</h3>", unsafe_allow_html=True)
            account_info = analyzer.account_info
            col1, col2 = st.columns(2)
            
            with col1:
                if 'account_number' in account_info:
                    st.text(f"Account Number: {account_info['account_number']}")
                if 'account_holder' in account_info:
                    st.text(f"Account Holder: {account_info['account_holder']}")
            
            with col2:
                if 'statement_period' in account_info:
                    st.text(f"Statement Period: {account_info['statement_period']}")
                if 'currency' in account_info:
                    st.text(f"Currency: {account_info['currency']}")
            
            st.markdown("---")
            
        # Get transaction summary for monthly data
        summary = analyzer.get_transaction_summary()
        
        # Display monthly summary if available
        monthly_summary = analyzer.get_monthly_summary()
        if monthly_summary:
            st.markdown("<h3 class='section-title'>Monthly Summary</h3>", unsafe_allow_html=True)
            
            # Create a DataFrame for monthly data
            monthly_data = []
            for month, data in monthly_summary.items():
                monthly_data.append({
                    'Month': month,
                    'Income': data.get('total_credit', 0),
                    'Expenses': data.get('total_debit', 0),
                    'Net': data.get('total_credit', 0) - data.get('total_debit', 0),
                    'Opening Balance': data.get('opening_balance', 0),
                    'Closing Balance': data.get('closing_balance', 0)
                })
            
            monthly_df = pd.DataFrame(monthly_data)
            
            # Display monthly summary table
            st.dataframe(
                monthly_df.style.format({
                    'Income': '₹{:,.2f}'.format,
                    'Expenses': '₹{:,.2f}'.format,
                    'Net': '₹{:,.2f}'.format,
                    'Opening Balance': '₹{:,.2f}'.format,
                    'Closing Balance': '₹{:,.2f}'.format
                }),
                use_container_width=True
            )
            
            # Monthly Income vs Expenses Chart
            if not monthly_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=monthly_df['Month'],
                    y=monthly_df['Income'],
                    name='Income',
                    marker_color='#2e7d32'
                ))
                fig.add_trace(go.Bar(
                    x=monthly_df['Month'],
                    y=monthly_df['Expenses'],
                    name='Expenses',
                    marker_color='#c62828'
                ))
                fig.update_layout(
                    title='Monthly Income vs Expenses',
                    xaxis_title='Month',
                    yaxis_title='Amount (₹)',
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Display expense categories if available
        if hasattr(analyzer, 'categorize_expenses'):
            try:
                expense_categories = analyzer.categorize_expenses()
                if expense_categories:
                    st.markdown("<h3 class='section-title'>Expense Categories</h3>", unsafe_allow_html=True)
                    
                    # Create a DataFrame for expense categories
                    categories_df = pd.DataFrame(
                        list(expense_categories.items()),
                        columns=['Category', 'Amount']
                    )
                    
                    # Display top expense categories
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        st.dataframe(
                            categories_df.style.format({'Amount': '₹{:,.2f}'.format}),
                            use_container_width=True
                        )
                    
                    with col2:
                        # Donut chart for expense categories
                        fig = px.pie(
                            categories_df,
                            values='Amount',
                            names='Category',
                            hole=0.5,
                            title='Expense Distribution by Category'
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate expense categories: {str(e)}")
        
        # Display recurring payments if available
        if hasattr(analyzer, 'detect_recurring_payments'):
            try:
                recurring_payments = analyzer.detect_recurring_payments()
                if recurring_payments:
                    st.markdown("<h3 class='section-title'>Recurring Payments</h3>", unsafe_allow_html=True)
                    
                    # Convert to DataFrame for better display
                    recurring_df = pd.DataFrame(recurring_payments)
                    
                    # Format amount column
                    if 'amount' in recurring_df.columns:
                        recurring_df['amount'] = recurring_df['amount'].apply(lambda x: f"₹{x:,.2f}")
                    
                    st.dataframe(recurring_df, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not analyze recurring payments: {str(e)}")
        
        # Display transaction history
        if hasattr(analyzer, 'df') and not analyzer.df.empty:
            st.markdown("<h3 class='section-title'>Transaction History</h3>", unsafe_allow_html=True)
            
            # Make a copy of the DataFrame for display
            display_df = analyzer.df.copy()
            
            # Handle duplicate column names by adding suffixes
            display_df.columns = [f"{col}_{i}" if list(display_df.columns).count(col) > 1 and list(display_df.columns).index(col) != i else col 
                               for i, col in enumerate(display_df.columns)]
            
            # Format amount columns if they exist
            amount_columns = ['amount', 'debit', 'credit', 'balance']
            for col in amount_columns:
                # Check for column name with or without suffix
                matching_cols = [c for c in display_df.columns if c.startswith(col)]
                for col_name in matching_cols:
                    def safe_format(x):
                        try:
                            if pd.isna(x) or str(x).strip() == '':
                                return ""
                            return f"₹{float(x):,.2f}"
                        except (ValueError, TypeError):
                            return str(x)  # Return original value if conversion fails
                            
                    display_df[col_name] = display_df[col_name].apply(safe_format)
            
            st.dataframe(display_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error displaying analysis: {str(e)}")
        st.exception(e)  # Show full error for debugging

def main():
    st.markdown("<h1 class='main-title'>Bank Statement Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Upload your bank statement PDF to analyze your financial transactions and gain valuable insights.</p>", unsafe_allow_html=True)
    
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
        if st.button("Analyze Statement"):
            with st.spinner('Processing and analyzing your bank statement...'):
                try:
                    # Extract data from PDF
                    extract_full_table_to_csv(temp_file, output_file)
                    
                    # Read the CSV data
                    df = pd.read_csv(output_file)
                    
                    # Initialize the analyzer with the DataFrame
                    analyzer = EnhancedBankStatementAnalyzer(df)
                    
                    # Save the processed data before displaying analysis
                    processed_csv = "output/processed_statement.csv"
                    analyzer.df.to_csv(processed_csv, index=False)
                    
                    # Create tabs for different analysis views
                    tab1, tab2 = st.tabs(["Basic Analysis", "Financial Insights"])
                    
                    with tab1:
                        # Display the basic analysis
                        display_analysis(analyzer)
                    
                    with tab2:
                        # Display financial insights
                        display_financial_insights(processed_csv)
                    
                    # Add download buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # CSV download button
                        with open(output_file, 'rb') as f:
                            st.download_button(
                                label="📥 Download CSV",
                                data=f,
                                file_name="bank_statement.csv",
                                mime="text/csv",
                                help="Download transaction data as CSV file"
                            )
                    
                    with col2:
                        # Excel download button
                        excel_file = "output/bank_statement_analysis.xlsx"
                        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                            # Save transaction data
                            analyzer.df.to_excel(writer, sheet_name='Transactions', index=False)
                            
                            # Add summary sheets if available
                            if hasattr(analyzer, 'get_transaction_summary'):
                                summary = analyzer.get_transaction_summary()
                                if summary:
                                    pd.DataFrame([summary]).to_excel(writer, sheet_name='Summary', index=False)
                            
                            if hasattr(analyzer, 'get_monthly_summary'):
                                monthly = analyzer.get_monthly_summary()
                                if monthly:
                                    monthly_df = pd.DataFrame(monthly).T.reset_index()
                                    monthly_df.columns = ['Month'] + list(monthly_df.columns[1:])
                                    monthly_df.to_excel(writer, sheet_name='Monthly Summary', index=False)
                            
                            # Add financial insights if available
                            try:
                                financial_analyzer = FinancialStatementAnalyzer(processed_csv)
                                report = financial_analyzer.generate_comprehensive_report()
                                
                                # Add transaction counts
                                pd.DataFrame([report['transaction_counts']]).to_excel(
                                    writer, sheet_name='Financial Summary', index=False
                                )
                                
                                # Add expense categories
                                if report['expense_categories']:
                                    pd.DataFrame(
                                        list(report['expense_categories'].items()),
                                        columns=['Category', 'Amount']
                                    ).to_excel(writer, sheet_name='Expense Categories', index=False)
                                
                                # Add monthly surplus data
                                if report['monthly_surplus']:
                                    surplus_data = []
                                    for month, data in report['monthly_surplus'].items():
                                        row = {'Month': month}
                                        row.update(data)
                                        surplus_data.append(row)
                                    pd.DataFrame(surplus_data).to_excel(
                                        writer, sheet_name='Monthly Surplus', index=False
                                    )
                                
                                # Add EMI analysis if available
                                if report['emi_analysis']:
                                    pd.DataFrame(report['emi_analysis']).to_excel(
                                        writer, sheet_name='Recurring Payments', index=False
                                    )
                                
                            except Exception as e:
                                st.warning(f"Could not include all financial insights in Excel: {str(e)}")
                        
                        # Add download button for Excel
                        with open(excel_file, 'rb') as f:
                            st.download_button(
                                label="📊 Download Excel Report",
                                data=f,
                                file_name="bank_statement_analysis.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help="Download comprehensive financial report in Excel format"
                            )
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    st.exception(e)  # Show full error for debugging
            
            # Clean up the temporary file
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    st.warning(f"Could not remove temporary file: {str(e)}")

if __name__ == "__main__":
    main()
