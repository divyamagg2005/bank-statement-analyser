import os
import sys
from pathlib import Path

def main():
    # Add the current directory to the path so we can import from ui
    sys.path.append(str(Path(__file__).parent))
    
    # Import streamlit here to avoid importing it unless needed
    import streamlit.web.cli as stcli
    
    # Set the streamlit config to not open the browser automatically
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    
    # Set the app to run on all network interfaces
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
    
    # Set the port (default is 8501)
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    
    # Run the Streamlit app
    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).parent / "ui" / "app.py"),
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--browser.gatherUsageStats=False",
    ]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()