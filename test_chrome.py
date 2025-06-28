#!/usr/bin/env python3
"""
Test script to check if Chrome/Kaleido is working for Plotly image generation.
Run this script to diagnose Chrome installation issues.
"""

import plotly.graph_objects as go
import os

def test_chrome_kaleido():
    """Test if Chrome and Kaleido are working for Plotly image generation."""
    
    print("Testing Chrome/Kaleido for Plotly image generation...")
    print("=" * 50)
    
    # Check environment variables
    chrome_bin = os.environ.get('CHROME_BIN')
    if chrome_bin:
        print(f"✓ CHROME_BIN environment variable is set: {chrome_bin}")
    else:
        print("✗ CHROME_BIN environment variable is not set")
    
    # Create a simple test figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3, 4, 5], y=[1, 4, 9, 16, 25], mode='lines+markers'))
    fig.update_layout(title="Test Chart", xaxis_title="X", yaxis_title="Y")
    
    # Test image generation
    try:
        # Create exports directory if it doesn't exist
        os.makedirs("exports/charts", exist_ok=True)
        
        # Try to save the image
        fig.write_image("exports/charts/test_chrome.png")
        print("✓ Successfully saved test image to exports/charts/test_chrome.png")
        
        # Check if file was created
        if os.path.exists("exports/charts/test_chrome.png"):
            file_size = os.path.getsize("exports/charts/test_chrome.png")
            print(f"✓ Test image file created successfully (size: {file_size} bytes)")
        else:
            print("✗ Test image file was not created")
            
    except Exception as e:
        print(f"✗ Failed to save image: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Install Chrome: sudo apt-get install google-chrome-stable")
        print("2. Set CHROME_BIN environment variable")
        print("3. For Streamlit Cloud, add chromium to packages.txt")
        print("4. Check if kaleido is installed: pip install kaleido")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_chrome_kaleido() 