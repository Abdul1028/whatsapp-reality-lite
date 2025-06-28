# Deployment Guide for WhatsApp Chat Analyzer

## Chrome/Kaleido Issue

The application uses Plotly's `fig.write_image()` function which requires Google Chrome to be installed for image generation. This can cause issues in deployment environments where Chrome is not available.

### Error Message
```
ChromeNotFoundError: Kaleido v1 and later requires Chrome to be installed.
```

## Solutions

### 1. Install Chrome in Deployment Environment (Recommended for full functionality)

#### For Ubuntu/Debian:
```bash
# Install Chrome
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list
apt-get update
apt-get install -y google-chrome-stable

# Install Chrome for Plotly
pip install kaleido
```

#### For Alpine Linux:
```bash
# Install Chrome
apk add --no-cache chromium
export CHROME_BIN=/usr/bin/chromium-browser
```

#### For Docker:
```dockerfile
# Add to your Dockerfile
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*
```

### 2. Use Alternative Approach (Current Implementation)

The application has been modified to handle missing Chrome gracefully:

- **Safe Image Saving**: The `safe_write_image()` function catches Chrome-related errors
- **Graceful Degradation**: Charts are still displayed in the app even if they can't be saved to files
- **PDF Export**: The export function handles missing images gracefully

### 3. Environment Variables

Set these environment variables in your deployment:

```bash
# For Streamlit Cloud
export CHROME_BIN=/usr/bin/google-chrome-stable

# For Heroku
heroku config:set CHROME_BIN=/app/.apt/usr/bin/google-chrome-stable
```

### 4. Streamlit Cloud Specific

For Streamlit Cloud, add this to your `packages.txt`:
```
chromium
chromium-chromedriver
```

## Current Status

The application will work without Chrome installed, but with these limitations:
- Charts won't be saved as image files
- PDF export will have placeholder text for missing charts
- All interactive charts will still display and function normally in the web app

## Testing

To test if Chrome is working:
```python
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
try:
    fig.write_image("test.png")
    print("Chrome/Kaleido is working!")
except Exception as e:
    print(f"Chrome/Kaleido issue: {e}")
```

## Alternative Solutions

If you can't install Chrome, consider these alternatives:

1. **Use Plotly's built-in download**: Users can download charts directly from the browser
2. **Convert to base64**: Embed images as base64 strings in the app
3. **Use different plotting libraries**: Matplotlib/Seaborn for static images
4. **Remove image saving**: Focus on interactive charts only

## Support

If you continue to have issues, check:
1. Chrome installation status
2. Environment variables
3. File permissions for the exports directory
4. Streamlit Cloud logs for specific error messages 