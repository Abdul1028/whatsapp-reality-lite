import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import numpy as np
import pylab as pl
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import preprocessor
from sklearn.preprocessing import OrdinalEncoder
from textblob import TextBlob
from urlextract import URLExtract
import emoji
import streamlit as st
import plotly.express as px
import pandas as pd
import nltk
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import ssl
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import re
import random

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('vader_lexicon')
nltk.download('punkt')

extract = URLExtract()
nltk.download('vader_lexicon')

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # # fetch number of media messages
    # # Fetch number of media messages (either '<Media omitted>\n' or 'image omitted')
    # num_media_messages = df[(df['message'] == '<Media omitted>\n')].shape[0]

    # Fetch number of media messages (either '<Media omitted>\n', 'video omitted', or 'image omitted')
    num_media_messages = df[df['message'].str.contains('<Media omitted>|video omitted|image omitted')].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)


def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})

    # Create a bar plot using Plotly Express
    fig = px.bar(x=x.index, y=x.values, labels={'x': 'User', 'y': 'Count'})
    fig.update_layout(title="Most Busy Users")
    fig.update_xaxes(title_text='User', tickangle=-45)
    fig.update_yaxes(title_text='Count')

    return fig,df

from wordcloud import WordCloud,STOPWORDS
import plotly.graph_objs as go
from plotly.offline import plot
from collections import Counter

def load_hinglish_stopwords():
    """
    Load Hinglish stopwords from Kaggle or fallback to a basic list
    """
    try:
        import kagglehub
        # Download latest version
        path = kagglehub.dataset_download("prxshetty/stop-words-hinglish")
        
        # Read the stopwords file
        with open(f"{path}/stopwords.txt", "r", encoding="utf-8") as f:
            hinglish_stopwords = set(word.strip() for word in f.readlines())
        
        print(f"Loaded {len(hinglish_stopwords)} Hinglish stopwords")
        return hinglish_stopwords
        
    except Exception as e:
        print(f"Error loading Hinglish stopwords: {e}")
        # Fallback to basic stopwords
        basic_stopwords = {
            'media', 'omitted', 'image', 'video', 'audio', 'sticker', 'gif',
            'http', 'https', 'www', 'com', 'message', 'deleted', 'ok', 'okay',
            'yes', 'no', 'hi', 'hello', 'hey', 'hmm', 'haha', 'lol', 'lmao',
            'thanks', 'thank', 'you', 'the', 'and', 'for', 'this', 'that',
            'have', 'has', 'had', 'not', 'with', 'from', 'your', 'which',
            'there', 'their', 'they', 'them', 'then', 'than', 'but', 'also',
            # Basic Hindi stopwords
            'hai', 'hain', 'ho', 'ki', 'ka', 'ke', 'ko', 'main', 'aur', 'par',
            'kya', 'se', 'ne', 'to', 'bhi', 'kuch', 'nahi', 'na', 'ab', 'ye',
            'yeh', 'woh', 'mein', 'tha', 'thi', 'the', 'raha', 'rahi', 'rahe'
        }
        return basic_stopwords

# Load stopwords once at module level
HINGLISH_STOPWORDS = load_hinglish_stopwords()

def create_plotly_wordcloud(selected_user, df):
    """
    Enhanced word cloud with Hinglish stopword handling using local file
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Load stopwords from the local file
    try:
        with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
            hinglish_stopwords = set(word.strip() for word in f.readlines())
        print(f"Loaded {len(hinglish_stopwords)} Hinglish stopwords from file")
    except Exception as e:
        print(f"Error loading stopwords from file: {e}")
        hinglish_stopwords = set()
    
    # Add additional chat-specific stopwords
    chat_stopwords = {
        'media', 'omitted', 'image', 'video', 'audio', 'sticker', 'gif',
        'http', 'https', 'www', 'com', 'message', 'deleted', 'ok', 'okay',
        'yes', 'no', 'hi', 'hello', 'hey', 'hmm', 'haha', 'lol', 'lmao',
        'thanks', 'thank', 'you', 'the', 'and', 'for', 'this', 'that',
        'have', 'has', 'had', 'not', 'with', 'from', 'your', 'which',
        'there', 'their', 'they', 'them', 'then', 'than', 'but', 'also'
    }
    
    # Combine all stopwords
    all_stopwords = hinglish_stopwords.union(chat_stopwords)
    
    # Process messages
    words = []
    for message in df['message']:
        # Skip media messages
        if 'omitted' in message.lower():
            continue
            
        # Clean and tokenize
        clean_message = re.sub(r'[^\w\s]', '', message.lower())
        message_words = clean_message.split()
        
        # Filter stopwords, short words, and words containing digits
        filtered_words = []
        for word in message_words:
            # Skip if word is in stopwords
            if word.lower() in all_stopwords:
                continue
                
            # Skip if word is too short
            if len(word) <= 2:
                continue
                
            # Skip if word contains any digits
            if any(char.isdigit() for char in word):
                continue
                
            filtered_words.append(word)
            
        words.extend(filtered_words)
    
    # Count word frequencies
    word_counts = {}
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    
    # Get top words
    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:100]
    print(f"Top words after filtering: {top_words[:10]}...")
    
    # Prepare data for word cloud
    word_cloud_data = pd.DataFrame(top_words, columns=['word', 'count'])
    
    # Check if we have any words to display
    if word_cloud_data.empty:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No significant words found after filtering stopwords",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Scale sizes for better visualization
    max_count = word_cloud_data['count'].max()
    word_cloud_data['size'] = (word_cloud_data['count'] / max_count * 50) + 10
    
    # Generate random positions with collision detection
    positions = []
    for _ in range(len(word_cloud_data)):
        attempts = 0
        while attempts < 100:  # Limit attempts to avoid infinite loop
            x = random.uniform(-0.8, 0.8)
            y = random.uniform(-0.8, 0.8)
            
            # Check for collisions with existing positions
            collision = False
            for pos in positions:
                # Simple distance check
                if ((x - pos[0])**2 + (y - pos[1])**2) < 0.02:  # Adjust this value for spacing
                    collision = True
                    break
            
            if not collision:
                positions.append((x, y))
                break
            
            attempts += 1
            
        # If we couldn't find a non-colliding position, just add one
        if attempts >= 100:
            positions.append((random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8)))
    
    word_cloud_data['x'] = [pos[0] for pos in positions]
    word_cloud_data['y'] = [pos[1] for pos in positions]
    
    # Create color scale based on frequency
    word_cloud_data['color'] = word_cloud_data['count'].rank(pct=True)
    
    # Create the figure
    fig = go.Figure()
    
    # Add text traces for each word
    for _, row in word_cloud_data.iterrows():
        # Get color from Viridis colorscale
        color_idx = min(int(row['color']*8), 7)  # Ensure index is within range
        
        fig.add_trace(go.Scatter(
            x=[row['x']],
            y=[row['y']],
            mode='text',
            text=[row['word']],
            textfont=dict(
                size=row['size'],
                color=px.colors.sequential.Viridis[color_idx]
            ),
            hoverinfo='text',
            hovertext=f"{row['word']}: {row['count']} occurrences",
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Word Cloud for {selected_user}' if selected_user != 'Overall' else 'Overall Word Cloud',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        hovermode='closest',
        margin=dict(l=20, r=20, t=50, b=20),
        template='plotly_white'
    )
    
    # Print some debug info
    print(f"Generated word cloud with {len(word_cloud_data)} words after filtering {len(all_stopwords)} stopwords and numbers")
    
    return fig

def most_common_words(selected_user, df):
    """
    Enhanced visualization of most common words with Hinglish stopword handling
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Load stopwords from the local file
    try:
        with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
            hinglish_stopwords = set(word.strip() for word in f.readlines())
        print(f"Loaded {len(hinglish_stopwords)} Hinglish stopwords from file")
    except Exception as e:
        print(f"Error loading stopwords from file: {e}")
        hinglish_stopwords = set()
    
    # Add additional chat-specific stopwords
    chat_stopwords = {
        'media', 'omitted', 'image', 'video', 'audio', 'sticker', 'gif',
        'http', 'https', 'www', 'com', 'message', 'deleted', 'ok', 'okay',
        'yes', 'no', 'hi', 'hello', 'hey', 'hmm', 'haha', 'lol', 'lmao',
        'thanks', 'thank', 'you', 'the', 'and', 'for', 'this', 'that',
        'have', 'has', 'had', 'not', 'with', 'from', 'your', 'which',
        'there', 'their', 'they', 'them', 'then', 'than', 'but', 'also'
    }
    
    # Combine all stopwords
    all_stopwords = hinglish_stopwords.union(chat_stopwords)
    
    # Process messages
    words = []
    for message in df['message']:
        # Skip media messages
        if 'omitted' in message.lower():
            continue
            
        # Clean and tokenize
        clean_message = re.sub(r'[^\w\s]', '', message.lower())
        message_words = clean_message.split()
        
        # Filter stopwords, short words, and words containing digits
        filtered_words = []
        for word in message_words:
            # Skip if word is in stopwords
            if word.lower() in all_stopwords:
                continue
                
            # Skip if word is too short
            if len(word) <= 2:
                continue
                
            # Skip if word contains any digits
            if any(char.isdigit() for char in word):
                continue
                
            filtered_words.append(word)
            
        words.extend(filtered_words)
    
    # Count word frequencies
    word_counts = {}
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    
    # Get top words
    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    print(f"Top common words after filtering: {top_words[:5]}...")
    
    # Check if we have any words to display
    if not top_words:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No significant words found after filtering stopwords",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    word_df = pd.DataFrame(top_words, columns=['word', 'count'])
    
    # Create horizontal bar chart
    fig = px.bar(
        word_df,
        y='word',
        x='count',
        orientation='h',
        color='count',
        color_continuous_scale='Viridis',
        title=f"Most Common Words for {selected_user}" if selected_user != 'Overall' else "Overall Most Common Words"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Count",
        yaxis_title="Word",
        yaxis=dict(autorange="reversed"),  # Highest count at top
        template='plotly_white'
    )
    
    # Print some debug info
    print(f"Generated common words chart with {len(word_df)} words after filtering {len(all_stopwords)} stopwords and numbers")
    
    return fig

def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])
        emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    fig = px.pie(emoji_df.head(8), labels={'0': 'Emoji', '1': 'Frequency'}, values=1, names=0,
                 title="Emoji Distribution")

    fig.write_image("exports/charts/emojis.png")
    return fig

def monthly_timeline(selected_user, df):
    """
    Creates an enhanced monthly timeline visualization with trend line
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Ensure we have date as datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Create year-month field for proper chronological sorting
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Group by year-month
    timeline = df.groupby('year_month').agg({
        'message': 'count',
        'user': 'nunique'  # Count unique users per month
    }).reset_index()
    
    # Convert period to string for display
    timeline['time'] = timeline['year_month'].astype(str)
    
    # Calculate 3-month moving average for trend
    timeline['trend'] = timeline['message'].rolling(window=3, min_periods=1).mean()
    
    # Create enhanced figure
    fig = go.Figure()
    
    # Add bar chart for message count
    fig.add_trace(go.Bar(
        x=timeline['time'],
        y=timeline['message'],
        name='Messages',
        marker_color='rgba(58, 71, 80, 0.6)',
        hovertemplate='%{y} messages<extra></extra>'
    ))
    
    # Add trend line
    fig.add_trace(go.Scatter(
        x=timeline['time'],
        y=timeline['trend'],
        mode='lines',
        name='3-Month Trend',
        line=dict(color='firebrick', width=2),
        hovertemplate='Trend: %{y:.1f}<extra></extra>'
    ))
    
    # Improve layout
    fig.update_layout(
        title={
            'text': f'Monthly Message Activity for {selected_user}' if selected_user != 'Overall' else 'Monthly Message Activity',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Month',
        yaxis_title='Number of Messages',
        legend_title='Metrics',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def daily_timeline(selected_user, df):
    """
    Creates an enhanced daily timeline with day of week highlighting
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Ensure we have date as datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Extract date only (no time)
    df['message_date'] = df['date'].dt.date
    
    # Add day of week
    df['day_of_week'] = df['date'].dt.day_name()
    
    # Group by date
    daily_counts = df.groupby('message_date').agg({
        'message': 'count',
        'day_of_week': 'first'  # Get day name
    }).reset_index()
    
    # Convert back to datetime for proper plotting
    daily_counts['message_date'] = pd.to_datetime(daily_counts['message_date'])
    
    # Calculate 7-day moving average
    daily_counts['7day_avg'] = daily_counts['message'].rolling(window=7, min_periods=1).mean()
    
    # Create color mapping for days of week
    day_colors = {
        'Monday': '#FFD700',    # Gold
        'Tuesday': '#87CEFA',   # Light Sky Blue
        'Wednesday': '#90EE90', # Light Green
        'Thursday': '#FFA07A',  # Light Salmon
        'Friday': '#DA70D6',    # Orchid
        'Saturday': '#FF6347',  # Tomato
        'Sunday': '#1E90FF'     # Dodger Blue
    }
    
    # Map colors to days
    daily_counts['color'] = daily_counts['day_of_week'].map(day_colors)
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot with colored points by day of week
    fig.add_trace(go.Scatter(
        x=daily_counts['message_date'],
        y=daily_counts['message'],
        mode='markers',
        marker=dict(
            size=8,
            color=daily_counts['color'],
            opacity=0.7
        ),
        name='Daily Count',
        hovertemplate='%{y} messages on %{x|%A, %b %d, %Y}<extra></extra>'
    ))
    
    # Add 7-day moving average
    fig.add_trace(go.Scatter(
        x=daily_counts['message_date'],
        y=daily_counts['7day_avg'],
        mode='lines',
        line=dict(color='rgba(0,0,0,0.7)', width=2),
        name='7-Day Average',
        hovertemplate='7-day avg: %{y:.1f}<extra></extra>'
    ))
    
    # Create day of week legend
    for day, color in day_colors.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=day,
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Daily Message Activity for {selected_user}' if selected_user != 'Overall' else 'Daily Message Activity',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Date',
        yaxis_title='Number of Messages',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def week_activity_map(selected_user, df):
    """
    Enhanced weekly activity heatmap showing hourly patterns
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Ensure we have date as datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Extract day of week and hour
    df['day_of_week'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    
    # Create pivot table for heatmap
    heatmap_data = df.pivot_table(
        index='day_of_week', 
        columns='hour', 
        values='message',
        aggfunc='count',
        fill_value=0
    )
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)
    
    # Create heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Hour of Day", y="Day of Week", color="Message Count"),
        x=[str(h) for h in range(24)],
        y=day_order,
        color_continuous_scale='Viridis',
        title=f"Weekly Activity Pattern for {selected_user}" if selected_user != 'Overall' else "Overall Weekly Activity Pattern"
    )
    
    # Add hour labels
    hour_labels = [f"{h}:00" for h in range(24)]
    fig.update_xaxes(tickvals=list(range(24)), ticktext=hour_labels)
    
    # Add annotations for peak times
    peak_day = heatmap_data.sum(axis=1).idxmax()
    peak_hour = heatmap_data.sum(axis=0).idxmax()
    peak_cell = heatmap_data.stack().idxmax()
    
    fig.add_annotation(
        x=peak_hour,
        y=peak_day,
        text="Peak Activity",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        coloraxis_colorbar=dict(
            title="Message Count",
        ),
        template='plotly_white'
    )
    
    return fig

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    busy_month = df['month'].value_counts()

    fig = px.bar(busy_month, x=busy_month.index, y=busy_month.values, color=busy_month.values,
                 color_continuous_scale='Viridis')
    fig.update_layout(xaxis_tickangle=-45)
    fig.write_image("exports/charts/month_activity.png")
    return fig

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap

def analyze_sentiment(message):
    blob = TextBlob(message)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

def message_length_analysis(selected_participant,df):
    filtered_df = df[df['user'] == selected_participant] if selected_participant != 'Overall' else df
    filtered_df['message_length'] = filtered_df['message'].apply(lambda msg: len(msg))
    average_length = filtered_df['message_length'].mean()
    st.write(f"Average Message Length for {selected_participant}: {average_length:.2f}")


# Function for busiest hours analysis
def busiest_hours_analysis(df):
    busiest_hours = df['hour'].value_counts()
    st.bar_chart(busiest_hours)


# Function for message count by month
def message_count_by_month(selected_participant,df):
    filtered_df = df[df['user'] == selected_participant] if selected_participant != 'Overall' else df
    message_count_per_month = filtered_df.groupby(['year', 'month']).count()['message'].reset_index()
    st.dataframe(message_count_per_month)


# Function for top emojis used
def top_emojis_used(selected_participant,df):
    filtered_df = df[df['user'] == selected_participant] if selected_participant != 'Overall' else df
    emojis = [c for message in filtered_df['message'] for c in message if c in emoji.EMOJI_DATA]
    top_emojis = Counter(emojis).most_common()
    st.write(f"Top Emojis Used by {selected_participant}: {top_emojis}")
    return top_emojis


# Function for greeting and farewell analysis
def greeting_farewell_analysis(selected_participant, df):
    filtered_df = df[df['user'] == selected_participant] if selected_participant != 'Overall' else df

    greetings = filtered_df['message'].apply(lambda msg: 'hello' in msg.lower() or 'hi' in msg.lower()).sum()
    farewells = filtered_df['message'].apply(lambda msg: 'goodbye' in msg.lower() or 'bye' in msg.lower()).sum()
    birthdays = filtered_df['message'].apply(
        lambda msg: 'happy birthday' in msg.lower() or 'happiest birthday' in msg.lower()).sum()

    total_messages = filtered_df.shape[0]
    greeting_percentage = (greetings / total_messages) * 100
    farewell_percentage = (farewells / total_messages) * 100
    birthday_percentage = (birthdays / total_messages) * 100

    # Create a pie chart using Plotly
    labels = ['Greetings', 'Farewells', 'Birthday Wishes']
    sizes = [greeting_percentage, farewell_percentage, birthday_percentage]
    colors = ['yellowgreen', 'lightskyblue', 'lightcoral']

    fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=.3)])
    fig.update_layout(title=f"Greeting, Farewell, and Birthday Wishes Analysis by {selected_participant}")

    greetings = filtered_df['message'].apply(lambda msg: 'hello' in msg.lower() or 'hi' in msg.lower()).sum()
    farewells = filtered_df['message'].apply(lambda msg: 'goodbye' in msg.lower() or 'bye' in msg.lower()).sum()
    birthdays = filtered_df['message'].apply(
        lambda msg: 'happy birthday' in msg.lower() or 'happiest birthday' in msg.lower()).sum()

    st.write(f"Total Greetings by {selected_participant}: {greetings}")
    st.write(f"Total Farewells by {selected_participant}: {farewells}")
    st.write(f"Total Birthday Wishes by {selected_participant}: {birthdays}")

    fig.write_image("exports/charts/greetings.png")
    return fig


# Function for topic analysis using LDA
with open('stop_hinglish.txt', 'r') as f:
    stop_words = set(f.read().splitlines())

# Function for topic analysis using LDA with heuristic topic naming
# Load the stop words from the file
with open('stop_hinglish.txt', 'r') as f:
    stop_words = set(f.read().splitlines())


#Only highest Reply time user display
def longest_reply_user(df):
    # Ordinal encoders will encode each user with its own number
    user_encoder = OrdinalEncoder()
    df['User Code'] = user_encoder.fit_transform(df['user'].values.reshape(-1, 1))

    # Find replies
    message_senders = df['User Code'].values
    sender_changed = (np.roll(message_senders, 1) - message_senders).reshape(1, -1)[0] != 0
    sender_changed[0] = False
    is_reply = sender_changed & ~df['user'].eq('group_notification')

    df['Is Reply'] = is_reply

    # Calculate times based on replies
    reply_times, indices = preprocessor.calculate_times_on_trues(df, 'Is Reply')
    reply_times_df_list = []
    reply_time_index = 0
    for i in range(0, len(df)):
        if i in indices:
            reply_times_df_list.append(reply_times[reply_time_index].astype("timedelta64[m]").astype("float"))
            reply_time_index += 1
        else:
            reply_times_df_list.append(0)

    df['Reply Time'] = reply_times_df_list

    # Calculate the maximum reply time for each user
    max_reply_times = df.groupby('user')['Reply Time'].max()

    # Find the user with the longest reply time
    max_reply_user = max_reply_times.idxmax()
    max_reply_time = max_reply_times.max()
    return max_reply_user, max_reply_time


#additional info about reply

def longest_reply_user2(df):

    # Filter out messages containing the specified strings
    omitted_strings = ["image omitted", "media omitted", "video omitted"]
    df = df[~df['message'].str.lower().str.contains('|'.join(omitted_strings))]

    # Ordinal encoders will encode each user with its own number
    user_encoder = OrdinalEncoder()
    df['User Code'] = user_encoder.fit_transform(df['user'].values.reshape(-1, 1))

    # Find replies
    message_senders = df['User Code'].values
    sender_changed = (np.roll(message_senders, 1) - message_senders).reshape(1, -1)[0] != 0
    sender_changed[0] = False
    is_reply = sender_changed & ~df['user'].eq('group_notification')

    df['Is Reply'] = is_reply

    # Calculate times based on replies
    reply_times, indices = preprocessor.calculate_times_on_trues(df, 'Is Reply')
    reply_times_df_list = []
    reply_time_index = 0
    for i in range(0, len(df)):
        if i in indices:
            reply_times_df_list.append(reply_times[reply_time_index].astype("timedelta64[m]").astype("float"))
            reply_time_index += 1
        else:
            reply_times_df_list.append(0)

    df['Reply Time'] = reply_times_df_list

    # Calculate the maximum reply time for each user
    max_reply_times = df.groupby('user')['Reply Time'].max()

    # Find the user with the longest reply time
    max_reply_user = max_reply_times.idxmax()
    max_reply_time_minutes = max_reply_times.max()  # Max reply time in minutes

    # Find the message to which the user replied the most late
    max_reply_message_index = df[df['Reply Time'] == max_reply_time_minutes].index[0]
    max_reply_message = df.loc[max_reply_message_index, 'message']

    reply = df.shift(1).loc[max_reply_message_index, 'message']

    return max_reply_user, max_reply_time_minutes, max_reply_message,reply



def top5_late_replies(df):
    # Filter out messages containing the specified strings
    omitted_strings = ["image omitted", "media omitted", "video omitted"]
    df = df[~df['message'].str.lower().str.contains('|'.join(omitted_strings))]

    # Ordinal encoders will encode each user with its own number
    user_encoder = OrdinalEncoder()
    df['User Code'] = user_encoder.fit_transform(df['user'].values.reshape(-1, 1))

    # Find replies
    message_senders = df['User Code'].values
    sender_changed = (np.roll(message_senders, 1) - message_senders).reshape(1, -1)[0] != 0
    sender_changed[0] = False
    is_reply = sender_changed & ~df['user'].eq('group_notification')

    df['Is Reply'] = is_reply

    # Calculate times based on replies
    reply_times, indices = preprocessor.calculate_times_on_trues(df, 'Is Reply')
    reply_times_df_list = []
    reply_time_index = 0
    for i in range(0, len(df)):
        if i in indices:
            reply_times_df_list.append(reply_times[reply_time_index].astype("timedelta64[m]").astype("float"))
            reply_time_index += 1
        else:
            reply_times_df_list.append(0)

    df['Reply Time'] = reply_times_df_list

    # Calculate the maximum reply time for each user
    max_reply_times = df.groupby('user')['Reply Time'].max()

    # Find the top 5 users with the longest reply times
    top_5_users = max_reply_times.nlargest(5)

    # Initialize lists to store results
    users = []
    reply_times = []
    max_reply_messages = []
    replies = []

    # Iterate over the top 5 users
    for user, max_reply_time in top_5_users.items():
        # Find the message to which the user replied the most late
        max_reply_message_index = df[df['Reply Time'] == max_reply_time].index[0]
        max_reply_message = df.loc[max_reply_message_index, 'message']

        # Get the reply to the max reply message
        reply = df.shift(1).loc[max_reply_message_index, 'message']

        # Append user, max reply time, max reply message, and reply to lists
        users.append(user)
        reply_times.append(max_reply_time)
        max_reply_messages.append(max_reply_message)
        replies.append(reply)

    return users, reply_times, max_reply_messages, replies


def top_texts_late_replies(df):
    # Filter out messages containing the specified strings
    omitted_strings = ["image omitted", "media omitted", "video omitted"]
    df = df[~df['message'].str.lower().str.contains('|'.join(omitted_strings))]

    # Ordinal encoders will encode each user with its own number
    user_encoder = OrdinalEncoder()
    df['User Code'] = user_encoder.fit_transform(df['user'].values.reshape(-1, 1))

    # Find replies
    message_senders = df['User Code'].values
    sender_changed = (np.roll(message_senders, 1) - message_senders).reshape(1, -1)[0] != 0
    sender_changed[0] = False
    is_reply = sender_changed & ~df['user'].eq('group_notification')

    df['Is Reply'] = is_reply

    # Calculate times based on replies
    reply_times, indices = preprocessor.calculate_times_on_trues(df, 'Is Reply')
    reply_times_df_list = []
    reply_time_index = 0
    for i in range(0, len(df)):
        if i in indices:
            reply_times_df_list.append(reply_times[reply_time_index].astype("timedelta64[m]").astype("float"))
            reply_time_index += 1
        else:
            reply_times_df_list.append(0)

    df['Reply Time'] = reply_times_df_list

    # Keep only messages with reply time greater than 2 days (very late replies)
    late_replies_df = df[df['Reply Time'] > (48 * 60)]  # 48 hours in minutes

    # Find the top 5 users with the longest reply times
    top_5_users = late_replies_df.groupby('user')['Reply Time'].max().nlargest(5)

    # Initialize lists to store results
    users = []
    reply_times = []
    max_reply_messages = []
    replies = []

    # Iterate over the top 5 users
    for user, max_reply_time in top_5_users.items():
        # Find the message to which the user replied the most late
        max_reply_message_index = df[df['Reply Time'] == max_reply_time].index[0]
        max_reply_message = df.loc[max_reply_message_index, 'message']

        # Get the reply to the max reply message
        reply = df.shift(1).loc[max_reply_message_index, 'message']

        # Append user, max reply time, max reply message, and reply to lists
        users.append(user)
        reply_times.append(max_reply_time)
        max_reply_messages.append(max_reply_message)
        replies.append(reply)

    return users, reply_times, max_reply_messages, replies




# shows everyone's reply time and also plots graph
def show_average_reply_time(df):
    """
    Enhanced visualization of reply times with user comparisons
    """
    # Create a copy to avoid modifying original
    df_copy = df.copy()
    
    # Handle case where date is both index and column
    if df_copy.index.name == 'date':
        if 'date' in df_copy.columns:
            # If date is both an index and a column, just drop the index
            df_copy = df_copy.reset_index(drop=True)
        else:
            # If date is only in the index, reset the index to get it as a column
            df_copy = df_copy.reset_index()
    
    # Ensure we have date as datetime
    if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
        df_copy['date'] = pd.to_datetime(df_copy['date'])
    
    # Sort by date
    df_copy = df_copy.sort_values('date')
    
    # Add previous user column
    df_copy['prev_user'] = df_copy['user'].shift(1)
    df_copy['prev_time'] = df_copy['date'].shift(1)
    
    # Calculate time difference in minutes
    df_copy['time_diff'] = (df_copy['date'] - df_copy['prev_time']).dt.total_seconds() / 60
    
    # Filter for actual replies (different user than previous message)
    replies = df_copy[(df_copy['user'] != df_copy['prev_user']) & 
                      (df_copy['time_diff'] <= 24*60) &  # Limit to 24 hours
                      (df_copy['time_diff'] > 0)]        # Ensure positive time
    
    # Group by user to get average reply time
    user_reply_times = replies.groupby('user').agg({
        'time_diff': ['mean', 'median', 'count']
    }).reset_index()
    
    # Flatten column names
    user_reply_times.columns = ['user', 'mean_reply_time', 'median_reply_time', 'reply_count']
    
    # Filter users with at least 5 replies
    user_reply_times = user_reply_times[user_reply_times['reply_count'] >= 5]
    
    # Sort by median reply time
    user_reply_times = user_reply_times.sort_values('median_reply_time')
    
    # Create figure
    fig = go.Figure()
    
    # Check if we have any data to display
    if user_reply_times.empty:
        # Create an empty figure with a message
        fig.add_annotation(
            text="Not enough reply data to analyze",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
    return fig

    # Add bar for median reply time
    fig.add_trace(go.Bar(
        x=user_reply_times['user'],
        y=user_reply_times['median_reply_time'],
        name='Median Reply Time',
        marker_color='#2196F3',
        hovertemplate='Median: %{y:.1f} minutes<br>Count: %{text}<extra></extra>',
        text=user_reply_times['reply_count']
    ))
    
    # Add markers for mean reply time
    fig.add_trace(go.Scatter(
        x=user_reply_times['user'],
        y=user_reply_times['mean_reply_time'],
        mode='markers',
        name='Mean Reply Time',
        marker=dict(
            color='#F44336',
            size=10,
            symbol='diamond'
        ),
        hovertemplate='Mean: %{y:.1f} minutes<extra></extra>'
    ))
    
    # Add overall average line
    if not user_reply_times.empty:
        overall_median = user_reply_times['median_reply_time'].median()
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=overall_median,
            x1=len(user_reply_times)-0.5,
            y1=overall_median,
            line=dict(
                color="green",
                width=2,
                dash="dash",
            )
        )
        
        # Add annotation for overall average
        fig.add_annotation(
            x=len(user_reply_times)-1,
            y=overall_median,
            text=f"Overall Median: {overall_median:.1f} min",
            showarrow=True,
            arrowhead=1,
            ax=50,
            ay=-30
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Average Reply Time by User',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='User',
        yaxis_title='Reply Time (minutes)',
        hovermode='closest',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add a second y-axis for reply count
    fig.update_layout(
        annotations=[
            dict(
                text="Lower is faster response",
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                showarrow=False
            )
        ]
    )
    
    return fig



def _create_wide_area_fig(df : pd.DataFrame, legend : bool = True):
    fig, ax = plt.subplots(figsize=(12,5))
    df.plot(
        alpha=0.6,
        cmap=plt.get_cmap('viridis'),
        ax=ax,
        stacked=True
    )
    ax.patch.set_alpha(0.0)
    fig.patch.set_alpha(0.0)
    if legend:
        ax.legend(df['user'])
    return fig


def create_narrow_pie_fig(df : pd.DataFrame):
    narrow_figsize = (6, 5)
    cmap = plt.get_cmap('viridis')
    fig1, ax = plt.subplots(figsize=narrow_figsize)
    df.plot(kind='pie', cmap=cmap, ax=ax, autopct='%1.1f%%', explode=[0.015] * len(df.index.unique()))
    centre_circle = plt.Circle((0, 0), 0.80, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    ax.patch.set_alpha(0.0)
    fig.patch.set_alpha(0.0)
    ax.set_ylabel('')
    return fig


def message_count_aggregated_graph(df):
    subject_df = df.groupby('user').count()['message'].sort_values(ascending=False)
    most_messages_winner = subject_df.index[subject_df.argmax()]
    # fig = create_narrow_pie_fig(subject_df)

    # Create a Pie chart
    fig = go.Figure(data=[go.Pie(labels=subject_df.index, values=subject_df.values)])
    fig.update_layout(title="Message Count Aggregated by User")

    fig.write_image("exports/charts/most_message_winner.png")

    return fig, most_messages_winner


def conversation_starter_graph(df):
    subject_df = df[df['Conv change']].groupby('user').count()['Reply Time']

    # Create a Pie chart
    fig = go.Figure(data=[go.Pie(labels=subject_df.index, values=subject_df.values)])
    fig.update_layout(title="Conversation Starter Count by User")

    most_messages_winner = subject_df.index[subject_df.argmax()]
    fig.write_image("exports/charts/conversation_starter_winner.png")
    return fig, most_messages_winner

def conversation_size_aggregated_graph( df):
    conversations_df = df.groupby('Conv code').agg(count=('Conv code', 'size'),
                                                   mean_date=('date', 'mean')).reset_index()
    conversations_df.index = conversations_df['mean_date']
    conversations_df = conversations_df.resample('W').mean().fillna(0)

    # Create a line plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=conversations_df.index, y=conversations_df['count'], mode='lines', fill='tozeroy'))
    fig.update_layout(title="Conversation Size Aggregated over Time",
                      xaxis_title="Date",
                      yaxis_title="Average Conversation Size")
    fig.write_image("exports/charts/average_conversation_size.png")
    return fig

def most_idle_date_time(df):
    df['date'] = pd.to_datetime(df['date'])

    # Group the data by date and calculate total idle time
    grouped_by_date = df.groupby(df['date'].dt.date)['Inter conv time'].sum()

    # Find the date(s) with the maximum idle time
    max_idle_date = grouped_by_date.idxmax()
    max_idle_time = grouped_by_date[max_idle_date]

    return max_idle_date,max_idle_time

def median_delay_btwn_convo(df):

    df['date'] = pd.to_datetime(df['date'])

    # Calculate reply delay within each conversation
    df['Reply Delay'] = df.groupby('Conv code')['date'].diff().dt.total_seconds() / 60.0  # Convert to minutes

    # Group by conversation and calculate median reply delay
    median_delay_per_conversation = df.groupby('Conv code')['Reply Delay'].median()

    # Calculate overall median delay across all conversations
    overall_median_delay = median_delay_per_conversation.median()

    print(f"Median Delay Between Conversations: {overall_median_delay:.2f} minutes")

def median_delay_between_conversations(user,df):
    if user != "Overall":
        df['date'] = pd.to_datetime(df['date'])
        # Filter DataFrame to include only messages sent by the specified user
        user_messages = df[df['user'] == user]

        # Calculate reply delay within each conversation for the user
        user_messages['Reply Delay'] = user_messages.groupby('Conv code')[
                                           'date'].diff().dt.total_seconds() / 60.0  # Convert to minutes

        # Display the median reply delay for the user
        median_delay_per_user = user_messages['Reply Delay'].median()

        # THIS IS NOT WORKING ONLY MEDIAN DELAY SHOWING RIGHT
        # TODO: UPDATE THE AVERAGE DELAY BETWEEN REPLIES IN CONVERSATION

        # # Compute average delay between conversations for the user
        # avg_delay_between_conversations = user_messages['Reply Delay'].mean()

        return median_delay_per_user
    return  None






import plotly.graph_objects as go

def analyze_and_plot_sentiment(selected_user, df):
    """
    Enhanced sentiment analysis with time trends and user comparisons
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Ensure we have date as datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Add month-year for grouping
    df['month_year'] = df['date'].dt.strftime('%Y-%m')
    
    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    # Calculate sentiment for each message
    sentiment_data = []
    for _, row in df.iterrows():
        sentiment = sentiment_analyzer.polarity_scores(row['message'])
        sentiment_data.append({
            'user': row['user'],
            'date': row['date'],
            'month_year': row['month_year'],
            'message': row['message'],
            'positive': sentiment['pos'],
            'negative': sentiment['neg'],
            'neutral': sentiment['neu'],
            'compound': sentiment['compound'],
            'sentiment_category': 'Positive' if sentiment['compound'] > 0.05 else 
                                 'Negative' if sentiment['compound'] < -0.05 else 'Neutral'
        })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    
    # Create monthly sentiment trends
    monthly_sentiment = sentiment_df.groupby('month_year').agg({
        'positive': 'mean',
        'negative': 'mean',
        'neutral': 'mean',
        'compound': 'mean',
        'message': 'count'
    }).reset_index()
    
    # Sort chronologically
    monthly_sentiment['date'] = pd.to_datetime(monthly_sentiment['month_year'] + '-01')
    monthly_sentiment = monthly_sentiment.sort_values('date')
    monthly_sentiment['month_year_formatted'] = monthly_sentiment['date'].dt.strftime('%b %Y')
    
    # Create sentiment distribution figure
    sentiment_counts = sentiment_df['sentiment_category'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    # Order categories
    category_order = ['Positive', 'Neutral', 'Negative']
    sentiment_counts['Sentiment'] = pd.Categorical(
        sentiment_counts['Sentiment'], 
        categories=category_order, 
        ordered=True
    )
    sentiment_counts = sentiment_counts.sort_values('Sentiment')
    
    # Calculate percentages
    total = sentiment_counts['Count'].sum()
    sentiment_counts['Percentage'] = (sentiment_counts['Count'] / total * 100).round(1)
    
    # Create distribution figure
    dist_fig = px.pie(
        sentiment_counts, 
        values='Count', 
        names='Sentiment',
        color='Sentiment',
        color_discrete_map={
            'Positive': '#4CAF50',
            'Neutral': '#2196F3',
            'Negative': '#F44336'
        },
        hole=0.4,
        title=f"Sentiment Distribution for {selected_user}" if selected_user != 'Overall' else "Overall Sentiment Distribution"
    )
    
    # Add percentage labels
    dist_fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent:.1%}<extra></extra>'
    )
    
    # Create trend figure
    trend_fig = go.Figure()
    
    # Add sentiment trend lines
    trend_fig.add_trace(go.Scatter(
        x=monthly_sentiment['month_year_formatted'],
        y=monthly_sentiment['positive'],
        mode='lines+markers',
        name='Positive',
        line=dict(color='#4CAF50', width=2)
    ))
    
    trend_fig.add_trace(go.Scatter(
        x=monthly_sentiment['month_year_formatted'],
        y=monthly_sentiment['negative'],
        mode='lines+markers',
        name='Negative',
        line=dict(color='#F44336', width=2)
    ))
    
    trend_fig.add_trace(go.Scatter(
        x=monthly_sentiment['month_year_formatted'],
        y=monthly_sentiment['compound'],
        mode='lines+markers',
        name='Overall Sentiment',
        line=dict(color='#2196F3', width=3)
    ))
    
    # Update trend layout
    trend_fig.update_layout(
        title={
            'text': f'Sentiment Trends for {selected_user}' if selected_user != 'Overall' else 'Overall Sentiment Trends',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Month',
        yaxis_title='Sentiment Score',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return dist_fig, trend_fig

def calculate_sentiment_percentage(selected_users, df):
    nltk.download('vader_lexicon')

    # Filter messages based on selected users
    if selected_users == 'Overall':
        selected_df = df  # Consider the whole dataframe if 'Overall' is selected
    else:
        if isinstance(selected_users, str):
            selected_users = [selected_users]  # Convert to list if only one user is selected
        selected_df = df[df['user'].isin(selected_users)]

    # Initialize the sentiment analyzer outside the function
    sid = SentimentIntensityAnalyzer()

    # Initialize counters for positive and negative sentiment scores
    positive_count = 0
    negative_count = 0

    # Dictionary to store positivity and negativity percentages for each user
    user_sentiment_percentages = {}

    # Tokenize and analyze sentiment for each message
    for user, messages in selected_df.groupby('user')['message']:
        # Reset counters for each user
        positive_count = 0
        negative_count = 0

        # Calculate sentiment scores for each message
        for message in messages:
            sentiment_score = sid.polarity_scores(message)['compound']
            if sentiment_score > 0:
                positive_count += 1
            elif sentiment_score < 0:
                negative_count += 1

        # Calculate positivity and negativity percentages for the user
        total_messages = len(messages)
        positivity_percentage = (positive_count / total_messages) * 100
        negativity_percentage = (negative_count / total_messages) * 100

        # Format percentages as strings with two decimal places and a percentage sign
        formatted_positivity = f"{positivity_percentage:.2f}%"
        formatted_negativity = f"{negativity_percentage:.2f}%"

        # Store the percentages in the dictionary
        user_sentiment_percentages[user] = (formatted_positivity, formatted_negativity)

    # Find the most positive and most negative users
    most_positive_user = max(user_sentiment_percentages, key=lambda x: user_sentiment_percentages[x][0])
    most_negative_user = min(user_sentiment_percentages, key=lambda x: user_sentiment_percentages[x][1])

    return user_sentiment_percentages, most_positive_user, most_negative_user



def create_messages_per_week_graph(df: pd.DataFrame):
    # Convert 'Date' column to datetime if it's not already
    df['date'] = pd.to_datetime(df['date'])

    # Makes the first graph
    date_df = df.groupby(df.index)[df['user']].sum().resample('W').sum()
    fig = _create_wide_area_fig(date_df)

    max_message_count = date_df[df['user']].sum(axis=1).max()
    max_message_count_date = date_df.index[date_df[df['user']].sum(axis=1).argmax()]

    return fig, max_message_count, max_message_count_date

def create_average_wpm_graph( df : pd.DataFrame):
    other_y_columns = [f"{subject}_mlength" for subject in df['user'].unique()]
    date_avg_df = df[other_y_columns].resample('W').mean()
    fig = _create_wide_area_fig(date_avg_df)
    return fig


def calculate_monthly_sentiment_trend(df):
    # Make a copy of the DataFrame to avoid modifying the original DataFrame
    df_copy = df.copy()
    nltk.download('vader_lexicon')

    # Initialize the sentiment analyzer outside the function
    sid = SentimentIntensityAnalyzer()

    # Tokenize and analyze sentiment for each message
    sentiment_scores = []
    for message in df_copy['message']:
        # Get sentiment score for the message
        sentiment_score = sid.polarity_scores(message)['compound']
        sentiment_scores.append(sentiment_score)

    # Add sentiment scores to the copied DataFrame
    df_copy['sentiment_score'] = sentiment_scores

    # Convert 'date' column to datetime if it's not already
    df_copy['date'] = pd.to_datetime(df_copy['date'])

    # Group data by month and calculate positivity and negativity percentages
    df_copy['month'] = df_copy['date'].dt.to_period('M')
    monthly_sentiment = df_copy.groupby('month').agg(
        positivity_percentage=('sentiment_score', lambda x: (x > 0).mean() * 100),
        negativity_percentage=('sentiment_score', lambda x: (x < 0).mean() * 100)
    )

    # Convert Period index to string for serialization
    monthly_sentiment.index = monthly_sentiment.index.astype(str)

    # Plot the trend
    fig = px.line(monthly_sentiment, x=monthly_sentiment.index, y=['positivity_percentage', 'negativity_percentage'],
                  title='Monthly Sentiment Trend',
                  labels={'month': 'Month', 'value': 'Percentage', 'variable': 'Sentiment'},
                  color_discrete_map={'positivity_percentage': 'blue', 'negativity_percentage': 'red'})
    fig.update_xaxes(type='category')  # Ensure x-axis is treated as categorical

    return fig




def create_pdf(figs):
    c = canvas.Canvas("plots_report.pdf", pagesize=letter)
    c.drawString(100, 750, "Whatsapp Chat Analysis Report")  # Title

    # Insert plots into PDF
    for fig in figs:
        c.drawImage(fig, 90, 200, width=400, height=500)
        c.showPage()  # Start a new page for each plot

    c.save()  # Save PDF

def export(selected_user,df):

    # fig1, _ = most_busy_users(df)
    # fig2 = most_common_words(selected_user, df)
    # fig3 = emoji_helper(selected_user, df)
    # fig4 = monthly_timeline(selected_user, df)
    # fig5 = daily_timeline(selected_user, df)
    # fig6 = week_activity_map(selected_user, df)
    # fig7 = month_activity_map(selected_user, df)
    # fig8 = greeting_farewell_analysis(selected_user, df)
    # fig9 = show_average_reply_time(df)
    # fig10, _ = message_count_aggregated_graph(df)
    # fig11, _ = conversation_starter_graph(df)
    # fig12 = conversation_size_aggregated_graph(df)
    # fig13, fig14 = analyze_and_plot_sentiment(selected_user, df)
    # fig15, _,_ = create_messages_per_week_graph(df)
    # fig16 = create_average_wpm_graph(df)
    # fig17 = calculate_monthly_sentiment_trend(df)

    create_pdf(["exports/charts/average_conversation_size.png", "exports/charts/average_reply_time.png", "exports/charts/commonwords.png", "exports/charts/conversation_starter_winner.png","exports/charts/emojis.png","exports/charts/fig1.png","exports/charts/greetings.png","exports/charts/month_activity.png","exports/charts/most_message_winner.png","exports/charts/negative.png","exports/charts/positive.png","exports/charts/week_activity.png"])

    st.toast('Data Exported successfully! Now you can download :)')

def calculate_average_late_reply_time(df, threshold_hours=48):
    """
    Calculate the average late reply time for all users and display it with a graph.
    
    Args:
        df: DataFrame containing the chat data
        threshold_hours: Number of hours to consider a reply as "late" (default: 48 hours)
        
    Returns:
        A plotly figure showing the average late reply times for all users
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Filter out messages containing media
    omitted_strings = ["image omitted", "media omitted", "video omitted"]
    df_copy = df_copy[~df_copy['message'].str.lower().str.contains('|'.join(omitted_strings))]
    
    # Ensure we have Reply Time calculated
    if 'Reply Time' not in df_copy.columns:
        # Ordinal encoders will encode each user with its own number
        user_encoder = OrdinalEncoder()
        df_copy['User Code'] = user_encoder.fit_transform(df_copy['user'].values.reshape(-1, 1))

        # Find replies
        message_senders = df_copy['User Code'].values
        sender_changed = (np.roll(message_senders, 1) - message_senders).reshape(1, -1)[0] != 0
        sender_changed[0] = False
        is_reply = sender_changed & ~df_copy['user'].eq('group_notification')

        df_copy['Is Reply'] = is_reply

        # Calculate times based on replies
        reply_times, indices = preprocessor.calculate_times_on_trues(df_copy, 'Is Reply')
        reply_times_df_list = []
        reply_time_index = 0
        for i in range(0, len(df_copy)):
            if i in indices:
                reply_times_df_list.append(reply_times[reply_time_index].astype("timedelta64[m]").astype("float"))
                reply_time_index += 1
            else:
                reply_times_df_list.append(0)

        df_copy['Reply Time'] = reply_times_df_list
    
    # Filter for late replies (greater than threshold)
    threshold_minutes = threshold_hours * 60
    late_replies = df_copy[df_copy['Reply Time'] > threshold_minutes]
    
    # Calculate average late reply time for each user
    avg_late_reply_times = late_replies.groupby('user')['Reply Time'].mean().reset_index()
    
    # Convert minutes to hours for better readability
    avg_late_reply_times['Reply Time (Hours)'] = avg_late_reply_times['Reply Time'] / 60
    
    # Sort by average reply time in descending order
    avg_late_reply_times = avg_late_reply_times.sort_values('Reply Time (Hours)', ascending=False)
    
    # Create a bar chart
    fig = px.bar(
        avg_late_reply_times, 
        x='user', 
        y='Reply Time (Hours)',
        title=f'Average Late Reply Time by User (Threshold: {threshold_hours} hours)',
        labels={'user': 'User', 'Reply Time (Hours)': 'Average Reply Time (Hours)'},
        color='Reply Time (Hours)',
        color_continuous_scale='Viridis'
    )
    
    # Add annotations for the values
    for i, row in avg_late_reply_times.iterrows():
        fig.add_annotation(
            x=row['user'],
            y=row['Reply Time (Hours)'],
            text=f"{row['Reply Time (Hours)']:.1f}h",
            showarrow=False,
            yshift=10
        )
    
    # Update layout for better readability
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_title='Average Reply Time (Hours)',
        xaxis_title='User',
        height=600,
        width=800
    )
    
    # Calculate overall average late reply time
    overall_avg = avg_late_reply_times['Reply Time (Hours)'].mean()
    
    # Add a horizontal line for the overall average
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=overall_avg,
        x1=len(avg_late_reply_times)-0.5,
        y1=overall_avg,
        line=dict(
            color="red",
            width=2,
            dash="dash",
        )
    )
    
    # Add annotation for the overall average
    fig.add_annotation(
        x=len(avg_late_reply_times)/2,
        y=overall_avg,
        text=f"Overall Average: {overall_avg:.1f} hours",
        showarrow=False,
        yshift=-20,
        font=dict(color="red")
    )
    
    fig.write_image("exports/charts/average_late_reply_time.png")
    return fig, avg_late_reply_times, overall_avg

def analyze_conversation_momentum(df):
    """
    Analyzes how conversations gain and lose momentum by measuring 
    message frequency changes within conversations.
    
    Returns:
        Dictionary with conversation momentum metrics
    """
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # Reset index to avoid ambiguity with 'date' column
    if df_copy.index.name == 'date':
        # Reset index but don't add the index as a column
        df_copy = df_copy.reset_index(drop=True)
    
    # Ensure we have conversation codes
    if 'Conv code' not in df_copy.columns:
        return None
    
    # Group by conversation and calculate metrics
    conv_metrics = []
    
    for conv_code, conv_df in df_copy.groupby('Conv code'):
        if len(conv_df) < 3:  # Skip very short conversations
            continue
            
        # Sort by date within conversation (using column, not index)
        conv_df = conv_df.sort_values('date')
        
        # Calculate time differences between consecutive messages
        conv_df['time_diff'] = conv_df['date'].diff().dt.total_seconds() / 60  # in minutes
        
        # Calculate metrics
        start_time = conv_df['date'].min()
        end_time = conv_df['date'].max()
        duration = (end_time - start_time).total_seconds() / 60  # in minutes
        msg_count = len(conv_df)
        
        # Skip conversations with zero duration (all messages at same time)
        if duration == 0:
            continue
            
        # Calculate message rate (messages per minute)
        msg_rate = msg_count / duration if duration > 0 else 0
        
        # Calculate acceleration (change in message rate over time)
        # Split conversation into thirds to see if it speeds up or slows down
        thirds = np.array_split(conv_df, 3)
        
        rates = []
        for third in thirds:
            if len(third) <= 1:
                rates.append(0)
                continue
                
            third_duration = (third['date'].max() - third['date'].min()).total_seconds() / 60
            third_rate = len(third) / third_duration if third_duration > 0 else 0
            rates.append(third_rate)
        
        # Calculate momentum (positive means conversation accelerated, negative means it slowed down)
        momentum = 0
        if len(rates) == 3 and rates[0] > 0:
            momentum = (rates[2] - rates[0]) / rates[0]  # Normalized change in rate
        
        # Get participants
        participants = conv_df['user'].unique().tolist()
        
        # Add to metrics
        conv_metrics.append({
            'conv_code': conv_code,
            'start_time': start_time,
            'end_time': end_time,
            'duration_minutes': duration,
            'message_count': msg_count,
            'message_rate': msg_rate,
            'momentum': momentum,
            'participants': participants,
            'participant_count': len(participants)
        })
    
    # Convert to DataFrame
    momentum_df = pd.DataFrame(conv_metrics)
    
    # Calculate additional metrics
    if not momentum_df.empty:
        avg_momentum = momentum_df['momentum'].mean()
        positive_momentum_pct = (momentum_df['momentum'] > 0).mean() * 100
        
        # Find users who tend to be in high-momentum conversations
        user_momentum = {}
        for _, row in momentum_df.iterrows():
            for user in row['participants']:
                if user not in user_momentum:
                    user_momentum[user] = []
                user_momentum[user].append(row['momentum'])
        
        user_avg_momentum = {user: np.mean(momentums) for user, momentums in user_momentum.items()}
        momentum_starters = sorted(user_avg_momentum.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'conversation_metrics': momentum_df,
            'avg_momentum': avg_momentum,
            'positive_momentum_pct': positive_momentum_pct,
            'user_momentum': user_avg_momentum,
            'momentum_starters': momentum_starters
        }
    
    return None

def analyze_topic_switching(df):
    """
    Analyzes how often conversations switch topics by detecting 
    shifts in vocabulary.
    
    Returns:
        Dictionary with topic switching metrics
    """
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # Reset index to avoid ambiguity with 'date' column
    if df_copy.index.name == 'date':
        # Reset index but don't add the index as a column
        df_copy = df_copy.reset_index(drop=True)
    
    # Ensure we have conversation codes
    if 'Conv code' not in df_copy.columns:
        return None
    
    # Load stop words
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = set(f.read().splitlines())
    
    # Process each conversation
    topic_switches = []
    user_switches = {}
    
    for conv_code, conv_df in df_copy.groupby('Conv code'):
        if len(conv_df) < 5:  # Skip very short conversations
            continue
            
        # Sort by date within conversation
        conv_df = conv_df.sort_values('date')
        
        # Process messages to extract keywords
        processed_messages = []
        
        for _, row in conv_df.iterrows():
            # Tokenize and filter out stop words
            words = [word.lower() for word in row['message'].split() 
                    if word.lower() not in stop_words and len(word) > 2]
            processed_messages.append(words)
        
        # Calculate similarity between consecutive messages
        similarities = []
        switch_points = []
        
        for i in range(1, len(processed_messages)):
            # Skip if either message has no words
            if not processed_messages[i] or not processed_messages[i-1]:
                similarities.append(1.0)  # Assume no change
                continue
                
            # Calculate Jaccard similarity (intersection over union)
            set1 = set(processed_messages[i-1])
            set2 = set(processed_messages[i])
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            similarity = intersection / union if union > 0 else 1.0
            similarities.append(similarity)
            
            # Consider it a topic switch if similarity is below threshold
            if similarity < 0.1:  # Threshold can be adjusted
                switch_points.append(i)
                
                # Record who initiated the switch
                switcher = conv_df.iloc[i]['user']
                if switcher not in user_switches:
                    user_switches[switcher] = 0
                user_switches[switcher] += 1
        
        # Calculate metrics
        switch_count = len(switch_points)
        switch_rate = switch_count / len(conv_df) if len(conv_df) > 0 else 0
        
        topic_switches.append({
            'conv_code': conv_code,
            'message_count': len(conv_df),
            'switch_count': switch_count,
            'switch_rate': switch_rate,
            'switch_points': switch_points
        })
    
    # Convert to DataFrame
    switches_df = pd.DataFrame(topic_switches)
    
    # Calculate additional metrics
    if not switches_df.empty:
        avg_switch_rate = switches_df['switch_rate'].mean()
        top_switchers = sorted(user_switches.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'topic_switches': switches_df,
            'avg_switch_rate': avg_switch_rate,
            'user_switches': user_switches,
            'top_switchers': top_switchers
        }
    
    return None

def analyze_initiator_responder_dynamics(df):
    """
    Analyzes who typically initiates conversations and who responds,
    revealing relationship dynamics.
    
    Returns:
        Dictionary with initiator-responder metrics
    """
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # Reset index to avoid ambiguity with 'date' column
    if df_copy.index.name == 'date':
        # Reset index but don't add the index as a column
        df_copy = df_copy.reset_index(drop=True)
    
    # Ensure we have conversation codes
    if 'Conv code' not in df_copy.columns or 'Conv change' not in df_copy.columns:
        return None
    
    # Initialize metrics
    initiations = {}
    responses = {}
    response_times = {}
    user_pairs = {}
    
    # Process each conversation
    for conv_code, conv_df in df_copy.groupby('Conv code'):
        if len(conv_df) < 2:  # Skip single-message conversations
            continue
            
        # Sort by date within conversation
        conv_df = conv_df.sort_values('date')
        
        # Get initiator (first message in conversation)
        initiator = conv_df.iloc[0]['user']
        if initiator not in initiations:
            initiations[initiator] = 0
        initiations[initiator] += 1
        
        # Get first responder (second message, if by different user)
        if len(conv_df) > 1 and conv_df.iloc[1]['user'] != initiator:
            responder = conv_df.iloc[1]['user']
            
            # Record response
            if responder not in responses:
                responses[responder] = 0
            responses[responder] += 1
            
            # Record response time
            response_time = (conv_df.iloc[1]['date'] - conv_df.iloc[0]['date']).total_seconds() / 60
            if responder not in response_times:
                response_times[responder] = []
            response_times[responder].append(response_time)
            
            # Record initiator-responder pair
            pair = f"{initiator} → {responder}"
            if pair not in user_pairs:
                user_pairs[pair] = 0
            user_pairs[pair] += 1
    
    # Calculate average response times
    avg_response_times = {user: np.mean(times) for user, times in response_times.items()}
    
    # Calculate initiation-response ratios
    all_users = set(initiations.keys()).union(set(responses.keys()))
    initiation_response_ratios = {}
    
    for user in all_users:
        init_count = initiations.get(user, 0)
        resp_count = responses.get(user, 0)
        
        # Avoid division by zero
        if resp_count > 0:
            ratio = init_count / resp_count
        elif init_count > 0:
            ratio = float('inf')  # Only initiates, never responds
        else:
            ratio = 0  # Neither initiates nor responds
            
        initiation_response_ratios[user] = ratio
    
    # Sort results
    top_initiators = sorted(initiations.items(), key=lambda x: x[1], reverse=True)
    top_responders = sorted(responses.items(), key=lambda x: x[1], reverse=True)
    fastest_responders = sorted(avg_response_times.items(), key=lambda x: x[1])
    most_common_pairs = sorted(user_pairs.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'initiations': initiations,
        'responses': responses,
        'avg_response_times': avg_response_times,
        'initiation_response_ratios': initiation_response_ratios,
        'top_initiators': top_initiators,
        'top_responders': top_responders,
        'fastest_responders': fastest_responders,
        'most_common_pairs': most_common_pairs
    }

def analyze_red_green_flags(df):
    """
    Analyzes conversations for GenZ red flags and green flags.
    
    Returns:
        Dictionary with red/green flag metrics
    """
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # Reset index if needed
    if df_copy.index.name == 'date':
        df_copy = df_copy.reset_index(drop=True)
    
    # Define red flag and green flag patterns (both English and Hindi/Hinglish)
    red_flags = {
        'ghosting': ['ghosted', 'ghosting', 'no reply', 'not replying', 'ignoring', 'ignore kiya'],
        'dry_texting': ['k', 'ok', 'hmm', 'hm', 'oh', 'achha', 'acha', 'thik hai', 'theek hai'],
        'late_replies': ['sorry for late reply', 'reply dene mein der', 'late reply', 'der se reply'],
        'mood_swings': ['mood swing', 'mood off', 'mood kharab', 'irritated', 'annoyed', 'gussa'],
        'possessiveness': ['jealous', 'possessive', 'controlling', 'nazar mat daal', 'kiski baat kar raha hai'],
        'breadcrumbing': ['maybe later', 'baad mein dekhte', 'will see', 'not sure', 'pata nahi']
    }
    
    green_flags = {
        'active_listening': ['i understand', 'samajh gaya', 'get it', 'makes sense', 'i see'],
        'respect': ['respect', 'izzat', 'sorry', 'my bad', 'galti meri thi', 'maaf kardo'],
        'communication': ['let\'s talk', 'baat karte hain', 'discuss', 'clear karna chahta hoon'],
        'appreciation': ['proud of you', 'well done', 'badhiya', 'sahi hai', 'awesome', 'mast'],
        'emotional_support': ['here for you', 'feel better', 'chinta mat karo', 'tension mat lo'],
        'consistency': ['always', 'hamesha', 'everyday', 'roz', 'regularly']
    }
    
    # Initialize counters
    user_red_flags = {user: {flag: 0 for flag in red_flags} for user in df_copy['user'].unique()}
    user_green_flags = {user: {flag: 0 for flag in green_flags} for user in df_copy['user'].unique()}
    
    # Count flags in messages
    for _, row in df_copy.iterrows():
        user = row['user']
        message = row['message'].lower()
        
        # Check for red flags
        for flag_type, patterns in red_flags.items():
            if any(pattern in message for pattern in patterns):
                user_red_flags[user][flag_type] += 1
        
        # Check for green flags
        for flag_type, patterns in green_flags.items():
            if any(pattern in message for pattern in patterns):
                user_green_flags[user][flag_type] += 1
    
    # Calculate total flags per user
    user_total_red = {user: sum(flags.values()) for user, flags in user_red_flags.items()}
    user_total_green = {user: sum(flags.values()) for user, flags in user_green_flags.items()}
    
    # Calculate flag ratios (green-to-red ratio)
    user_flag_ratios = {}
    for user in user_total_red:
        red_count = user_total_red[user]
        green_count = user_total_green[user]
        
        if red_count > 0:
            ratio = green_count / red_count
        elif green_count > 0:
            ratio = float('inf')  # All green, no red
        else:
            ratio = 0  # No flags detected
            
        user_flag_ratios[user] = ratio
    
    # Get most common flag types
    all_red_flags = {flag_type: sum(user_flags[flag_type] for user_flags in user_red_flags.values()) 
                    for flag_type in red_flags}
    all_green_flags = {flag_type: sum(user_flags[flag_type] for user_flags in user_green_flags.values()) 
                      for flag_type in green_flags}
    
    # Sort users by flag counts
    most_red_flags = sorted(user_total_red.items(), key=lambda x: x[1], reverse=True)
    most_green_flags = sorted(user_total_green.items(), key=lambda x: x[1], reverse=True)
    best_ratios = sorted(user_flag_ratios.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'user_red_flags': user_red_flags,
        'user_green_flags': user_green_flags,
        'user_total_red': user_total_red,
        'user_total_green': user_total_green,
        'user_flag_ratios': user_flag_ratios,
        'all_red_flags': all_red_flags,
        'all_green_flags': all_green_flags,
        'most_red_flags': most_red_flags,
        'most_green_flags': most_green_flags,
        'best_ratios': best_ratios
    }

def analyze_vibe_check(df):
    """
    Performs a GenZ-style "vibe check" on conversations and users.
    
    Returns:
        Dictionary with vibe metrics
    """
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # Reset index if needed
    if df_copy.index.name == 'date':
        df_copy = df_copy.reset_index(drop=True)
    
    # Define vibe indicators (both English and Hindi/Hinglish)
    positive_vibes = [
        'lit', 'fire', 'aag', 'vibe', 'mood', 'slay', 'queen', 'king', 'goat', 
        'legend', 'cool', 'awesome', 'mast', 'badhiya', 'zabardast', 'dope', 
        'sick', 'epic', 'love', 'pyaar', 'lob', 'xoxo', 'haha', 'lol', 'lmao',
        'rofl', 'lmfao', '😂', '🔥', '❤️', '💯', '👑', '🙌', '✨', '😍'
    ]
    
    negative_vibes = [
        'cringe', 'ew', 'eww', 'yikes', 'yuck', 'gross', 'ganda', 'bekar', 
        'boring', 'lame', 'mid', 'basic', 'chee', 'chi', 'ugh', 'meh', 'whatever',
        'whatever', 'idgaf', 'idc', 'don\'t care', 'don\'t give', 'whatever', 
        'whatever', '🙄', '😒', '💀', '😬', '🤮', '👎'
    ]
    
    # Initialize counters
    user_vibes = {user: {'positive': 0, 'negative': 0} for user in df_copy['user'].unique()}
    
    # Count vibe indicators in messages
    for _, row in df_copy.iterrows():
        user = row['user']
        message = row['message'].lower()
        
        # Check for positive vibes
        for vibe in positive_vibes:
            if vibe in message:
                user_vibes[user]['positive'] += 1
        
        # Check for negative vibes
        for vibe in negative_vibes:
            if vibe in message:
                user_vibes[user]['negative'] += 1
    
    # Calculate vibe ratios and scores
    vibe_scores = {}
    for user, vibes in user_vibes.items():
        positive = vibes['positive']
        negative = vibes['negative']
        total = positive + negative
        
        if total > 0:
            ratio = positive / total
            # Scale to -100 to +100 range
            score = (ratio * 2 - 1) * 100
        else:
            ratio = 0
            score = 0
            
        vibe_scores[user] = {
            'positive_count': positive,
            'negative_count': negative,
            'total_count': total,
            'positive_ratio': ratio,
            'vibe_score': score
        }
    
    # Sort users by vibe score
    best_vibes = sorted([(user, data['vibe_score']) for user, data in vibe_scores.items()], 
                        key=lambda x: x[1], reverse=True)
    
    # Calculate overall chat vibe
    total_positive = sum(vibes['positive'] for vibes in user_vibes.values())
    total_negative = sum(vibes['negative'] for vibes in user_vibes.values())
    total_vibes = total_positive + total_negative
    
    if total_vibes > 0:
        overall_ratio = total_positive / total_vibes
        overall_score = (overall_ratio * 2 - 1) * 100
    else:
        overall_ratio = 0
        overall_score = 0
    
    return {
        'user_vibes': user_vibes,
        'vibe_scores': vibe_scores,
        'best_vibes': best_vibes,
        'overall_ratio': overall_ratio,
        'overall_score': overall_score
    }

def analyze_genz_slang(df):
    """
    Analyzes the use of GenZ slang and trending terms in conversations.
    
    Returns:
        Dictionary with slang usage metrics
    """
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # Reset index if needed
    if df_copy.index.name == 'date':
        df_copy = df_copy.reset_index(drop=True)
    
    # Define GenZ slang categories (both English and Hindi/Hinglish)
    slang_categories = {
        'abbreviations': ['lol', 'lmao', 'rofl', 'idk', 'idc', 'tbh', 'fyi', 'imo', 'btw', 'omg', 'wtf', 'af'],
        'internet_slang': ['sus', 'cap', 'no cap', 'based', 'cringe', 'simp', 'vibe', 'slay', 'stan', 'tea', 'shade'],
        'hinglish_slang': ['bro', 'yaar', 'bhai', 'dude', 'scene', 'chill', 'chill maar', 'bindaas', 'lit af'],
        'emoji_usage': ['😂', '💀', '🔥', '👀', '🙄', '😭', '✨', '💯', '🥺', '👉👈', '🤡'],
        'trendy_phrases': ['main character energy', 'living my best life', 'rent free', 'understood the assignment',
                          'it\'s giving', 'ate and left no crumbs', 'periodt', 'as you should']
    }
    
    # Initialize counters
    user_slang = {user: {category: 0 for category in slang_categories} 
                 for user in df_copy['user'].unique()}
    
    # Count slang usage in messages
    for _, row in df_copy.iterrows():
        user = row['user']
        message = row['message'].lower()
        
        # Check for slang in each category
        for category, terms in slang_categories.items():
            for term in terms:
                if term in message:
                    user_slang[user][category] += 1
    
    # Calculate total slang usage per user
    user_total_slang = {user: sum(categories.values()) for user, categories in user_slang.items()}
    
    # Calculate slang usage per message
    user_message_counts = df_copy['user'].value_counts().to_dict()
    user_slang_density = {user: user_total_slang[user] / user_message_counts[user] 
                         for user in user_total_slang if user_message_counts[user] > 0}
    
    # Get most used slang categories
    category_totals = {category: sum(user_cats[category] for user_cats in user_slang.values()) 
                      for category in slang_categories}
    
    # Sort users by slang usage
    most_slang_users = sorted(user_total_slang.items(), key=lambda x: x[1], reverse=True)
    highest_density_users = sorted(user_slang_density.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'user_slang': user_slang,
        'user_total_slang': user_total_slang,
        'user_slang_density': user_slang_density,
        'category_totals': category_totals,
        'most_slang_users': most_slang_users,
        'highest_density_users': highest_density_users
    }

def analyze_reply_pairs(df):
    """
    Analyzes which users reply to each other most frequently.
    
    Returns:
        Dictionary with reply pair metrics
    """
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # Reset index if needed
    if df_copy.index.name == 'date':
        df_copy = df_copy.reset_index(drop=True)
    
    # Ensure we have the necessary data
    if 'user' not in df_copy.columns:
        return None
    
    # Filter out group notifications
    df_copy = df_copy[df_copy['user'] != 'group_notification']
    
    # Create a shifted dataframe to identify who replied to whom
    df_copy['previous_user'] = df_copy['user'].shift(1)
    
    # Filter out self-replies (same user sending consecutive messages)
    reply_pairs = df_copy[df_copy['user'] != df_copy['previous_user']]
    
    # Count reply pairs
    pair_counts = reply_pairs.groupby(['user', 'previous_user']).size().reset_index(name='count')
    
    # Sort by count in descending order
    pair_counts = pair_counts.sort_values('count', ascending=False)
    
    # Calculate total replies for each user
    user_total_replies = pair_counts.groupby('user')['count'].sum().to_dict()
    
    # Calculate percentage of replies for each pair
    pair_counts['percentage'] = pair_counts.apply(
        lambda row: (row['count'] / user_total_replies[row['user']]) * 100 
        if row['user'] in user_total_replies and user_total_replies[row['user']] > 0 
        else 0, 
        axis=1
    )
    
    # Format the pairs for better readability
    formatted_pairs = []
    for _, row in pair_counts.iterrows():
        formatted_pairs.append({
            'replier': row['user'],
            'replied_to': row['previous_user'],
            'count': row['count'],
            'percentage': row['percentage'],
            'pair_text': f"{row['user']} → {row['previous_user']}"
        })
    
    # Find the most frequent replier for each user
    most_frequent_repliers = {}
    for user in df_copy['user'].unique():
        user_pairs = [p for p in formatted_pairs if p['replied_to'] == user]
        if user_pairs:
            most_frequent_replier = max(user_pairs, key=lambda x: x['count'])
            most_frequent_repliers[user] = most_frequent_replier
    
    # Find the person each user replies to most
    most_replied_to = {}
    for user in df_copy['user'].unique():
        user_pairs = [p for p in formatted_pairs if p['replier'] == user]
        if user_pairs:
            most_replied = max(user_pairs, key=lambda x: x['count'])
            most_replied_to[user] = most_replied
    
    return {
        'pair_counts': formatted_pairs,
        'most_frequent_repliers': most_frequent_repliers,
        'most_replied_to': most_replied_to
    }

def analyze_conversation_influence(df):
    """
    Analyzes which users have the most influence on conversation direction and engagement.
    
    Returns:
        Dictionary with influence metrics
    """
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # Reset index if needed
    if df_copy.index.name == 'date':
        df_copy = df_copy.reset_index(drop=True)
    
    # Ensure we have conversation codes
    if 'Conv code' not in df_copy.columns:
        return None
    
    # Initialize metrics
    user_metrics = {user: {
        'messages': 0,
        'conversations': set(),
        'responses_received': 0,
        'conversation_extensions': 0,
        'avg_responses_per_message': 0,
        'conversation_revival': 0
    } for user in df_copy['user'].unique() if user != 'group_notification'}
    
    # Process each conversation
    for conv_code, conv_df in df_copy.groupby('Conv code'):
        if len(conv_df) < 3:  # Skip very short conversations
            continue
            
        # Sort by date
        conv_df = conv_df.sort_values('date')
        
        # Get unique participants
        participants = conv_df['user'].unique()
        
        # Update conversation count for each participant
        for user in participants:
            if user != 'group_notification':
                user_metrics[user]['conversations'].add(conv_code)
                user_metrics[user]['messages'] += len(conv_df[conv_df['user'] == user])
        
        # Analyze message flow
        for i in range(1, len(conv_df)):
            current_user = conv_df.iloc[i]['user']
            prev_user = conv_df.iloc[i-1]['user']
            
            # Skip group notifications
            if current_user == 'group_notification' or prev_user == 'group_notification':
                continue
                
            # Count responses received
            user_metrics[prev_user]['responses_received'] += 1
            
            # Check for conversation extensions (3+ messages after user's message)
            if i < len(conv_df) - 2:
                next_users = conv_df.iloc[i:i+3]['user'].tolist()
                if len(set(next_users)) > 1 and current_user != prev_user:
                    user_metrics[prev_user]['conversation_extensions'] += 1
        
        # Check for conversation revivals (messages after 3+ hours of inactivity)
        for i in range(1, len(conv_df)):
            time_diff = (conv_df.iloc[i]['date'] - conv_df.iloc[i-1]['date']).total_seconds() / 3600
            if time_diff > 3:  # 3+ hours of inactivity
                revival_user = conv_df.iloc[i]['user']
                if revival_user != 'group_notification':
                    user_metrics[revival_user]['conversation_revival'] += 1
    
    # Calculate derived metrics
    for user, metrics in user_metrics.items():
        # Convert conversation set to count
        metrics['conversation_count'] = len(metrics['conversations'])
        del metrics['conversations']  # Remove the set to make it JSON serializable
        
        # Calculate average responses per message
        if metrics['messages'] > 0:
            metrics['avg_responses_per_message'] = metrics['responses_received'] / metrics['messages']
        
        # Calculate influence score (weighted combination of metrics)
        metrics['influence_score'] = (
            0.3 * metrics['avg_responses_per_message'] * 10 +  # Normalized to ~0-10 scale
            0.3 * metrics['conversation_extensions'] / max(metrics['messages'], 1) * 10 +
            0.4 * metrics['conversation_revival'] / max(metrics['conversation_count'], 1) * 10
        )
    
    # Sort users by influence score
    most_influential = sorted([(user, metrics['influence_score']) 
                              for user, metrics in user_metrics.items()], 
                             key=lambda x: x[1], reverse=True)
    
    return {
        'user_metrics': user_metrics,
        'most_influential': most_influential
    }

def analyze_mood_shifters(df):
    """
    Identifies users who tend to shift the emotional tone of conversations.
    
    Returns:
        Dictionary with mood shifter metrics
    """
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # Reset index if needed
    if df_copy.index.name == 'date':
        df_copy = df_copy.reset_index(drop=True)
    
    # Ensure we have conversation codes
    if 'Conv code' not in df_copy.columns:
        return None
    
    # Add sentiment scores to messages
    df_copy['sentiment_score'] = df_copy['message'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Initialize metrics
    user_shifts = {user: {
        'positive_shifts': 0,
        'negative_shifts': 0,
        'total_messages': 0,
        'avg_sentiment': 0
    } for user in df_copy['user'].unique() if user != 'group_notification'}
    
    # Process each conversation
    for conv_code, conv_df in df_copy.groupby('Conv code'):
        if len(conv_df) < 5:  # Skip very short conversations
            continue
            
        # Sort by date
        conv_df = conv_df.sort_values('date')
        
        # Calculate rolling average sentiment (3-message window)
        conv_df['rolling_sentiment'] = conv_df['sentiment_score'].rolling(window=3, min_periods=1).mean()
        
        # Detect significant sentiment shifts
        for i in range(3, len(conv_df)):
            current_user = conv_df.iloc[i]['user']
            
            # Skip group notifications
            if current_user == 'group_notification':
                continue
                
            # Calculate sentiment change
            before_sentiment = conv_df.iloc[i-3:i]['rolling_sentiment'].mean()
            after_sentiment = conv_df.iloc[i:i+3]['sentiment_score'].mean() if i+3 <= len(conv_df) else conv_df.iloc[i:]['sentiment_score'].mean()
            
            sentiment_shift = after_sentiment - before_sentiment
            
            # Record significant shifts (threshold can be adjusted)
            if abs(sentiment_shift) > 0.2:
                if sentiment_shift > 0:
                    user_shifts[current_user]['positive_shifts'] += 1
                else:
                    user_shifts[current_user]['negative_shifts'] += 1
        
        # Update message counts and average sentiment
        for user in user_shifts:
            user_messages = conv_df[conv_df['user'] == user]
            user_shifts[user]['total_messages'] += len(user_messages)
            if len(user_messages) > 0:
                user_shifts[user]['avg_sentiment'] += user_messages['sentiment_score'].mean() * len(user_messages)
    
    # Finalize metrics
    for user, metrics in user_shifts.items():
        if metrics['total_messages'] > 0:
            metrics['avg_sentiment'] /= metrics['total_messages']
            
        metrics['total_shifts'] = metrics['positive_shifts'] + metrics['negative_shifts']
        
        if metrics['total_messages'] > 0:
            metrics['shift_rate'] = metrics['total_shifts'] / metrics['total_messages']
        else:
            metrics['shift_rate'] = 0
            
        if metrics['total_shifts'] > 0:
            metrics['positive_shift_ratio'] = metrics['positive_shifts'] / metrics['total_shifts']
        else:
            metrics['positive_shift_ratio'] = 0
    
    # Sort users by total shifts and shift rate
    top_shifters = sorted([(user, metrics['total_shifts'], metrics['shift_rate']) 
                          for user, metrics in user_shifts.items() if metrics['total_messages'] > 10], 
                         key=lambda x: (x[1], x[2]), reverse=True)
    
    # Identify mood lifters and dampeners
    mood_lifters = sorted([(user, metrics['positive_shifts'], metrics['positive_shift_ratio']) 
                          for user, metrics in user_shifts.items() 
                          if metrics['total_shifts'] > 5 and metrics['positive_shift_ratio'] > 0.6], 
                         key=lambda x: (x[1], x[2]), reverse=True)
    
    mood_dampeners = sorted([(user, metrics['negative_shifts'], 1 - metrics['positive_shift_ratio']) 
                            for user, metrics in user_shifts.items() 
                            if metrics['total_shifts'] > 5 and metrics['positive_shift_ratio'] < 0.4], 
                           key=lambda x: (x[1], x[2]), reverse=True)
    
    return {
        'user_shifts': user_shifts,
        'top_shifters': top_shifters,
        'mood_lifters': mood_lifters,
        'mood_dampeners': mood_dampeners
    }

def analyze_conversation_compatibility(df):
    """
    Analyzes which pairs of users have the most engaging and positive interactions.
    
    Returns:
        Dictionary with compatibility metrics
    """
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # Reset index if needed
    if df_copy.index.name == 'date':
        df_copy = df_copy.reset_index(drop=True)
    
    # Filter out group notifications
    df_copy = df_copy[df_copy['user'] != 'group_notification']
    
    # Get all unique users
    all_users = df_copy['user'].unique()
    
    # Initialize compatibility metrics for all user pairs
    user_pairs = {}
    for i, user1 in enumerate(all_users):
        for user2 in all_users[i+1:]:  # Only process each pair once
            pair_key = f"{user1}_{user2}"
            user_pairs[pair_key] = {
                'user1': user1,
                'user2': user2,
                'conversations': 0,
                'messages': 0,
                'direct_replies': 0,
                'avg_reply_time': 0,
                'sentiment_alignment': 0,
                'topic_similarity': 0,
                'compatibility_score': 0
            }
    
    # Add sentiment scores to messages
    df_copy['sentiment_score'] = df_copy['message'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    # Create a shifted dataframe to identify direct replies
    df_copy['previous_user'] = df_copy['user'].shift(1)
    df_copy['previous_sentiment'] = df_copy['sentiment_score'].shift(1)
    df_copy['reply_time'] = (df_copy['date'] - df_copy['date'].shift(1)).dt.total_seconds() / 60
    
    # Process conversations
    if 'Conv code' in df_copy.columns:
        for conv_code, conv_df in df_copy.groupby('Conv code'):
            # Get participants in this conversation
            participants = conv_df['user'].unique()
            
            # Update conversation count for each pair in this conversation
            for i, user1 in enumerate(participants):
                for user2 in participants[i+1:]:
                    pair_key = f"{user1}_{user2}"
                    if pair_key in user_pairs:
                        user_pairs[pair_key]['conversations'] += 1
                        
                    # Also count the reverse pair
                    reverse_key = f"{user2}_{user1}"
                    if reverse_key in user_pairs:
                        user_pairs[reverse_key]['conversations'] += 1
    
    # Process direct replies
    direct_replies = df_copy[df_copy['user'] != df_copy['previous_user']]
    
    for _, row in direct_replies.iterrows():
        user1 = row['previous_user']
        user2 = row['user']
        
        # Skip if either user is missing
        if pd.isna(user1) or pd.isna(user2):
            continue
        
        # Get pair key (ensure consistent ordering)
        if user1 < user2:
            pair_key = f"{user1}_{user2}"
        else:
            pair_key = f"{user2}_{user1}"
        
        # Skip if pair not in our dictionary
        if pair_key not in user_pairs:
            continue
        
        # Update metrics
        user_pairs[pair_key]['direct_replies'] += 1
        user_pairs[pair_key]['messages'] += 1
        
        # Update reply time if valid
        if not pd.isna(row['reply_time']) and row['reply_time'] > 0 and row['reply_time'] < 1440:  # Less than 1 day
            current_avg = user_pairs[pair_key]['avg_reply_time']
            current_count = user_pairs[pair_key]['direct_replies']
            
            # Update running average
            if current_count > 1:
                user_pairs[pair_key]['avg_reply_time'] = (current_avg * (current_count - 1) + row['reply_time']) / current_count
            else:
                user_pairs[pair_key]['avg_reply_time'] = row['reply_time']
        
        # Update sentiment alignment
        if not pd.isna(row['sentiment_score']) and not pd.isna(row['previous_sentiment']):
            # Higher score when sentiments are similar (both positive or both negative)
            sentiment_diff = abs(row['sentiment_score'] - row['previous_sentiment'])
            sentiment_alignment = 1 - min(sentiment_diff, 1)  # 0-1 scale, higher is more aligned
            
            current_alignment = user_pairs[pair_key]['sentiment_alignment']
            current_count = user_pairs[pair_key]['direct_replies']
            
            # Update running average
            if current_count > 1:
                user_pairs[pair_key]['sentiment_alignment'] = (current_alignment * (current_count - 1) + sentiment_alignment) / current_count
            else:
                user_pairs[pair_key]['sentiment_alignment'] = sentiment_alignment
    
    # Calculate compatibility scores
    for pair_key, metrics in user_pairs.items():
        if metrics['direct_replies'] > 5:  # Only consider pairs with sufficient interaction
            # Normalize reply time (faster is better, up to a point)
            reply_time_score = max(0, 1 - (metrics['avg_reply_time'] / 120))  # 0-2 hours scale
            
            # Calculate compatibility score (weighted combination of metrics)
            metrics['compatibility_score'] = (
                0.4 * metrics['sentiment_alignment'] +  # 0-1 scale
                0.3 * reply_time_score +  # 0-1 scale
                0.3 * min(1, metrics['direct_replies'] / 50)  # 0-1 scale, capped at 50 replies
            ) * 100  # Convert to 0-100 scale
    
    # Sort pairs by compatibility score
    most_compatible = sorted([(metrics['user1'], metrics['user2'], metrics['compatibility_score']) 
                             for pair_key, metrics in user_pairs.items() 
                             if metrics['direct_replies'] > 5], 
                            key=lambda x: x[2], reverse=True)
    
    return {
        'user_pairs': user_pairs,
        'most_compatible': most_compatible
    }

def generate_message_embeddings(df, model_name='all-MiniLM-L6-v2'):
    """
    Generates embeddings for messages using a sentence transformer model
    """
    # Initialize the model
    model = SentenceTransformer(model_name)
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Generate embeddings (in batches to manage memory)
    embeddings = []
    batch_size = 100
    for i in range(0, len(df_copy), batch_size):
        batch = df_copy['message'].iloc[i:i+batch_size].tolist()
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    
    # Add embeddings as a new column
    df_copy['embedding'] = embeddings
    
    return df_copy

def analyze_personality(df):
    """
    Analyzes personality traits based on message patterns
    """
    # Features that correlate with personality traits
    personality_features = {
        'openness': ['vocabulary_diversity', 'topic_diversity', 'question_frequency'],
        'conscientiousness': ['message_length', 'response_time', 'greeting_usage'],
        'extraversion': ['message_frequency', 'emoji_usage', 'exclamation_usage'],
        'agreeableness': ['positive_sentiment', 'agreement_phrases', 'thank_you_frequency'],
        'neuroticism': ['negative_sentiment', 'anxiety_words', 'uncertainty_phrases']
    }
    
    # Calculate features for each user
    user_features = {}
    for user in df['user'].unique():
        if user == 'group_notification':
            continue
            
        user_df = df[df['user'] == user]
        user_features[user] = {
            'vocabulary_diversity': calculate_vocabulary_diversity(user_df),
            'topic_diversity': calculate_topic_diversity(user_df),
            'question_frequency': calculate_question_frequency(user_df),
            'message_length': calculate_avg_message_length(user_df),
            'response_time': calculate_avg_response_time(user_df, df),
            'greeting_usage': calculate_greeting_usage(user_df),
            'message_frequency': calculate_message_frequency(user_df, df),
            'emoji_usage': calculate_emoji_usage(user_df),
            'exclamation_usage': calculate_exclamation_usage(user_df),
            'positive_sentiment': calculate_positive_sentiment(user_df),
            'agreement_phrases': calculate_agreement_phrases(user_df),
            'thank_you_frequency': calculate_thank_you_frequency(user_df),
            'negative_sentiment': calculate_negative_sentiment(user_df),
            'anxiety_words': calculate_anxiety_words(user_df),
            'uncertainty_phrases': calculate_uncertainty_phrases(user_df)
        }
    
    # Calculate personality scores
    personality_scores = {}
    for user, features in user_features.items():
        personality_scores[user] = {}
        for trait, trait_features in personality_features.items():
            # Calculate weighted average of features for this trait
            trait_score = sum(features[feature] for feature in trait_features) / len(trait_features)
            # Normalize to 0-100 scale
            personality_scores[user][trait] = min(100, max(0, trait_score * 100))
    
    return personality_scores

def summarize_conversations(df):
    """
    Generates simple summaries for conversations without requiring external models
    """
    # Group by conversation
    summaries = {}
    if 'Conv code' in df.columns:
        for conv_code, conv_df in df.groupby('Conv code'):
            if len(conv_df) < 5:  # Skip very short conversations
                continue
                
            # Get basic conversation info
            participants = conv_df['user'].unique().tolist()
            message_count = len(conv_df)
            date_range = (conv_df['date'].min(), conv_df['date'].max())
            duration = (date_range[1] - date_range[0]).total_seconds() / 3600  # hours
            
            # Get most active user
            most_active_user = conv_df['user'].value_counts().idxmax()
            most_active_count = conv_df['user'].value_counts().max()
            
            # Get sentiment of conversation
            sentiment_analyzer = SentimentIntensityAnalyzer()
            sentiment_scores = []
            for message in conv_df['message']:
                sentiment = sentiment_analyzer.polarity_scores(message)
                sentiment_scores.append(sentiment['compound'])
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            sentiment_label = "positive" if avg_sentiment > 0.05 else "negative" if avg_sentiment < -0.05 else "neutral"
            
            # Extract key topics (simple word frequency)
            all_words = []
            for message in conv_df['message']:
                words = [word.lower() for word in message.split() if len(word) > 4]
                all_words.extend(words)
            
            word_counts = {}
            for word in all_words:
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
            
            # Get top topics
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            top_topics = [word for word, count in top_words]
            
            # Create a simple summary
            summary = f"A {sentiment_label} conversation between {', '.join(participants)} lasting {duration:.1f} hours. "
            summary += f"{most_active_user} was most active with {most_active_count} messages. "
            
            if top_topics:
                summary += f"Main topics: {', '.join(top_topics)}."
            
            summaries[conv_code] = {
                'summary': summary,
                'participants': participants,
                'message_count': message_count,
                'date_range': date_range,
                'sentiment': sentiment_label,
                'top_topics': top_topics
            }
    
    return summaries

# Add these functions before the analyze_personality function

def calculate_vocabulary_diversity(df):
    """Calculate vocabulary diversity (unique words / total words)"""
    all_words = []
    for message in df['message']:
        words = message.lower().split()
        all_words.extend(words)
    
    if len(all_words) == 0:
        return 0
    
    unique_words = set(all_words)
    return len(unique_words) / max(len(all_words), 1)

def calculate_topic_diversity(df):
    """Estimate topic diversity based on message content variety"""
    # Simple implementation - could be improved with actual topic modeling
    if len(df) < 3:
        return 0
    
    # Use average cosine similarity between messages as a proxy for topic diversity
    # Lower similarity = higher diversity
    all_messages = " ".join(df['message'])
    words = set(all_messages.lower().split())
    
    # More unique words relative to message count suggests more topics
    return min(1.0, len(words) / (len(df) * 5))  # Normalize to 0-1

def calculate_question_frequency(df):
    """Calculate frequency of questions in messages"""
    question_count = 0
    for message in df['message']:
        if '?' in message:
            question_count += 1
    
    return question_count / max(len(df), 1)

def calculate_avg_message_length(df):
    """Calculate average message length in words"""
    if len(df) == 0:
        return 0
    
    total_words = 0
    for message in df['message']:
        words = message.split()
        total_words += len(words)
    
    avg_length = total_words / len(df)
    # Normalize to 0-1 scale (assuming most messages are under 50 words)
    return min(1.0, avg_length / 50)

def calculate_avg_response_time(user_df, full_df):
    """Calculate average response time"""
    # This is a simplified version - ideally would use conversation threading
    if 'Conv code' not in full_df.columns or len(user_df) < 3:
        return 0.5  # Default middle value if we can't calculate
    
    # Get conversations this user participated in
    user = user_df['user'].iloc[0]
    conv_codes = user_df['Conv code'].unique()
    
    total_response_time = 0
    response_count = 0
    
    for conv_code in conv_codes:
        try:
            # Make a copy with reset index to avoid the ambiguity
            conv_df = full_df[full_df['Conv code'] == conv_code].copy()
            
            # Ensure we have a date column
            if 'date' not in conv_df.columns and conv_df.index.name == 'date':
                # If date is only in the index, reset the index to get it as a column
                conv_df = conv_df.reset_index()
            
            # Sort by date
            conv_df = conv_df.sort_values('date')
            
            # Find messages by this user and calculate response time
            for i in range(1, len(conv_df)):
                if conv_df.iloc[i]['user'] == user and conv_df.iloc[i-1]['user'] != user:
                    # This is a response by our user
                    time_diff = (conv_df.iloc[i]['date'] - conv_df.iloc[i-1]['date']).total_seconds() / 60
                    if time_diff < 60*24:  # Only count responses within 24 hours
                        total_response_time += time_diff
                        response_count += 1
        except Exception as e:
            print(f"Error processing conversation {conv_code}: {e}")
            continue
    
    if response_count == 0:
        return 0.5  # Default
    
    avg_time = total_response_time / response_count
    # Normalize: faster responses (lower time) = higher conscientiousness
    # 0 min = 1.0, 60 min = 0.0
    return max(0, min(1.0, 1 - (avg_time / 60)))

def calculate_greeting_usage(df):
    """Calculate frequency of greetings in messages"""
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 
                'namaste', 'hola', 'sup', 'yo', 'hii', 'hiii', 'hiiii']
    
    greeting_count = 0
    for message in df['message'].str.lower():
        if any(greeting in message for greeting in greetings):
            greeting_count += 1
    
    return greeting_count / max(len(df), 1)

def calculate_message_frequency(user_df, full_df):
    """Calculate message frequency relative to group average"""
    if len(full_df) == 0 or len(user_df) == 0:
        return 0
    
    # Get date range
    date_range = (full_df['date'].max() - full_df['date'].min()).total_seconds() / 86400  # days
    if date_range < 1:
        return 0.5  # Not enough data
    
    # Calculate messages per day
    user_msgs_per_day = len(user_df) / date_range
    
    # Get average messages per day per user
    unique_users = full_df['user'].nunique()
    avg_msgs_per_day = len(full_df) / (date_range * unique_users)
    
    if avg_msgs_per_day == 0:
        return 0.5
    
    # Normalize: higher frequency = higher extraversion
    ratio = user_msgs_per_day / avg_msgs_per_day
    return min(1.0, ratio / 3)  # Cap at 3x average

def calculate_emoji_usage(df):
    """Calculate emoji usage frequency"""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    
    emoji_count = 0
    for message in df['message']:
        emojis = emoji_pattern.findall(message)
        emoji_count += len(emojis)
    
    # Normalize by message count
    return min(1.0, emoji_count / max(len(df), 1))

def calculate_exclamation_usage(df):
    """Calculate exclamation mark usage frequency"""
    exclamation_count = 0
    for message in df['message']:
        exclamation_count += message.count('!')
    
    # Normalize by message count
    return min(1.0, exclamation_count / max(len(df), 1))

def calculate_positive_sentiment(df):
    """Calculate average positive sentiment in messages"""
    if len(df) == 0:
        return 0.5
    
    sentiment_analyzer = SentimentIntensityAnalyzer()
    total_positive = 0
    
    for message in df['message']:
        sentiment = sentiment_analyzer.polarity_scores(message)
        total_positive += sentiment['pos']
    
    return total_positive / len(df)

def calculate_agreement_phrases(df):
    """Calculate frequency of agreement phrases"""
    agreement_phrases = ['yes', 'yeah', 'sure', 'agree', 'correct', 'right', 'ok', 'okay', 
                        'definitely', 'absolutely', 'indeed', 'exactly', 'true', 'yep', 'yup']
    
    agreement_count = 0
    for message in df['message'].str.lower():
        if any(phrase in message.split() for phrase in agreement_phrases):
            agreement_count += 1
    
    return agreement_count / max(len(df), 1)

def calculate_thank_you_frequency(df):
    """Calculate frequency of thank you phrases"""
    thank_phrases = ['thank', 'thanks', 'thx', 'ty', 'appreciate', 'grateful']
    
    thank_count = 0
    for message in df['message'].str.lower():
        if any(phrase in message for phrase in thank_phrases):
            thank_count += 1
    
    return thank_count / max(len(df), 1)

def calculate_negative_sentiment(df):
    """Calculate average negative sentiment in messages"""
    if len(df) == 0:
        return 0.5
    
    sentiment_analyzer = SentimentIntensityAnalyzer()
    total_negative = 0
    
    for message in df['message']:
        sentiment = sentiment_analyzer.polarity_scores(message)
        total_negative += sentiment['neg']
    
    return total_negative / len(df)

def calculate_anxiety_words(df):
    """Calculate frequency of anxiety-related words"""
    anxiety_words = ['worry', 'worried', 'anxious', 'anxiety', 'stress', 'stressed',
                    'nervous', 'fear', 'afraid', 'scared', 'panic', 'tension', 'concerned']
    
    anxiety_count = 0
    for message in df['message'].str.lower():
        if any(word in message.split() for word in anxiety_words):
            anxiety_count += 1
    
    return anxiety_count / max(len(df), 1)

def calculate_uncertainty_phrases(df):
    """Calculate frequency of uncertainty phrases"""
    uncertainty_phrases = ['maybe', 'perhaps', 'not sure', 'uncertain', 'doubt', 'confused',
                          'possibly', 'might', 'could be', 'don\'t know', 'unsure', 'unclear']
    
    uncertainty_count = 0
    for message in df['message'].str.lower():
        if any(phrase in message for phrase in uncertainty_phrases):
            uncertainty_count += 1
    
    return uncertainty_count / max(len(df), 1)

def analyze_relationship(df, user1, user2):
    """
    Analyzes the relationship between two users based on their interactions
    
    Returns:
        Dictionary with relationship metrics
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Reset index if needed, but check if the column already exists
    if df_copy.index.name == 'date':
        if 'date' not in df_copy.columns:
            df_copy = df_copy.reset_index()
        else:
            # If date is both an index and a column, just drop the index
            df_copy = df_copy.reset_index(drop=True)
    
    # Filter messages from these two users
    users_df = df_copy[(df_copy['user'] == user1) | (df_copy['user'] == user2)]
    
    # If we have conversation codes, use them
    if 'Conv code' in df_copy.columns:
        # Find conversations where both users participated
        user1_convs = set(df_copy[df_copy['user'] == user1]['Conv code'].unique())
        user2_convs = set(df_copy[df_copy['user'] == user2]['Conv code'].unique())
        shared_convs = user1_convs.intersection(user2_convs)
        
        # Filter to just those conversations
        relationship_df = df_copy[df_copy['Conv code'].isin(shared_convs)]
        conversation_count = len(shared_convs)
    else:
        # Without conversation codes, just use the messages from both users
        relationship_df = users_df
        conversation_count = 1  # Can't determine actual count
    
    # Calculate direct replies
    relationship_df = relationship_df.sort_values('date')
    relationship_df['previous_user'] = relationship_df['user'].shift(1)
    relationship_df['reply_time'] = (relationship_df['date'] - relationship_df['date'].shift(1)).dt.total_seconds() / 60
    
    # Filter to just direct replies between the two users
    direct_replies = relationship_df[
        ((relationship_df['user'] == user1) & (relationship_df['previous_user'] == user2)) |
        ((relationship_df['user'] == user2) & (relationship_df['previous_user'] == user1))
    ]
    
    # Calculate average response time
    valid_reply_times = direct_replies[
        (direct_replies['reply_time'] > 0) & 
        (direct_replies['reply_time'] < 1440)  # Less than 1 day
    ]['reply_time']
    
    avg_response_time = valid_reply_times.mean() if len(valid_reply_times) > 0 else 0
    
    # Calculate sentiment alignment
    sentiment_analyzer = SentimentIntensityAnalyzer()
    user1_sentiments = []
    user2_sentiments = []
    
    for _, row in relationship_df[relationship_df['user'] == user1].iterrows():
        sentiment = sentiment_analyzer.polarity_scores(row['message'])
        user1_sentiments.append(sentiment['compound'])
    
    for _, row in relationship_df[relationship_df['user'] == user2].iterrows():
        sentiment = sentiment_analyzer.polarity_scores(row['message'])
        user2_sentiments.append(sentiment['compound'])
    
    # Calculate correlation between sentiments
    if len(user1_sentiments) > 5 and len(user2_sentiments) > 5:
        # Truncate to same length
        min_len = min(len(user1_sentiments), len(user2_sentiments))
        correlation = np.corrcoef(user1_sentiments[:min_len], user2_sentiments[:min_len])[0, 1]
        sentiment_alignment = (correlation + 1) * 5  # Convert from -1,1 to 0-10 scale
    else:
        sentiment_alignment = 5  # Default middle value
    
    # Create timeline of message frequency
    if len(relationship_df) > 0:
        try:
            # Create a daily count of messages
            daily_counts = relationship_df.groupby(relationship_df['date'].dt.date).size().reset_index()
            daily_counts.columns = ['date', 'message_count']
            
            # Convert date back to datetime
            daily_counts['date'] = pd.to_datetime(daily_counts['date'])
            
            timeline = daily_counts
        except Exception as e:
            print(f"Error creating timeline: {e}")
            # Fallback to empty timeline
            timeline = pd.DataFrame(columns=['date', 'message_count'])
    else:
        timeline = pd.DataFrame(columns=['date', 'message_count'])
    
    # Extract common topics (simple implementation)
    common_words = []
    for message in relationship_df['message']:
        words = [word.lower() for word in message.split() if len(word) > 3]
        common_words.extend(words)
    
    word_counts = {}
    for word in common_words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    
    # Get top topics
    common_topics = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Calculate compatibility score
    compatibility_score = (
        0.4 * min(10, direct_replies.shape[0] / 10) +  # More replies = higher score (up to 10)
        0.3 * (10 - min(10, avg_response_time / 60)) +  # Faster replies = higher score
        0.3 * sentiment_alignment  # Higher sentiment alignment = higher score
    ) * 10  # Scale to 0-100
    
    return {
        'conversation_count': conversation_count,
        'direct_replies_count': len(direct_replies),
        'avg_response_time': avg_response_time,
        'sentiment_alignment': sentiment_alignment,
        'timeline': timeline,
        'common_topics': common_topics,
        'compatibility_score': compatibility_score
    }

def predict_future_activity(df, forecast_months=3):
    """
    Predicts future message activity based on historical data
    
    Args:
        df: DataFrame containing chat data
        forecast_months: Number of months to forecast
        
    Returns:
        Tuple of (plotly figure, conclusions dictionary)
    """
    try:
        # Make sure Prophet is installed
        import prophet
        from prophet import Prophet
    except ImportError:
        # If Prophet isn't installed, add it to requirements and show a message
        st.error("Prophet package is required for predictions. Please install it with: pip install prophet")
        return None, None
    
    # Prepare data for Prophet (needs 'ds' for dates and 'y' for values)
    # Group by day to get message counts
    daily_counts = df.groupby(df['date'].dt.date).size().reset_index()
    daily_counts.columns = ['ds', 'y']
    
    # Convert date to datetime
    daily_counts['ds'] = pd.to_datetime(daily_counts['ds'])
    
    # Create and fit the model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'  # Usually better for message data
    )
    
    # Add special seasonality for chat patterns if needed
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Fit the model
    model.fit(daily_counts)
    
    # Create future dataframe for predictions
    future_period = forecast_months * 30  # Approximate days in forecast months
    future = model.make_future_dataframe(periods=future_period)
    
    # Make predictions
    forecast = model.predict(future)
    
    # Get the last date of historical data
    last_date = daily_counts['ds'].max()
    
    # Convert to plotly for Streamlit
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create plotly figure
    plotly_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add historical data
    plotly_fig.add_trace(
        go.Scatter(
            x=daily_counts['ds'],
            y=daily_counts['y'],
            mode='lines',
            name='Historical Messages',
            line=dict(color='blue')
        )
    )
    
    # Add forecast
    plotly_fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Predicted Messages',
            line=dict(color='green')
        )
    )
    
    # Add uncertainty intervals
    plotly_fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        )
    )
    
    plotly_fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(0, 176, 0, 0.2)',
            line=dict(width=0),
            name='Prediction Interval'
        )
    )
    
    # Add a shape for the vertical line instead of using add_vline
    plotly_fig.add_shape(
        type="line",
        x0=last_date,
        y0=0,
        x1=last_date,
        y1=forecast['yhat'].max() * 1.2,  # Make line extend beyond the highest point
        line=dict(
            color="red",
            width=2,
            dash="dash",
        )
    )
    
    # Add annotation for the forecast start
    plotly_fig.add_annotation(
        x=last_date,
        y=forecast['yhat'].max() * 1.1,
        text="Forecast Start",
        showarrow=True,
        arrowhead=1,
        ax=50,
        ay=-40
    )
    
    # Update layout
    plotly_fig.update_layout(
        title='Message Activity Forecast',
        xaxis_title='Date',
        yaxis_title='Number of Messages',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Generate textual conclusions
    conclusions = generate_forecast_conclusions(daily_counts, forecast, forecast_months)
    
    return plotly_fig, conclusions

def generate_forecast_conclusions(historical_data, forecast, forecast_months):
    """
    Generate textual conclusions based on the forecast results
    
    Args:
        historical_data: DataFrame with historical message counts
        forecast: DataFrame with forecast results from Prophet
        forecast_months: Number of months forecasted
    
    Returns:
        Dictionary with various conclusions
    """
    # Split forecast into historical and future periods
    historical_end = historical_data['ds'].max()
    historical_forecast = forecast[forecast['ds'] <= historical_end]
    future_forecast = forecast[forecast['ds'] > historical_end]
    
    # Calculate overall trend
    first_month = future_forecast.head(30)
    last_month = future_forecast.tail(30)
    first_month_avg = first_month['yhat'].mean()
    last_month_avg = last_month['yhat'].mean()
    
    if last_month_avg > first_month_avg * 1.1:
        trend = "increasing"
        trend_pct = ((last_month_avg / first_month_avg) - 1) * 100
    elif last_month_avg < first_month_avg * 0.9:
        trend = "decreasing"
        trend_pct = (1 - (last_month_avg / first_month_avg)) * 100
    else:
        trend = "stable"
        trend_pct = abs(((last_month_avg / first_month_avg) - 1) * 100)
    
    # Find peak activity days
    weekday_averages = {}
    for i in range(7):
        weekday_data = future_forecast[future_forecast['ds'].dt.weekday == i]
        weekday_averages[i] = weekday_data['yhat'].mean()
    
    most_active_day = max(weekday_averages.items(), key=lambda x: x[1])
    least_active_day = min(weekday_averages.items(), key=lambda x: x[1])
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    most_active_day_name = day_names[most_active_day[0]]
    least_active_day_name = day_names[least_active_day[0]]
    
    # Calculate expected total messages
    total_expected_messages = int(future_forecast['yhat'].sum())
    daily_average = future_forecast['yhat'].mean()
    
    # Find expected peak day
    peak_day = future_forecast.loc[future_forecast['yhat'].idxmax()]
    peak_day_date = peak_day['ds'].strftime('%B %d, %Y')
    peak_day_count = int(peak_day['yhat'])
    
    # Calculate seasonality
    if 'yearly' in forecast.columns and 'weekly' in forecast.columns:
        yearly_amplitude = forecast['yearly'].max() - forecast['yearly'].min()
        weekly_amplitude = forecast['weekly'].max() - forecast['weekly'].min()
        
        if yearly_amplitude > weekly_amplitude:
            seasonality = "yearly patterns are stronger than weekly patterns"
        else:
            seasonality = "weekly patterns are stronger than yearly patterns"
    else:
        seasonality = "both weekly and monthly patterns influence your messaging habits"
    
    # Compare with historical data
    historical_daily_avg = historical_data['y'].mean()
    future_daily_avg = future_forecast['yhat'].mean()
    
    if future_daily_avg > historical_daily_avg * 1.2:
        historical_comparison = f"significantly higher than your historical average of {historical_daily_avg:.1f} messages per day"
        change_pct = ((future_daily_avg / historical_daily_avg) - 1) * 100
    elif future_daily_avg < historical_daily_avg * 0.8:
        historical_comparison = f"significantly lower than your historical average of {historical_daily_avg:.1f} messages per day"
        change_pct = (1 - (future_daily_avg / historical_daily_avg)) * 100
    else:
        historical_comparison = f"similar to your historical average of {historical_daily_avg:.1f} messages per day"
        change_pct = abs(((future_daily_avg / historical_daily_avg) - 1) * 100)
    
    # Compile conclusions
    conclusions = {
        "trend": {
            "direction": trend,
            "percentage": f"{trend_pct:.1f}%",
            "description": f"Your message activity is predicted to be {trend} over the next {forecast_months} months, with a {trend_pct:.1f}% {trend} trend."
        },
        "activity_patterns": {
            "most_active_day": most_active_day_name,
            "least_active_day": least_active_day_name,
            "description": f"You're most active on {most_active_day_name}s and least active on {least_active_day_name}s."
        },
        "volume": {
            "total_expected": total_expected_messages,
            "daily_average": f"{daily_average:.1f}",
            "description": f"You're expected to send approximately {total_expected_messages} messages over the next {forecast_months} months, averaging {daily_average:.1f} messages per day."
        },
        "peak": {
            "date": peak_day_date,
            "count": peak_day_count,
            "description": f"Your peak messaging day is expected to be {peak_day_date} with approximately {peak_day_count} messages."
        },
        "seasonality": {
            "pattern": seasonality,
            "description": f"Analysis shows that {seasonality}."
        },
        "historical_comparison": {
            "change": f"{change_pct:.1f}%",
            "description": f"Your predicted daily average of {future_daily_avg:.1f} messages is {historical_comparison} (a {change_pct:.1f}% {'increase' if future_daily_avg > historical_daily_avg else 'decrease'})."
        }
    }
    
    return conclusions

