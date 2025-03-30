from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json
import helper
import preprocessor
import nltk
import ssl


#Layout
st.set_page_config(
    page_title="Whatsapp Chat Analyzer",
    layout="wide",
    initial_sidebar_state="expanded")


def display_chat_message(message):
    st.markdown(f'<div class="chat-message">{message}</div>', unsafe_allow_html=True)


chat_history = []

#custom css for button and webview
st.markdown("""
    <style>
    
    .big-font {
    font-size:80px !important;
}

     .message-container {
        display: flex;
        flex-direction: column;
        margin-bottom: 10px;
    }
        
    .metric-container .metric{
    color: green;
    background-color: #6092e7;
    border: solid 2px #000000;
    padding: 10px 10px 10px 70px;
    border-radius: 20px;
    color: black;
}

    .user-message {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        max-width: 60%;
        align-self: flex-end;

    }
    .other-message {
        background-color: #F1F0F0;
        padding: 10px;
        border-radius: 10px;
        max-width: 60%;
        align-self:flex-start;

    }

    .notification-message {
        background-color: #F1F0F0;
        padding: 10px;
        border-radius: 10px;
        max-width: 60%;
        align-self:center;

    }

    .sender {
        font-size: 12px;
        color: red;
    }
    .time {
        font-size: 10px;
        color: black;
        text-align: right;

    }

       .chat-message {
        background-color: #DCF8C6;
        color: #000000;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        max-width: 70%;
    }
    .chat-message:nth-child(odd) {
        align-self: flex-start;
        text-align: left;
    }
    .chat-message:nth-child(even) {
        align-self: flex-end;
        text-align: right;
        background-color: #DCF8C6;
        color: #000000;
            
    }



    </style>
""", unsafe_allow_html=True)

github_link = ""
export_file_path = "plots_report.pdf"


@st.cache_data
def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)


def display_chat_message(sender, message, sentiment):
    st.markdown(
        f"<div class='chat-message'>Sender: {sender}<br>Message: {message}<br>Sentiment: {sentiment}</div>",
        unsafe_allow_html=True
    )


#Options Menu
with st.sidebar:
    selected = option_menu( 'Chat Analyzer', ["Intro", 'Search','About','Login'],icons=['play-btn','search','info-circle','gear'],menu_icon='intersect', default_index=0)
    lottie = load_lottiefile("lottie_jsons/sidebar.json")
    st_lottie(lottie, key='loc')

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('vader_lexicon')

# Add this function definition at the top of main.py, before it's used
def generate_personality_description(traits):
    """Generate a description based on personality traits"""
    descriptions = {
        'openness': {
            'high': "You're curious, creative, and open to new experiences. You enjoy exploring new ideas and have diverse interests.",
            'medium': "You have a healthy balance between tradition and innovation. You're open to new ideas while valuing stability.",
            'low': "You're practical and grounded. You prefer familiar routines and concrete, straightforward communication."
        },
        'conscientiousness': {
            'high': "You're organized, reliable, and detail-oriented. You respond promptly and take your commitments seriously.",
            'medium': "You balance structure with flexibility. You're generally reliable while allowing room for spontaneity.",
            'low': "You have a relaxed, flexible approach to life. You prefer to go with the flow rather than stick to rigid plans."
        },
        'extraversion': {
            'high': "You're outgoing, energetic, and enjoy social interaction. You communicate frequently and expressively.",
            'medium': "You balance social time with alone time. You engage well with others but also value your independence.",
            'low': "You're thoughtful and reserved. You prefer deeper one-on-one conversations to group chats."
        },
        'agreeableness': {
            'high': "You're friendly, cooperative, and considerate. You focus on harmony and positive interactions.",
            'medium': "You're generally cooperative while maintaining healthy boundaries. You can be both kind and assertive.",
            'low': "You're direct and straightforward. You prioritize honesty over social niceties and don't shy away from disagreement."
        },
        'neuroticism': {
            'high': "You're emotionally sensitive and expressive. You experience a wide range of emotions and share them openly.",
            'medium': "You have a balanced emotional response. You're neither overly sensitive nor completely detached.",
            'low': "You're calm and emotionally stable. You tend to stay composed even in stressful situations."
        }
    }
    
    result = ""
    for trait, score in traits.items():
        if score > 70:
            level = 'high'
        elif score > 30:
            level = 'medium'
        else:
            level = 'low'
        
        result += descriptions[trait][level] + "\n\n"
    
    return result


if selected == "Intro":
    c1, c2 = st.columns((2, 1))
    c1.title("""Whatsapp Chat Analyser""")
    c1.subheader("""Discover trends, analyse your chat history and judge your friends!""")
    c1.markdown(
        f"Dont worry, we wont peek, we're not about that, in fact, you can check the code in here: [link]({github_link})")

    uploaded_file = c1.file_uploader(label="""Upload your Whatsapp chat, don't worry, we won't peek""",
                                     key="notniq")



    with c2:
        lottie = load_lottiefile("lottie_jsons/chat_icon.json")
        st_lottie(lottie, key='loc2')


    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        df = preprocessor.preprocess(data)
        print("YOUR DATA FRAME IS: ", df)
        st.write(df)

        # Assuming 'date' is a datetime column in your DataFrame
        start_date = df['date'].min()
        end_date = df.iloc[-1]['date']

        # Convert datetime to timestamp for slider
        start_date_timestamp = int(start_date.timestamp())
        end_date_timestamp = int(end_date.timestamp())

        # Convert timestamp back to datetime
        start_datee = datetime.utcfromtimestamp(start_date_timestamp).strftime('%Y-%m-%d')
        end_datee = datetime.utcfromtimestamp(end_date_timestamp).strftime('%Y-%m-%d')

        # Convert to datetime objects for showing them in slider
        date_object1 = datetime.strptime(start_datee, "%Y-%m-%d")
        date_object2 = datetime.strptime(end_datee, "%Y-%m-%d")


        # Create a date slider
        selected_date_range_timestamp = st.slider(
            'Select date',
            min_value=date_object1,
            value=(date_object1, date_object2),
            max_value=date_object2,
            format="YYYY-MM-DD",  # Display format
        )

        # Display the selected date range

        selected_start_date = selected_date_range_timestamp[0].strftime("%Y-%m-%d")
        selected_end_date = selected_date_range_timestamp[1].strftime("%Y-%m-%d")
        st.write(f'Start Date: {selected_date_range_timestamp[0].strftime("%Y-%m-%d")} & End Date: {selected_date_range_timestamp[1].strftime("%Y-%m-%d")}')

        # Filter data based on selected date range
        df = df[(df['date'] >= selected_start_date) & (df['date'] <= selected_end_date)]
        # st.write(df)

        # fetch unique users
        user_list = df['user'].unique().tolist()
        # user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")

        c3, c4, c5 = st.columns((1, 2, 1))

        with c3:
            selected_user = st.selectbox("Select Participants for analysis", user_list, )
        with c4:
            selected_participants = st.multiselect("Select Participants to view there conversation", user_list,
                                                   key="new2")
        with c5:
            selected_participant_for_displaying_messsage = st.selectbox("select participant for viewing: ",selected_participants)

        placeholder = st.empty()

        if selected_user:
            with placeholder.container():



                # Stats Area
                num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
                st.title("Top Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.header("Total Messages")
                    st.metric(" ",num_messages)
                with col2:
                    st.header("Total Words")
                    st.metric(" ",words)
                with col3:
                    st.header("Media Shared")
                    st.metric(" ",num_media_messages)
                with col4:
                    st.header("Links Shared")
                    st.metric(" ",num_links)

                # working but shows a lot of graphs
                # figures = helper.calculate_monthly_sentiment_trends(selected_user, df)
                # # Display the figures in Streamlit
                # for fig in figures:
                #     st.plotly_chart(fig)

                # Write stylish text
                st.markdown(
                    f"""
                    <div style='text-align: left; color: white; padding-top: 40px; padding-bottom:20px; font-size: 24px;' >
                        You're first conversation started on <b>{start_date}</b> and last text was at <b>{end_date}</b>
                        <br>
                        You guys did a whopping <b>{num_messages}</b> messages in this journey 
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write("Time Analysis")

                selected = option_menu('Select Time period', ["Monthly", 'Daily', 'Weekly',''],
                                        menu_icon='intersect',
                                       default_index=3,orientation="horizontal")

                if selected == "Monthly":
                    # monthly timeline
                    st.title("Monthly Timeline")
                    fig = helper.monthly_timeline(selected_user, df)
                    st.plotly_chart(fig)

                elif selected == "Daily":
                    # daily timeline
                    st.title("Daily Timeline")
                    fig = helper.daily_timeline(selected_user, df)
                    st.plotly_chart(fig)

                else:
                    st.info("Select the analysis time period")

                st.header("Most busy day")
                fig = helper.week_activity_map(selected_user, df)
                st.plotly_chart(fig)

                st.header("Most busy month")
                fig= helper.month_activity_map(selected_user, df)
                st.plotly_chart(fig)



                # finding the busiest users in the group(Group level)
                if selected_user == 'Overall':
                    st.title('Most Busy Users')
                    fig, new_df = helper.most_busy_users(df)
                    st.plotly_chart(fig)
                    fig.write_image("exports/charts/fig1.png")



                    st.dataframe(new_df)

                ###End of sentimental analysis
                st.title("Busiest Hours")
                helper.busiest_hours_analysis(df)

                st.title("Wordcloudd")

                wordcloud_fig = helper.create_plotly_wordcloud(selected_user, df)
                st.plotly_chart(wordcloud_fig)


                # most common words
                fig = helper.most_common_words(selected_user, df)
                st.title('Most common words')

                st.plotly_chart(fig)

                # emoji analysis
                st.title("Emoji Analysis")
                fig = helper.emoji_helper(selected_user, df)
                st.plotly_chart(fig)

                fig = helper.show_average_reply_time(df)
                st.plotly_chart(fig)


                ##Start of sentimental analysis
                # Perform sentiment analysis on the selected messages
                positive_fig, negative_fig = helper.analyze_and_plot_sentiment(selected_user, df)
                # Display the positive and negative sentiment figures
                st.plotly_chart(positive_fig)
                st.plotly_chart(negative_fig)

                st.info("Brief analayis for negativity and positivity percentagr of users! ")

                user_sentiment_percentages, most_positive, least_positive = helper.calculate_sentiment_percentage(
                    selected_user, df)
                for user, percentages in user_sentiment_percentages.items():
                    st.write(f"User: {user}")
                    st.write(f"Positivity Percentage: ", percentages[0])
                    st.write(f"Negativity Percentage: ", percentages[1])
                    st.write("---")  # Separator between users

                st.write("Most positive User: ", most_positive)
                st.write("Least positive: ", least_positive)


                st.info("Sentimental trend followed by you in time period of months")

                f = helper.calculate_monthly_sentiment_trend(df)
                st.plotly_chart(f)



                c_11, c_12 = st.columns((1, 1))
                fig1, most_messages_winner = helper.message_count_aggregated_graph(df)
                c_11.subheader("Who talks the most?")
                c_11.markdown(
                    f"How many messages has each one sent in your convo? apparently **{most_messages_winner}** did")
                with c_11:
                    st.plotly_chart(fig1)

                # Show who starts the conversations
                c_11, c_12 = st.columns((1, 1))
                fig1, most_messages_winner = helper.conversation_starter_graph(df)
                c_11.subheader("Who's starts the conversations?")
                c_11.markdown(f"This clearly shows that **{most_messages_winner}** started all the convos")
                with c_11:
                    st.plotly_chart(fig1)

                fig = helper.conversation_size_aggregated_graph(df)
                st.subheader("How long are your conversations?")
                st.markdown(
                    f"This is how many messages (on average) your conversations had, the more of them there are, the more messages you guys exchanged everytime one of you started the convo!")
                st.plotly_chart(fig)

                c1, c2 = st.columns(2)

                emoji_df = helper.top_emojis_used(selected_user, df)

                with c1:
                    st.dataframe(emoji_df)

                with c2:
                    helper.message_count_by_month(selected_user, df)

                fig = helper.greeting_farewell_analysis(selected_user, df)
                st.plotly_chart(fig)


                max_user, max_time = helper.longest_reply_user(df)

                st.write(max_user, " takes the most time to reply wiz ", max_time)
                user,time,msg,reply = helper.longest_reply_user2(df)
                st.write(f"User with longest reply time: {user}")
                st.write(f"Longest reply time (minutes): {time}")
                st.write(f"Message to which the user replied the most late: {msg}")
                st.write(f"Replied message: {reply}")

                user, time, msg, reply = helper.top5_late_replies(df)
                st.write(f"User with longest reply time: {user}")
                st.write(f"Longest reply time (minutes): {time}")
                st.write(f"Message to which the user replied the most late: {msg}")
                st.write(f"Replied message: {reply}")

                user, time, msg, reply = helper.top_texts_late_replies(df)
                st.write(f"User with longest reply time: {user}")
                st.write(f"Longest reply time (minutes): {time}")
                st.write(f"Message to which the user replied the most late: {msg}")
                st.write(f"Replied message: {reply}")

                # Add the new function call here
                st.subheader("Average Late Reply Time Analysis")
                late_reply_fig, avg_late_reply_times, overall_avg = helper.calculate_average_late_reply_time(df)
                st.plotly_chart(late_reply_fig)
                st.write(f"Overall average late reply time: {overall_avg:.1f} hours")
                
                # Display the data in a table
                st.write("Average Late Reply Times by User:")
                st.dataframe(avg_late_reply_times)

                helper.message_length_analysis(selected_user, df)



                max_idle_date,max_idle_time = helper.most_idle_date_time(df)
                st.write(f"The date(s) when the group was idle for the most time:")
                st.write(f"Date: {max_idle_date}, Total Idle Time : {max_idle_time / 86400 :.2f} days")

                median_delay_per_user = helper.median_delay_between_conversations(selected_user,df)
                if median_delay_per_user is not None :
                    st.write(f"Median Reply Delay for {selected_user}: {median_delay_per_user:.2f} minutes")

                # Conversation Momentum Analysis
                st.header("Conversation Momentum Analysis")
                momentum_data = helper.analyze_conversation_momentum(df)

                if momentum_data:
                    # Display overall metrics
                    st.subheader("Overall Momentum Metrics")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Average Conversation Momentum", f"{momentum_data['avg_momentum']:.2f}")
                        st.write(f"Positive momentum in {momentum_data['positive_momentum_pct']:.1f}% of conversations")

                    # Plot momentum starters
                    momentum_starters_df = pd.DataFrame(momentum_data['momentum_starters'],
                                                        columns=['User', 'Average Momentum'])

                    if not momentum_starters_df.empty:
                        fig = px.bar(momentum_starters_df.head(10),
                                     x='User', y='Average Momentum',
                                     title="Top Conversation Momentum Drivers",
                                     color='Average Momentum',
                                     color_continuous_scale='RdBu')
                        st.plotly_chart(fig)

                        st.write("Users with positive momentum tend to keep conversations engaging and accelerating, "
                                 "while negative momentum indicates conversations that slow down.")

                    # Plot conversation metrics
                    conv_metrics_df = momentum_data['conversation_metrics']
                    if not conv_metrics_df.empty:
                        fig = px.scatter(conv_metrics_df,
                                         x='duration_minutes', y='message_rate',
                                         size='message_count', color='momentum',
                                         hover_name='conv_code',
                                         title="Conversation Dynamics",
                                         labels={'duration_minutes': 'Duration (minutes)',
                                                 'message_rate': 'Messages per Minute',
                                                 'momentum': 'Momentum'},
                                         color_continuous_scale='RdBu')
                        st.plotly_chart(fig)
                else:
                    st.info("Not enough conversation data for momentum analysis.")

                # Topic Switching Analysis
                st.header("Topic Switching Analysis")
                topic_data = helper.analyze_topic_switching(df)

                if topic_data:
                    # Display overall metrics
                    st.subheader("Topic Switching Patterns")
                    st.write(f"Average topic switch rate: {topic_data['avg_switch_rate']:.2f} switches per message")

                    # Plot top topic switchers
                    switchers_df = pd.DataFrame(topic_data['top_switchers'],
                                                columns=['User', 'Topic Switches'])

                    if not switchers_df.empty:
                        fig = px.bar(switchers_df.head(10),
                                     x='User', y='Topic Switches',
                                     title="Top Topic Switchers",
                                     color='Topic Switches',
                                     color_continuous_scale='Viridis')
                        st.plotly_chart(fig)

                        st.write(
                            "Users who frequently switch topics may be driving the conversation in new directions.")
                else:
                    st.info("Not enough conversation data for topic switching analysis.")

                # Initiator-Responder Dynamics
                st.header("Conversation Initiator-Responder Dynamics")
                dynamics_data = helper.analyze_initiator_responder_dynamics(df)

                if dynamics_data:
                    # Display metrics in columns
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Top Conversation Initiators")
                        initiators_df = pd.DataFrame(dynamics_data['top_initiators'],
                                                     columns=['User', 'Initiations'])
                        st.dataframe(initiators_df.head(5))

                    with col2:
                        st.subheader("Top Responders")
                        responders_df = pd.DataFrame(dynamics_data['top_responders'],
                                                     columns=['User', 'Responses'])
                        st.dataframe(responders_df.head(5))

                    # Plot initiation-response ratios
                    ratios_df = pd.DataFrame(list(dynamics_data['initiation_response_ratios'].items()),
                                             columns=['User', 'Initiation-Response Ratio'])

                    fig = px.bar(ratios_df.sort_values('Initiation-Response Ratio', ascending=False),
                                 x='User', y='Initiation-Response Ratio',
                                 title="Initiation vs Response Behavior",
                                 color='Initiation-Response Ratio',
                                 color_continuous_scale='Viridis')
                    st.plotly_chart(fig)

                    st.write("A high ratio means the user tends to start conversations more than respond to others. "
                             "A low ratio means they primarily respond to others' messages.")

                    # Plot most common initiator-responder pairs
                    pairs_df = pd.DataFrame(dynamics_data['most_common_pairs'],
                                            columns=['Pair', 'Count'])

                    fig = px.bar(pairs_df.head(10),
                                 x='Pair', y='Count',
                                 title="Most Common Initiator-Responder Pairs",
                                 color='Count',
                                 color_continuous_scale='Viridis')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig)

                    st.write("These pairs show the most common conversation initiation patterns, "
                             "revealing who tends to respond to whom.")
                else:
                    st.info("Not enough conversation data for initiator-responder analysis.")

                # Red Flag/Green Flag Analysis
                st.header("🚩 Red Flag / Green Flag Analysis 🚩")
                flag_data = helper.analyze_red_green_flags(df)

                if flag_data:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Top Red Flag Users")
                        red_flag_df = pd.DataFrame(flag_data['most_red_flags'],
                                                   columns=['User', 'Red Flag Count'])
                        st.dataframe(red_flag_df.head(5))

                        # Plot red flag types
                        red_types_df = pd.DataFrame(list(flag_data['all_red_flags'].items()),
                                                    columns=['Flag Type', 'Count'])
                        fig = px.bar(red_types_df.sort_values('Count', ascending=False),
                                     x='Flag Type', y='Count',
                                     title="Most Common Red Flags",
                                     color='Count',
                                     color_continuous_scale='Reds')
                        st.plotly_chart(fig)

                    with col2:
                        st.subheader("Top Green Flag Users")
                        green_flag_df = pd.DataFrame(flag_data['most_green_flags'],
                                                     columns=['User', 'Green Flag Count'])
                        st.dataframe(green_flag_df.head(5))

                        # Plot green flag types
                        green_types_df = pd.DataFrame(list(flag_data['all_green_flags'].items()),
                                                      columns=['Flag Type', 'Count'])
                        fig = px.bar(green_types_df.sort_values('Count', ascending=False),
                                     x='Flag Type', y='Count',
                                     title="Most Common Green Flags",
                                     color='Count',
                                     color_continuous_scale='Greens')
                        st.plotly_chart(fig)

                    # Plot green-to-red ratio
                    ratio_df = pd.DataFrame(flag_data['best_ratios'],
                                            columns=['User', 'Green-to-Red Ratio'])
                    fig = px.bar(ratio_df.head(10),
                                 x='User', y='Green-to-Red Ratio',
                                 title="Green-to-Red Flag Ratio by User",
                                 color='Green-to-Red Ratio',
                                 color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig)

                    st.write("A higher green-to-red ratio indicates more positive communication patterns.")

                # Vibe Check Analysis
                st.header("✨ GenZ Vibe Check ✨")
                vibe_data = helper.analyze_vibe_check(df)

                if vibe_data:
                    # Display overall vibe score
                    st.metric("Overall Chat Vibe Score", f"{vibe_data['overall_score']:.1f}")

                    # Create vibe score gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=vibe_data['overall_score'],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Chat Vibe Meter"},
                        gauge={
                            'axis': {'range': [-100, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [-100, -50], 'color': "red"},
                                {'range': [-50, 0], 'color': "orange"},
                                {'range': [0, 50], 'color': "lightgreen"},
                                {'range': [50, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': vibe_data['overall_score']
                            }
                        }
                    ))
                    st.plotly_chart(fig)

                    # Plot user vibe scores
                    vibe_scores_df = pd.DataFrame([(user, data['vibe_score'])
                                                   for user, data in vibe_data['vibe_scores'].items()],
                                                  columns=['User', 'Vibe Score'])
                    vibe_scores_df = vibe_scores_df.sort_values('Vibe Score', ascending=False)

                    fig = px.bar(vibe_scores_df,
                                 x='User', y='Vibe Score',
                                 title="User Vibe Scores",
                                 color='Vibe Score',
                                 color_continuous_scale='RdBu',
                                 range_color=[-100, 100])
                    st.plotly_chart(fig)

                    st.write("Higher vibe scores indicate more positive GenZ communication patterns.")

                # GenZ Slang Analysis
                st.header("💯 GenZ Slang Analysis 💯")
                slang_data = helper.analyze_genz_slang(df)

                if slang_data:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Top Slang Users")
                        slang_users_df = pd.DataFrame(slang_data['most_slang_users'],
                                                      columns=['User', 'Slang Count'])
                        st.dataframe(slang_users_df.head(5))

                    with col2:
                        st.subheader("Highest Slang Density")
                        density_df = pd.DataFrame(slang_data['highest_density_users'],
                                                  columns=['User', 'Slang per Message'])
                        st.dataframe(density_df.head(5))

                    # Plot slang categories
                    categories_df = pd.DataFrame(list(slang_data['category_totals'].items()),
                                                 columns=['Category', 'Count'])
                    fig = px.pie(categories_df,
                                 values='Count', names='Category',
                                 title="GenZ Slang Categories",
                                 color_discrete_sequence=px.colors.sequential.Plasma_r)
                    st.plotly_chart(fig)

                    # Plot user slang breakdown
                    user_slang_data = []
                    for user, categories in slang_data['user_slang'].items():
                        for category, count in categories.items():
                            if count > 0:  # Only include non-zero counts
                                user_slang_data.append({
                                    'User': user,
                                    'Category': category,
                                    'Count': count
                                })

                    user_slang_df = pd.DataFrame(user_slang_data)
                    if not user_slang_df.empty:
                        fig = px.bar(user_slang_df,
                                     x='User', y='Count', color='Category',
                                     title="Slang Usage by User and Category",
                                     barmode='stack')
                        st.plotly_chart(fig)

                    st.write("This analysis shows who uses the most GenZ slang and which types they prefer.")

                # Reply Pair Analysis
                st.header("👥 Reply Pair Analysis 👥")
                reply_pair_data = helper.analyze_reply_pairs(df)

                if reply_pair_data:
                    # Display top reply pairs
                    st.subheader("Top Reply Pairs")

                    # Convert to DataFrame for display
                    pairs_df = pd.DataFrame(reply_pair_data['pair_counts'])
                    top_pairs = pairs_df.head(10)

                    # Create a more readable display format
                    display_df = pd.DataFrame({
                        'Replier': top_pairs['replier'],
                        'Replied To': top_pairs['replied_to'],
                        'Count': top_pairs['count'],
                        'Percentage': top_pairs['percentage'].apply(lambda x: f"{x:.1f}%")
                    })

                    st.dataframe(display_df)

                    # Plot top reply pairs
                    fig = px.bar(top_pairs,
                                 x='pair_text', y='count',
                                 title="Most Frequent Reply Pairs",
                                 color='count',
                                 color_continuous_scale='Viridis')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig)

                    # Display who replies to whom the most
                    st.subheader("Who Replies to Whom the Most")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("Person who replies to each user the most:")
                        for user, replier_data in reply_pair_data['most_frequent_repliers'].items():
                            st.write(
                                f"**{user}** is most frequently replied to by **{replier_data['replier']}** ({replier_data['count']} times)")

                    with col2:
                        st.write("Person each user replies to the most:")
                        for user, replied_data in reply_pair_data['most_replied_to'].items():
                            st.write(
                                f"**{user}** most frequently replies to **{replied_data['replied_to']}** ({replied_data['count']} times)")

                    # Create a network graph of reply relationships
                    st.subheader("Reply Relationship Network")

                    # Get top N pairs for each user to avoid overcrowding
                    top_n_per_user = 2
                    network_pairs = []

                    for user in df['user'].unique():
                        if user == 'group_notification':
                            continue

                        user_pairs = [p for p in reply_pair_data['pair_counts'] if p['replier'] == user]
                        if user_pairs:
                            top_user_pairs = sorted(user_pairs, key=lambda x: x['count'], reverse=True)[:top_n_per_user]
                            network_pairs.extend(top_user_pairs)

                    # Create network nodes and edges
                    nodes = list(set([p['replier'] for p in network_pairs] + [p['replied_to'] for p in network_pairs]))

                    # Create a network graph using plotly
                    edge_x = []
                    edge_y = []
                    edge_weights = []

                    # Simple circular layout
                    node_positions = {}
                    n = len(nodes)
                    for i, node in enumerate(nodes):
                        angle = 2 * np.pi * i / n
                        node_positions[node] = (np.cos(angle), np.sin(angle))

                    for pair in network_pairs:
                        x0, y0 = node_positions[pair['replier']]
                        x1, y1 = node_positions[pair['replied_to']]

                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                        edge_weights.append(pair['count'])

                    # Create edges trace
                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=1, color='#888'),
                        hoverinfo='none',
                        mode='lines')

                    # Create nodes trace
                    node_x = [node_positions[node][0] for node in nodes]
                    node_y = [node_positions[node][1] for node in nodes]

                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=nodes,
                        textposition="top center",
                        marker=dict(
                            showscale=True,
                            colorscale='YlGnBu',
                            size=15,
                            colorbar=dict(
                                thickness=15,
                                title='Node Connections',
                                xanchor='left',
                                titleside='right'
                            )
                        ),
                        hoverinfo='text')

                    # Count connections for each node
                    node_connections = {node: 0 for node in nodes}
                    for pair in network_pairs:
                        node_connections[pair['replier']] += 1
                        node_connections[pair['replied_to']] += 1

                    node_trace.marker.color = [node_connections[node] for node in nodes]
                    node_trace.text = [f"{node}<br># of connections: {node_connections[node]}" for node in nodes]

                    # Create the figure
                    fig = go.Figure(data=[edge_trace, node_trace],
                                    layout=go.Layout(
                                        title="Reply Network",
                                        showlegend=False,
                                        hovermode='closest',
                                        margin=dict(b=20, l=5, r=5, t=40),
                                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                    )

                    st.plotly_chart(fig)

                    st.write(
                        "This network shows who replies to whom most frequently. Thicker lines indicate more frequent replies.")
                else:
                    st.info("Not enough data for reply pair analysis.")

                # Conversation Influence Analysis
                st.header("🌟 Conversation Influence Analysis 🌟")
                influence_data = helper.analyze_conversation_influence(df)

                if influence_data:
                    # Display most influential users
                    st.subheader("Most Influential Users")

                    influence_df = pd.DataFrame(influence_data['most_influential'],
                                                columns=['User', 'Influence Score'])

                    fig = px.bar(influence_df.head(10),
                                 x='User', y='Influence Score',
                                 title="Top Conversation Influencers",
                                 color='Influence Score',
                                 color_continuous_scale='Viridis')
                    st.plotly_chart(fig)

                    # Display detailed metrics for top users
                    st.subheader("Influence Metrics for Top Users")

                    top_users = [user for user, _ in influence_data['most_influential'][:5]]
                    metrics_df = pd.DataFrame([
                        {
                            'User': user,
                            'Messages': influence_data['user_metrics'][user]['messages'],
                            'Conversations': influence_data['user_metrics'][user]['conversation_count'],
                            'Responses Received': influence_data['user_metrics'][user]['responses_received'],
                            'Conversation Extensions': influence_data['user_metrics'][user]['conversation_extensions'],
                            'Conversation Revivals': influence_data['user_metrics'][user]['conversation_revival'],
                            'Responses per Message': f"{influence_data['user_metrics'][user]['avg_responses_per_message']:.2f}"
                        }
                        for user in top_users
                    ])

                    st.dataframe(metrics_df)

                    st.write(
                        "Influential users tend to receive more responses, extend conversations, and revive dead chats.")

                # Mood Shifter Analysis
                st.header("😊 Conversation Mood Shifters 😊")
                mood_data = helper.analyze_mood_shifters(df)

                if mood_data:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Top Mood Lifters")
                        if mood_data['mood_lifters']:
                            lifters_df = pd.DataFrame(mood_data['mood_lifters'],
                                                      columns=['User', 'Positive Shifts', 'Positive Ratio'])
                            lifters_df['Positive Ratio'] = lifters_df['Positive Ratio'].apply(
                                lambda x: f"{x * 100:.1f}%")
                            st.dataframe(lifters_df)
                        else:
                            st.info("No significant mood lifters detected")

                    with col2:
                        st.subheader("Top Mood Dampeners")
                        if mood_data['mood_dampeners']:
                            dampeners_df = pd.DataFrame(mood_data['mood_dampeners'],
                                                        columns=['User', 'Negative Shifts', 'Negative Ratio'])
                            dampeners_df['Negative Ratio'] = dampeners_df['Negative Ratio'].apply(
                                lambda x: f"{x * 100:.1f}%")
                            st.dataframe(dampeners_df)
                        else:
                            st.info("No significant mood dampeners detected")

                    # Plot overall mood shifting activity
                    shifters_df = pd.DataFrame(mood_data['top_shifters'][:10],
                                               columns=['User', 'Total Shifts', 'Shift Rate'])

                    fig = px.bar(shifters_df,
                                 x='User', y='Total Shifts',
                                 title="Top Conversation Mood Shifters",
                                 color='Shift Rate',
                                 color_continuous_scale='RdBu')
                    st.plotly_chart(fig)

                    st.write(
                        "Mood shifters change the emotional tone of conversations. Lifters make conversations more positive, while dampeners make them more negative.")

                # Conversation Compatibility Analysis
                st.header("❤️ Conversation Compatibility Analysis ❤️")
                compatibility_data = helper.analyze_conversation_compatibility(df)

                if compatibility_data:
                    # Display most compatible pairs
                    st.subheader("Most Compatible User Pairs")

                    if compatibility_data['most_compatible']:
                        compatible_df = pd.DataFrame(compatibility_data['most_compatible'][:10],
                                                     columns=['User 1', 'User 2', 'Compatibility Score'])
                        compatible_df['Pair'] = compatible_df.apply(lambda row: f"{row['User 1']} & {row['User 2']}",
                                                                    axis=1)

                        fig = px.bar(compatible_df,
                                     x='Pair', y='Compatibility Score',
                                     title="Most Compatible User Pairs",
                                     color='Compatibility Score',
                                     color_continuous_scale='Viridis')
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig)

                        # Create a compatibility matrix heatmap
                        st.subheader("Compatibility Matrix")

                        # Get unique users from the compatible pairs
                        unique_users = set()
                        for user1, user2, _ in compatibility_data['most_compatible']:
                            unique_users.add(user1)
                            unique_users.add(user2)

                        # Create matrix data
                        matrix_data = []
                        for user1 in unique_users:
                            row_data = []
                            for user2 in unique_users:
                                if user1 == user2:
                                    row_data.append(100)  # Perfect compatibility with self
                                else:
                                    # Find pair in compatibility data
                                    if user1 < user2:
                                        pair_key = f"{user1}_{user2}"
                                    else:
                                        pair_key = f"{user2}_{user1}"

                                    if pair_key in compatibility_data['user_pairs'] and \
                                            compatibility_data['user_pairs'][pair_key]['direct_replies'] > 5:
                                        row_data.append(
                                            compatibility_data['user_pairs'][pair_key]['compatibility_score'])
                                    else:
                                        row_data.append(0)  # No significant interaction

                            matrix_data.append(row_data)

                        # Create heatmap
                        fig = px.imshow(matrix_data,
                                        x=list(unique_users),
                                        y=list(unique_users),
                                        color_continuous_scale='Viridis',
                                        title="User Compatibility Matrix")

                        st.plotly_chart(fig)

                        st.write(
                            "Higher compatibility scores indicate user pairs who interact frequently, respond quickly to each other, and share similar emotional tones in their messages.")
                    else:
                        st.info("Not enough interaction data to determine compatibility")
                else:
                    st.info("Not enough data for compatibility analysis")

                # AI Features Section
                st.header("🤖 AI-Powered Insights")

                # Add a tab interface for different AI features
                ai_tabs = st.tabs(
                    ["Personality Insights", "Conversation Summaries", "Relationship Analysis", "Activity Forecast"])

                with ai_tabs[0]:
                    st.subheader("Personality Insights")

                    if st.button("Generate Personality Insights"):
                        with st.spinner("Analyzing personalities..."):
                            personality_data = helper.analyze_personality(df)

                            # Display personality radar charts
                            for user, traits in personality_data.items():
                                st.write(f"### {user}'s Personality Profile")

                                # Create radar chart data
                                categories = list(traits.keys())
                                values = list(traits.values())

                                fig = go.Figure()
                                fig.add_trace(go.Scatterpolar(
                                    r=values,
                                    theta=categories,
                                    fill='toself',
                                    name=user
                                ))

                                fig.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                            range=[0, 100]
                                        )
                                    ),
                                    showlegend=False
                                )

                                st.plotly_chart(fig)

                                # Add personality description
                                st.write(generate_personality_description(traits))

                with ai_tabs[1]:
                    st.subheader("Conversation Summaries")

                    if st.button("Generate Conversation Summaries"):
                        with st.spinner("Summarizing conversations..."):
                            summaries = helper.summarize_conversations(df)

                            for conv_code, data in summaries.items():
                                with st.expander(
                                        f"Conversation from {data['date_range'][0].date()} ({data['message_count']} messages)"):
                                    st.write(f"**Summary:** {data['summary']}")
                                    st.write(f"**Participants:** {', '.join(data['participants'])}")

                                    # Add a button to view the full conversation
                                    if st.button(f"View Full Conversation", key=f"view_{conv_code}"):
                                        st.dataframe(df[df['Conv code'] == conv_code][['date', 'user', 'message']])

                with ai_tabs[2]:
                    st.subheader("Relationship Analysis")

                    # Let user select two participants
                    col1, col2 = st.columns(2)
                    with col1:
                        user1 = st.selectbox("Select first user", df['user'].unique())
                    with col2:
                        user2 = st.selectbox("Select second user", [u for u in df['user'].unique() if u != user1])

                    if st.button("Analyze Relationship"):
                        with st.spinner("Analyzing relationship..."):
                            relationship_data = helper.analyze_relationship(df, user1, user2)

                            # Display relationship metrics
                            st.write(f"### Relationship between {user1} and {user2}")

                            # Create metrics row
                            metric_cols = st.columns(4)
                            with metric_cols[0]:
                                st.metric("Compatibility Score", f"{relationship_data['compatibility_score']:.1f}/100")
                            with metric_cols[1]:
                                st.metric("Conversation Count", relationship_data['conversation_count'])
                            with metric_cols[2]:
                                st.metric("Avg Response Time", f"{relationship_data['avg_response_time']:.1f} min")
                            with metric_cols[3]:
                                st.metric("Sentiment Alignment", f"{relationship_data['sentiment_alignment']:.1f}/10")

                            # Display conversation timeline
                            st.write("#### Conversation Timeline")
                            fig = px.line(relationship_data['timeline'], x='date', y='message_count',
                                          title="Message Frequency Over Time")
                            st.plotly_chart(fig)

                            # Display common topics
                            st.write("#### Common Topics")
                            topic_df = pd.DataFrame(relationship_data['common_topics'],
                                                    columns=['Topic', 'Frequency'])
                            fig = px.bar(topic_df, x='Topic', y='Frequency',
                                         title="Topics Discussed")
                            st.plotly_chart(fig)

                with ai_tabs[3]:
                    st.subheader("Future Activity Prediction")

                    # Let user select forecast period
                    forecast_months = st.slider("Forecast months ahead", min_value=1, max_value=12, value=3)

                    if st.button("Generate Activity Forecast"):
                        with st.spinner("Analyzing message patterns and generating forecast..."):
                            forecast_fig, conclusions = helper.predict_future_activity(df, forecast_months)

                            if forecast_fig:
                                st.plotly_chart(forecast_fig, use_container_width=True)

                                st.info(
                                    "This forecast is based on historical message patterns, including daily, weekly, and monthly trends. The shaded area represents the prediction uncertainty range.")

                                # Display textual conclusions
                                if conclusions:
                                    st.write("## Forecast Conclusions")

                                    # Overall trend
                                    st.write("### Overall Trend")
                                    st.write(conclusions["trend"]["description"])

                                    # Message volume
                                    st.write("### Message Volume")
                                    st.write(conclusions["volume"]["description"])

                                    # Activity patterns
                                    st.write("### Activity Patterns")
                                    st.write(conclusions["activity_patterns"]["description"])

                                    # Peak activity
                                    st.write("### Peak Activity")
                                    st.write(conclusions["peak"]["description"])

                                    # Historical comparison
                                    st.write("### Comparison to Historical Data")
                                    st.write(conclusions["historical_comparison"]["description"])

                                    # Seasonality
                                    st.write("### Seasonality")
                                    st.write(conclusions["seasonality"]["description"])

                                # Add some insights about the forecast methodology
                                with st.expander("About the Forecast Methodology"):
                                    st.write(
                                        "This forecast uses Facebook's Prophet model, which is designed for time series forecasting with strong seasonal patterns.")
                                    st.write("The model analyzes:")
                                    st.write("- Weekly patterns (like weekend vs weekday activity)")
                                    st.write("- Monthly trends (increasing or decreasing engagement)")
                                    st.write("- Special events or holidays that might affect messaging patterns")
                                    st.write("- Long-term growth or decline trends")

                # double_text_count = helper.double_text_counts(selected_user,df)
                # st.write(double_text_count)

                # if st.button("Export all data"):
                #     helper.export(selected_user, df)
                #     with open(export_file_path, 'r') as f:
                #         dl_button = download_button(f.read(), 'exported_data.pdf', 'Download your data!')
                #         st.markdown(dl_button, unsafe_allow_html=True)

                if st.button("Export all data"):
                    if "user" in st.session_state:
                        with st.spinner("Exporting data..."):
                            helper.export(selected_user, df)
                        with open(export_file_path, 'r') as f:
                            dl_button = download_button(f.read(), 'exported_data.pdf', 'Download your data!')
                            st.markdown(dl_button, unsafe_allow_html=True)
                            st.success("You can Download your data now!")
                    else:
                        st.error("You need to login to export your data please login! :)")



        if selected_participant_for_displaying_messsage:
            placeholder.empty()
            st.header(f"You are viewing as {selected_participant_for_displaying_messsage}")

            # Convert the 'date' column to pandas datetime object
            df['date'] = pd.to_datetime(df['date'])

            # # Sort the DataFrame by 'date' and 'user'
            # df = df.sort_values(by=['date', 'user'])
            
            # Iterate through messages and display them in a chat-like format
            for _, group in df[df['user'].isin(selected_participants)].iterrows():
                sender = group['user']
                message = group['message']
                time = group['date']
                sentiment = helper.analyze_sentiment(message)  # assuming you have a helper function

                # Apply custom CSS class based on the sender
                if sender == selected_participant_for_displaying_messsage:
                    st.markdown(
                        f'<div class="message-container"><div class="user-message">{message}</div>'
                        f'<div class="time"  >{time} </div></div>',
                        unsafe_allow_html=True
                    )
                elif sender == "group_notification":
                    st.markdown(
                        f'<div class="message-container">'
                        f'<div class="notification-message">{message}</div></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="message-container"><div class="sender">{sender}</div>'
                        f'<div class="other-message">{message}</div><div class="time"  style="text-align:left " >{time}</div></div>'
                        ,
                        unsafe_allow_html=True
                    )

            selected_participant_for_sentiments = placeholder.multiselect(f"Show Messages and Sentiments", user_list)

            if selected_participant_for_sentiments:
                st.header(f"Message sentiments of {selected_participant_for_sentiments}")
                filtered_df = df[df['user'].isin(selected_participant_for_sentiments)]

                for _, group in filtered_df.iterrows():
                    sender = group['user']
                    message = group['message']
                    sentiment = helper.analyze_sentiment(message)  # assuming you have a helper function
                    display_chat_message(sender, message, sentiment)

# Add after the existing analyses
