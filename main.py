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

                st.title("Wordcloud")

                # df_wc = helper.create_wordcloud(selected_user, df)
                # fig, ax = plt.subplots()
                # ax.imshow(df_wc)
                # plt.axis("off")
                # st.pyplot(fig)

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

                helper.message_length_analysis(selected_user, df)



                max_idle_date,max_idle_time = helper.most_idle_date_time(df)
                st.write(f"The date(s) when the group was idle for the most time:")
                st.write(f"Date: {max_idle_date}, Total Idle Time : {max_idle_time / 86400 :.2f} days")

                median_delay_per_user = helper.median_delay_between_conversations(selected_user,df)
                if median_delay_per_user is not None :
                    st.write(f"Median Reply Delay for {selected_user}: {median_delay_per_user:.2f} minutes")

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




