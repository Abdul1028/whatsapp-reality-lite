import re

import numpy as np
import pandas as pd
from dateutil.parser import parse
from sklearn.preprocessing import OrdinalEncoder

#THIS ONE WORKS WELL WITH BOTH ANDROID AND IOS FORMAT

def preprocess(data):
    # Define patterns for different formats
    android_12hr_pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s(?:am|pm|AM|PM)\s*-\s*'
    android_24hr_pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s*-\s*'
    ios_12hr_pattern = r'\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s(?:AM|PM)\]'
    ios_24hr_pattern = r'\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\]'
    
    # Try to detect the format
    if re.search(android_24hr_pattern, data):
        pattern = android_24hr_pattern
        messages = re.split(pattern, data)[1:]
        dates = re.findall(pattern, data)
        df = pd.DataFrame({'user_message': messages, 'message_date': dates})
        df['message_date'] = df['message_date'].apply(lambda x: x.strip('- '))
    
    elif re.search(android_12hr_pattern, data):
        pattern = android_12hr_pattern
        messages = re.split(pattern, data)[1:]
        dates = re.findall(pattern, data)
        df = pd.DataFrame({'user_message': messages, 'message_date': dates})
        df['message_date'] = df['message_date'].apply(lambda x: x.strip('- '))
    
    elif re.search(ios_12hr_pattern, data):
        pattern = ios_12hr_pattern
        messages = re.split(pattern, data)[1:]
        dates = re.findall(pattern, data)
        df = pd.DataFrame({'user_message': messages, 'message_date': dates})
        df['message_date'] = df['message_date'].apply(lambda x: re.sub(r'[\[\]]', '', x))
    
    elif re.search(ios_24hr_pattern, data):
        pattern = ios_24hr_pattern
        messages = re.split(pattern, data)[1:]
        dates = re.findall(pattern, data)
        df = pd.DataFrame({'user_message': messages, 'message_date': dates})
        df['message_date'] = df['message_date'].apply(lambda x: re.sub(r'[\[\]]', '', x))
    
    else:
        raise ValueError("Unsupported data format")

    # Parse the dates using dateutil.parser.parse with fuzzy matching
    df['date'] = df['message_date'].apply(lambda x: parse(x, fuzzy=True))
    df.index = df['date']
    # df['Formatted Date'] = df.index.strftime('%b - %y').values

    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df['Message Length'] = df['message'].apply(lambda x: len(x.split(' ')))

    conv_codes, conv_changes = cluster_into_conversations(df)
    df['Conv code'] = conv_codes
    df['Conv change'] = conv_changes

    is_reply, sender_changes = find_replies(df)
    df['Is reply'] = is_reply
    df['Sender change'] = sender_changes

    for subject in df['user'].unique():
        df[subject] = df['user'].apply(lambda x: 1 if x == subject else 0)
        df[f"{subject}_mlength"] = df[subject].values * df['Message Length']

    df.drop(columns=['user_message', 'message_date'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute



    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period


    # Add logic for finding replies and calculating times
    df = add_reply_logic(df)

    # Calculate times based on replies
    reply_times, indices = calculate_times_on_trues(df, 'Is Reply')
    reply_times_df_list = []
    reply_time_index = 0
    for i in range(0, len(df)):
        if i in indices:
            reply_times_df_list.append(reply_times[reply_time_index].astype("timedelta64[m]").astype("float"))
            reply_time_index += 1
        else:
            reply_times_df_list.append(0)

    df['Reply Time'] = reply_times_df_list

    inter_conv_times, indices = calculate_times_on_trues(df, 'Conv change')
    inter_conv_times_df_list = []
    inter_conv_time_index = 0
    for i in range(0, len(df)):
        if i in indices:
            inter_conv_times_df_list.append(
                inter_conv_times[inter_conv_time_index].astype("timedelta64[m]").astype("float"))
            inter_conv_time_index = inter_conv_time_index + 1
        else:
            inter_conv_times_df_list.append(0)

    df['Inter conv time'] = inter_conv_times_df_list

    df.to_csv("my_csv_data.csv", index=False)

    return df

def add_reply_logic(df):
    # Ordinal encoders will encode each user with its own number
    user_encoder = OrdinalEncoder()
    df['User Code'] = user_encoder.fit_transform(df['user'].values.reshape(-1, 1))

    # Find replies
    message_senders = df['User Code'].values
    sender_changed = (np.roll(message_senders, 1) - message_senders).reshape(1, -1)[0] != 0
    sender_changed[0] = False
    is_reply = sender_changed & ~df['user'].eq('group_notification')

    df['Is Reply'] = is_reply



    return df

def calculate_times_on_trues(df : pd.DataFrame, column : str):
    assert(column in df.columns)
    true_indices = np.where(df[column])[0]
    inter_conv_time = [df.index.values[ind] - df.index.values[ind-1] for ind in true_indices]
    return inter_conv_time, true_indices


##Section to get the conversation changes data not working here

def cluster_into_conversations(df: pd.DataFrame, inter_conversation_threshold_time: int = 60):
    threshold_time_mins = np.timedelta64(inter_conversation_threshold_time, 'm')

    # This calculates the time between the current message and the previous one
    conv_delta = df.index.values - np.roll(df.index.values, 1)
    conv_delta[0] = 0

    # This detects where the time between messages is higher than the threshold
    conv_changes = conv_delta > threshold_time_mins
    conv_changes_indices = np.where(conv_changes)[0]
    conv_codes = []

    if len(conv_changes_indices) == 0:
        # If there are no changes in conversation, assign all messages to a single conversation
        conv_codes = [0] * len(df)
    else:
        # Otherwise, assign conversation codes based on the detected changes
        last_conv_change = 0
        for i, conv_change in enumerate(conv_changes_indices):
            conv_codes.extend([i] * (conv_change - last_conv_change))
            last_conv_change = conv_change
        conv_codes.extend([len(conv_changes_indices)] * (len(df) - conv_changes_indices[-1]))

    # This serves to ensure that the conversation codes and the number of messages are aligned
    conv_codes = pad_list_to_value(conv_codes, len(df), conv_codes[-1])
    conv_changes = pad_list_to_value(conv_changes, len(df), False)

    return conv_codes, conv_changes


def pad_list_to_value(input_list : list, length : int, value):
    assert(length >= len(input_list))
    output_list = list(input_list)
    padding = [value]*(length - len(output_list))
    output_list.extend(padding)
    return np.array(output_list)


def find_replies(df : pd.DataFrame):
    # These are sanity checks in order to see if I made any ordering mistakes
    assert('Conv code' in df.columns)
    assert('Conv change' in df.columns)
    assert('user' in df.columns)
    # Ordinal encoders will encode each subject with its own number
    message_senders = OrdinalEncoder().fit_transform(df['user'].values.reshape(-1,1))
    # This compares the current subject with the previous subject
    # In a way that computers can optimize
    sender_changed = (np.roll(message_senders, 1) - message_senders).reshape(1, -1)[0] != 0
    sender_changed[0] = False
    # This checks if the reply isn't within a different conversation
    is_reply = sender_changed & ~df['Conv change']
    return is_reply, sender_changed


