import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pm4py
import altair as alt
import os

#Removing a warning that will come due to deprication
st.set_option('deprecation.showPyplotGlobalUse', False)

#Set the title for the sidebar
st.sidebar.subheader('Features for batch detection')

#Set the title for the app
st.title("Batching")

#Import a file
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.sidebar.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)
filename = file_selector()
st.write('You selected `%s`' % filename)


#Read in the data
#data = pd.read_csv('event_log.csv', sep=";")
data = pd.read_csv(filename, sep=";")

#Set some text to introduce the original event log
st.subheader("This is the original event log")

#Show the original event log
st.dataframe(data)

#######
# Time level batch detection - features selection
#######

# Select resources
resource = data['resource'].unique()
resource_selected = st.sidebar.multiselect('1. Select resources', resource)
mask_resource = data['resource'].isin(resource_selected)
data = data[mask_resource]

# Select activities
activity = data['activity'].unique()
activity_selected = st.sidebar.multiselect('2. Select activities', activity)
mask_activity = data['activity'].isin(activity_selected)
data = data[mask_activity]

#Drop the columns selected from the multiselect
#Select multiple columns
cols_drop = st.multiselect("Select columns to drop", data.columns)
#Drop columns
data = data.drop(cols_drop, axis=1)

#Show the new dataframe
#Text to introduce the new dataframe
st.text("Once features have been selected, the event log looks like this")
#Show the new dataframe
st.dataframe(data)

#Change the time between activites that should form a batch
m=st.sidebar.slider("3. Time in minutes between activites that should form a batch",1,60)

#######
# Time level batch detection - features selection end
#######

#######
# Time level batch detection
#######

# Sort events by time start and time end
data["start time"] = pd.to_datetime(data["start time"], format="%Y-%m-%d %H:%M:%S")
data["end time"] = pd.to_datetime(data["end time"], format="%Y-%m-%d %H:%M:%S")
data = data.sort_values(['start time','end time'])

events = data[["start time", "end time"]].to_dict("r")
for e in events:
    e["start time"] = e["start time"].timestamp()
    e["end time"] = e["end time"].timestamp()
intervals = [(e["start time"], e["end time"], set([(e["start time"], e["end time"])])) for e in events]
intervals.sort()


# Merge overlaping intervals
continue_cycle = True
while continue_cycle:
    continue_cycle = False
    i = 0
    while i < len(intervals)-1:
        if intervals[i][1] > intervals[i+1][0]:
            # decide to merge interval i and i+1
            new_interval = (min(intervals[i][0], intervals[i+1][0]), max(intervals[i][1], intervals[i+1][1]), intervals[i][2].union(intervals[i+1][2]))
            # add the new interval to the list
            intervals.append(new_interval)
            # remove the i+1 interval
            del intervals[i+1]
            # remove the i interval
            del intervals[i]
            # sort the intervals
            intervals.sort()
            # set the variable continue_cycle to True
            continue_cycle = True
            # interrupt the current iteration on the intervals
            break
        i = i + 1

# Here we merge intrval with time in minutes defind before
continue_cycle = True
while continue_cycle:
    continue_cycle = False
    i = 0
    while i < len(intervals)-1:
        if intervals[i+1][0] - intervals[i][1] <= m*60:
            # decide to merge interval i and i+1
            new_interval = (min(intervals[i][0], intervals[i+1][0]), max(intervals[i][1], intervals[i+1][1]), intervals[i][2].union(intervals[i+1][2]))
            # add the new interval to the list
            intervals.append(new_interval)
            # remove the i+1 interval
            del intervals[i+1]
            # remove the i interval
            del intervals[i]
            # sort the intervals
            intervals.sort()
            # set the variable continue_cycle to True
            continue_cycle = True
            # interrupt the current iteration on the intervals
            break
        i = i + 1

# Here is batching
from pandas import DataFrame
batch = [len(interval[2]) for interval in intervals]
df_batches = DataFrame (batch,columns=['Number of events'])

# Types of batching

def check_batch_type(batch):
    events_batch = sorted(list(batch[2]))
    # take the minimum of the left-extreme of each interval
    min_left_events = min(ev[0] for ev in events_batch)
    # take the maximum of the left-extreme of each interval
    max_left_events = max(ev[0] for ev in events_batch)
    # take the minimum of the right-extreme of each interval
    min_right_events = min(ev[1] for ev in events_batch)
    # take the maximum of the right-extreme of each interval
    max_right_events = max(ev[1] for ev in events_batch)
    
    # CONDITION 1 - All the events in the batch have identical start and end timestamps
    if min_left_events == max_left_events and min_right_events == max_right_events:
        return "Simultaneous"
    # CONDITION 4 - All the events in the batch have identical start timestamp:
    if min_left_events == max_left_events:
        return "Batching on Start"
    # CONDITION 5 - All the events in the batch have identical end timestamp:
    if min_right_events == max_right_events:
        return "Batching on End"
    
    # now we could be in the SEQUENTIAL batching or the CONCURRENT batching
    # in order to be in the SEQUENTIAL, we need that for all the consecutive events the end of the first is equal to the start of the second
    is_sequential = True
    i = 0
    while i < len(events_batch)-1:
        # if there are two consecutive events that are not sequentially matched, then we automatically fall inside the CONCURRENT batching
        if events_batch[i][1] != events_batch[i+1][0]:
            is_sequential = False
            break
        i = i + 1
    if is_sequential:
        return "Sequential batching"
    else:
        return "Concurrent batching"

# 
df_types= pd.DataFrame(columns=['batch_type', 'len'])

# check the type for each batch (interval of length at least equal to two)
for interv in intervals:
    if len(interv[2]) >= 2:
        batch_type = check_batch_type(interv)
        #print(batch_type)
        df_types = df_types.append({'batch_type': batch_type, 'len': len(interv[2])}, ignore_index=True)




# STATISTICS

# total number of batches with two and more events
df_batches_new = df_batches[df_batches['Number of events']!=1]



df_end = df_types.loc[df_types['batch_type'] == 'Batching on End']
df_seq = df_types.loc[df_types['batch_type'] == 'Sequential batching']
df_con = df_types.loc[df_types['batch_type'] == 'Concurrent batching']
df_start = df_types.loc[df_types['batch_type'] == 'Batching on Start']
df_sim = df_types.loc[df_types['batch_type'] == 'Simultaneous batching']


st.subheader('Total number of batches')
count_column = df_types['batch_type'].count()
count_column

st.text('Batching types')

# total number of batches per each batching type and in total
sum_end_batch = df_end['batch_type'].count()
sum_start_batch = df_start['batch_type'].count()
sum_seq_batch = df_seq['batch_type'].count()
sum_sim_batch = df_sim['batch_type'].count()
sum_con_batch = df_con['batch_type'].count()

# Table
batching_types = {'concurrent': [sum_con_batch], 'sequential': [sum_seq_batch], 'simultaneous': [sum_sim_batch], 'on start': [sum_start_batch], 'on end': [sum_end_batch]}
df_batching_event = pd.DataFrame(data=batching_types)
df_batching_event


# total number of batched events per each batching type and in total
sum_end = df_end['len'].sum()
sum_start = df_start['len'].sum()
sum_seq = df_seq['len'].sum()
sum_con = df_con['len'].sum()
sum_sim = df_sim['len'].sum()
sum_all_batches = sum_end + sum_start + sum_seq + sum_con + sum_sim

st.subheader('Total number of batched events, all types')
sum_all_batches

st.text('Batched events per types')
# Table
batching_events_types = {'concurrent': [sum_con], 'sequential': [sum_seq], 'simultaneous': [sum_sim], 'on start': [sum_start], 'on end': [sum_end]}
df_batching_event_types = pd.DataFrame(data=batching_events_types)
df_batching_event_types

# total number of batched events, all batching types
#st.text('Total number of batched events, all batching types')
#sum_all_event = df_types['len'].sum()
#sum_all_event

st.text('Batches and batched events')
df_batches_new



#######
# Time level batch detection end
#######


#######
# Case level batch detection
#######

def load_dataframe():
    df = pm4py.read_csv("event_log.csv", sep=";")
    df = pm4py.objects.log.util.dataframe_utils.convert_timestamp_columns_in_df(df)
    df = df.sort_values(['start time', 'end time'])
    df["event id"] = df.index.astype(str)
    df = df.reset_index()
    return df


def get_groups_from_dataframe(df):
    return df.groupby(["activity", "resource"]).size().to_dict()


def merge_overlapping_intervals(intervals):
    continue_cycle = True
    while continue_cycle:
        continue_cycle = False
        i = 0
        while i < len(intervals) - 1:
            if intervals[i][1] > intervals[i + 1][0]:
                # decide to merge interval i and i+1
                new_interval = (min(intervals[i][0], intervals[i + 1][0]), max(intervals[i][1], intervals[i + 1][1]),
                                intervals[i][2].union(intervals[i + 1][2]))
                # add the new interval to the list
                intervals.append(new_interval)
                # remove the i+1 interval
                del intervals[i + 1]
                # remove the i interval
                del intervals[i]
                # sort the intervals
                intervals.sort()
                # set the variable continue_cycle to True
                continue_cycle = True
                # interrupt the current iteration on the intervals
                break
            i = i + 1
    return intervals


def merge_near_intervals(intervals, max_allowed_distance):
    continue_cycle = True
    while continue_cycle:
        continue_cycle = False
        i = 0
        while i < len(intervals) - 1:
            if intervals[i + 1][0] - intervals[i][1] <= max_allowed_distance:
                # decide to merge interval i and i+1
                new_interval = (min(intervals[i][0], intervals[i + 1][0]), max(intervals[i][1], intervals[i + 1][1]),
                                intervals[i][2].union(intervals[i + 1][2]))
                # add the new interval to the list
                intervals.append(new_interval)
                # remove the i+1 interval
                del intervals[i + 1]
                # remove the i interval
                del intervals[i]
                # sort the intervals
                intervals.sort()
                # set the variable continue_cycle to True
                continue_cycle = True
                # interrupt the current iteration on the intervals
                break
            i = i + 1
    return intervals


def check_batch_type(batch):
    events_batch = sorted(list(batch[2]))
    # take the minimum of the left-extreme of each interval
    min_left_events = min(ev[0] for ev in events_batch)
    # take the maximum of the left-extreme of each interval
    max_left_events = max(ev[0] for ev in events_batch)
    # take the minimum of the right-extreme of each interval
    min_right_events = min(ev[1] for ev in events_batch)
    # take the maximum of the right-extreme of each interval
    max_right_events = max(ev[1] for ev in events_batch)

    # CONDITION 1 - All the events in the batch have identical start and end timestamps
    if min_left_events == max_left_events and min_right_events == max_right_events:
        return "Simultaneous"
    # CONDITION 4 - All the events in the batch have identical start timestamp:
    if min_left_events == max_left_events:
        return "Batching on Start"
    # CONDITION 5 - All the events in the batch have identical end timestamp:
    if min_right_events == max_right_events:
        return "Batching on End"

    # now we could be in the SEQUENTIAL batching or the CONCURRENT batching
    # in order to be in the SEQUENTIAL, we need that for all the consecutive events the end of the first is equal to the start of the second
    is_sequential = True
    i = 0
    while i < len(events_batch) - 1:
        # if there are two consecutive events that are not sequentially matched, then we automatically fall inside the CONCURRENT batching
        if events_batch[i][1] != events_batch[i + 1][0]:
            is_sequential = False
            break
        i = i + 1
    if is_sequential:
        return "Sequential batching"
    else:
        return "Concurrent batching"



def get_events_from_dataframe(dataframe):
    all_events = [(x["start time"].timestamp(), x["end time"].timestamp(), x["case"]) for x in
                  dataframe[["start time", "end time", "case"]].to_dict("r")]
    all_events.sort()
    return all_events



def find_batches_with_type_and_count_cases(df, case_dict):
    events = get_events_from_dataframe(df)
    intervals = [(e[0], e[1], {(e[0], e[1], e[2])}) for e in
                 events]
    intervals.sort()
    intervals = merge_overlapping_intervals(intervals)
    intervals = merge_near_intervals(intervals, 15 * 60)
    batches = [x for x in intervals if len(x[2]) > 1]
    for batch in batches:
        batch_type = check_batch_type(batch)
        cases = set(x[2] for x in batch[2])
        for case in cases:
            case_dict[batch_type][case] = case_dict[batch_type][case] + 1


def measure_service_time(list_events):
    # we start measuring the service time
    service_time_total = 0.0

    # take the first event (the one in list_events[0]) and consider its start time (list_events[0][0]) and its end time (list_events[0][1])
    this_start = list_events[0][0]
    this_end = list_events[0][1]

    i = 1
    while i < len(list_events):
        # for the i-th event, consider its start time (list_events[i][0]) and end time (list_events[i][1])
        curr_start = list_events[i][0]
        curr_end = list_events[i][1]

        # if the current event start is greater than the previously recorded end, its time to add the difference between this_end and this_start to the service time
        if curr_start > this_end:
            service_time_total = service_time_total + this_end - this_start
            this_start = curr_start
            this_end = curr_end
        else:
            # otherwise, the events are overlapping, so update the completion time by the new end timestamp (take the maximum between the old recorded one and the new)
            this_end = max(this_end, curr_end)
        i = i + 1

    # at the end of the iteration, add the current values of this_start and this_end that still are not recorded in the service time
    service_time_total = service_time_total + this_end - this_start

    return service_time_total


dataframe = load_dataframe()
all_events = get_events_from_dataframe(dataframe)
cases = set(dataframe["case"].unique())
activities_resources = get_groups_from_dataframe(dataframe)
case_dict = {"Simultaneous": {}, "Batching on Start": {}, "Batching on End": {}, "Sequential batching": {},
             "Concurrent batching": {}, "Lead Time": {}, "Service Time": {}, "Flow Time": {}, "Flow Rate": {}}


for case in cases:
    case_events = [x for x in all_events if x[2] == case]
    lead_time = max(x[1] for x in case_events) - min(x[0] for x in case_events)
    service_time = measure_service_time(case_events)
    case_dict["Simultaneous"][case] = 0
    case_dict["Batching on Start"][case] = 0
    case_dict["Batching on End"][case] = 0
    case_dict["Sequential batching"][case] = 0
    case_dict["Concurrent batching"][case] = 0
    case_dict["Lead Time"][case] = lead_time
    case_dict["Service Time"][case] = service_time
    case_dict["Flow Time"][case] = lead_time - service_time
    case_dict["Flow Rate"][case] = float(service_time) / float(lead_time) if lead_time > 0 else 0.0


for act_res in activities_resources:
    filtered_dataframe = dataframe[dataframe["activity"] == act_res[0]]
    filtered_dataframe = filtered_dataframe[filtered_dataframe["resource"] == act_res[1]]
    find_batches_with_type_and_count_cases(filtered_dataframe, case_dict)

case_dataframe = pd.DataFrame(case_dict)
case_dataframe["case"] = case_dataframe.index.astype(str)
case_dataframe = case_dataframe.reset_index()

#Set some text to show the originald dataframe
st.subheader("Case level batching")

#Show the final dataframe
st.dataframe(case_dataframe)

######
#Case level batch detection end


# Case elvel batching statistics
st.text("Case level bathcing statistics")
df = pd.DataFrame(case_dataframe,columns=['Simultaneous','Batching on Start','Batching on End','Sequential batching', 'Concurrent batching'])
sum_column = df.sum()
st.dataframe(sum_column)


#Plotting the data

#df_scatterplot = pd.DataFrame(case_dataframe, columns=["Concurrent batching", "Lead time", "Service Time"])


source = case_dataframe
alt.Chart(source).mark_circle(size=60).encode(
    x='Service Time',
    y='Lead Time',
    #color='Origin',
    #tooltip=['Concurrent batching', 'Lead time', 'Service Time']
).interactive()	