import numpy as np
import sys
import os

"""
1. Convert your edf file to ascii file using their edfviewer or edf2asc executable
2. Run this code from terminal as follows:
python parse_eye_data.py  <ascii file name> <number of trials this ascii file recorded>"""

TRACKING_START = 'TrackingStart' #Your message for when the trial starts
TRACKING_END = 'TrackingEnd' #Your message for when the trial ends
EFIX = 'EFIX' #Message marker for fixation events in the ascii file

def expand_efix(chunk_parts, chunk_list):
    """This function's purpose is to re-label all the fixation events with a flag
    value of 1. In general, if you come across an entry in the csv file which
    has the fixation flag =1, that means that event happened when the participant
    fixated their eyes"""
    ts_start = int(chunk_parts[2])
    ts_end = int(chunk_parts[3])
    efix_entries = [c for c in chunk_list if int(c[1]) <= ts_end and int(c[1]) >= ts_start]
    for entry in efix_entries:
        entry[len(entry) - 1] = 1

rand_p = 0

with open(sys.argv[1]) as ascii_file:
    start_chunk = False
    chunk_index = 0

    chunk_list = []

    p_values = np.arange(1, int(sys.argv[2]) + 1) #Replace this with your trial order
    np.random.shuffle(p_values) # Comment this if you have an actual trial order

    for line in ascii_file:
        line = line.replace('\n', '').replace('\r', '').replace('\t', ' ').replace('...', '')
        chunk_parts = [p for p in line.split(' ') if p != '']

        # This works if your data structure is something like this:
        # Data samples : Timestamp Xloc Yloc Pupil_Size
        # Event samples : EFIX R   start_time	end_time	difference	  Xloc	  Yloc	   Pupil_Size
        if len(chunk_parts) >= 3:
            if chunk_parts[2] == TRACKING_START:
                start_chunk = True
                rand_p = p_values[chunk_index]
                chunk_index += 1
            elif chunk_parts[2] == TRACKING_END:
                start_chunk = False

        if start_chunk == True:
            if chunk_parts[0] == EFIX:
                expand_efix(chunk_parts, chunk_list)
                continue
            else:
                try:
                    time_stamp = int(chunk_parts[0])
                    chunk_list.append([rand_p, time_stamp, float(chunk_parts[1]), float(chunk_parts[2]), float(chunk_parts[3]), 0])
                except ValueError:
                    if chunk_parts[0].isdigit() and chunk_parts[1] == '.' and chunk_parts[2] == '.':
                        chunk_list.append([rand_p, int(chunk_parts[0]), -230491, -230491, -230491, -230491])
                        continue

with open(os.path.splitext(sys.argv[1])[0]+'.csv', 'w') as csv_file:
    csv_file.write('Trial Number, Timestamp, X, Y, Pupil Area, Fixation flag\n')
    for l in chunk_list:
        csv_file.write(str(l).replace('[', '').replace(']', '') + '\n')
