"""
File: MergeSingleTraces_openworld.py
Author: jzx-bupt
"""
import pickle
import numpy as np
import os
import random
import copy

from tqdm import tqdm

input_dir = r'D:\论文工作\多页指纹攻击\datasets\tmp/'
output_dir = r'D:\论文工作\多页指纹攻击\datasets\tbb_er1_valid\tbb_2tabs_duration/'

valid_category_num = 50
trace_length = 5120     # just used for 'BAPM' or 'BOTH'
padding = False  # whether padding on COCO_WF-format
page_num = 2
overlapped_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
pure_ratio_threshold = 0.1  # used for 'index' or 'duration'
expand_rate = 1
task_type = 'multi'    # 'binary' or 'multi'
merge_type = 'duration'     # 'index' or 'duration' or 'delay'
min_delay = 2   # just used for delay
max_delay = 6   # just used for delay
save_format = 'CONJUNCTION'    # 'BAPM' or 'COCO_WF' or 'BOTH' or 'CONJUNCTION'
complete_randomization = True  # allow continuous same classes or not
devide_num = 10000  # a new file is generated to prevent memory overflow if the samples' number exceeds this value


def rm_zero(sequence):
    index = len(sequence)
    # Traverse the 'sequence' in reverse order until the current element is not equal to 0.
    # Set 'index' to the current packet index + 1 (=the order of the packet in the sequence).
    # In the end, 'index' will be the index of the last packet in 'sequence' with a value not equal to 0 + 1.
    # The returned value 'sequence[:index]' represents the front part of the sequence without padding values.
    # In other words, all elements in the original sequence after the index-th position are 0.
    for i in range(len(sequence) - 1, -1, -1):
        if sequence[i] != 0:
            index = i + 1
            break
    return sequence[:index]


# Selecting an overlapping starting point based on the number of packets
def merge_with_index(times, datas, ratio):
    time = times[0]
    data = datas[0]
    split_index = int(len(time) * (1 - ratio))
    split_time = time[split_index]
    # Add the clean segment to the merged segments
    merged_time = list(time[:split_index])
    merged_data = list(data[:split_index])
    start = 0
    merged_anno = [{"bbox": [start, 0, 0, 1]}]

    res_time = time[split_index:]
    res_data = data[split_index:]

    for time, data in list(zip(times, datas))[1:]:
        time = [(t + split_time) for t in time]
        data = [(d * 1) for d in data]
        index1 = index2 = 0

        while index1 < len(res_time) and index2 < len(time):
            # Traverse the time interval sequences of the front and back pages with index1 and index2 respectively
            # Arrange both sequences in the final merged list in ascending order of total time intervals
            # The second page only adds the front mixed part and the middle clean segment, with the rear part used for the next page's mixing
            if res_time[index1] <= time[index2]:
                merged_time.append(res_time[index1])
                merged_data.append(res_data[index1])
                index1 += 1
                # When the first page is traversed, modify the attributes in its annotation dictionary
                if index1 == len(res_time):
                    end = len(merged_time) - 1
                    merged_anno[-1]["area"] = end - merged_anno[-1]["bbox"][0]
                    merged_anno[-1]["bbox"][2] = merged_anno[-1]["area"]
            else:
                merged_time.append(time[index2])
                merged_data.append(data[index2])
                index2 += 1
                if index2 == 1:
                    start = len(merged_time) - 1

        # When a mixed segment is finished, add a new dictionary in annotation to store the attributes of the second page
        merged_anno.append({"bbox": [start, 0, 0, 1]})

        # index1 represents the length of the sequence of the previous page that has been traversed
        # index2 represents the length of the sequence of the current page that has been traversed
        # If the sequence of time intervals of the previous page is not completely traversed (while the second page is already traversed), or both the first and second pages are traversed at the same time, interrupt the merge
        if index1 < len(res_time) or index2 == len(time):
            return None

        # If the starting point of the rear part of the current page used for overlap is earlier than the end point of the traversed segment, it means that this page does not contain clean segments after merging, interrupt the merge
        split_index = int(len(time) * (1 - ratio))
        if split_index <= index2:
            return None

        # If the proportion of clean segments in the current page is less than the preset threshold, interrupt the merge
        if (split_index - index2) / len(time) < pure_ratio_threshold:
            return None

        # Append the middle clean segments from the second page to the mixed sequence
        if index2 < len(time):
            merged_time.extend(time[index2:split_index])
            merged_data.extend(data[index2:split_index])

        res_time = time[split_index:]
        res_data = data[split_index:]

    merged_time.extend(res_time)
    merged_data.extend(res_data)

    # Modify the attributes in the last page's annotation dictionary
    end = len(merged_time) - 1
    merged_anno[-1]["area"] = end - merged_anno[-1]["bbox"][0]
    merged_anno[-1]["bbox"][2] = merged_anno[-1]["area"]

    return merged_time, merged_data, merged_anno


# Selecting an overlapping starting point based on the length of loading time
def merge_with_durationscale(times, datas, ratio):
    time = times[0]
    data = datas[0]
    split_index = 0
    split_time = np.max(time) * (1 - ratio)
    for i, packet_time in enumerate(time):
        if packet_time >= split_time:
            split_index = i
            break
    # Add the clean segment to the merged segments
    merged_time = list(time[:split_index])
    merged_data = list(data[:split_index])
    start = 0
    merged_anno = [{"bbox": [start, 0, 0, 1]}]

    res_time = time[split_index:]
    res_data = data[split_index:]

    for time, data in list(zip(times, datas))[1:]:
        time = [(t + split_time) for t in time]
        data = [(d * 1) for d in data]
        index1 = index2 = 0

        while index1 < len(res_time) and index2 < len(time):
            # Traverse the time interval sequences of the front and back pages with index1 and index2 respectively
            # Arrange both sequences in the final merged list in ascending order of total time intervals
            # The second page only adds the front mixed part and the middle clean segment, with the rear part used for the next page's mixing
            if res_time[index1] <= time[index2]:
                merged_time.append(res_time[index1])
                merged_data.append(res_data[index1])
                index1 += 1
                # When the first page is traversed, modify the attributes in its annotation dictionary
                if index1 == len(res_time):
                    end = len(merged_time) - 1
                    merged_anno[-1]["area"] = end - merged_anno[-1]["bbox"][0]
                    merged_anno[-1]["bbox"][2] = merged_anno[-1]["area"]
            else:
                merged_time.append(time[index2])
                merged_data.append(data[index2])
                index2 += 1
                if index2 == 1:
                    start = len(merged_time) - 1

        # When a mixed segment is finished, add a new dictionary in annotation to store the attributes of the second page
        merged_anno.append({"bbox": [start, 0, 0, 1]})

        # index1 represents the length of the sequence of the previous page that has been traversed
        # index2 represents the length of the sequence of the current page that has been traversed
        # If the sequence of time intervals of the previous page is not completely traversed (while the second page is already traversed), or both the first and second pages are traversed at the same time, interrupt the merge
        if index1 < len(res_time) or index2 == len(time):
            return None

        # If the starting point of the rear part of the current page used for overlap is earlier than the end point of the traversed segment, it means that this page does not contain clean segments after merging, interrupt the merge
        split_time = np.max(time) * (1 - ratio)
        if split_time <= time[index2]:
            return None

        # If the proportion of clean segments in the current page is less than the preset threshold, interrupt the merge
        if (split_time - time[index2]) / (np.max(time) - np.min(time)) < pure_ratio_threshold:
            return None

        for i, packet_time in enumerate(time):
            if packet_time >= split_time:
                split_index = i
                break

        # Append the middle clean segments from the second page to the mixed sequence
        if index2 < len(time):
            merged_time.extend(time[index2:split_index])
            merged_data.extend(data[index2:split_index])

        res_time = time[split_index:]
        res_data = data[split_index:]

    merged_time.extend(res_time)
    merged_data.extend(res_data)

    # Modify the attributes in the last page's annotation dictionary
    end = len(merged_time) - 1
    merged_anno[-1]["area"] = end - merged_anno[-1]["bbox"][0]
    merged_anno[-1]["bbox"][2] = merged_anno[-1]["area"]

    return merged_time, merged_data, merged_anno


# Selecting an overlapping starting point based on continuous access delay
def merge_with_delaytime(times, datas, min_interval, max_interval):
    delay_cum = 0
    for page_idx, time in enumerate(times):
        if page_idx == 0:
            delay_cum += random.uniform(min_interval, max_interval)
        else:
            for t_idx, t in enumerate(time):
                time[t_idx] += delay_cum
            times[page_idx] = time
            delay_cum += random.uniform(min_interval, max_interval)
            delay_cum += random.uniform(min_interval, max_interval)
    # print(times)
    # raise ValueError
    pointers = [0] * page_num
    merged_time, merged_data = [], []
    while True:
        # Find the page number 'i' where the current minimum element is located (from the timestamp sequences of 'page_num' pages)
        min_time = float('inf')
        min_index = -1
        for i in range(page_num):
            # 'times[i]' represents the timestamp sequence of the i-th page
            # 'pointers[i]' represents the pointer (current index in the sequence) of the i-th page's sequence
            if pointers[i] < len(times[i]) and times[i][pointers[i]] < min_time:
                min_time = times[i][pointers[i]]
                min_index = i
        # If all elements have been merged, exit the loop
        if min_index == -1:
            break
        # Add the smallest timestamp to 'merged_time'
        merged_time.append(min_time)
        # Add the data packet direction corresponding to the smallest timestamp to 'merged_data'
        merged_data.append(datas[min_index][pointers[min_index]])
        # Move the pointer to the next position in the sequence containing the smallest timestamp
        pointers[min_index] += 1

    return merged_time, merged_data, None



def load_raw_data(file):
    if file.split('.')[-1] == 'npz':    # handle datasets format of BAPM
        raw_times = np.load(input_dir + file)['time']
        raw_datas = np.load(input_dir + file)['data']
        raw_labels = np.load(input_dir + file)['label']
    else:
        with open(input_dir + file, 'rb') as f:
            raw_dict = pickle.load(f)
        raw_times = raw_dict['time']
        raw_datas = raw_dict['data']
        raw_labels = raw_dict['label']
    return raw_times, raw_datas, raw_labels


def save_merged_traces(merged_info, file):
    print('saving merged traces')
    merged_times, merged_datas, merged_annos, merged_labels, bapm_labels = merged_info
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if save_format in {'BAPM', 'BOTH'}:
        # Process the mixed sequence into a uniform length, padding or trimming as needed
        padded_times, padded_datas = [], []
        for merged_time, merged_data in zip(merged_times, merged_datas):
            padded_time = copy.deepcopy(merged_time)
            padded_data = copy.deepcopy(merged_data)
            for _ in range(page_num * trace_length - len(merged_time)):
                padded_time.append(0)
                padded_data.append(0)
            if len(merged_time) > page_num * trace_length:
                padded_time = merged_time[:page_num * trace_length]
                padded_data = merged_data[:page_num * trace_length]
            padded_times.append(padded_time)
            padded_datas.append(padded_data)
        bapm_times = np.array(padded_times).astype(np.float64)
        bapm_datas = np.array(padded_datas).astype(np.int8)
        bapm_labels = np.array(bapm_labels).astype(np.int8)
        np.savez(output_dir + 'BAPM_' + file, time=bapm_times, data=bapm_datas, label=bapm_labels)
    if save_format in {'COCO_WF', 'BOTH', 'CONJUNCTION'}:
        traces, annotations, categories = [], [], []
        anno_count = 0
        if merge_type != 'delay':
            for i in range(len(merged_times)):
                traces.append({'id': i,
                               'file_name': str(i) + '.npy',
                               'width': len(merged_times[i]) if not padding else page_num * trace_length,
                               'height': 1})
                for label, anno in zip(merged_labels[i], merged_annos[i]):
                    annotations.append({'id': anno_count,
                                        'area': anno['area'],
                                        'iscrowd': 0,  # necessary attribute for computeIoU
                                        'trace_id': i,
                                        'bbox': anno['bbox'],
                                        'category_id': int(label)})  # Object of type int8 is not JSON serializable
                    anno_count += 1
            for i in range(valid_category_num):
                categories.append({'id': i,
                                   'name': str(i),
                                   'supercategory': 'none'})
        # Store the timestamp sequences and direction sequences of all samples in a single pickle file
        if not padding:
            for i in range(len(merged_times)):
                merged_times[i] = np.array(merged_times[i]).astype(np.float64)
                merged_datas[i] = np.array(merged_datas[i]).astype(np.int8)
        else:
            for i in range(len(merged_times)):
                for _ in range(page_num * trace_length - len(merged_times[i])):
                    merged_times[i].append(0)
                    merged_datas[i].append(0)
                if len(merged_times[i]) > page_num * trace_length:
                    merged_times[i] = merged_times[i][:page_num * trace_length]
                    merged_datas[i] = merged_datas[i][:page_num * trace_length]
                merged_times[i] = np.array(merged_times[i]).astype(np.float64)
                merged_datas[i] = np.array(merged_datas[i]).astype(np.int8)
        import pickle
        with open(os.path.join(output_dir, 'COCO_' + file.split('.')[0]), 'wb') as f:
            pickle.dump({"time": merged_times, "data": merged_datas}, f)
        # Store COCO format annotation information
        merge_dict = {"traces": traces,
                      "annotations": annotations,
                      "categories": categories}
        import json
        with open(os.path.join(output_dir, file.split('.')[0] + '_annotations.json'), 'w') as json_file:
            json.dump(merge_dict, json_file)
        if save_format == 'CONJUNCTION' and len(bapm_labels) > 0:
            bapm_labels = np.array(bapm_labels).astype(np.int8)
            np.savez(output_dir + 'BAPM_' + file, label=bapm_labels)


def merge_single_traces():
    file_list = os.listdir(input_dir)
    for file in file_list:
        merged_datas, merged_times, merged_labels, merged_annos, bapm_labels = [], [], [], [], []
        raw_times, raw_datas, raw_labels = load_raw_data(file)
        total_num = len(raw_times)
        # Count the number of unmonitored traces
        bg_num = 0
        for label in reversed(list(raw_labels)):
            if label in range(valid_category_num):
                break
            bg_num += 1
        fg_num = total_num - bg_num
        file_count = 0
        for er in range(expand_rate):
            # Iterate over all monitored single-page trace samples in the training set or test set, i represents its index
            for i in tqdm(range(fg_num)):
                merged_sequence = None
                picked_indexes = []
                # Randomly select multiple single-page traces to merge
                while not merged_sequence:
                    if complete_randomization:
                        picked_indexes = [np.random.randint(0, total_num) for _ in range(page_num)]
                    else:
                        picked_indexes = [np.random.randint(0, total_num)]
                        for j in range(1, page_num):
                            # Pick an index of background pages if the former page belongs to foreground pages
                            if picked_indexes[j-1] < fg_num:
                                picked_indexes.append(np.random.randint(fg_num, total_num))
                            # Pick an index of foreground pages if the former page belongs to background pages
                            else:
                                picked_indexes.append(np.random.randint(0, fg_num))
                        # Randomly replace a monitored trace with the current trace
                        random_index = np.random.randint(0, page_num)
                        while picked_indexes[random_index] >= fg_num:
                            random_index = np.random.randint(0, page_num)
                        picked_indexes[random_index] = i
                    # List structure does not support using more than one index to get a subset
                    picked_times = [raw_times[j] for j in picked_indexes]
                    picked_datas = [raw_datas[j] for j in picked_indexes]
                    times = [rm_zero(rt) for rt in picked_times]
                    datas = [rm_zero(rd) for rd in picked_datas]
                    ratio = random.choice(overlapped_ratios)
                    if merge_type == 'index':
                        merged_sequence = merge_with_index(times, datas, ratio)
                    elif merge_type == 'duration':
                        merged_sequence = merge_with_durationscale(times, datas, ratio)
                    else:
                        merged_sequence = merge_with_delaytime(times, datas, min_delay, max_delay)
                merged_time, merged_data, merged_anno = merged_sequence
                merged_label = raw_labels[picked_indexes]
                bapm_label = [valid_category_num if label == -1 else label for label in merged_label]
                # Remove the non-object annotations for object detection
                if save_format != 'BAPM':
                    remaining_anno_id = []
                    for j, index in enumerate(picked_indexes):
                        if index < fg_num:
                            remaining_anno_id.append(j)
                    if merged_anno:
                        merged_anno = [anno for index, anno in enumerate(merged_anno) if index in remaining_anno_id]
                    merged_label = [label for index, label in enumerate(merged_label) if index in remaining_anno_id]
                if task_type == 'binary':
                    merged_label = [0 if label in range(valid_category_num) else 1 for label in merged_label]

                merged_times.append(merged_time)
                merged_datas.append(merged_data)
                merged_annos.append(merged_anno)
                merged_labels.append(merged_label)
                bapm_labels.append(bapm_label)
                if len(merged_times) == devide_num:
                    save_merged_traces([merged_times, merged_datas, merged_annos, merged_labels, []],
                                       file + '_' + str(file_count))
                    merged_datas, merged_times, merged_labels, merged_annos = [], [], [], []
                    file_count += 1

        save_merged_traces([merged_times, merged_datas, merged_annos, merged_labels, bapm_labels],
                           file + '_' + str(file_count))



if __name__ == '__main__':
    merge_single_traces()


