import os
import pandas as pd

def analyze_event_duration(data_path, features_path, subset="train"):
    # 加载charades的文本文件
    fname = os.path.join(data_path, f'charades_sta_{subset}.txt')
    with open(fname, 'r') as f:
        L = f.readlines()
    L = [x.strip().split('##') for x in L]
    desc = [x[1] for x in L]
    L = [x[0].split(' ') for x in L]
    ID = [x[0] for x in L]
    dur = [(float(x[1]), float(x[2])) for x in L]
    dur = [(x[1], x[0]) if x[0] > x[1] else (x[0], x[1]) for x in dur]

    # 加载视频信息
    video_dict = {}
    for root, dub_dir, video_files in os.walk(features_path):
        for video_file in video_files:
            video_id_ = os.path.splitext(video_file)[0]
            if video_id_ not in ID:
                continue
            file_path_ = os.path.join(root, video_file)
            video_dict[video_id_] = {'video': file_path_, 'sentences': [], 'start': [], 'end': []}

    for i in range(len(ID)):
        if ID[i] not in video_dict:
            continue
        video_dict[ID[i]]['sentences'].append(desc[i])
        video_dict[ID[i]]['start'].append(dur[i][0])
        video_dict[ID[i]]['end'].append(dur[i][1])

    # 加载视频时长信息
    df = pd.read_csv(os.path.join(data_path, f'Charades_v1_{subset}.csv'))
    for k in range(len(df)):
        id_ = df.id[k]
        if id_ not in video_dict:
            continue
        video_dict[id_]['length'] = df.length[k]

    # 统计低于视频总时长1/4的视频数量，并按事件数量分组
    below_quarter_count = 0
    event_counts = {1: 0, 2: 0, 3: 0, 'more_than_3': 0}  # 统计不同事件数量的视频

    for video_id, video_info in video_dict.items():
        video_length = video_info['length']  # 视频总时长
        event_durations = [end - start for start, end in zip(video_info['start'], video_info['end'])]  # 每个事件的时长
        total_event_duration = sum(event_durations)  # 所有事件的总时长
        
        # 判断事件总时长是否低于视频总时长的1/4
        if total_event_duration < video_length / 4:
            below_quarter_count += 1
            num_events = len(video_info['start'])  # 当前视频的事件数量
            
            if num_events == 1:
                event_counts[1] += 1
            elif num_events == 2:
                event_counts[2] += 1
            elif num_events == 3:
                event_counts[3] += 1
            else:
                event_counts['more_than_3'] += 1
    
    # 输出统计结果
    print(f"低于视频总时长1/4的视频数量: {below_quarter_count}")
    print(f"事件数量为 1 的视频数量: {event_counts[1]}")
    print(f"事件数量为 2 的视频数量: {event_counts[2]}")
    print(f"事件数量为 3 的视频数量: {event_counts[3]}")
    print(f"事件数量大于 3 的视频数量: {event_counts['more_than_3']}")

# 使用时，请确保替换成你实际的data_path和features_path路径
data_path = "/home/zyl/MeVTR_data_and_models/charades/annotation"
features_path = "/home/zyl/MeVTR_data_and_models/charades/Charades_v2_3"

# 调用分析函数，分析训练集
analyze_event_duration(data_path, features_path, subset="train")