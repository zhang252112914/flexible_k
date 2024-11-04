import random
import numpy as np
import torch

# process the pairs into a sorted and justified list
def process_pairs(segment_num, pairs, duration):
    if segment_num == 0:
        return segment_num, pairs
    
    
    pairs.sort(key=lambda x: x[0])    
    new_pairs = []
    
    for pair in pairs:
        if pair[0] >= pair[1] or pair[0] > duration or pair[1] < 0:
            segment_num -= 1
            continue
        
        start = max(0, pair[0])
        end = min(duration, pair[1])

        new_pairs.append((start, end))
    
    return segment_num, new_pairs

def shuffle_video_events(segment_num, pairs, video, video_mask, duration):
    if segment_num == 0:
        return video, video_mask
    if segment_num == 1:
        return video, video_mask
    
    shuffle_pairs = pairs.copy()
    shuffle_pairs.sort(key=lambda x: x[0])
    shuffle_pairs, segment_num = process_pairs(segment_num, shuffle_pairs, duration)

    random.shuffle(shuffle_pairs)
    frame_num = np.sum(video_mask)
    new_video = np.zeros(video.shape, dtype=video.dtype)
    new_start = 0

    for i in range(segment_num):
        pair = shuffle_pairs[i]
        start_time = pair[0]
        end_time = pair[1]
        start_frame = int(start_time / duration * frame_num)
        end_frame = int(end_time / duration * frame_num)
        segment_length = end_frame - start_frame

        if new_start + segment_length > new_video.shape[1]:
            segment_length = new_video.shape[1] - new_start
            end_frame = start_frame + segment_length
        new_video[0][new_start:new_start+segment_length, ...] = video[0][start_frame:end_frame, ...]
        new_start += segment_length
    return new_video, video_mask

def video_expansion(raw_video, pairs, i, duration):
    new_video = raw_video.tolist()
    intersection = (pairs[i+1][0], pairs[i][1])
    intersection_length = intersection[1]-intersection[0]

    start_idx = int(intersection[0]*len(new_video)/duration)
    end_idx = int(intersection[1]*len(new_video)/duration)
    intersection_video = new_video[start_idx:end_idx]
    new_video = new_video[:end_idx]+ intersection_video + new_video[end_idx:]
    new_duration = duration+intersection_length

    new_pairs = []
    new_pairs.extend(pairs[:i+1])
    for j in range(i+1, len(pairs)):
        new_start = pairs[j][0] + intersection_length
        new_end = pairs[j][1] + intersection_length
        new_pairs.append((new_start, new_end))

    new_video = torch.tensor(np.stack(new_video),dtype=raw_video.dtype, device=raw_video.device)
    return new_video, new_duration, new_pairs