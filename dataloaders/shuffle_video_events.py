import random
import numpy as np

def shuffle_video_events(segment_num, pairs, video, video_mask, duration):
    shuffle_pairs = pairs.copy()
    shuffle_pairs.sort(key=lambda x: x[0])
    random.shuffle(shuffle_pairs)
    frame_num = np.sum(video_mask)
    new_video = np.zeros(video.shape)
    new_start = 0
    for i in range(segment_num):
        pair = shuffle_pairs[i]
        start_time = pair[0]
        end_time = pair[1]
        start_frame = int(start_time / duration * frame_num)
        end_frame = int(end_time / duration * frame_num)
        new_video[new_start:new_start + end_frame - start_frame] = video[start_frame:end_frame]
        new_start += (end_frame - start_frame)
    return new_video, video_mask