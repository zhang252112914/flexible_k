import random
import numpy as np

def shuffle_video_events(segment_num, pairs, video, video_mask, duration):
    if segment_num == 0:
        return video, video_mask
    if segment_num == 1:
        return video, video_mask
    
    shuffle_pairs = pairs.copy()
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