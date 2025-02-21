from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import json
import torch
import math
from dataloaders.rawvideo_util import RawVideoExtractor


class ActivityNetMeDataLoader(Dataset):
    max_text_per_video = 27

    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            K = 16,
            fps = 3,
            min_dur=None,
            max_dur=None
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]
        self.K = K
        self.fps = fps

        self.subset = subset
        assert self.subset in ["train", "val"]

        video_id_path_dict = {"train": os.path.join(self.data_path, "train_ids.json"),
                              "val": os.path.join(self.data_path, "val_ids.json")}

        video_json_path_dict = {"train": os.path.join(self.data_path, "train.json"),
                                "val": os.path.join(self.data_path, "val_1.json")}

        video_id_list = self._get_video_id_single(video_id_path_dict[self.subset]) # all video ids in the subset

        video_dict = {} # video_id: video_path
        for root, dub_dir, video_files in os.walk(self.features_path):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])
                if video_id_ not in video_id_list:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_
        self.video_dict = video_dict

        self.video_id_list = video_id_list

        # Get iterator video ids
        self.video_id2idx_dict = {pseudo_video_id: id_ for id_, pseudo_video_id in enumerate(self.video_id_list)} # video_id: index
        # Get all captions
        with open(video_json_path_dict[self.subset], 'r') as fname:
            self.dat = json.load(fname)
        # min_dur is always none
        if min_dur is not None and max_dur is not None:
            dat1 = {}
            for key in self.dat:
                dur = self.dat[key]['duration']
                if min_dur <= dur <= max_dur:
                    dat1[key] = self.dat[key]
            self.dat = dat1

        self.video_text_pairs = []
        for video_id in self.video_id_list:
            if video_id not in self.dat:
                continue
            d = self.dat[video_id]['sentences']
            t = self.dat[video_id]['timestamps']
            self.video_text_pairs.append((video_id, self.dat[video_id]['duration'], d, t)) #[(video_id, length, sentences, timestamps(start, end))]

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.video_text_pairs)

    def _get_video_id_single(self, path):
        video_id_list = []
        with open(path, 'r') as f:
            json_data = json.load(f)

        for video_id in json_data:
            if video_id in video_id_list:
                print("reduplicate.")
            else:
                video_id_list.append(video_id)
        return video_id_list

    def _get_text(self, sentences):
        k = self.max_text_per_video
        n = len(sentences)
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        group_mask = np.zeros((k,), dtype=np.long)

        for i in range(n):
            words = self.tokenizer.tokenize(sentences[i])

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            group_mask[i] = 1

        return pairs_text, pairs_mask, group_mask

    def _get_rawvideo(self, idx, dur, s, e):
        video_mask = np.zeros((1, self.max_frames), dtype=np.long)
        max_video_length = [0] * 1

        # Pair x L x T x 3 x H x W
        video = np.zeros((1, self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)
        video_path = self.video_dict[idx]
        try:
            for i in range(1):
                # Should be optimized by gathering all asking of this video
                raw_video_data = self.rawVideoExtractor.get_video_data(video_path, dur, s, e)

                if len(raw_video_data.shape) > 3:
                    raw_video_data_clip = raw_video_data
                    # L x T x 3 x H x W
                    raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                    if self.max_frames < raw_video_slice.shape[0]:
                        if self.slice_framepos == 0:
                            video_slice = raw_video_slice[:self.max_frames, ...]
                        elif self.slice_framepos == 1:
                            video_slice = raw_video_slice[-self.max_frames:, ...]
                        else:
                            sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                            video_slice = raw_video_slice[sample_indx, ...]
                    else:
                        video_slice = raw_video_slice

                    video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                    slice_len = video_slice.shape[0]
                    max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                    if slice_len < 1:
                        pass
                    else:
                        video[i][:slice_len, ...] = video_slice
                else:
                    print("video path: {} error. video id: {}, start: {}, end: {}".format(video_path, idx, s, e))
        except Exception as excep:
            print("video path: {} error. video id: {}, start: {}, end: {}, Error: {}".format(video_path, idx, s, e,
                                                                                             excep))
            raise excep

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def _get_vt_mask(self, v_mask, dur, ts):
        res = np.zeros((self.max_text_per_video, self.max_frames), dtype=int)
        n_frames = np.count_nonzero(v_mask)
        n_texts = len(ts)
        frame_time = np.linspace(0, dur, num=n_frames)
        for k in range(n_texts):
            start, end = ts[k][0], ts[k][1]
            temp = np.ones((n_frames,), dtype=int)
            temp[frame_time < start] = 0
            temp[frame_time > end] = 0
            res[k, :n_frames] = temp
        return res
    # 检查原有的视频，对应事件是否能够产生足够数量的帧
    def check_and_expand_events(self, length, starts, ends):

        max_frames = self.max_frames
        selected_frames = self.K
        ratio = selected_frames / max_frames

        n = len(starts)
        total_duration = 0 #total_duration is the total number of seconds
        for i in range(n):
            total_duration += (ends[i] - starts[i])
        
        valid_rate = total_duration / length if length*self.fps > max_frames else total_duration*self.fps / max_frames
        if valid_rate*64 >= 16:
            return starts, ends
        
        else:
            sum_increment = (ratio - valid_rate) * length if self.fps*length > max_frames else math.ceil((selected_frames-total_duration*self.fps) / self.fps) # sum_increment是总共要增加的秒数

            for i in range(n):

                increment = math.ceil(sum_increment * (ends[i] - starts[i]) / total_duration) # 按照不同event的占比划分increment
                half_increment = increment / 2
                if starts[i] - half_increment >= 0 and ends[i] + half_increment <= length:
                    starts[i] = math.floor(starts[i] - half_increment)
                    ends[i] = math.ceil(ends[i] + half_increment)
                elif starts[i] - half_increment < 0: 
                    ends[i] = math.ceil(ends[i] + increment - starts[i])
                    starts[i] = 0
                else: 
                    starts[i] = math.floor(starts[i] - increment + (length - ends[i]))
                    ends[i] = length

            return starts, ends
    
    def generate_event_range(self, length, starts, ends):
        n = len(starts)
        k = self.max_text_per_video
        ranges = torch.zeros((k, 2), dtype=torch.long)
        for i in range(n):
            start_p = max(0, starts[i])
            end_p = min(length, ends[i])

            # add a step to check if the range of event can generate enough frames
            start = math.floor(start_p * self.max_frames / length)
            end = math.ceil(end_p * self.max_frames / length)
            ranges[i] = torch.tensor([start, end])
        return ranges

    def __getitem__(self, feature_idx):
        video_id, duration, sentences, timestamps = self.video_text_pairs[feature_idx]
        sentences_num = len(sentences)
        starts = []
        ends = []
        for timestamp in timestamps:
            starts.append(timestamp[0])
            ends.append(timestamp[1])
        starts, ends = self.check_and_expand_events(duration, starts, ends)
        pairs_text, pairs_mask, group_mask = self._get_text(sentences)
        video, video_mask = self._get_rawvideo(video_id, duration, 0, duration)
        vt_mask = self._get_vt_mask(video_mask, duration, timestamps)
        ranges = self.generate_event_range(duration, starts, ends)
        return pairs_text, pairs_mask, group_mask, video, video_mask, vt_mask, sentences_num, ranges
