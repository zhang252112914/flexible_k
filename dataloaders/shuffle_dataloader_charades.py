from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import json
from dataloaders.rawvideo_util import RawVideoExtractor
import dataloaders.shuffle_video_events as sve

class ShuffleCharadesMeDataloader(Dataset):
    max_text_per_video = 12

    def __init__(
            self,
            subset,
            data_path,
            features_path,
            tokenizer,
            shuffle_events,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.data_path = data_path
        self.features_path = features_path
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]
        subset = 'test' if subset == 'val' else subset
        self.subset = subset
        assert self.subset in ["train", "val", "test"]
        self.shuffle_events = shuffle_events

        fname = os.path.join(self.data_path, f'charades_sta_{subset}.txt')
        with open(fname, 'r') as f:
            L = f.readlines()
        L = [x.strip().split('##') for x in L]
        desc = [x[1] for x in L]
        L = [x[0].split(' ') for x in L]
        ID = [x[0] for x in L]
        dur = [(float(x[1]), float(x[2])) for x in L]
        dur = [(x[1], x[0]) if x[0]>x[1] else (x[0], x[1]) for x in dur]

        video_dict = {}
        for root, dub_dir, video_files in os.walk(self.features_path):
            for video_file in video_files:
                video_id_ = os.path.splitext(video_file)[0]
                if video_id_ not in ID:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = {'video': file_path_, 'sentences': [], 'start': [], 'end': [], 'segment_num': 1,  'pair': []}
        for i in range(len(ID)):
            if ID[i] not in video_dict:
                continue
            video_dict[ID[i]]['sentences'].append(desc[i])
            video_dict[ID[i]]['start'].append(dur[i][0])
            video_dict[ID[i]]['end'].append(dur[i][1])
        for key in video_dict:
            start = video_dict[key]['start']
            end = video_dict[key]['end']
            video_segments = list(zip(start, end))
            video_segments = list(set(video_segments))
            video_dict[key]['segment_num'] = len(video_segments)
            video_dict[key]['pair'] = video_segments            


        df = pd.read_csv(os.path.join(self.data_path, f'Charades_v1_{subset}.csv'))
        for k in range(len(df)):
            id_ = df.id[k]
            if id_ not in video_dict:
                continue
            video_dict[id_]['length'] = df.length[k]
        self.dat = []
        for key in video_dict:
            self.dat.append(video_dict[key])

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.dat)

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

    def _get_rawvideo(self, video_path, dur, s, e, pairs, segment_num):
        video_mask = np.zeros((1, self.max_frames), dtype=np.long)
        max_video_length = [0] * 1

        # Pair x L x T x 3 x H x W
        video = np.zeros((1, self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)
        # video_path = self.video_dict[idx]
        try:
            for i in range(1):
                # Should be optimized by gathering all asking of this video
                raw_video_data = self.rawVideoExtractor.get_video_data(video_path, dur, s, e)

                if segment_num > 1:
                    for j in range(segment_num-1):
                        if pairs[j][1] > pairs[j+1][0]:
                            raw_video_data, dur, pairs = sve.video_expansion(raw_video_data, pairs, j, dur)

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
                    print("video path: {} error. video id: {}, start: {}, end: {}".format(video_path, id, s, e))
        except Exception as excep:
            print("video path: {} error. video id: {}, start: {}, end: {}, Error: {}".format(video_path, id, s, e,
                                                                                             excep))
            raise excep

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask, pairs, segment_num, dur

    def _get_vt_mask(self, v_mask, dur, ts, te):
        res = np.zeros((self.max_text_per_video, self.max_frames), dtype=int)
        n_frames = np.count_nonzero(v_mask)
        n_texts = len(ts)
        frame_time = np.linspace(0, dur, num=n_frames)
        for k in range(n_texts):
            start, end = ts[k], te[k]
            temp = np.ones((n_frames,), dtype=int)
            temp[frame_time < start] = 0
            temp[frame_time > end] = 0
            res[k, :n_frames] = temp
        return res

    def __getitem__(self, item):
        dat = self.dat[item]
        pairs_text, pairs_mask, group_mask = self._get_text(dat['sentences'])
        duration = dat['length']
        dat['segment_num'], dat['pair'] = sve.process_pairs(dat['segment_num'], dat['pair'], duration)

        video, video_mask, dat['pairs'], dat['segment_num'], duration = self._get_rawvideo(dat['video'], duration, 0, duration, dat['pair'], dat['segment_num'])
        
        if self.shuffle_events:
            video, video_mask = sve.shuffle_video_events(dat['segment_num'], dat['pairs'], video, video_mask, duration)
        
        vt_mask = self._get_vt_mask(video_mask, duration, dat['start'], dat['end'])
        ranges=[]
        return pairs_text, pairs_mask, group_mask, video, video_mask, vt_mask, len(dat['sentences']), ranges
