import torch
import torch.nn as nn
import math

def global_attention(frame_features, global_feature):
    bz, num_frames, embed_dim = frame_features.shape
    Q = global_feature.unsqueeze(1) # (bz, 1, embed_dim)
    K = frame_features
    V = frame_features

    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(embed_dim) # (bz, 1, num_frames)

    attn_weights = torch.softmax(scores, dim=-1) # (bz, 1, num_frames)

    attended_feature = torch.bmm(attn_weights, V) # (bz, 1, embed_dim)
    attended_feature = attended_feature.squeeze(1) # (bz, embed_dim)

    return attended_feature