#modified
import torch
import torch.nn.functional  as F


def pick_frames_event(sequence_output, visual_output, group_mask, video_mask, pick_arrangement, K, sentence_num, onlyone, ranges, similarity_metric='cosine'):
    
    batch_size, max_sentence_per_video, embedding_dim = sequence_output.shape
    _, num_frames, video_embedding_dim = visual_output.shape

    picked_frames = torch.full((batch_size, K), -1, device=visual_output.device)


    for i in range(batch_size):
        frame_indices = []
        n = 0
        for j in range(max_sentence_per_video):
            if group_mask[i][j] == 0:
                continue
            # k是安排的帧数，K是总共的帧数（是一个参数）
            k = pick_arrangement[i][n]
            n += 1
            sentence_embedding = sequence_output[i, j].unsqueeze(0)
            start, end = ranges[i][j]

            # 检查对应区域是否具有足够的帧数
            # 检查两次的必要性在于，第二次是避免舍入造成的帧数不足

            duration = end - start
            if duration < min(K, k):
                make_up = min(K, k) - duration
                new_start = max(0, start - make_up)
                if new_start == 0:
                    end = min(num_frames, end + make_up - start)
                    start = 0
                else:
                    start = new_start

            #print (start - end, min(k, K))

            frame_embeddings = visual_output[i][start:end]
            frame_mask = video_mask[i][start:end]
            frame_embeddings = frame_embeddings[frame_mask == 1]
            similarity = F.cosine_similarity(sentence_embedding, frame_embeddings, dim=1)

            top_k_indices = similarity.topk(min(K, k), largest=True).indices + start
            frame_indices.append(top_k_indices)
        frame_indices = torch.cat(frame_indices)[:K]
        frame_indices = frame_indices.sort()[0]  # 排序以保持帧顺序一致
        picked_frames[i, :len(frame_indices)] = frame_indices
    return picked_frames.long()

def get_global_representation(visual_output, video_mask):
    video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
    visual_output = visual_output * video_mask_un
    video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
    video_mask_un_sum[video_mask_un_sum == 0.] = 1.
    video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
    return video_out

def add_global_info(visual_output, video_mask, global_visual_output, vt_mask):
    batch_size, num_picked_frames, embedding_dim = visual_output.shape
    
    global_visual_output = global_visual_output.unsqueeze(1)  # (batch_size, 1, embedding_dim)
    updated_visual_output = torch.cat([global_visual_output, visual_output], dim=1)
    
    global_mask = torch.ones((batch_size, 1), dtype=video_mask.dtype, device=video_mask.device)
    updated_video_mask = torch.cat([global_mask, video_mask], dim=1)
    
    text_length = vt_mask.shape[2]
    global_vt_mask = torch.ones((batch_size, 1, text_length), dtype=vt_mask.dtype, device=vt_mask.device)
    updated_vt_mask = torch.cat([global_vt_mask, vt_mask], dim=1)
    
    return updated_visual_output, updated_video_mask, updated_vt_mask