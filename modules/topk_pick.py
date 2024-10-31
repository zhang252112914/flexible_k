#modified
import torch
import torch.nn.functional  as F


def pick_frames(sequence_output, visual_output, group_mask, video_mask, pick_arrangement, K, sentence_num,similarity_metric='cosine'):
    
    batch_size, max_sentence_per_video, embedding_dim = sequence_output.shape
    _, num_frames, video_embedding_dim = visual_output.shape

    picked_frames = torch.zeros((batch_size, K),device=visual_output.device)


    for i in range(batch_size):
        frame_indices = []
        n = 0
        for j in range(max_sentence_per_video):
            if group_mask[i][j] == 0:
                continue
            k = pick_arrangement[i][n]
            n += 1
            sentence_embedding = sequence_output[i, j].unsqueeze(0)
            frame_embeddings = visual_output[i][video_mask[i] == 1]
            similarity = F.cosine_similarity(sentence_embedding, frame_embeddings, dim=1)
            top_k_indices = similarity.topk(min(K, pick_arrangement[i][j]), largest=True).indices
            frame_indices.append(top_k_indices)
        frame_indices = torch.cat(frame_indices)[:K]
        frame_indices = frame_indices.sort()[0]  # 排序以保持帧顺序一致
        picked_frames[i, :len(frame_indices)] = frame_indices
    return picked_frames.long()