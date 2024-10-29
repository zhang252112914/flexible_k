import torch
import torch.nn.functional  as F

def pick_frames(text_output, visual_output, group_mask, video_mask ,pick_arrangement, similarity_metric='cosine'):
    
    # Normalize the embeddings if using cosine similarity
    if similarity_metric == 'cosine':
        text_output = F.normalize(text_output, p=2, dim=-1)
        visual_output = F.normalize(visual_output, p=2, dim=-1)
        similarity_matrix = torch.matmul(text_output, visual_output.transpose(1, 2))  # (batch_size, num_sentences, num_frames)
    elif similarity_metric == 'euclidean':
        similarity_matrix = -torch.cdist(text_output, visual_output, p=2)  # 转为负值，便于选择 top-k 最大值
    
