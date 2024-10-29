import torch
import torch.nn.functional  as F


def pick_frames(sentence_embeddings, visual_output, group_mask, video_mask, pick_arrangement, similarity_metric='cosine'):
    """
    Selects top-K frames most similar to each sentence based on `Top-K` strategy, reducing `visual_output` and 
    `video_mask` to include only the selected frames.
    
    Args:
        sentence_embeddings (torch.Tensor): Text embeddings for each sentence, shape (batch_size, max_text_per_video, embedding_dim)
        visual_output (torch.Tensor): Video frame embeddings, shape (batch_size, num_frames, embedding_dim)
        group_mask (torch.Tensor): Mask for valid sentences, shape (batch_size, max_text_per_video)
        video_mask (torch.Tensor): Initial video frame mask, shape (batch_size, num_frames)
        pick_arrangement (List[List[int]]): Number of frames to pick for each sentence per batch item, with variable lengths
        similarity_metric (str): Similarity metric to use ('cosine' or 'euclidean')

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Reduced `visual_output` and `video_mask` with shapes (batch_size, K, embedding_dim) and (batch_size, K)
    """
    batch_size, num_sentences, embedding_dim = sentence_embeddings.shape
    _, num_frames, _ = visual_output.shape

    # Normalize embeddings if using cosine similarity
    if similarity_metric == 'cosine':
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=-1)
        visual_output = F.normalize(visual_output, p=2, dim=-1)

    # Determine the total number of frames selected per batch item, K
    K = max(sum(arrangement) for arrangement in pick_arrangement)  # Assuming max frames across batch items

    # Initialize tensors to store the reduced visual output and mask in final shape
    reduced_visual_output = torch.zeros((batch_size, K, embedding_dim), device=visual_output.device)
    reduced_video_mask = torch.zeros((batch_size, K), device=video_mask.device, dtype=video_mask.dtype)

    # Process each batch item
    for b in range(batch_size):
        batch_selected_indices = []  # Store indices of selected frames for each sentence in the batch

        # Restrict valid frames to those initially marked as valid by `video_mask`
        valid_frame_indices = video_mask[b].nonzero(as_tuple=True)[0]

        for s, frames_to_select in enumerate(pick_arrangement[b]):
            # Only process valid sentences
            if group_mask[b, s] > 0:
                # Calculate similarity for each valid sentence independently
                similarity_scores = torch.matmul(sentence_embeddings[b, s], visual_output[b, valid_frame_indices].T)
                
                # Select top-K indices within valid frames
                topk_indices = torch.topk(similarity_scores, frames_to_select, dim=-1).indices
                selected_indices = valid_frame_indices[topk_indices]
                batch_selected_indices.extend(selected_indices.tolist())

        # Remove duplicates and sort the indices
        batch_selected_indices = sorted(set(batch_selected_indices))
        batch_selected_indices_tensor = torch.tensor(batch_selected_indices, dtype=torch.long, device=visual_output.device)
        
        # Populate the reduced tensors for the current batch item
        num_selected = min(K, len(batch_selected_indices_tensor))  # Handle cases where fewer than K frames are available
        reduced_visual_output[b, :num_selected] = visual_output[b, batch_selected_indices_tensor[:num_selected]]
        reduced_video_mask[b, :num_selected] = 1  # Set selected frames as valid in the mask

    return reduced_visual_output, reduced_video_mask