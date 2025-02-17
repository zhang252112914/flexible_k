import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_mha_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.visualize = config.xpool_visualize
        self.counter = 0
        self.pic_id = 0
        self.save_dir = config.xpool_visualize_dir
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    
    def forward(self, text_embeds, video_embeds, text_mask):
        """
        Input
            text_embeds: batch_size x embed_dim
            video_embeds: batch_size x num_frames x embed_dim
        Output
            out: batch_size x num_frames x embed_dim
        """
        batch_size, num_texts, _ = text_embeds.shape
        q = self.q_proj(text_embeds)
        q = q.reshape(batch_size ,num_texts, self.num_heads, self.head_dim)
        q = q.permute(0,2,3,1) # batch_size x num_heads x head_dim x num_texts

        _, num_frames, _ = video_embeds.shape
        k = self.k_proj(video_embeds)
        k = k.reshape(batch_size, num_frames, self.num_heads, self.head_dim)
        k = k.permute(0,2,1,3) # batch_size x num_heads x num_frames x head_dim

        v = self.v_proj(video_embeds)
        v = v.reshape(batch_size, num_frames, self.num_heads, self.head_dim)
        v = v.permute(0,1,3,2) # batch_size x num_frames x head_dim x num_heads

        attention_logits = k @ q # batch_size x num_heads x num_frames x num_texts
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=-1)  # batch_size x num_heads x num_frames x num_texts

        attention_weights = attention_weights * text_mask.unsqueeze(1).unsqueeze(1) # batch_size x 1 x 1 x num_texts
        if self.visualize and self.counter % 10 == 0:
            self.save_attention_weights(attention_weights)
            self.pic_id += 1
        self.counter += 1
        
        attention_weights = attention_weights.permute(0,2,1,3) # batch_size x num_frames x num_heads x num_texts

        attention = v @ attention_weights # batch_size x num_frames x head_dim x num_texts
        valid = text_mask.unsqueeze(1).unsqueeze(1).sum(dim=-1) # batch_size x 1 x 1
        attention = attention.sum(dim=-1) / valid # batch_size x num_frames x head_dim
        #attention = attention.mean(dim=-1) # batch_size x num_frames x head_dim

        o = self.out_proj(attention)
        return o

    
    def save_attention_weights(self, attention_weights):
        """Save attention weights as images to the disk"""
        # Assuming batch_size = 1 for visualization
        attention_weights = attention_weights.squeeze(1)  # remove head_dim
        attention_weights = attention_weights[0].cpu().detach()  # select the first batch
    
        fig, ax = plt.subplots(figsize=(5, 10))

        ax.imshow(attention_weights.cpu().detach().numpy(), cmap='viridis', aspect='auto')
        ax.set_title(f'Attention Weights')
        ax.set_xlabel('Texts')
        ax.set_ylabel('Frame Indices')

        # Save image to file with a unique name based on the forward call count
        image_path = f"{self.save_dir}/attention_weights_{self.pic_id}.png"
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close(fig)


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.embed_dim = config.embed_dim
        dropout = config.transformer_dropout

        self.cross_attn = MultiHeadedAttention(config)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
            
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)


    def forward(self, text_embeds, video_embeds, text_mask):
        """
        Input
            text_embeds: batch_size x embed_dim
            video_embeds: batch_size x num_frames x embed_dim
        Output
            out: batch_size x num_frames x embed_dim
        """
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        # num_vids x num_texts x embed_dim
        attn_out = self.cross_attn(text_embeds, video_embeds, text_mask)     # multi-head attention
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        return out
