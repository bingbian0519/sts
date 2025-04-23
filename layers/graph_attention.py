import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphAttentionLayerWithEmbedding(nn.Module):
    """
    GCNConv + 多头自注意力 + 可学习关节嵌入
    """
    def __init__(self, in_channels, out_channels, num_joints,
                 heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = (out_channels // heads) ** 0.5
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

        # 关节嵌入
        self.joint_embed = nn.Embedding(num_joints, out_channels)

        # 轻量 GCNConv 用于局部平滑
        self.gcn = GCNConv(out_channels, out_channels, add_self_loops=False)

    def forward(self, x, edge_index, joint_ids):
        """
        x: (N, C_in)        节点特征
        edge_index: (2, E)  边索引
        joint_ids: (N,)     关节 id 0~num_joints-1
        """
        # ---------- 1 关节嵌入 ----------
        x = x + self.joint_embed(joint_ids)

        # ---------- 2 计算多头自注意力 ----------
        Q = self.q_proj(x).view(x.size(0), self.heads, -1)   # (N, H, d_k)
        K = self.k_proj(x).view(x.size(0), self.heads, -1)
        V = self.v_proj(x).view(x.size(0), self.heads, -1)

        scores = torch.einsum('nhd,mhd->hnm', Q, K) / self.scale   # (H, N, N)
        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)

        out = torch.einsum('hnm,mhd->nhd', attn, V).reshape(x.size(0), -1)  # (N, C_out)

        # ---------- 3 走一次 GCNConv ----------
        out = self.gcn(out, edge_index)

        return out
