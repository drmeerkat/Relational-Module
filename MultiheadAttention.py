import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    '''Multi-Head Attention module'''
    
    def __init__(self, d_model, d_kq, d_v, n_heads, drop_prob=0.0):
        super(MultiHeadAttention, self).__init__()
        
        init_ = lambda m: self.param_init(m, nn.init.orthogonal_, nn.init.calculate_gain('relu'))

        self.d_model = d_model
        self.d_k = d_kq
        self.d_q = d_kq
        self.d_v = d_v
        self.n_heads = n_heads
        self.linear_k = init_(nn.Linear(self.d_model, self.d_k*n_heads, bias=False))
        self.linear_q = init_(nn.Linear(self.d_model, self.d_q*n_heads, bias=False))
        self.linear_v = init_(nn.Linear(self.d_model, self.d_v*n_heads, bias=False))
        self.normalize = np.sqrt(self.d_k)
        self.linear_output = nn.Sequential(
#                                 nn.Linear(self.d_v*n_heads, self.d_model*2, bias=False),
#                                 nn.LeakyReLU(),
#                                 nn.Linear(self.d_model*2, self.d_model, bias=False)
                                  nn.Linear(self.d_v*n_heads, self.d_model, bias=False),
                             )
        
        # Assume that the dimension of linear_k/q/v are all the same
        self.layer_norm_embed = nn.LayerNorm(self.d_k*n_heads, eps=1e-6)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.atten_dropout = nn.Dropout(drop_prob)
        
    def param_init(self, module, weigth_init, gain=1):
        weigth_init(module.weight.data, gain=gain)

        return module
    
    def forward(self, entity_embeds_raw):
        b_sz, num_entities = entity_embeds_raw.size(0), entity_embeds_raw.size(1)
        # (batch_size, num_entities, d_model) -> (batch_size*num_entities, d_model)
        entity_embeds = entity_embeds_raw.reshape(-1, self.d_model)
        # (batch_size*num_entities, d_k*n_heads) -> (batch_size, num_entities, n_heads, d_k)
        embed_q = self.layer_norm_embed(self.linear_q(entity_embeds)).view(b_sz, num_entities, self.n_heads, self.d_q)
        embed_k = self.layer_norm_embed(self.linear_k(entity_embeds)).view(b_sz, num_entities, self.n_heads, self.d_k)
        embed_v = self.layer_norm_embed(self.linear_v(entity_embeds)).view(b_sz, num_entities, self.n_heads, self.d_v)

        residual_v = embed_v
        # swap n_head dim with num_entities
        # ->(batch_size, n_heads, num_entities, d_k)
        embed_q2 = embed_q.transpose(1,2)
        embed_k2 = embed_k.transpose(1,2)
        embed_v2 = embed_v.transpose(1,2)
        
        # Scaled Dot Product Attention(for each head)
        tmp = torch.matmul(embed_q2, embed_k2.transpose(2, 3))/self.normalize
        # -> (batch_size, n_heads, num_entities, num_entities_prob)
        weights = self.atten_dropout(F.softmax(tmp, dim=-1))
        # weights = self.atten_dropout(F.softmax(tmp, dim=1)) #this is the previous old/wrong implementation
        new_v = torch.matmul(weights, embed_v2)
        
        # Concatenate over head dimensioins
        # (batch_size, n_heads, num_entities, d_k) -> (batch_size, num_entities, n_heads*d_k)
        new_v = new_v.transpose(1, 2).contiguous().view(b_sz, num_entities, -1)
        new_v = self.linear_output(new_v)
        
        # residual
        output = new_v + entity_embeds_raw
        # output = new_v + residual_v.view(b_sz, num_entities, -1)
        # output = self.layer_norm_embed(output).view(b_sz, num_entities, new_v.shape[-1])
        output = self.layer_norm(output).view(b_sz, num_entities, self.d_model)
        
        return output


class RelationalAttention(nn.Base):
    '''
    Transpose the input data batch BxCxHxW into object embeddings of shape Bx(HxW)xC and process with multihead dot product attention. 
    The output shape will be BxHxWx(C+2) since the module will attach two extra dimensions to represent the location of each object in the original HxW frame.

    The output shape is
    BxHxWx(C+2) if maxout=False
    Bx(C+2) if maxout=True
    '''

    def __init__(self, d_model, d_kq, d_v, n_heads, drop_prob=0.0, maxout=False):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_kq
        self.d_q = d_kq
        self.d_v = d_v
        self.n_heads = n_heads
        self.maxout = maxout
        self.gnn = MultiHeadAttention(d_model, d_kq, d_v, n_heads, drop_prob)
        # self.gnn = MultiHeadAttention(512, 256, 256, 4)

    def forward(self, x):
        # add object position as additional GNN feature
        b_sz, n_channel, height, width = x.size()
        entity_embed = x.reshape(b_sz, n_channel, -1).transpose(1, 2)
        coord = []
        for i in range(height*width):
            # add coordinate and normalize
            coord.append([float(i//width)/height, (i%width)/width])
        coord = torch.tensor(coord, device=entity_embed.device).view(1, -1, 2).repeat(b_sz, 1, 1)
        entity_embed = torch.cat((entity_embed, coord), dim=2)
        
        out = F.relu(self.gnn(entity_embed))
        if self.maxout:
            out = torch.max(out, dim=1)[0]

        return out
