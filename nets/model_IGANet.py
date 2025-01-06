import sys
#from xml.etree.ElementTree import QName
#sys.path.append("..")
import torch
import torch.nn as nn
from einops import rearrange
from nets.graph_frames import Graph
from functools import partial
from einops import rearrange, repeat
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import copy
def _get_clones(module, N):
    #init_weights(module)
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, is_activation_last=False):
        super().__init__()
        self.num_layers = num_layers
        self.is_activation_last = is_activation_last
        if not isinstance(hidden_dim, list):
            h = [hidden_dim] * (num_layers - 1)
        else:
            assert isinstance(hidden_dim, list), 'hidden_dim arg should be list or a number'
            assert len(hidden_dim) == num_layers-1, 'len(hidden_dim) != num_layers-1'
            h = hidden_dim

        # self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        all_dims = [input_dim] + h + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(all_dims[i], all_dims[i+1]) for i in range(num_layers))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))

        if self.is_activation_last:
            x = F.relu(self.layers[-1](x))
        else:
            x = self.layers[-1](x)

        return x
    

class linear_block(nn.Module):
    def __init__(self, ch_in, ch_out, drop=0.1):
        super(linear_block,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(ch_in, ch_out),
            nn.GELU(),
            nn.Dropout(drop)
        )
    def forward(self,x):
        x = self.linear(x)
        return x

class encoder(nn.Module): # 2,256,512
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        dim_0 = 2
        dim_2 = 64
        dim_3 = 128
        dim_4 = 256
        dim_5 = 512
        
        self.fc1 = nn.Linear(dim_0, dim_2)   
        self.fc3 = nn.Linear(dim_2, dim_3)
        self.fc4 = nn.Linear(dim_3, dim_4)
        self.fc5 = nn.Linear(dim_4, dim_5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
    
class uMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        self.linear512_256 = linear_block(in_features,256,drop)
        self.linear256_256 = linear_block(256,256,drop) 
        self.linear256_512 = linear_block(256,in_features,drop)

    def forward(self, x):
        # down          
        x = self.linear512_256(x)
        res_256 = x 
        # mid
        x = self.linear256_256(x)
        x = x + res_256
        # up
        x = self.linear256_512(x) 
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # 
        self.attn_drop = nn.Dropout(attn_drop) # p=0
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)  # 0

    def forward(self, x, f,joints_pos_embed=None):
        B, N, C = x.shape # b,j,c
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)   #3*B*self.num_heads*N*C


        q, k, v = qkv[0], qkv[1], qkv[2]  
        # if joints_pos_embed is not None:
        #     joints_pos_embed = torch.cat( [joints_pos_embed,torch.zeros((42,C-256), device = joints_pos_embed.device)],dim=-1 ).reshape(N,self.num_heads, C // self.num_heads).unsqueeze(0).permute(0,2,1,3)
        #     q= q+joints_pos_embed
        #     k= k+joints_pos_embed #forgot k


        attn = (q @ k.transpose(-2, -1)) * self.scale # b,heads,17,4 @ b,heads,4,17 = b,heads,17,17
        attn = attn.softmax(dim=-1) 
        attn = self.attn_drop(attn)

        f = f.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous() # b,j,h,c -> b,h,j,c
        x = (attn @ v)
        attn2gcn = x.clone().permute(0, 2, 1, 3).contiguous().reshape(B, N, C).contiguous()
        x = x + f
        x = x.transpose(1, 2).reshape(B, N, C).contiguous() 
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn2gcn
    
    
class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()

        self.adj = adj # 4,17,17
        self.kernel_size = adj.size(0)
        #
        self.conv1d = nn.Conv1d(in_channels, out_channels * self.kernel_size, kernel_size=1)
    def forward(self, x): # b,j,c
        # conv1d
        x = rearrange(x,"b j c -> b c j") 
        x = self.conv1d(x)   # b,c*kernel_size,j = b,c*4,j
        x = rearrange(x,"b ck j -> b ck 1 j")
        b, kc, t, v = x.size()
        x = x.view(b, self.kernel_size, kc//self.kernel_size, t, v) # b,k, kc/k, 1, j 
        # x = torch.einsum('bkctv, kvw->bctw', (x, self.adj))   # bctw   b,c,1,j 
        x = torch.einsum('bkctv,kvw->bctw',x,self.adj)   # bctw   b,c,1,j 
        x = x.contiguous()
        x = rearrange(x, 'b c 1 j -> b j c') 
        return x.contiguous()
    
class My_GCN(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):
        super(My_GCN, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        #self.adj_sq = adj_sq
        self.activation = activation
        #self.scale_identity = scale_identity
        #self.I = Parameter(torch.eye(number_of_nodes, requires_grad=False).unsqueeze(0))

    def laplacian(self, A_hat):
        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L
    
    
    def laplacian_batch(self, A_hat):
        #batch, N = A.shape[:2]
        #if self.adj_sq:
        #    A = torch.bmm(A, A)  # use A^2 to increase graph connectivity
        #I = torch.eye(N).unsqueeze(0).to(device)
        #I = self.I
        #if self.scale_identity:
        #    I = 2 * I  # increase weight of self connections
        #A_hat = A + I
        batch, N = A_hat.shape[:2]
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N).contiguous()
        return L


    def forward(self, X, A):
        batch = X.size(0)
        #A = self.laplacian(A)
        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)
        #X = self.fc(torch.bmm(A_hat, X))
        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X))
        if self.activation is not None:
            X = self.activation(X)
        return X

class Block(nn.Module): # drop=0.1
    def __init__(self, length, dim, adj, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,joints_pos_embed = None,use_origin_gcn = True):
        # length =17, dim = args.channel = 512, tokens_dim = args.token_dim=256, channels_dim = args.d_hid = 1024
        super().__init__()
        self.use_origin_gcn = use_origin_gcn
        self.adj = adj
        #only update weight between two hand
        #self.adj.weight.register_hook(lambda grad: grad.mul_(gradient_mask))
        self.joints_pos_embed = joints_pos_embed
        # GCN
        self.norm1 = norm_layer(length)
        if use_origin_gcn:
            self.GCN_Block1 = GCN(dim, dim, adj)
            self.GCN_Block2 = GCN(dim, dim, adj)
        else:
            self.GCN_Block1 = My_GCN(dim, dim)
            self.GCN_Block2 = My_GCN(dim, dim)

        self.adj = adj
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # attention
        self.norm_att1=norm_layer(dim)
        self.num_heads = 8
        qkv_bias =  True
        qk_scale = None
        attn_drop = 0.2
        proj_drop = 0.25
        self.attn = Attention(
            dim, num_heads=self.num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop) 

        # 512,1024
        self.norm2 = norm_layer(dim)
        self.uMLP = uMLP(in_features=dim, hidden_features=256, act_layer=act_layer, drop=0.20)
        
        gcn2attn_p = 0.15
        Attn2gcn_p = 0.15
        self.gcn2Attn_drop = nn.Dropout(p = gcn2attn_p)
        self.Attn2gcn_drop = nn.Dropout(p = Attn2gcn_p)
        # self.s_gcn2attn = nn.Parameter(torch.tensor([(0.5)], dtype=torch.float32), requires_grad=True) 
        # self.s_attn2gcn = nn.Parameter(torch.tensor([(0.8)], dtype=torch.float32), requires_grad=True) 
        self.s_gcn2attn = 0.5
        self.s_attn2gcn = 0.8

    def forward(self, x):
        # B,J,dim 
        res1 = x # b,j,c
        x_atten = x.clone()
        x_gcn_1 = x.clone()
        # GCN
        x_gcn_1 = rearrange(x_gcn_1,"b j c -> b c j").contiguous() 
        x_gcn_1 = self.norm1(x_gcn_1) # b,c,j
        x_gcn_1 = rearrange(x_gcn_1,"b j c -> b c j").contiguous()
        if self.use_origin_gcn:
            x_gcn_1 = self.GCN_Block1(x_gcn_1)  # b,j,c
        else:
            x_gcn_1 = self.GCN_Block1(x_gcn_1,self.adj)

        # Atten
        x_atten = self.norm_att1(x_atten)
        if self.joints_pos_embed is None:
            x_atten, attn2gcn = self.attn(x_atten, f= self.gcn2Attn_drop(x_gcn_1*self.s_gcn2attn))
        else:
            x_atten, attn2gcn = self.attn(x_atten, f= self.gcn2Attn_drop(x_gcn_1*self.s_gcn2attn),joints_pos_embed=self.joints_pos_embed)

        
        if self.use_origin_gcn:
            x_gcn_2 = self.GCN_Block2(x_gcn_1 + self.Attn2gcn_drop(attn2gcn*self.s_attn2gcn))  # b, j, c
        else:
            x_gcn_2 = self.GCN_Block2(x_gcn_1 + self.Attn2gcn_drop(attn2gcn*self.s_attn2gcn),self.adj)  # b, j, c

        x = res1 + self.drop_path(x_gcn_2 + x_atten)

        # uMLP
        res2 = x  # b,j,c
        x = self.norm2(x)
        x =  res2 + self.drop_path(self.uMLP(x))  
        return x




        # # B,J,dim 
        # res1 = x # b,j,c
        # x_atten = x.clone()
        # x_gcn_1 = x.clone()
        # # GCN
        # x_gcn_1 = rearrange(x_gcn_1,"b j c -> b c j").contiguous() 
        # x_gcn_1 = self.norm1(x_gcn_1) # b,c,j
        # x_gcn_1 = rearrange(x_gcn_1,"b j c -> b c j").contiguous()
        # x_gcn_1 = self.GCN_Block1(x_gcn_1)  # b,j,c

        # # Atten
        # # x_atten = self.norm_att1(x_atten)
        # # x_atten, attn2gcn = self.attn(x_atten, f= self.gcn2Attn_drop(x_gcn_1*self.s_gcn2attn))
        
        # x_gcn_2 = self.GCN_Block2(x_gcn_1)  # b, j, c

        # x = res1 + self.drop_path(x_gcn_2)

        # # uMLP
        # res2 = x  # b,j,c
        # x = self.norm2(x)
        # x =  res2 + self.drop_path(self.uMLP(x))  
        # return x
        
class IGANet(nn.Module):
    def __init__(self, depth, embed_dim, adj, drop_rate=0.10, length=27,joints_pos_embed = None ,use_origin_gcn = True):
        super().__init__()
        
        drop_path_rate = 0.2
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        self.blocks = nn.ModuleList([
            Block(
                length, embed_dim, adj, 
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,joints_pos_embed = joints_pos_embed,use_origin_gcn = use_origin_gcn)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

class Model(nn.Module):
    def __init__(self, args,return_refine_embeding=False,mode='joints',joints_pos_embed = None):
        super().__init__()
        self.return_refine_embeding = return_refine_embeding
        if mode == 'joints':
            self.graph = Graph('interhand_gt', 'spatial', pad=1)
        elif mode == 'bone':
            self.graph = Graph('interhand_gt_bone', 'spatial', pad=1)
        else:
            raise NotImplementedError
        
        if args.use_origin_gcn:
            self.A = nn.Parameter(torch.tensor(self.graph.A, dtype=torch.float32), requires_grad=False)
        else:
            self.adjacency = self.graph.adjacency  #未归一化
            self.A = nn.Parameter(torch.tensor(self.adjacency,dtype=torch.float32), requires_grad=True )
        
            # #only update weight between two hand
            # gradient_mask = torch.ones(args.keypoint_num,args.keypoint_num)
            # single_hand = torch.zeros(args.keypoint_num//2,args.keypoint_num//2)
            # gradient_mask[:args.keypoint_num//2,:args.keypoint_num//2] = single_hand
            # gradient_mask[args.keypoint_num//2:,args.keypoint_num//2:] = single_hand
            # self.register_buffer('gradient_mask',gradient_mask)
            # self.A.register_hook(lambda grad: grad.mul_(self.gradient_mask)) #原地置零梯度


        
        #self.encoder = encoder(2,args.lift_channel//2,args.channel)
        #  
        self.IGANet = IGANet(args.lift_layers, args.lift_channel, self.A, length=args.keypoint_num,joints_pos_embed = joints_pos_embed,use_origin_gcn = args.use_origin_gcn) # 256

        #modified
        if args.pred_2d_depth:
            self.fcn = nn.Linear(args.lift_channel, 1)
        else:
            #self.fcn = MLP(input_dim=288, hidden_dim=[128,64], output_dim=3, num_layers=3)
            self.fcn = nn.Linear(args.lift_channel, 3)


    def forward(self, x):
        #x = rearrange(x, 'b f j c -> (b f) j c').contiguous() # B 17 2
        #print(self.A)
        # encoder
        #x = self.encoder(x)     # B 17 512

        x = self.IGANet(x)    # B 17 512

        if self.return_refine_embeding:
            refine_embeding = x
        
        # regression6
        x = self.fcn(x)         # B 17 3

        # x = rearrange(x, 'b j c -> b 1 j c').contiguous() # B, 1, 17, 3


        if  self.return_refine_embeding:
            return x,refine_embeding
        else:
            return x






    
class GCN_attention_guide(nn.Module):
    def __init__(self, in_channels, out_channels,spatial_adj):
        super().__init__()

        self.kernel_size = 4
        self.spatial_adj = spatial_adj
        #
        self.conv1d = nn.Conv1d(in_channels, out_channels * self.kernel_size, kernel_size=1)

        self.s_context = nn.Parameter(torch.tensor([(1)], dtype=torch.float32), requires_grad=True)
        
        #self.s_context = nn.Parameter(torch.tensor([1,1], dtype=torch.float32), requires_grad=True) 

    def forward(self, x, adj): # b,j,c
        bs = adj.shape[0]
        # conv1d
        x = rearrange(x,"b j c -> b c j") 
        x = self.conv1d(x)   # b,c*kernel_size,j = b,c*4,j
        x = rearrange(x,"b ck j -> b ck 1 j")
        b, kc, t, v = x.size()
        x = x.view(b, self.kernel_size, kc//self.kernel_size, t, v) # b,k, kc/k, 1, j 
        # #只有内容关系
        adj = torch.cat((adj*self.s_context,self.spatial_adj.unsqueeze(0).repeat(bs,1,1,1)),dim=1)
        # #包含位置关系
        #adj = torch.cat((adj*(self.s_context.view(1,-1,1,1)),self.spatial_adj.unsqueeze(0).repeat(bs,1,1,1)),dim=1)
        x = torch.einsum('bkctv, bkvw->bctw', (x,adj))  # bctw   b,c,1,j  其中t=1为时间帧的长度
        x = x.contiguous()
        x = rearrange(x, 'b c 1 j -> b j c') 
        return x.contiguous()


class Block_attention_guide(nn.Module): # drop=0.1
    def __init__(self, length, dim, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,use_origin_gcn = True,spatial_adj=None):
        # length =17, dim = args.channel = 512, tokens_dim = args.token_dim=256, channels_dim = args.d_hid = 1024
        super().__init__()
        self.use_origin_gcn = use_origin_gcn
        #only update weight between two hand
        #self.adj.weight.register_hook(lambda grad: grad.mul_(gradient_mask))
        # GCN
        self.norm1 = norm_layer(length)
        if use_origin_gcn:
            self.GCN_Block1 = GCN_attention_guide(dim, dim,spatial_adj)
            self.GCN_Block2 = GCN_attention_guide(dim, dim,spatial_adj)
        else:
            self.GCN_Block1 = My_GCN(dim, dim)
            self.GCN_Block2 = My_GCN(dim, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # attention
        self.norm_att1=norm_layer(dim)
        self.num_heads = 8
        qkv_bias =  True
        qk_scale = None
        attn_drop = 0.2
        proj_drop = 0.25
        self.attn = Attention(
            dim, num_heads=self.num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop) 

        # 512,1024
        self.norm2 = norm_layer(dim)
        self.uMLP = uMLP(in_features=dim, hidden_features=256, act_layer=act_layer, drop=0.20)
        
        gcn2attn_p = 0.15
        Attn2gcn_p = 0.15
        self.gcn2Attn_drop = nn.Dropout(p = gcn2attn_p)
        self.Attn2gcn_drop = nn.Dropout(p = Attn2gcn_p)
        self.s_gcn2attn = nn.Parameter(torch.tensor([(0.5)], dtype=torch.float32), requires_grad=False) 
        self.s_attn2gcn = nn.Parameter(torch.tensor([(0.8)], dtype=torch.float32), requires_grad=False) 

    def forward(self, x, adj):
        # # B,J,dim 
        # res1 = x # b,j,c
        # x_atten = x.clone()
        # x_gcn_1 = x.clone()
        # # GCN
        # x_gcn_1 = rearrange(x_gcn_1,"b j c -> b c j").contiguous() 
        # x_gcn_1 = self.norm1(x_gcn_1) # b,c,j
        # x_gcn_1 = rearrange(x_gcn_1,"b j c -> b c j").contiguous()
        # if self.use_origin_gcn:
        #     x_gcn_1 = self.GCN_Block1(x_gcn_1,adj)  # b,j,c
        # else:
        #     x_gcn_1 = self.GCN_Block1(x_gcn_1,adj)

        # # Atten
            
        # x_atten = self.norm_att1(x_atten)
        # x_atten, attn2gcn = self.attn(x_atten, f= self.gcn2Attn_drop(x_gcn_1*self.s_gcn2attn))
        
        # if self.use_origin_gcn:
        #     x_gcn_2 = self.GCN_Block2(x_gcn_1 + self.Attn2gcn_drop(attn2gcn*self.s_attn2gcn),adj)  # b, j, c
        # else:
        #     x_gcn_2 = self.GCN_Block2(x_gcn_1 + self.Attn2gcn_drop(attn2gcn*self.s_attn2gcn),adj)  # b, j, c

        # x = res1 + self.drop_path(x_gcn_2 + x_atten)

        # # uMLP
        # res2 = x  # b,j,c
        # x = self.norm2(x)
        # x =  res2 + self.drop_path(self.uMLP(x))  
        # return x


        # B,J,dim 
        res1 = x # b,j,c
        x_atten = x.clone()
        x_gcn_1 = x.clone()
        # GCN
        x_gcn_1 = rearrange(x_gcn_1,"b j c -> b c j").contiguous() 
        x_gcn_1 = self.norm1(x_gcn_1) # b,c,j
        x_gcn_1 = rearrange(x_gcn_1,"b j c -> b c j").contiguous()
        x_gcn_1 = self.GCN_Block1(x_gcn_1,adj)  # b,j,c

        # Atten
        # x_atten = self.norm_att1(x_atten)
        # x_atten, attn2gcn = self.attn(x_atten, f= self.gcn2Attn_drop(x_gcn_1*self.s_gcn2attn))
        
        x_gcn_2 = self.GCN_Block2(x_gcn_1,adj)  # b, j, c

        x = res1 + self.drop_path(x_gcn_2)

        # uMLP
        res2 = x  # b,j,c
        x = self.norm2(x)
        x =  res2 + self.drop_path(self.uMLP(x))  
        return x



class IGANet_attention_guide_layer(nn.Module):
    def __init__(self, depth, embed_dim, drop_rate=0.10, length=27,use_origin_gcn = True,spatial_adj=None):
        super().__init__()
        
        drop_path_rate = 0.2
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        self.blocks = nn.ModuleList([
            Block_attention_guide(
                length, embed_dim, 
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,use_origin_gcn = use_origin_gcn,spatial_adj=spatial_adj)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)


    def forward(self, x, adj):
        for blk in self.blocks:
            x = blk(x,adj)

        x = self.norm(x)
        return x




class IGANet_attention_guide(nn.Module):
    def __init__(self, args,return_refine_embeding=False,mode='joints'):
        super().__init__()
        self.return_refine_embeding = return_refine_embeding
        if mode == 'joints':
            self.graph = Graph('interhand_gt', 'spatial', pad=1)
        elif mode == 'bone':
            self.graph = Graph('interhand_gt_bone', 'spatial', pad=1)
        else:
            raise NotImplementedError
        
        if args.use_origin_gcn:
            self.A = nn.Parameter(torch.tensor(self.graph.A, dtype=torch.float32), requires_grad=False)
        # else:
        #     self.adjacency = self.graph.adjacency  #未归一化
        #     self.A = nn.Parameter(torch.tensor(self.adjacency,dtype=torch.float32), requires_grad=True )
        
        #     #only update weight between two hand
        #     gradient_mask = torch.ones(args.keypoint_num,args.keypoint_num)
        #     single_hand = torch.zeros(args.keypoint_num//2,args.keypoint_num//2)
        #     gradient_mask[:args.keypoint_num//2,:args.keypoint_num//2] = single_hand
        #     gradient_mask[args.keypoint_num//2:,args.keypoint_num//2:] = single_hand
        #     self.register_buffer('gradient_mask',gradient_mask)
        #     self.A.register_hook(lambda grad: grad.mul_(self.gradient_mask)) #原地置零梯度


        
        #self.encoder = encoder(2,args.lift_channel//2,args.channel)
        #  
        self.IGANet = IGANet_attention_guide_layer(args.lift_layers, args.lift_channel, length=args.keypoint_num,use_origin_gcn = args.use_origin_gcn,spatial_adj=self.A) # 256
        self.IGANet_list   = _get_clones(self.IGANet,1)
        #modified
        if args.pred_2d_depth:
            self.fcn = nn.Linear(args.lift_channel, 1)
        else:
            self.fcn = nn.Linear(args.lift_channel, 3)
            #self.fcn = MLP(16, 32, 64, 128)

    def forward(self, x, adj):
        #x = rearrange(x, 'b f j c -> (b f) j c').contiguous() # B 17 2
        #print(self.A)
        # encoder
        #x = self.encoder(x)     # B 17 512
        for i in range(len(x)):
            x = self.IGANet_list[i](x[i],adj)    # B 17 512

        if self.return_refine_embeding:
            refine_embeding = x
        
        # regression
        x = self.fcn(x)         # B 17 3

        x = rearrange(x, 'b j c -> b 1 j c').contiguous() # B, 1, 17, 3


        if  self.return_refine_embeding:
            return x,refine_embeding
        else:
            return x
