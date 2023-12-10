import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import copy
import math
from collections import OrderedDict

def clones(module, N):
	# 克隆N个完全相同的SubLayer，使用了copy.deepcopy
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) \
		/ math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = F.softmax(scores, dim = -1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class PositionalEncoding(nn.Module): 
	def __init__(self, d_model, dropout, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		
		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) *
			-(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + torch.tensor(self.pe[:, :x.size(1)], requires_grad=False)
		return self.dropout(x)

class PositionwiseFeedForward(nn.Module): 
	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x):
		return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, dim_feat, dim_model, dropout=0.1):
        super(Embeddings, self).__init__()
        self.w_1 = nn.Linear(dim_feat, 512)
        self.w_2 = nn.Linear(512, dim_model)
        self.dropout = nn.Dropout(dropout)
		 
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        if self.adapter == None:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

class Encoder(nn.Module):
		"Encoder是N个EncoderLayer的stack"
		def __init__(self, layer, N):
				super(Encoder, self).__init__()
				# layer是一个SubLayer，我们clone N个
				self.layers = clones(layer, N)
				# 再加一个LayerNorm层
				self.norm = LayerNorm(layer.size)
		
		def forward(self, x, mask):
				"逐层进行处理"
				for layer in self.layers:
					x = layer(x, mask)
				# 最后进行LayerNorm，后面会解释为什么最后还有一个LayerNorm。
				return self.norm(x)

class EncoderLayer(nn.Module):
		"EncoderLayer由self-attn和feed forward组成"
		def __init__(self, size, self_attn, feed_forward, dropout):
				super(EncoderLayer, self).__init__()
				self.self_attn = self_attn
				self.feed_forward = feed_forward
				self.sublayer = clones(SublayerConnection(size, dropout), 2)
				self.size = size

		def forward(self, x, mask):
				"Follow Figure 1 (left) for connections."
				# x = self.sublayer[0](x)
				x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, need_weights=False))
				return self.sublayer[1](x, self.feed_forward)

class SublayerConnection(nn.Module):
		"""
		LayerNorm + sublayer(Self-Attenion/Dense) + dropout + 残差连接
		"""
		def __init__(self, size, dropout):
				super(SublayerConnection, self).__init__()
				self.norm = LayerNorm(size)
				self.dropout = nn.Dropout(dropout)
		
		def forward(self, x, sublayer):
				"sublayer是传入的参数,参考DecoderLayer,它可以当成函数调用,这个函数的有一个输入参数"
				x = sublayer(self.norm(x))
				if type(x) is tuple:
					x = x[0] 
				return x + self.dropout(x)

class Decoder(nn.Module): 
		def __init__(self, layer, N):
				super(Decoder, self).__init__()
				self.layers = clones(layer, N)
				self.norm = LayerNorm(layer.size)
		
		def forward(self, x, memory, src_mask, tgt_mask):
				for layer in self.layers:
					x = layer(x, memory, src_mask, tgt_mask)
				return self.norm(x)

class DecoderLayer(nn.Module):
		"Decoder包括self-attn, src-attn, 和feed forward "
		def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
				super(DecoderLayer, self).__init__()
				self.size = size
				self.self_attn = self_attn
				self.src_attn = src_attn
				self.feed_forward = feed_forward
				self.sublayer = clones(SublayerConnection(size, dropout), 3)
		
		def forward(self, x, memory, src_mask, tgt_mask): 
				m = memory
				if len(tgt_mask.shape) == 3:
					tgt_mask = tgt_mask.squeeze(0)
				x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, attn_mask=tgt_mask, need_weights=False))
				x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
				return self.sublayer[2](x, self.feed_forward)

class EncoderDecoder(nn.Module):
		def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
				super(EncoderDecoder, self).__init__()
				# encoder和decoder都是构造的时候传入的，这样会非常灵活
				self.encoder = encoder
				self.decoder = decoder
				# 源语言和目标语言的embedding
				self.src_embed = src_embed
				self.tgt_embed = tgt_embed
				# generator后面会讲到，就是根据Decoder的隐状态输出当前时刻的词
				# 基本的实现就是隐状态输入一个全连接层，全连接层的输出大小是词的个数
				# 然后接一个softmax变成概率。
				self.generator = generator
		
		def forward(self, src, tgt, src_mask, tgt_mask):
				# 首先调用encode方法对输入进行编码，然后调用decode方法解码
				return self.decode(self.encode(src, src_mask), src_mask,
					tgt, tgt_mask)
		
		def encode(self, src, src_mask):
				# 调用encoder来进行编码，传入的参数embedding的src和src_mask
				return self.encoder(self.src_embed(src), src_mask)
		
		def decode(self, memory, src_mask, tgt, tgt_mask):
				# 调用decoder
				return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
	# 根据Decoder的隐状态输出一个词
	# d_model是Decoder输出的大小，pose是姿态参数
	def __init__(self, d_model, pose_dim):
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, pose_dim)
	
	# 全连接
	def forward(self, x):
		return self.proj(x)

def build_attention_mask(context_length):
		# lazily create causal attention mask, with full attention between the vision tokens
		# pytorch uses additive attention mask; fill with -inf
		mask = torch.empty(context_length, context_length)
		mask.fill_(float("-inf"))
		mask.triu_(1)  # zero out the lower diagonal
		return mask

def build_attention_mask_batch(batch, head, context_length):
		# lazily create causal attention mask, with full attention between the vision tokens
		# pytorch uses additive attention mask; fill with -inf
		mask = torch.empty(context_length, context_length)
		mask.fill_(float("-inf"))
		mask.triu_(1)  # zero out the lower diagonal
		mask = torch.repeat_interleave(mask.unsqueeze(0), head, dim=0)
		mask = torch.repeat_interleave(mask.unsqueeze(0), batch, dim=0)
		return mask.reshape(-1, context_length, context_length)

def make_model(src_feat, tgt_feat, layers=6, d_model=512, d_ff=2048, n_head=8, dropout=0.1): 
		c = copy.deepcopy
		attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
		ff = PositionwiseFeedForward(d_model, d_ff, dropout)
		position = PositionalEncoding(d_model, dropout)
		model = EncoderDecoder(
			Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), layers),
			Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), layers),
			# nn.Sequential(Embeddings(src_feat, d_model), c(position)),
			# nn.Sequential(Embeddings(tgt_feat, d_model), c(position)),
			nn.Sequential(Embeddings(src_feat, d_model)),
			nn.Sequential(Embeddings(tgt_feat, d_model)),
			Generator(d_model, tgt_feat))
		
		# 随机初始化参数，这非常重要
		for p in model.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform(p)
		return model

def subsequent_mask(size):
		"Mask out subsequent positions."
		attn_shape = (1, size, size)
		subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
		return torch.from_numpy(subsequent_mask) == 0

if __name__=='__main__':
	# context_length = 10
	# mask = build_attention_mask(context_length)
	# print(mask)
	from timesformer.models.vit import TimeSformer
	transformer = make_model(src_feat=1024, tgt_feat=13*6,
							layers=5, d_model=256, n_head=8,
							dropout=0.1)
	backbone = TimeSformer(img_size=224, num_classes=1024, num_frames=8, attention_type='divided_space_time',
						pretrained_model="/data/newhome/litianyi/model/TimeSformer/TimeSformer_divST_8x32_224_K600.pyth")
	print(backbone)
	for name, parameter in backbone.named_parameters():
		print(name)
		if "head" in name:
			parameter.requires_grad = False
	batch = 1
	length = 15
	inputs = torch.ones(batch, length, 8, 3, 224, 224)
	inputs = inputs.reshape(-1, 8, 3, 224, 224).permute(0, 2, 1, 3, 4)
	feature = backbone(inputs).unsqueeze(0)  ## (1, 15, 2048)
	tgt_mask = build_attention_mask(length+1)
	target = torch.ones(batch, length+1, 13, 6).reshape(batch, length+1, -1)
	src_mask = None
	output = transformer.forward(feature, target, src_mask, tgt_mask)
	output = transformer.generator(output)
	print("output shape: ", output.shape)