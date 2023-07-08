import torch
import torch.nn.functional as F
from functools import wraps
from torch import nn, einsum
from einops import rearrange, repeat


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None, device=None):
        super().__init__()
        self.fn = fn.to(device)
        self.norm = nn.LayerNorm(dim).to(device)
        self.norm_context = nn.LayerNorm(context_dim).to(device) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, device=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim).to(device),
        ).to(device)

    def forward(self, x):
        device = x.device
        return self.net(x).to(device)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, device=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False).to(device)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False).to(device)
        self.to_out = nn.Linear(inner_dim, query_dim).to(device)

    def forward(self, x, context=None, mask=None):
        device = x.device
        h = self.heads

        q = self.to_q(x).to(device)
        context = default(context, x)  # return context if exists(context) else x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h).to(device)
            sim.masked_fill_(~mask, max_neg_value).to(device)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1).to(device)

        out = einsum('b i j, b j d -> b i d', attn, v).to(device)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h).to(device)
        return self.to_out(out).to(device)


class LSRE(nn.Module):
    r"""
    This class implements the LSRE model proposed in my paper

    For more details, please refer to the papers `Online portfolio management via deep reinforcement learning with
    high-frequency data <https://www.sciencedirect.com/science/article/abs/pii/S030645732200348X>` and `Perceiver IO: A
    General Architecture for Structured Inputs & Outputs <https://arxiv.org/abs/2107.14795>`
    """

    def __init__(
            self,
            *,
            depth,
            dim,
            num_latents,
            latent_dim,
            cross_heads,
            latent_heads,
            cross_dim_head,
            latent_dim_head,
            weight_tie_layers=True,
            device,
            args
    ):
        super().__init__()
        self.args = args
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim)).to(device)
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head, device=device),
                    context_dim=dim, device=device).to(device),
            PreNorm(latent_dim, FeedForward(latent_dim, device=device).to(device), device=device)
        ])
        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head,
                                                    device=device).to(device), device=device)
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, device=device).to(device), device=device)
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

    def forward(
            self,
            data,
            mask=None
    ):
        b, *_, device = *data.shape, data.device

        # latents
        x = repeat(self.latents, 'n d -> b n d', b=b)
        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(x, context=data, mask=mask) + x
        x = cross_ff(x) + x

        # layers
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x  # x.shape = torch.Size([num_assets, num_latents, latent_dim]), where num_latents = 1

        assert x.shape[1] == 1  # I set num_latents = 1 in my paper
        return x.squeeze(1)  # [num_assets, 1, latent_dim] -> [num_assets, latent_dim]


class LSRE_CAAN(nn.Module):
    r"""
    This class implements the LSRE_CAAN model proposed in my paper

    For more details, please refer to the paper `Online portfolio management via deep reinforcement learning with
    high-frequency data <https://www.sciencedirect.com/science/article/abs/pii/S030645732200348X>`
    """

    def __init__(
            self,
            *,
            num_assets,
            num_feats,
            args,
            **kwargs
    ):
        super().__init__()
        device = args.device
        self.args = args
        self.num_feats = num_feats
        self.num_assets = num_assets
        self.latent_dim = args.latent_dim
        self.dim = num_feats
        self.preset_size = args.preset_size
        self.window_size = args.window_size
        self.token_emb = nn.Linear(self.num_feats, self.dim).to(device)
        self.pos_emb = nn.Embedding(args.window_size, self.dim).to(device)

        self.lsre = LSRE(
            depth=args.depth,  # 1
            dim=self.dim,  # num_feats
            queries_dim=args.latent_dim,  # 32
            num_latents=args.num_latents,  # 1
            latent_dim=self.latent_dim,  # 32
            cross_heads=args.cross_heads,  # 1
            latent_heads=args.latent_heads,  # 1
            cross_dim_head=args.cross_dim_head,  # 64
            latent_dim_head=args.latent_dim_head,  # 32
            device=device,
            args=args,
            **kwargs
        )
        value_dim = self.latent_dim
        self.linear_query = torch.nn.Linear(value_dim, value_dim).to(device)
        self.linear_key = torch.nn.Linear(value_dim, value_dim).to(device)
        self.linear_value = torch.nn.Linear(value_dim, value_dim).to(device)
        self.linear_winner = torch.nn.Linear(value_dim, 1).to(device)

    def forward(
            self,
            x
    ):
        x = x.squeeze(0)  # [1, num_assets, window_size, num_feats] -> [num_assets, window_size, num_feats]
        n, d, device = x.shape[1], x.shape[2], x.device  # n: window size; d: number of features

        # LSRE
        # x = self.token_emb(x)  # optional
        pos_emb = self.pos_emb(torch.arange(n, device=device))
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        x = x + pos_emb
        stock_rep = self.lsre(x, mask=None, queries=None)  # [num_assets, window_size, num_feats] -> [num_assets,
        # num_latents]

        # CAAN
        x = stock_rep  # [num_assets, latent_dim]
        query = self.linear_query(x)  # [num_assets, latent_dim]
        key = self.linear_key(x)
        value = self.linear_value(x)

        beta = torch.matmul(query, key.transpose(0, 1)) / torch.sqrt(torch.tensor(float(query.shape[-1])))  # [S, S]
        beta = F.softmax(beta, dim=-1).unsqueeze(-1)
        stock_rep = torch.sum(value.unsqueeze(0) * beta, dim=1)  # [S, H]

        final_scores = self.linear_winner(stock_rep).squeeze()  # [S]

        # Portfolio Management
        if self.Prop_winners != 1:
            # Prop_winners: proportion of winners, i.e. G in Section 4.2
            num_winners = int(self.num_assets * self.Prop_winners)
            assert num_winners != 0 and num_winners <= self.num_assets
            rank = torch.argsort(final_scores)
            winners = set(rank.detach().cpu().numpy()[-num_winners:])  # <class 'set'>
            winners_mask = torch.Tensor([0 if i in winners else 1 for i in range(rank.shape[0])]).to(device)
            portfolio = F.softmax(final_scores - 1e9 * winners_mask, dim=0)
        else:
            portfolio = F.softmax(final_scores, dim=0)

        return portfolio  # with size [num_assets] or [num_assets, 1]
