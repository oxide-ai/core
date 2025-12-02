use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{embedding, linear, linear_no_bias, Embedding, Linear, Module, VarBuilder};
use serde::Deserialize;

/// Phi-3 Model Configuration
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

impl Default for Config {
    fn default() -> Self {
        // Phi-3-mini-4k-instruct config
        Self {
            vocab_size: 32064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(32),
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        }
    }
}

impl Config {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Key-Value Cache for a single layer
/// Stores cached K and V tensors to avoid recomputation
#[derive(Debug, Clone)]
pub struct LayerKVCache {
    pub key_cache: Option<Tensor>,
    pub value_cache: Option<Tensor>,
}

impl LayerKVCache {
    pub fn new() -> Self {
        Self {
            key_cache: None,
            value_cache: None,
        }
    }

    /// Append new K/V to cache and return concatenated tensors
    pub fn append(&mut self, new_k: Tensor, new_v: Tensor) -> Result<(Tensor, Tensor)> {
        let k = if let Some(ref cached_k) = self.key_cache {
            Tensor::cat(&[cached_k, &new_k], 2)?
        } else {
            new_k.clone()
        };

        let v = if let Some(ref cached_v) = self.value_cache {
            Tensor::cat(&[cached_v, &new_v], 2)?
        } else {
            new_v.clone()
        };

        self.key_cache = Some(k.clone());
        self.value_cache = Some(v.clone());

        Ok((k, v))
    }

    pub fn clear(&mut self) {
        self.key_cache = None;
        self.value_cache = None;
    }

    pub fn seq_len(&self) -> usize {
        self.key_cache
            .as_ref()
            .map(|k| k.dim(2).unwrap_or(0))
            .unwrap_or(0)
    }
}

/// KV Cache for all layers
#[derive(Debug, Clone)]
pub struct KVCache {
    caches: Vec<LayerKVCache>,
}

impl KVCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            caches: (0..num_layers).map(|_| LayerKVCache::new()).collect(),
        }
    }

    pub fn get_layer_cache(&mut self, layer_idx: usize) -> &mut LayerKVCache {
        &mut self.caches[layer_idx]
    }

    pub fn clear(&mut self) {
        for cache in &mut self.caches {
            cache.clear();
        }
    }

    pub fn seq_len(&self) -> usize {
        self.caches.first().map(|c| c.seq_len()).unwrap_or(0)
    }
}

/// Pre-computed rotary embeddings
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    /// Pre-compute rotary embeddings for all positions
    pub fn new(cfg: &Config, device: &Device) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let theta = cfg.rope_theta;
        let max_seq_len = cfg.max_position_embeddings;

        // Pre-compute inverse frequencies
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;

        // Pre-compute all positions
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        Ok(Self { cos, sin })
    }

    /// Get rotary embeddings for specific positions
    pub fn forward(&self, seq_len: usize, start_pos: usize) -> Result<(Tensor, Tensor)> {
        let end_pos = start_pos + seq_len;
        let cos = self.cos.i(start_pos..end_pos)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.i(start_pos..end_pos)?.unsqueeze(0)?.unsqueeze(0)?;
        Ok((cos, sin))
    }
}

/// RMS Normalization layer
#[derive(Debug)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(candle_core::D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(candle_core::D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        Ok(x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&self.weight)?)
    }
}

/// Rotary Position Embedding
fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(candle_core::D::Minus1)?;
    let xs1 = xs.narrow(candle_core::D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(candle_core::D::Minus1, last_dim / 2, last_dim / 2)?;
    Ok(Tensor::cat(&[&xs2.neg()?, &xs1], candle_core::D::Minus1)?)
}

fn apply_rotary_emb(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    let q_embed = (q.broadcast_mul(cos)? + rotate_half(q)?.broadcast_mul(sin)?)?;
    let k_embed = (k.broadcast_mul(cos)? + rotate_half(k)?.broadcast_mul(sin)?)?;
    Ok((q_embed, k_embed))
}

/// Multi-Head Attention with KV Cache support
#[derive(Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads();
        let head_dim = cfg.head_dim();

        // Try to load fused QKV weights (common in Phi-3)
        let (q_proj, k_proj, v_proj) = if let Ok(qkv) = vb.pp("qkv_proj").get(
            (num_heads * head_dim + 2 * num_kv_heads * head_dim, hidden_sz), 
            "weight"
        ) {
            let q_size = num_heads * head_dim;
            let kv_size = num_kv_heads * head_dim;
            
            // Split QKV: [q_size + k_size + v_size, hidden_sz]
            let q = qkv.narrow(0, 0, q_size)?;
            let k = qkv.narrow(0, q_size, kv_size)?;
            let v = qkv.narrow(0, q_size + kv_size, kv_size)?;

            (
                Linear::new(q, None),
                Linear::new(k, None),
                Linear::new(v, None),
            )
        } else {
            // Fallback to separate weights
            let q_proj = linear_no_bias(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
            let k_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
            let v_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
            (q_proj, k_proj, v_proj)
        };

        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    /// Forward pass with KV cache
    /// Only processes new tokens (xs) and uses cached K/V for previous tokens
    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: &mut LayerKVCache,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        // Project new tokens
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // Reshape for multi-head attention
        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let new_k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let new_v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings to Q and new K
        let (q, new_k) = apply_rotary_emb(&q, &new_k, cos, sin)?;

        // Append to cache and get full K, V
        let (k, v) = kv_cache.append(new_k, new_v)?;

        // Scaled dot-product attention
        let att = (q.matmul(&k.transpose(2, 3)?)? / (self.head_dim as f64).sqrt())?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v)?;

        let out = out
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        Ok(self.o_proj.forward(&out)?)
    }
}

/// MLP (Feed-Forward Network)
#[derive(Debug)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;

        // Try to load fused GateUp weights (common in Phi-3)
        let (gate_proj, up_proj) = if let Ok(gate_up) = vb.pp("gate_up_proj").get(
            (2 * intermediate_sz, hidden_sz), 
            "weight"
        ) {
            // Split GateUp: [2 * intermediate_sz, hidden_sz]
            // First half is gate, second half is up
            let gate = gate_up.narrow(0, 0, intermediate_sz)?;
            let up = gate_up.narrow(0, intermediate_sz, intermediate_sz)?;
            
            (Linear::new(gate, None), Linear::new(up, None))
        } else {
            // Fallback to separate weights
            let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
            let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
            (gate_proj, up_proj)
        };

        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let hidden = (gate * up)?;
        Ok(self.down_proj.forward(&hidden)?)
    }
}

/// Transformer Decoder Layer with KV Cache
#[derive(Debug)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(cfg, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: &mut LayerKVCache,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin, kv_cache)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        Ok((xs + residual)?)
    }
}

/// Main Phi-3 Model with KV Caching
#[derive(Debug)]
pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: RotaryEmbedding,
    device: Device,
    config: Config,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(cfg, vb_l.pp(layer_idx))?;
            layers.push(layer);
        }

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;

        // Pre-compute rotary embeddings
        let rope = RotaryEmbedding::new(cfg, vb.device())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            device: vb.device().clone(),
            config: cfg.clone(),
        })
    }

    /// Forward pass with KV cache
    /// Only processes new tokens efficiently using cached K/V from previous steps
    pub fn forward(&self, input_ids: &Tensor, kv_cache: &mut KVCache) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;

        // Get position for RoPE (start from cache length)
        let start_pos = kv_cache.seq_len();
        let (cos, sin) = self.rope.forward(seq_len, start_pos)?;

        // Process through all layers with KV cache
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_cache = kv_cache.get_layer_cache(layer_idx);
            xs = layer.forward(&xs, &cos, &sin, layer_cache)?;
        }

        let xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs)?;

        Ok(logits)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
