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

/// Multi-Head Attention
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

        let q_proj = linear(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
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

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = apply_rotary_emb(&q, &k, cos, sin)?;

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

        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
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

/// Transformer Decoder Layer
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
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        Ok((xs + residual)?)
    }
}

/// Main Phi-3 Model
#[derive(Debug)]
pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
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

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            config: cfg.clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;

        // Create rotary embeddings
        let (cos, sin) = self.create_rotary_embeddings(seq_len, pos)?;

        for layer in &self.layers {
            xs = layer.forward(&xs, &cos, &sin)?;
        }

        let xs = self.norm.forward(&xs)?;
        let logits = self.lm_head.forward(&xs)?;

        Ok(logits)
    }

    fn create_rotary_embeddings(&self, seq_len: usize, start_pos: usize) -> Result<(Tensor, Tensor)> {
        let head_dim = self.config.head_dim();
        let theta = self.config.rope_theta;

        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), &self.device)?;

        let t = Tensor::arange(start_pos as u32, (start_pos + seq_len) as u32, &self.device)?
            .to_dtype(DType::F32)?
            .reshape((seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        let cos = freqs.cos()?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = freqs.sin()?.unsqueeze(0)?.unsqueeze(0)?;

        Ok((cos, sin))
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
