//! Per-stage attribution for a decode step.
//!
//! # Why this is not a synchronise-between-stages profiler
//!
//! The first version of this put a `cudaDeviceSynchronize` between stages. It
//! reported a **448-byte embedding gather at 0.486 ms/token** — a hundredfold
//! overstatement — and inflated the total by 58%. Two hours of optimisation
//! planning came off that breakdown before the numbers were checked against
//! reality.
//!
//! Events are recorded *into the stream*: the host never blocks, the pipeline is
//! not broken, and the timestamps come from the GPU's own clock. One
//! synchronise per step reads them all.
//!
//! # What it is still not good for
//!
//! Choosing between two kernels that differ by a few percent. Recording an event
//! at every stage boundary serialises those boundaries, and a per-shape kernel
//! rule selected from this profile measured *slower* end to end than the one it
//! replaced. Use it to decide **which stage to work on**; use `whetstone tune`
//! to decide **which kernel**.


use whetstone_kernels::decode;

use crate::engine::Engine;
use crate::error::Result;

/// A class of kernel in the decode step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Input embedding gather.
    Embed,
    /// Any of the RMSNorms.
    RmsNorm,
    /// The fused query/key/value projection.
    Qkv,
    /// Rotary embedding and the KV cache append.
    Rope,
    /// Attention against the cache.
    Attention,
    /// Attention output projection.
    OProj,
    /// The fused SwiGLU gate/up projection.
    GateUp,
    /// The SwiGLU product.
    Swiglu,
    /// MLP down projection.
    DownProj,
    /// Output projection over the vocabulary.
    LmHead,
    /// Argmax.
    Sample,
    /// The position advance, and boundary markers with no work in them.
    Advance,
}

/// Seconds spent per class of kernel, summed over profiled steps.
///
/// Attribution only — see [`Engine::profile`] for why the totals are inflated.
#[derive(Debug, Clone, Default)]
pub struct Profile {
    /// Decode steps measured.
    pub steps: usize,
    /// Input embedding gather.
    pub embed: f64,
    /// All RMSNorms: two per block plus the final one.
    pub rmsnorm: f64,
    /// Query, key and value projections.
    pub qkv: f64,
    /// Rotary embedding and the KV cache append.
    pub rope: f64,
    /// Attention against the cache.
    pub attention: f64,
    /// Attention output projection.
    pub o_proj: f64,
    /// SwiGLU gate and up projections.
    pub gate_up: f64,
    /// The SwiGLU product itself.
    pub swiglu: f64,
    /// MLP down projection.
    pub down_proj: f64,
    /// Output projection over the vocabulary.
    pub lm_head: f64,
    /// Argmax over the vocabulary.
    pub sample: f64,
    /// The position advance, and any gap the boundary markers caught.
    pub advance: f64,
}

impl Profile {
    fn bucket(&mut self, s: Stage) -> &mut f64 {
        match s {
            Stage::Embed => &mut self.embed,
            Stage::RmsNorm => &mut self.rmsnorm,
            Stage::Qkv => &mut self.qkv,
            Stage::Rope => &mut self.rope,
            Stage::Attention => &mut self.attention,
            Stage::OProj => &mut self.o_proj,
            Stage::GateUp => &mut self.gate_up,
            Stage::Swiglu => &mut self.swiglu,
            Stage::DownProj => &mut self.down_proj,
            Stage::LmHead => &mut self.lm_head,
            Stage::Sample => &mut self.sample,
            Stage::Advance => &mut self.advance,
        }
    }
}

impl Profile {
    /// Every stage, largest first, as `(name, ms per token, share of total)`.
    pub fn breakdown(&self) -> Vec<(&'static str, f64, f64)> {
        let n = self.steps.max(1) as f64;
        let mut rows = vec![
            ("gate+up proj", self.gate_up),
            ("down proj", self.down_proj),
            ("lm_head", self.lm_head),
            ("q/k/v proj", self.qkv),
            ("o proj", self.o_proj),
            ("attention", self.attention),
            ("rmsnorm", self.rmsnorm),
            ("swiglu", self.swiglu),
            ("rope + kv", self.rope),
            ("sample", self.sample),
            ("embed", self.embed),
            ("advance", self.advance),
        ];
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let total: f64 = rows.iter().map(|r| r.1).sum();
        rows.into_iter()
            .map(|(name, s)| (name, s / n * 1e3, if total > 0.0 { s / total } else { 0.0 }))
            .collect()
    }

    /// Total profiled milliseconds per token.
    pub fn total_ms(&self) -> f64 {
        let n = self.steps.max(1) as f64;
        (self.embed
            + self.rmsnorm
            + self.qkv
            + self.rope
            + self.attention
            + self.o_proj
            + self.gate_up
            + self.swiglu
            + self.down_proj
            + self.lm_head
            + self.sample
            + self.advance)
            / n
            * 1e3
    }
}


impl Engine {
    /// Times each class of kernel across `steps` decode steps, using
    /// stream-ordered CUDA events.
    ///
    /// Events are recorded *into the stream*, so the host never blocks and the
    /// pipeline is never broken. That matters: the first version of this
    /// function synchronised between stages, and the interference was large
    /// enough to report a 448-byte embedding gather as half a millisecond of
    /// work and to inflate the total by 70%. What you get here is the GPU's own
    /// clock on its own timeline.
    ///
    /// Events still serialise the stream at each boundary, so a stage that would
    /// have overlapped its neighbour is charged separately. Whetstone's decode
    /// step is a strict dependency chain, so there is nothing to overlap and the
    /// distortion is the per-event marker, around a microsecond per token in
    /// total.
    pub fn profile(&mut self, token: u32, steps: usize) -> Result<Profile> {
        let mut p = Profile::default();
        let c = &self.weights.config;
        let eps = c.rms_norm_eps;
        let n_q = c.num_attention_heads;
        let layers = self.weights.layers.len();

        // One event per stage boundary within a step: 10 stages per block, plus
        // the embedding, the final norm, the head and the argmax.
        let per_step = 10 * layers + 5;
        let mut events: Vec<decode::Event> = Vec::with_capacity(per_step + 1);
        for _ in 0..=per_step {
            events.push(decode::Event::new()?);
        }

        // Which bucket each interval belongs to, built once in stage order.
        let mut labels: Vec<Stage> = Vec::with_capacity(per_step);
        labels.push(Stage::Embed);
        for _ in 0..layers {
            labels.extend([
                Stage::RmsNorm,
                Stage::Qkv,
                Stage::Rope,
                Stage::Attention,
                Stage::OProj,
                Stage::RmsNorm,
                Stage::GateUp,
                Stage::Swiglu,
                Stage::DownProj,
                Stage::Advance,
            ]);
        }
        labels.extend([Stage::RmsNorm, Stage::LmHead, Stage::Sample, Stage::Advance]);
        debug_assert_eq!(labels.len(), per_step);

        self.acts.token.set(token as i32)?;

        for _ in 0..steps {
            if self.pos >= self.max_seq {
                self.reset()?;
            }
            let a = &mut self.acts;
            let mut e = 0usize;

            events[e].record()?;
            e += 1;

            self.weights.embed.gather(&a.token, &mut a.x)?;
            events[e].record()?;
            e += 1;

            for (l, layer) in self.weights.layers.iter().enumerate() {
                decode::rmsnorm(&a.x, &layer.input_norm, &mut a.h, eps)?;
                events[e].record()?;
                e += 1;

                layer.qkv_proj.forward(&a.h, layer.qkv_bias.as_ref(), &mut a.qkv, false)?;
                events[e].record()?;
                e += 1;

                decode::rope_cache(&mut a.qkv, &mut self.caches[l], &self.rope, n_q, &a.pos_dev)?;
                events[e].record()?;
                e += 1;

                decode::attn_decode(&a.qkv, &mut self.caches[l], &mut a.attn, n_q, &a.pos_dev)?;
                events[e].record()?;
                e += 1;

                layer.o_proj.forward(&a.attn, None, &mut a.x, true)?;
                events[e].record()?;
                e += 1;

                decode::rmsnorm(&a.x, &layer.post_attn_norm, &mut a.h, eps)?;
                events[e].record()?;
                e += 1;

                layer.gate_up_proj.forward(&a.h, None, &mut a.gate_up, false)?;
                events[e].record()?;
                e += 1;

                decode::swiglu(&a.gate_up, &mut a.act)?;
                events[e].record()?;
                e += 1;

                layer.down_proj.forward(&a.act, None, &mut a.x, true)?;
                events[e].record()?;
                e += 1;

                // Nothing here; the boundary keeps the label table uniform.
                events[e].record()?;
                e += 1;
            }

            decode::rmsnorm(&a.x, &self.weights.final_norm, &mut a.h, eps)?;
            events[e].record()?;
            e += 1;

            match &self.weights.lm_head {
                Some(head) => head.forward(&a.h, None, &mut a.logits, false)?,
                None => self.weights.embed.project(&a.h, &mut a.logits)?,
            }
            events[e].record()?;
            e += 1;

            decode::argmax(&a.logits, a.token.buffer_mut())?;
            events[e].record()?;
            e += 1;

            a.pos_dev.advance(self.max_seq)?;
            events[e].record()?;
            e += 1;
            debug_assert_eq!(e, per_step + 1);

            decode::stream_sync()?;

            for (i, stage) in labels.iter().enumerate() {
                let ms = events[i].elapsed_ms(&events[i + 1])? as f64 / 1e3;
                *p.bucket(*stage) += ms;
            }

            // Reset the token so every profiled step is identical; otherwise the
            // sequence wanders and attention's cost drifts with it.
            a.token.set(token as i32)?;
            self.pos += 1;
            p.steps += 1;
        }

        Ok(p)
    }
}
