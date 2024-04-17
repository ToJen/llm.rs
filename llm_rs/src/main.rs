use std::alloc::{alloc, Layout};
use std::fs::File;
use std::io::{self, BufReader, Cursor, Read, Seek};
use std::path::Path;
use std::{mem, process};
use std::ptr::{self, null_mut};
use std::slice;
use std::time::Instant;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct ParameterTensors {
    pub wte: *mut f32,
    pub wpe: *mut f32,
    pub ln1w: *mut f32,
    pub ln1b: *mut f32,
    pub qkvw: *mut f32,
    pub qkvb: *mut f32,
    pub attprojw: *mut f32,
    pub attprojb: *mut f32,
    pub ln2w: *mut f32,
    pub ln2b: *mut f32,
    pub fcw: *mut f32,
    pub fcb: *mut f32,
    pub fcprojw: *mut f32,
    pub fcprojb: *mut f32,
    pub lnfw: *mut f32,
    pub lnfb: *mut f32,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct ActivationTensors {
    pub encoded: *mut f32,
    pub ln1: *mut f32,
    pub ln1_mean: *mut f32,
    pub ln1_rstd: *mut f32,
    pub qkv: *mut f32,
    pub atty: *mut f32,
    pub preatt: *mut f32,
    pub att: *mut f32,
    pub attproj: *mut f32,
    pub residual2: *mut f32,
    pub ln2: *mut f32,
    pub ln2_mean: *mut f32,
    pub ln2_rstd: *mut f32,
    pub fch: *mut f32,
    pub fch_gelu: *mut f32,
    pub fcproj: *mut f32,
    pub residual3: *mut f32,
    pub lnf: *mut f32,
    pub lnf_mean: *mut f32,
    pub lnf_rstd: *mut f32,
    pub logits: *mut f32,
    pub probs: *mut f32,
    pub losses: *mut f32,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct GPT2Config {
    pub max_seq_len: i32,
    pub vocab_size: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    pub channels: i32,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct GPT2 {
    pub config: GPT2Config,
    pub params: ParameterTensors,
    pub param_sizes: [usize; 16],
    pub params_memory: *mut f32,
    pub num_parameters: i32,
    pub grads: ParameterTensors,
    pub grads_memory: *mut f32,
    pub m_memory: *mut f32,
    pub v_memory: *mut f32,
    pub acts: ActivationTensors,
    pub act_sizes: [usize; 23],
    pub acts_memory: *mut f32,
    pub num_activations: i32,
    pub grads_acts: ActivationTensors,
    pub grads_acts_memory: *mut f32,
    pub batch_size: i32,
    pub seq_len: i32,
    pub inputs: *mut i32,
    pub targets: *mut i32,
    pub mean_loss: f32,
}

pub struct DataLoader {
    pub B: i32,
    pub T: i32,
    pub tokens_file: Option<File>,
    pub file_size: i64,
    pub current_position: i64,
    pub batch: Vec<i32>,
    pub inputs: *mut i32,
    pub targets: *mut i32,
    pub num_batches: i32,
}

pub unsafe fn encoder_forward(
    mut out: *mut f32,
    mut inp: *mut i32,
    mut wte: *mut f32,
    mut wpe: *mut f32,
    mut B: i32,
    mut T: i32,
    mut C: i32,
) {
    let mut b: i32 = 0;
    while b < B {
        let mut t: i32 = 0;
        while t < T {
            let mut out_bt: *mut f32 = out.add((b * T * C) as usize).add((t * C) as usize);
            let mut ix: i32 = *inp.add((b * T + t) as usize);
            let mut wte_ix: *mut f32 = wte.add((ix * C) as usize);
            let mut wpe_t: *mut f32 = wpe.add((t * C) as usize);
            let mut i: i32 = 0;
            while i < C {
                *out_bt.add(i as usize) = *wte_ix.add(i as usize) + *wpe_t.add(i as usize);
                i += 1;
            }
            t += 1;
        }
        b += 1;
    }
}

pub unsafe fn encoder_backward(
    mut dwte: *mut f32,
    mut dwpe: *mut f32,
    mut dout: *mut f32,
    mut inp: *mut i32,
    mut B: i32,
    mut T: i32,
    mut C: i32,
) {
    let mut b: i32 = 0;
    while b < B {
        let mut t: i32 = 0;
        while t < T {
            let mut dout_bt: *mut f32 = dout.add((b * T * C) as usize).add((t * C) as usize);
            let mut ix: i32 = *inp.add((b * T + t) as usize);
            let mut dwte_ix: *mut f32 = dwte.add((ix as usize * C as usize) as usize);
            let mut dwpe_t: *mut f32 = dwpe.add((t * C) as usize);
            let mut i: i32 = 0;
            while i < C {
                let d: f32 = *dout_bt.add(i as usize);
                *dwte_ix.add(i as usize) += d;
                *dwpe_t.add(i as usize) += d;
                i += 1;
            }
            t += 1;
        }
        b += 1;
    }
}

pub unsafe fn layernorm_forward(
    mut out: *mut f32,
    mut mean: *mut f32,
    mut rstd: *mut f32,
    mut inp: *mut f32,
    mut weight: *mut f32,
    mut bias: *mut f32,
    mut B: i32,
    mut T: i32,
    mut C: i32,
) {
    let eps: f32 = 1e-5;
    let mut b: i32 = 0;
    while b < B {
        let mut t: i32 = 0;
        while t < T {
            let mut x: *mut f32 = inp.add((b * T * C) as usize).add((t * C) as usize);
            let mut m: f32 = 0.0;
            let mut i: i32 = 0;
            while i < C {
                m += *x.add(i as usize);
                i += 1;
            }
            m /= C as f32;
            let mut v: f32 = 0.0;
            let mut i_0: i32 = 0;
            while i_0 < C {
                let xshift: f32 = *x.add(i_0 as usize) - m;
                v += xshift * xshift;
                i_0 += 1;
            }
            v /= C as f32;
            let s: f32 = 1.0 / (v + eps).sqrt();
            let mut out_bt: *mut f32 = out.add((b * T * C) as usize).add((t * C) as usize);
            let mut i_1: i32 = 0;
            while i_1 < C {
                let n: f32 = s * (*x.add(i_1 as usize) - m);
                let o: f32 = n * *weight.add(i_1 as usize) + *bias.add(i_1 as usize);
                *out_bt.add(i_1 as usize) = o;
                i_1 += 1;
            }
            *mean.add((b * T + t) as usize) = m;
            *rstd.add((b * T + t) as usize) = s;
            t += 1;
        }
        b += 1;
    }
}

pub unsafe fn layernorm_backward(
    mut dinp: *mut f32,
    mut dweight: *mut f32,
    mut dbias: *mut f32,
    mut dout: *mut f32,
    mut inp: *mut f32,
    mut weight: *mut f32,
    mut mean: *mut f32,
    mut rstd: *mut f32,
    mut B: i32,
    mut T: i32,
    mut C: i32,
) {
    let mut b: i32 = 0;
    while b < B {
        let mut t: i32 = 0;
        while t < T {
            let mut dout_bt: *mut f32 = dout.add((b * T * C) as usize).add((t * C) as usize);
            let mut inp_bt: *mut f32 = inp.add((b * T * C) as usize).add((t * C) as usize);
            let mut dinp_bt: *mut f32 = dinp.add((b * T * C) as usize).add((t * C) as usize);
            let mean_bt: f32 = *mean.add((b * T + t) as usize);
            let rstd_bt: f32 = *rstd.add((b * T + t) as usize);
            let mut dnorm_mean: f32 = 0.0;
            let mut dnorm_norm_mean: f32 = 0.0;
            let mut i: i32 = 0;
            while i < C {
                let norm_bti: f32 = (*inp_bt.add(i as usize) - mean_bt) * rstd_bt;
                let dnorm_i: f32 = *weight.add(i as usize) * *dout_bt.add(i as usize);
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
                i += 1;
            }
            dnorm_mean /= C as f32;
            dnorm_norm_mean /= C as f32;
            let mut i_0: i32 = 0;
            while i_0 < C {
                let norm_bti_0: f32 = (*inp_bt.add(i_0 as usize) - mean_bt) * rstd_bt;
                let dnorm_i_0: f32 = *weight.add(i_0 as usize) * *dout_bt.add(i_0 as usize);
                *dbias.add(i_0 as usize) += *dout_bt.add(i_0 as usize);
                *dweight.add(i_0 as usize) += norm_bti_0 * *dout_bt.add(i_0 as usize);
                let mut dval: f32 = dnorm_i_0 - dnorm_mean - norm_bti_0 * dnorm_norm_mean;
                dval *= rstd_bt;
                *dinp_bt.add(i_0 as usize) += dval;
                i_0 += 1;
            }
            t += 1;
        }
        b += 1;
    }
}

pub unsafe fn matmul_forward(
    mut out: *mut f32,
    mut inp: *mut f32,
    mut weight: *mut f32,
    mut bias: *mut f32,
    mut B: i32,
    mut T: i32,
    mut C: i32,
    mut OC: i32,
) {
    let mut b: i32 = 0;
    while b < B {
        let mut t: i32 = 0;
        while t < T {
            let mut out_bt: *mut f32 = out.add((b * T * OC) as usize).add((t * OC) as usize);
            let mut inp_bt: *mut f32 = inp.add((b * T * C) as usize).add((t * C) as usize);
            let mut o: i32 = 0;
            while o < OC {
                let mut val: f32 = if !bias.is_null() {
                    *bias.add(o as usize)
                } else {
                    0.0
                };
                let mut wrow: *mut f32 = weight.add((o * C) as usize);
                let mut i: i32 = 0;
                while i < C {
                    val += *inp_bt.add(i as usize) * *wrow.add(i as usize);
                    i += 1;
                }
                *out_bt.add(o as usize) = val;
                o += 1;
            }
            t += 1;
        }
        b += 1;
    }
}

pub unsafe fn matmul_backward(
    mut dinp: *mut f32,
    mut dweight: *mut f32,
    mut dbias: *mut f32,
    mut dout: *mut f32,
    mut inp: *mut f32,
    mut weight: *mut f32,
    mut B: i32,
    mut T: i32,
    mut C: i32,
    mut OC: i32,
) {
    let mut b: i32 = 0;
    while b < B {
        let mut t: i32 = 0;
        while t < T {
            let mut dout_bt: *mut f32 = dout.add((b * T * OC) as usize).add((t * OC) as usize);
            let mut dinp_bt: *mut f32 = dinp.add((b * T * C) as usize).add((t * C) as usize);
            let mut o: i32 = 0;
            while o < OC {
                let mut wrow: *mut f32 = weight.add((o * C) as usize);
                let d: f32 = *dout_bt.add(o as usize);
                let mut i: i32 = 0;
                while i < C {
                    *dinp_bt.add(i as usize) += *wrow.add(i as usize) * d;
                    i += 1;
                }
                o += 1;
            }
            t += 1;
        }
        b += 1;
    }

    let mut o_0: i32 = 0;
    while o_0 < OC {
        let mut b_0: i32 = 0;
        while b_0 < B {
            let mut t_0: i32 = 0;
            while t_0 < T {
                let mut dout_bt_0: *mut f32 = dout.add((b_0 * T * OC) as usize).add((t_0 * OC) as usize);
                let mut inp_bt: *mut f32 = inp.add((b_0 * T * C) as usize).add((t_0 * C) as usize);
                let mut dwrow: *mut f32 = dweight.add((o_0 * C) as usize);
                let d_0: f32 = *dout_bt_0.add(o_0 as usize);
                if !dbias.is_null() {
                    *dbias.add(o_0 as usize) += d_0;
                }
                let mut i_0: i32 = 0;
                while i_0 < C {
                    *dwrow.add(i_0 as usize) += *inp_bt.add(i_0 as usize) * d_0;
                    i_0 += 1;
                }
                t_0 += 1;
            }
            b_0 += 1;
        }
        o_0 += 1;
    }
}

pub unsafe fn attention_forward(
    mut out: *mut f32,
    mut preatt: *mut f32,
    mut att: *mut f32,
    mut inp: *mut f32,
    mut B: i32,
    mut T: i32,
    mut C: i32,
    mut NH: i32,
) {
    let C3: i32 = C * 3;
    let hs: i32 = C / NH;
    let scale: f32 = (1.0f64 / (hs as f32).sqrt() as f64) as f32;
    let mut b: i32 = 0;
    while b < B {
        let mut t: i32 = 0;
        while t < T {
            let mut h: i32 = 0;
            while h < NH {
                let mut query_t: *mut f32 = inp.add((b * T * C3) as usize + (t * C3) as usize + (h * hs) as usize);
                let mut preatt_bth: *mut f32 = preatt.add((b * NH * T * T) as usize + (h * T * T) as usize + (t * T) as usize);
                let mut att_bth: *mut f32 = att.add((b * NH * T * T) as usize + (h * T * T) as usize + (t * T) as usize);
                let mut maxval: f32 = -10000.0;
                let mut t2: i32 = 0;
                while t2 <= t {
                    let mut key_t2: *mut f32 = inp.add((b * T * C3) as usize + (t2 * C3) as usize + (h * hs) as usize + C as usize);
                    let mut val: f32 = 0.0;
                    let mut i: i32 = 0;
                    while i < hs {
                        val += *query_t.add(i as usize) * *key_t2.add(i as usize);
                        i += 1;
                    }
                    val *= scale;
                    if val > maxval {
                        maxval = val;
                    }
                    *preatt_bth.add(t2 as usize) = val;
                    t2 += 1;
                }
                let mut expsum: f32 = 0.0;
                let mut t2_0: i32 = 0;
                while t2_0 <= t {
                    let expv: f32 = (*preatt_bth.add(t2_0 as usize)).exp() * (-maxval).exp();
                    expsum += expv;
                    *att_bth.add(t2_0 as usize) = expv;
                    t2_0 += 1;
                }
                let expsum_inv: f32 = if expsum == 0.0 { 0.0 } else { 1.0 / expsum };
                let mut t2_1: i32 = 0;
                while t2_1 < T {
                    if t2_1 <= t {
                        *att_bth.add(t2_1 as usize) *= expsum_inv;
                    } else {
                        *att_bth.add(t2_1 as usize) = 0.0;
                    }
                    t2_1 += 1;
                }
                let mut out_bth: *mut f32 = out.add((b * T * C) as usize + (t * C) as usize + (h * hs) as usize);
                let mut i_0: i32 = 0;
                while i_0 < hs {
                    *out_bth.add(i_0 as usize) = 0.0;
                    i_0 += 1;
                }
                let mut t2_2: i32 = 0;
                while t2_2 <= t {
                    let mut value_t2: *mut f32 = inp.add((b * T * C3) as usize + (t2_2 * C3) as usize + (h * hs) as usize + (C * 2) as usize);
                    let att_btht2: f32 = *att_bth.add(t2_2 as usize);
                    let mut i_1: i32 = 0;
                    while i_1 < hs {
                        *out_bth.add(i_1 as usize) += att_btht2 * *value_t2.add(i_1 as usize);
                        i_1 += 1;
                    }
                    t2_2 += 1;
                }
                h += 1;
            }
            t += 1;
        }
        b += 1;
    }
}

pub unsafe fn attention_backward(
    mut dinp: *mut f32,
    mut dpreatt: *mut f32,
    mut datt: *mut f32,
    mut dout: *mut f32,
    mut inp: *mut f32,
    mut att: *mut f32,
    mut B: i32,
    mut T: i32,
    mut C: i32,
    mut NH: i32,
) {
    let C3: i32 = C * 3;
    let hs: i32 = C / NH;
    let scale: f32 = (1.0f64 / (hs as f32).sqrt() as f64) as f32;
    let mut b: i32 = 0;
    while b < B {
        let mut t: i32 = 0;
        while t < T {
            let mut h: i32 = 0;
            while h < NH {
                let mut att_bth: *mut f32 = att.add((b * NH * T * T) as usize + (h * T * T) as usize + (t * T) as usize);
                let mut datt_bth: *mut f32 = datt.add((b * NH * T * T) as usize + (h * T * T) as usize + (t * T) as usize);
                let mut dpreatt_bth: *mut f32 = dpreatt.add((b * NH * T * T) as usize + (h * T * T) as usize + (t * T) as usize);
                let mut dquery_t: *mut f32 = dinp.add((b * T * C3) as usize + (t * C3) as usize + (h * hs) as usize);
                let mut query_t: *mut f32 = inp.add((b * T * C3) as usize + (t * C3) as usize + (h * hs) as usize);
                let mut dout_bth: *mut f32 = dout.add((b * T * C) as usize + (t * C) as usize + (h * hs) as usize);
                let mut t2: i32 = 0;
                while t2 <= t {
                    let mut value_t2: *mut f32 = inp.add((b * T * C3) as usize + (t2 * C3) as usize + (h * hs) as usize + (C * 2) as usize);
                    let mut dvalue_t2: *mut f32 = dinp.add((b * T * C3) as usize + (t2 * C3) as usize + (h * hs) as usize + (C * 2) as usize);
                    let mut i: i32 = 0;
                    while i < hs {
                        *datt_bth.add(t2 as usize) += *value_t2.add(i as usize) * *dout_bth.add(i as usize);
                        *dvalue_t2.add(i as usize) += *att_bth.add(t2 as usize) * *dout_bth.add(i as usize);
                        i += 1;
                    }
                    t2 += 1;
                }
                let mut t2_0: i32 = 0;
                while t2_0 <= t {
                    let mut t3: i32 = 0;
                    while t3 <= t {
                        let indicator: f32 = if t2_0 == t3 { 1.0 } else { 0.0 };
                        let local_derivative: f32 = *att_bth.add(t2_0 as usize) * (indicator - *att_bth.add(t3 as usize));
                        *dpreatt_bth.add(t3 as usize) += local_derivative * *datt_bth.add(t2_0 as usize);
                        t3 += 1;
                    }
                    t2_0 += 1;
                }
                let mut t2_1: i32 = 0;
                while t2_1 <= t {
                    let mut key_t2: *mut f32 = inp.add((b * T * C3) as usize + (t2_1 * C3) as usize + (h * hs) as usize + C as usize);
                    let mut dkey_t2: *mut f32 = dinp.add((b * T * C3) as usize + (t2_1 * C3) as usize + (h * hs) as usize + C as usize);
                    let mut i_0: i32 = 0;
                    while i_0 < hs {
                        *dquery_t.add(i_0 as usize) += *key_t2.add(i_0 as usize) * *dpreatt_bth.add(t2_1 as usize) * scale;
                        *dkey_t2.add(i_0 as usize) += *query_t.add(i_0 as usize) * *dpreatt_bth.add(t2_1 as usize) * scale;
                        i_0 += 1;
                    }
                    t2_1 += 1;
                }
                h += 1;
            }
            t += 1;
        }
        b += 1;
    }
}

pub unsafe fn gelu_forward(
    mut out: *mut f32,
    mut inp: *mut f32,
    mut N: i32,
) {
    let mut i: i32 = 0;
    while i < N {
        let x: f32 = *inp.offset(i as isize);
        let cube: f32 = 0.044715 * x * x * x;
        let sqrt_pi_over_two: f32 = (2.0 / std::f64::consts::PI as f32).sqrt();
        *out.offset(i as isize) = 0.5 * x * (1.0 + (sqrt_pi_over_two * (x + cube)).tanh());
        i += 1;
    }
}

pub unsafe fn gelu_backward(
    mut dinp: *mut f32,
    mut inp: *mut f32,
    mut dout: *mut f32,
    mut N: i32,
) {
    let mut i: i32 = 0;
    while i < N {
        let x: f32 = *inp.offset(i as isize);
        let cube: f32 = 0.044715 * x * x * x;
        let sqrt_pi_over_two: f32 = (2.0 / std::f64::consts::PI as f32).sqrt();
        let tanh_arg: f32 = sqrt_pi_over_two * (x + cube);
        let tanh_out: f32 = tanh_arg.tanh();
        let cosh_out: f32 = tanh_arg.cosh();
        let sech_out: f32 = 1.0 / (cosh_out * cosh_out);
        let local_grad: f32 = 0.5 * (1.0 + tanh_out)
            + x * 0.5 * sech_out * sqrt_pi_over_two * (1.0 + 3.0 * 0.044715 * x * x);
        *dinp.offset(i as isize) += local_grad * *dout.offset(i as isize);
        i += 1;
    }
}

pub unsafe fn residual_forward(
    mut out: *mut f32,
    mut inp1: *mut f32,
    mut inp2: *mut f32,
    mut N: i32,
) {
    let mut i: i32 = 0;
    while i < N {
        *out.offset(i as isize) = *inp1.offset(i as isize) + *inp2.offset(i as isize);
        i += 1;
    }
}

pub unsafe fn residual_backward(
    mut dinp1: *mut f32,
    mut dinp2: *mut f32,
    mut dout: *mut f32,
    mut N: i32,
) {
    let mut i: i32 = 0;
    while i < N {
        *dinp1.offset(i as isize) += *dout.offset(i as isize);
        *dinp2.offset(i as isize) += *dout.offset(i as isize);
        i += 1;
    }
}

pub unsafe fn softmax_forward(
    mut probs: *mut f32,
    mut logits: *mut f32,
    mut B: i32,
    mut T: i32,
    mut V: i32,
) {
    let mut b: i32 = 0;
    while b < B {
        let mut t: i32 = 0;
        while t < T {
            let mut logits_bt: *mut f32 = logits.add((b * T * V) as usize + (t * V) as usize);
            let mut probs_bt: *mut f32 = probs.add((b * T * V) as usize + (t * V) as usize);
            let mut maxval: f32 = -10000.0;
            let mut i: i32 = 0;
            while i < V {
                let val = *logits_bt.offset(i as isize);
                if val > maxval {
                    maxval = val;
                }
                i += 1;
            }
            let mut sum: f32 = 0.0;
            let mut i_0: i32 = 0;
            while i_0 < V {
                let exp_val = (*logits_bt.offset(i_0 as isize) - maxval).exp();
                *probs_bt.offset(i_0 as isize) = exp_val;
                sum += exp_val;
                i_0 += 1;
            }
            let mut i_1: i32 = 0;
            while i_1 < V {
                *probs_bt.offset(i_1 as isize) /= sum;
                i_1 += 1;
            }
            t += 1;
        }
        b += 1;
    }
}

pub unsafe fn crossentropy_forward(
    mut losses: *mut f32,
    mut probs: *mut f32,
    mut targets: *mut i32,
    mut B: i32,
    mut T: i32,
    mut V: i32,
) {
    let mut b: i32 = 0;
    while b < B {
        let mut t: i32 = 0;
        while t < T {
            let mut probs_bt: *mut f32 = probs.add((b * T * V) as usize + (t * V) as usize);
            let ix: i32 = *targets.add((b * T + t) as usize);
            *losses.add((b * T + t) as usize) = -(*probs_bt.add(ix as usize)).ln();
            t += 1;
        }
        b += 1;
    }
}

pub unsafe fn crossentropy_softmax_backward(
    mut dlogits: *mut f32,
    mut dlosses: *mut f32,
    mut probs: *mut f32,
    mut targets: *mut i32,
    mut B: i32,
    mut T: i32,
    mut V: i32,
) {
    let mut b: i32 = 0;
    while b < B {
        let mut t: i32 = 0;
        while t < T {
            let mut dlogits_bt: *mut f32 = dlogits.add((b * T * V) as usize + (t * V) as usize);
            let mut probs_bt: *mut f32 = probs.add((b * T * V) as usize + (t * V) as usize);
            let dloss: f32 = *dlosses.add((b * T + t) as usize);
            let ix: i32 = *targets.add((b * T + t) as usize);
            let mut i: i32 = 0;
            while i < V {
                let p: f32 = *probs_bt.add(i as usize);
                let indicator: f32 = if i == ix { 1.0 } else { 0.0 };
                *dlogits_bt.add(i as usize) += (p - indicator) * dloss;
                i += 1;
            }
            t += 1;
        }
        b += 1;
    }
}

pub unsafe fn malloc_and_point_parameters(
    params: *mut ParameterTensors,
    param_sizes: *mut usize,
) -> *mut f32 {
    let mut num_parameters: usize = 0;
    for i in 0..16 {
        num_parameters += *param_sizes.add(i);
    }

    let layout = Layout::array::<f32>(num_parameters).unwrap();
    let params_memory = alloc(layout) as *mut f32;
    if params_memory.is_null() {
        return null_mut();
    }

    let ptrs = &mut [
        &mut (*params).wte,
        &mut (*params).wpe,
        &mut (*params).ln1w,
        &mut (*params).ln1b,
        &mut (*params).qkvw,
        &mut (*params).qkvb,
        &mut (*params).attprojw,
        &mut (*params).attprojb,
        &mut (*params).ln2w,
        &mut (*params).ln2b,
        &mut (*params).fcw,
        &mut (*params).fcb,
        &mut (*params).fcprojw,
        &mut (*params).fcprojb,
        &mut (*params).lnfw,
        &mut (*params).lnfb,
    ];

    let mut offset = 0;
    for (tensor, i) in ptrs.iter_mut().zip(0..16) {
        **tensor = params_memory.add(offset);
        offset += *param_sizes.add(i);
    }

    params_memory
}

pub unsafe fn malloc_and_point_activations(
    mut acts: *mut ActivationTensors,
    mut act_sizes: *mut usize,
) -> *mut f32 {
    let mut num_activations: usize = 0;
    let mut i: usize = 0;
    while i < 23 {
        num_activations += *act_sizes.offset(i as isize);
        i += 1;
    }
    let mut acts_memory: *mut f32 = libc::malloc(
        num_activations * std::mem::size_of::<f32>()
    ) as *mut f32;
    let ptrs: [*mut *mut f32; 23] = [
        &mut (*acts).encoded,
        &mut (*acts).ln1,
        &mut (*acts).ln1_mean,
        &mut (*acts).ln1_rstd,
        &mut (*acts).qkv,
        &mut (*acts).atty,
        &mut (*acts).preatt,
        &mut (*acts).att,
        &mut (*acts).attproj,
        &mut (*acts).residual2,
        &mut (*acts).ln2,
        &mut (*acts).ln2_mean,
        &mut (*acts).ln2_rstd,
        &mut (*acts).fch,
        &mut (*acts).fch_gelu,
        &mut (*acts).fcproj,
        &mut (*acts).residual3,
        &mut (*acts).lnf,
        &mut (*acts).lnf_mean,
        &mut (*acts).lnf_rstd,
        &mut (*acts).logits,
        &mut (*acts).probs,
        &mut (*acts).losses,
    ];
    let mut acts_memory_iterator: *mut f32 = acts_memory;
    let mut i_0: usize = 0;
    while i_0 < 23 {
        *ptrs[i_0] = acts_memory_iterator;
        acts_memory_iterator = acts_memory_iterator.offset(*act_sizes.offset(i_0 as isize) as isize);
        i_0 += 1;
    }
    acts_memory
}

pub unsafe fn gpt2_build_from_checkpoint(
    model: *mut GPT2,
    checkpoint_path: &str,
) -> io::Result<()> {
    let file = File::open(Path::new(checkpoint_path))
        .map_err(|_| {
            println!("Error opening model file");
            io::Error::new(io::ErrorKind::NotFound, "Failed to open file")
        })?;

    let mut reader = BufReader::new(file);
    let mut model_header = [0i32; 256];
    let header_bytes = slice::from_raw_parts_mut(model_header.as_mut_ptr() as *mut u8, 256 * std::mem::size_of::<i32>());
    reader.read_exact(header_bytes)?;

    if model_header[0] != 20240326 {
        println!("Bad magic model file");
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad magic number"));
    }

    if model_header[1] != 1 {
        println!("Bad version in model file");
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Unsupported version"));
    }

    let maxT = model_header[2];
    let V = model_header[3];
    let L = model_header[4];
    let NH = model_header[5];
    let C = model_header[6];

    (*model).config.max_seq_len = maxT;
    (*model).config.vocab_size = V;
    (*model).config.num_layers = L;
    (*model).config.num_heads = NH;
    (*model).config.channels = C;

    println!("[GPT-2]");
    println!("max_seq_len: {}", maxT);
    println!("vocab_size: {}", V);
    println!("num_layers: {}", L);
    println!("num_heads: {}", NH);
    println!("channels: {}", C);

    (*model).params_memory = malloc_and_point_parameters(
        &mut (*model).params,
        (*model).param_sizes.as_mut_ptr(),
    );

    let num_parameters = (*model).param_sizes.iter().sum::<usize>();
    let param_size_bytes = num_parameters * std::mem::size_of::<f32>();
    let params_bytes = slice::from_raw_parts_mut((*model).params_memory as *mut u8, param_size_bytes);
    reader.read_exact(params_bytes)?;

    Ok(())
}

pub unsafe fn gpt2_forward(
    model: *mut GPT2,
    inputs: *mut i32,
    targets: *mut i32,
    B: i32,
    T: i32,
) {
    if (*model).params_memory.is_null() {
        eprintln!("Error: model was not initialized properly.");
        process::exit(1);
    }

    let V = (*model).config.vocab_size;
    let L = (*model).config.num_layers;
    let NH = (*model).config.num_heads;
    let C = (*model).config.channels;

    if (*model).acts_memory.is_null() {
        (*model).batch_size = B;
        (*model).seq_len = T;

        // Calculation for the activation sizes based on the configuration
        let act_size_calculations: [usize; 23] = [
            (B * T * C) as usize,
            (L * B * T * C) as usize,
            (L * B * T) as usize,
            (L * B * T) as usize,
            (L * B * T * 3 * C) as usize,
            (L * B * T * C) as usize,
            (L * B * NH * T * T) as usize,
            (L * B * NH * T * T) as usize,
            (L * B * T * C) as usize,
            (L * B * T * C) as usize,
            (L * B * T * C) as usize,
            (L * B * T) as usize,
            (L * B * T) as usize,
            (L * B * T * 4 * C) as usize,
            (L * B * T * 4 * C) as usize,
            (L * B * T * C) as usize,
            (L * B * T * C) as usize,
            (B * T * C) as usize,
            (B * T) as usize,
            (B * T) as usize,
            (B * T * V) as usize,
            (B * T * V) as usize,
            (B * T) as usize,
        ];

        for (i, &size) in act_size_calculations.iter().enumerate() {
            (*model).act_sizes[i] = size;
        }

        let num_activations: usize = act_size_calculations.iter().sum();
        println!("num_activations: {}", num_activations);
        (*model).num_activations = num_activations as i32;

        (*model).acts_memory = malloc_and_point_activations(&mut (*model).acts, (*model).act_sizes.as_mut_ptr());

        // Allocate memory for inputs and targets only if needed
        (*model).inputs = Box::into_raw(vec![0i32; (B * T) as usize].into_boxed_slice()) as *mut i32;
        (*model).targets = Box::into_raw(vec![0i32; (B * T) as usize].into_boxed_slice()) as *mut i32;
    } else if B > (*model).batch_size || T > (*model).seq_len {
        eprintln!("Error: batch size or sequence length is inadequately large");
        eprintln!("Model: B={} T={}, Desired: B={} T={}", (*model).batch_size, (*model).seq_len, B, T);
        process::exit(1);
    }

    // Copy inputs and targets into model memory
    std::ptr::copy(inputs, (*model).inputs, (B * T) as usize);
    if !targets.is_null() {
        std::ptr::copy(targets, (*model).targets, (B * T) as usize);
    }

    // Placeholder for actual layer computations
    for l in 0..L {
        println!("Processing layer {}", l);
    }

    // Update model loss if targets are provided
    if !targets.is_null() {
        let mut mean_loss: f32 = 0.0;
        let losses_ptr = std::slice::from_raw_parts((*model).acts.losses, (B * T) as usize);
        for &loss in losses_ptr.iter() {
            mean_loss += loss;
        }
        mean_loss /= (B * T) as f32;
        (*model).mean_loss = mean_loss;
    } else {
        (*model).mean_loss = -1.0;
    }
}

pub unsafe fn gpt2_zero_grad(mut model: *mut GPT2) {
    if !ptr::eq((*model).grads_memory, ptr::null()) {
        let size = ((*model).num_parameters as usize) * mem::size_of::<f32>();
        ptr::write((*model).grads_memory as *mut f32, 0.0f32);
        ptr::write(((*model).grads_memory as *mut f32).offset(size as isize), 0.0f32);
    }
    if !ptr::eq((*model).grads_acts_memory, ptr::null()) {
        let size = ((*model).num_activations as usize) * mem::size_of::<f32>();
        ptr::write((*model).grads_acts_memory as *mut f32, 0.0f32);
        ptr::write(((*model).grads_acts_memory as *mut f32).offset(size as isize), 0.0f32);
    }
}


pub unsafe fn gpt2_backward(model: *mut GPT2) {
    if (*model).mean_loss == -1.0 {
        eprintln!("Error: must forward with targets before backward.");
        process::exit(1);
    }

    if (*model).grads_memory.is_null() {
        (*model).grads_memory = malloc_and_point_parameters(&mut (*model).grads, (*model).param_sizes.as_mut_ptr());
        (*model).grads_acts_memory = malloc_and_point_activations(&mut (*model).grads_acts, (*model).act_sizes.as_mut_ptr());
        gpt2_zero_grad(model);
    }

    let B = (*model).batch_size;
    let T = (*model).seq_len;
    let V = (*model).config.vocab_size as usize;
    let L = (*model).config.num_layers as usize;
    let NH = (*model).config.num_heads;
    let C = (*model).config.channels;

    // Create slices for easier manipulation
    let grads_acts_losses = slice::from_raw_parts_mut((*model).grads_acts.losses, (B * T) as usize);
    let dloss_mean = 1.0 / (B * T) as f32;

    for loss in grads_acts_losses.iter_mut() {
        *loss = dloss_mean;
    }

    crossentropy_softmax_backward(
        (*model).grads_acts.logits,
        (*model).grads_acts.losses,
        (*model).acts.probs,
        (*model).targets,
        B as i32,
        T as i32,
        V as i32,
    );
    matmul_backward(
        (*model).grads_acts.lnf,
        (*model).grads.wte,
        ptr::null_mut(), // Using null_mut() for the bias pointer, as 0 was passed
        (*model).grads_acts.logits,
        (*model).acts.lnf,
        (*model).params.wte,
        B as i32,
        T as i32,
        C as i32,
        V as i32,
    );

    let mut residual = (*model).acts.residual3.add((L - 1) * (B * T * C) as usize);
    let mut dresidual = (*model).grads_acts.residual3.add((L - 1) * (B * T * C) as usize);

    layernorm_backward(
        dresidual,
        (*model).grads.lnfw,
        (*model).grads.lnfb,
        (*model).grads_acts.lnf,
        residual,
        (*model).params.lnfw,
        (*model).acts.lnf_mean,
        (*model).acts.lnf_rstd,
        B as i32,
        T as i32,
        C as i32,
    );

    let mut l = L as i32 - 1;
    while l >= 0 {

        residual = if l == 0 {
            (*model).acts.encoded
        } else {
            (*model).acts.residual3.offset(((l - 1) * B * T * C) as isize)
        };
        dresidual = if l == 0 {
            (*model).grads_acts.encoded
        } else {
            (*model).grads_acts.residual3.offset(((l - 1) * B * T * C) as isize)
        };

        let l_ln1w = (*model).params.ln1w.offset((l * C) as isize);
        let l_qkvw = (*model).params.qkvw.offset((l * 3 * C * C) as isize);
        let l_attprojw = (*model).params.attprojw.offset((l * C * C) as isize);
        let l_ln2w = (*model).params.ln2w.offset((l * C) as isize);
        let l_fcw = (*model).params.fcw.offset((l * 4 * C * C) as isize);
        let l_fcprojw = (*model).params.fcprojw.offset((l * C * 4 * C) as isize);

        let dl_ln1w = (*model).grads.ln1w.offset((l * C) as isize);
        let dl_ln1b = (*model).grads.ln1b.offset((l * C) as isize);
        let dl_qkvw = (*model).grads.qkvw.offset((l * 3 * C * C) as isize);
        let dl_qkvb = (*model).grads.qkvb.offset((l * 3 * C) as isize);
        let dl_attprojw = (*model).grads.attprojw.offset((l * C * C) as isize);
        let dl_attprojb = (*model).grads.attprojb.offset((l * C) as isize);
        let dl_ln2w = (*model).grads.ln2w.offset((l * C) as isize);
        let dl_ln2b = (*model).grads.ln2b.offset((l * C) as isize);
        let dl_fcw = (*model).grads.fcw.offset((l * 4 * C * C) as isize);
        let dl_fcb = (*model).grads.fcb.offset((l * 4 * C) as isize);
        let dl_fcprojw = (*model).grads.fcprojw.offset((l * C * 4 * C) as isize);
        let dl_fcprojb = (*model).grads.fcprojb.offset((l * C) as isize);

        let l_ln1 = (*model).acts.ln1.offset((l * B * T * C) as isize);
        let l_ln1_mean = (*model).acts.ln1_mean.offset((l * B * T) as isize);
        let l_ln1_rstd = (*model).acts.ln1_rstd.offset((l * B * T) as isize);
        let l_qkv = (*model).acts.qkv.offset((l * B * T * 3 * C) as isize);
        let l_atty = (*model).acts.atty.offset((l * B * T * C) as isize);
        let l_att = (*model).acts.att.offset((l * B * NH * T * T) as isize);
        let l_residual2 = (*model).acts.residual2.offset((l * B * T * C) as isize);
        let l_ln2 = (*model).acts.ln2.offset((l * B * T * C) as isize);
        let l_ln2_mean = (*model).acts.ln2_mean.offset((l * B * T) as isize);
        let l_ln2_rstd = (*model).acts.ln2_rstd.offset((l * B * T) as isize);
        let l_fch = (*model).acts.fch.offset((l * B * T * 4 * C) as isize);
        let l_fch_gelu = (*model).acts.fch_gelu.offset((l * B * T * 4 * C) as isize);

        let dl_ln1 = (*model).grads_acts.ln1.offset((l * B * T * C) as isize);
        let dl_qkv = (*model).grads_acts.qkv.offset((l * B * T * 3 * C) as isize);
        let dl_atty = (*model).grads_acts.atty.offset((l * B * T * C) as isize);
        let dl_preatt = (*model).grads_acts.preatt.offset((l * B * NH * T * T) as isize);
        let dl_att = (*model).grads_acts.att.offset((l * B * NH * T * T) as isize);
        let dl_attproj = (*model).grads_acts.attproj.offset((l * B * T * C) as isize);
        let dl_residual2 = (*model).grads_acts.residual2.offset((l * B * T * C) as isize);
        let dl_ln2 = (*model).grads_acts.ln2.offset((l * B * T * C) as isize);
        let dl_fch = (*model).grads_acts.fch.offset((l * B * T * 4 * C) as isize);
        let dl_fch_gelu = (*model).grads_acts.fch_gelu.offset((l * B * T * 4 * C) as isize);
        let dl_fcproj = (*model).grads_acts.fcproj.offset((l * B * T * C) as isize);
        let dl_residual3 = (*model).grads_acts.residual3.offset((l * B * T * C) as isize);

        residual_backward(dl_residual2, dl_fcproj, dl_residual3, (B * T * C) as i32);
        matmul_backward(
            dl_fch_gelu,
            dl_fcprojw,
            dl_fcprojb,
            dl_fcproj,
            l_fch_gelu,
            l_fcprojw,
            B as i32,
            T as i32,
            4 * C as i32,
            C as i32,
        );
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, (B * T * 4 * C) as i32);
        matmul_backward(
            dl_ln2,
            dl_fcw,
            dl_fcb,
            dl_fch,
            l_ln2,
            l_fcw,
            B as i32,
            T as i32,
            C as i32,
            4 * C as i32,
        );
        layernorm_backward(
            dl_residual2,
            dl_ln2w,
            dl_ln2b,
            dl_ln2,
            l_residual2,
            l_ln2w,
            l_ln2_mean,
            l_ln2_rstd,
            B as i32,
            T as i32,
            C as i32,
        );

        residual_backward(dresidual, dl_attproj, dl_residual2, (B * T * C) as i32);
        matmul_backward(
            dl_atty,
            dl_attprojw,
            dl_attprojb,
            dl_attproj,
            l_atty,
            l_attprojw,
            B as i32,
            T as i32,
            C as i32,
            C as i32,
        );
        attention_backward(
            dl_qkv,
            dl_preatt,
            dl_att,
            dl_atty,
            l_qkv,
            l_att,
            B as i32,
            T as i32,
            C as i32,
            NH,
        );
        matmul_backward(
            dl_ln1,
            dl_qkvw,
            dl_qkvb,
            dl_qkv,
            l_ln1,
            l_qkvw,
            B as i32,
            T as i32,
            C as i32,
            3 * C as i32,
        );
        layernorm_backward(
            dresidual,
            dl_ln1w,
            dl_ln1b,
            dl_ln1,
            residual,
            l_ln1w,
            l_ln1_mean,
            l_ln1_rstd,
            B as i32,
            T as i32,
            C as i32,
        );

        l -= 1;
    }

    encoder_backward(
        (*model).grads.wte,
        (*model).grads.wpe,
        (*model).grads_acts.encoded,
        (*model).inputs,
        B as i32,
        T as i32,
        C as i32
    );

}

pub unsafe fn gpt2_update(
    model: *mut GPT2,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: i32,
) {
    if (*model).m_memory.is_null() {
        (*model).m_memory = vec![0.0; (*model).num_parameters as usize].as_mut_ptr();
        (*model).v_memory = vec![0.0; (*model).num_parameters as usize].as_mut_ptr();
    }

    let num_params = (*model).num_parameters as isize;
    let params = std::slice::from_raw_parts_mut((*model).params_memory, num_params as usize);
    let grads = std::slice::from_raw_parts((*model).grads_memory, num_params as usize);
    let ms = std::slice::from_raw_parts_mut((*model).m_memory, num_params as usize);
    let vs = std::slice::from_raw_parts_mut((*model).v_memory, num_params as usize);

    for i in 0..num_params as usize {
        let grad = grads[i];
        let param = params[i];

        let m = beta1 * ms[i] + (1.0 - beta1) * grad;
        let v = beta2 * vs[i] + (1.0 - beta2) * grad * grad;

        let bias_correction1 = 1.0 - beta1.powi(t);
        let bias_correction2 = 1.0 - beta2.powi(t);

        let m_hat = m / bias_correction1;
        let v_hat = v / (v.sqrt() / bias_correction2 + eps);

        ms[i] = m;
        vs[i] = v;

        params[i] -= learning_rate * (m_hat / (v_hat.sqrt() + eps) + weight_decay * param);
    }
}

pub unsafe fn gpt2_free(model: *mut GPT2) {
    if !(*model).params_memory.is_null() {
        let _ = Box::from_raw((*model).params_memory);
    }
    if !(*model).grads_memory.is_null() {
        let _ = Box::from_raw((*model).grads_memory);
    }
    if !(*model).m_memory.is_null() {
        let _ = Box::from_raw((*model).m_memory);
    }
    if !(*model).v_memory.is_null() {
        let _ = Box::from_raw((*model).v_memory);
    }
    if !(*model).acts_memory.is_null() {
        let _ = Box::from_raw((*model).acts_memory);
    }
    if !(*model).grads_acts_memory.is_null() {
        let _ = Box::from_raw((*model).grads_acts_memory);
    }
    if !(*model).inputs.is_null() {
        let _ = Box::from_raw((*model).inputs);
    }
    if !(*model).targets.is_null() {
        let _ = Box::from_raw((*model).targets);
    }
}

pub fn dataloader_init(
    loader: &mut DataLoader,
    filename: &str,
    B: i32,
    T: i32,
) {
    loader.B = B;
    loader.T = T;

    let file = match File::open(filename) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("Error opening tokens file");
            std::process::exit(1);
        }
    };

    let file_size = file.metadata().unwrap().len(); // keep as u64

    let num_elements = (B * T + 1) as usize; // calculate total number of elements
    let total_size_needed = num_elements * std::mem::size_of::<i32>();

    if (file_size as usize) < total_size_needed {
        eprintln!("Error: file size is too small for the batch size and sequence length");
        std::process::exit(1);
    }

    loader.tokens_file = Some(file);
    loader.file_size = file_size as i64;
    loader.current_position = 0;

    loader.batch = vec![0i32; num_elements];
    loader.inputs = loader.batch.as_mut_ptr();
    loader.targets = unsafe { loader.inputs.add(1) };

    // Safely calculate the number of batches
    let num_batches = file_size as usize / total_size_needed;

    // Check for potential integer overflow before casting to i32
    if num_batches > i32::MAX as usize {
        eprintln!("Error: Computed number of batches exceeds i32 maximum");
        std::process::exit(1);
    }

    loader.num_batches = num_batches as i32;
}

pub fn dataloader_reset(loader: &mut DataLoader) {
    loader.current_position = 0;
}

pub unsafe fn dataloader_next_batch(loader: *mut DataLoader) {
    let B = (*loader).B as usize;
    let T = (*loader).T as usize;
    let increment = (B * T + 1) * std::mem::size_of::<i32>();

    if (*loader).current_position as usize + increment > (*loader).file_size as usize {
        (*loader).current_position = 0;
    }

    let file = &mut *(*loader).tokens_file.as_mut().unwrap();
    file.seek(std::io::SeekFrom::Start((*loader).current_position as u64)).unwrap();

    // Assuming `batch` is a Vec<i32> and we use as_mut_ptr() to get a raw pointer
    let buffer = std::slice::from_raw_parts_mut((*loader).batch.as_mut_ptr(), B * T + 1);
    file.read_exact(std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut u8, increment)).unwrap();

    (*loader).current_position += increment as i64;
}

pub fn dataloader_free(loader: &mut DataLoader) {
    // Close the file by dropping the Option<File>, which automatically calls close()
    loader.tokens_file.take(); // This takes the file out of the Option and drops it, effectively closing the file

    // Clear the batch vector to free its memory
    loader.batch.clear();
    loader.batch.shrink_to_fit(); // Optional: this can help return memory to the system

    // Note: There is no need to manually free loader.inputs and loader.targets as they are just pointers into loader.batch
}

pub fn random_u32(state: &mut u64) -> u32 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ((*state).wrapping_mul(0x2545F4914F6CDD1Du64) >> 32) as u32
}

pub fn random_f32(state: &mut u64) -> f32 {
    (random_u32(state) >> 8) as f32 / 16777216.0
}

pub fn sample_mult(probabilities: &[f32], coin: f32) -> i32 {
    let mut cdf = 0.0_f32;
    for (i, &prob) in probabilities.iter().enumerate() {
        cdf += prob;
        if coin < cdf {
            return i as i32;
        }
    }
    probabilities.len() as i32 - 1 // in case of rounding errors
}


fn main() {
    let mut model = GPT2 {
        config: GPT2Config {
            max_seq_len: 0,
            vocab_size: 0,
            num_layers: 0,
            num_heads: 0,
            channels: 0,
        },
        params: ParameterTensors {
            wte: ptr::null_mut(),
            wpe: ptr::null_mut(),
            ln1w: ptr::null_mut(),
            ln1b: ptr::null_mut(),
            qkvw: ptr::null_mut(),
            qkvb: ptr::null_mut(),
            attprojw: ptr::null_mut(),
            attprojb: ptr::null_mut(),
            ln2w: ptr::null_mut(),
            ln2b: ptr::null_mut(),
            fcw: ptr::null_mut(),
            fcb: ptr::null_mut(),
            fcprojw: ptr::null_mut(),
            fcprojb: ptr::null_mut(),
            lnfw: ptr::null_mut(),
            lnfb: ptr::null_mut(),
        },
        param_sizes: [0; 16],
        params_memory: ptr::null_mut(),
        num_parameters: 0,
        grads: ParameterTensors {
            wte: ptr::null_mut(),
            wpe: ptr::null_mut(),
            ln1w: ptr::null_mut(),
            ln1b: ptr::null_mut(),
            qkvw: ptr::null_mut(),
            qkvb: ptr::null_mut(),
            attprojw: ptr::null_mut(),
            attprojb: ptr::null_mut(),
            ln2w: ptr::null_mut(),
            ln2b: ptr::null_mut(),
            fcw: ptr::null_mut(),
            fcb: ptr::null_mut(),
            fcprojw: ptr::null_mut(),
            fcprojb: ptr::null_mut(),
            lnfw: ptr::null_mut(),
            lnfb: ptr::null_mut(),
        },
        grads_memory: ptr::null_mut(),
        m_memory: ptr::null_mut(),
        v_memory: ptr::null_mut(),
        acts: ActivationTensors {
            encoded: ptr::null_mut(),
            ln1: ptr::null_mut(),
            ln1_mean: ptr::null_mut(),
            ln1_rstd: ptr::null_mut(),
            qkv: ptr::null_mut(),
            atty: ptr::null_mut(),
            preatt: ptr::null_mut(),
            att: ptr::null_mut(),
            attproj: ptr::null_mut(),
            residual2: ptr::null_mut(),
            ln2: ptr::null_mut(),
            ln2_mean: ptr::null_mut(),
            ln2_rstd: ptr::null_mut(),
            fch: ptr::null_mut(),
            fch_gelu: ptr::null_mut(),
            fcproj: ptr::null_mut(),
            residual3: ptr::null_mut(),
            lnf: ptr::null_mut(),
            lnf_mean: ptr::null_mut(),
            lnf_rstd: ptr::null_mut(),
            logits: ptr::null_mut(),
            probs: ptr::null_mut(),
            losses: ptr::null_mut(),
        },
        act_sizes: [0; 23],
        acts_memory: ptr::null_mut(),
        num_activations: 0,
        grads_acts: ActivationTensors {
            encoded: ptr::null_mut(),
            ln1: ptr::null_mut(),
            ln1_mean: ptr::null_mut(),
            ln1_rstd: ptr::null_mut(),
            qkv: ptr::null_mut(),
            atty: ptr::null_mut(),
            preatt: ptr::null_mut(),
            att: ptr::null_mut(),
            attproj: ptr::null_mut(),
            residual2: ptr::null_mut(),
            ln2: ptr::null_mut(),
            ln2_mean: ptr::null_mut(),
            ln2_rstd: ptr::null_mut(),
            fch: ptr::null_mut(),
            fch_gelu: ptr::null_mut(),
            fcproj: ptr::null_mut(),
            residual3: ptr::null_mut(),
            lnf: ptr::null_mut(),
            lnf_mean: ptr::null_mut(),
            lnf_rstd: ptr::null_mut(),
            logits: ptr::null_mut(),
            probs: ptr::null_mut(),
            losses: ptr::null_mut(),
        },
        grads_acts_memory: ptr::null_mut(),
        batch_size: 0,
        seq_len: 0,
        inputs: ptr::null_mut(),
        targets: ptr::null_mut(),
        mean_loss: 0.0,
    };

    let filename = "gpt2_124M.bin"; // Rust string literal
    unsafe {
        gpt2_build_from_checkpoint(&mut model, filename).expect("build GPT2 from checkpoint");
    }

    let tiny_stories_train = "data/TinyStories_train.bin";
    let tiny_stories_val = "data/TinyStories_val.bin";
    let tiny_shakespeare_train = "data/tiny_shakespeare_train.bin";
    let tiny_shakespeare_val = "data/tiny_shakespeare_val.bin";

    // Check if the training data file exists, falling back if necessary
    let train_tokens = if Path::new(tiny_shakespeare_train).exists() {
        tiny_shakespeare_train
    } else {
        tiny_stories_train
    };

    // Check if the validation data file exists, falling back if necessary
    let val_tokens = if Path::new(tiny_shakespeare_val).exists() {
        tiny_shakespeare_val
    } else {
        tiny_stories_val
    };

    // Assume these are set based on your application's requirements
    let B: i32 = 4;
    let T: i32 = 64;

// Initialize the DataLoader for training
    let mut train_loader = DataLoader {
        B: 0,
        T: 0,
        tokens_file: None,
        file_size: 0,
        current_position: 0,
        batch: Vec::new(),
        inputs: std::ptr::null_mut(),
        targets: std::ptr::null_mut(),
        num_batches: 0,
    };
    dataloader_init(&mut train_loader, train_tokens, B, T);
    println!("train dataset num_batches: {}", train_loader.num_batches);

// Initialize the DataLoader for validation
    let mut val_loader = DataLoader {
        B: 0,
        T: 0,
        tokens_file: None,
        file_size: 0,
        current_position: 0,
        batch: Vec::new(),
        inputs: std::ptr::null_mut(),
        targets: std::ptr::null_mut(),
        num_batches: 0,
    };
    dataloader_init(&mut val_loader, val_tokens, B, T);
    println!("val dataset num_batches: {}", val_loader.num_batches);

    // If you need a specific number of batches for validation (as a fixed value)
    let val_num_batches: i32 = 10;

    let mut rng_state: u64 = 1337;
    let gen_max_length: i32 = 64;
    let mut gen_tokens: [i32; 64] = [0; 64];
    let mut start = std::time::Instant::now();
    let mut end = std::time::Instant::now();
    let mut step: i32 = 0;

    while step <= 20 {

        if step % 10 == 0 {
            let mut val_loss: f32 = 0.0;
            dataloader_reset(&mut val_loader);
            let mut i: i32 = 0;
            while i < val_num_batches {
                unsafe {
                    dataloader_next_batch(&mut val_loader);
                    gpt2_forward(&mut model, val_loader.inputs, val_loader.targets, B, T);
                }
                val_loss += model.mean_loss;
                i += 1;
            }
            val_loss /= val_num_batches as f32;
            println!("val loss {}", val_loss);
        }

        if step > 0 && step % 20 == 0 {
            gen_tokens[0] = 50256;
            let mut t: i32 = 1;

            while t < gen_max_length {
                unsafe {
                    gpt2_forward(
                        &mut model,
                        gen_tokens.as_mut_ptr(),
                        std::ptr::null_mut(),
                        1,
                        t,
                    );
                }
                // Access the probabilities safely, ensuring conversion to usize where needed
                let offset = ((t - 1) as usize) * (model.config.vocab_size as usize);
                let probs_slice = unsafe {
                    std::slice::from_raw_parts(
                        model.acts.probs.add(offset),
                        model.config.vocab_size as usize
                    )
                };
                let coin = random_f32(&mut rng_state);
                let next_token = sample_mult(probs_slice, coin);
                gen_tokens[t as usize] = next_token;
                t += 1;
            }
            print!("generated: ");
            for &token in &gen_tokens[..gen_max_length as usize] {
                print!("{} ", token);
            }
            println!();
        }

        start = Instant::now();
        unsafe {
            dataloader_next_batch(&mut train_loader);
            gpt2_forward(&mut model, train_loader.inputs, train_loader.targets, B, T);
            gpt2_zero_grad(&mut model);
            gpt2_backward(&mut model);
            gpt2_update(&mut model, 1e-4, 0.9, 0.999, 1e-8, 0.0, step + 1);
        }
        let duration = start.elapsed();
        let time_elapsed_ms = duration.as_secs_f64() * 1000.0;
        println!("step {}: train loss {:.3} (took {:.2} ms)", step, model.mean_loss, time_elapsed_ms);

        step += 1;
    }

    dataloader_free(&mut train_loader);
    dataloader_free(&mut val_loader);
    unsafe { gpt2_free(&mut model); }
}


