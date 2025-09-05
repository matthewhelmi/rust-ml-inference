use anyhow::{Result, anyhow};
use image::{self, GenericImageView, imageops::FilterType};
use ndarray::{Array, Array4, Axis};
use std::fs;
use std::time::Instant;

// ===== Imports for Tract =====
use tract_onnx::prelude::Tensor as TractTensor;
use tract_onnx::prelude::*;

// ===== Import for ONNX Runtime (ort) =====
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor as OrtTensor;

// ===== Imports for Burn =====
//use burn::backend::ndarray::NdArray;
//use burn::tensor::Tensor as BurnTensor;

// The following should be created by build.rs
mod generated {
    pub mod resnet50 {
        include!(concat!(env!("OUT_DIR"), "/burn_model/resnet50.rs"));
    }
}
use generated::resnet50 as resnet_model;

// ----------------- Preprocessing -----------------
// TODO: Does Burn have some of these image processing functions?
fn resize_shorter_side_to(img: &image::DynamicImage, short: u32) -> image::DynamicImage {
    let (w, h) = img.dimensions();
    if w < h {
        let new_w = short;
        let new_h = ((h as f32) * (short as f32) / (w as f32)).round() as u32;
        img.resize_exact(new_w, new_h, FilterType::Triangle)
    } else {
        let new_h = short;
        let new_w = ((w as f32) * (short as f32) / (h as f32)).round() as u32;
        img.resize_exact(new_w, new_h, FilterType::Triangle)
    }
}

fn center_crop(img: &image::DynamicImage, crop_w: u32, crop_h: u32) -> image::DynamicImage {
    let (w, h) = img.dimensions();
    let x = ((w - crop_w) / 2).max(0);
    let y = ((h - crop_h) / 2).max(0);
    img.crop_imm(x, y, crop_w, crop_h)
}

// Resize shorter side to 256 (FilterType::Triangle - is it the same as bilinear?)
// Center crop to 224x224
// To RGB, to f32 in [0,1], normalise per channel
// Shape: NCHW (1,3,224,224)
fn preprocess_imagenet(path: &str) -> Result<Array4<f32>> {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD:  [f32; 3] = [0.229, 0.224, 0.225];

    let img_rgb8 = image::open(path)
        .map_err(|e| anyhow!("Failed to open image {}: {e}", path))?
        .to_rgb8();

    // Work in DynamicImage space for ops
    let img_dyn = image::DynamicImage::ImageRgb8(img_rgb8);

    // Resize (keep aspect ratio) then center crop to 224x224
    let resized = resize_shorter_side_to(&img_dyn, 256);
    let cropped = center_crop(&resized, 224, 224).to_rgb8();

    // HWC -> CHW, f32, normalise
    let (h, w) = (224usize, 224usize);
    let mut chw = Array::zeros((3, h, w));
    for y in 0..h {
        for x in 0..w {
            let px = cropped.get_pixel(x as u32, y as u32);
            let r = px[0] as f32 / 255.0;
            let g = px[1] as f32 / 255.0;
            let b = px[2] as f32 / 255.0;

            chw[[0, y, x]] = (r - MEAN[0]) / STD[0];
            chw[[1, y, x]] = (g - MEAN[1]) / STD[1];
            chw[[2, y, x]] = (b - MEAN[2]) / STD[2];
        }
    }

    // Add batch dim: (1, 3, 224, 224)
    let nchw = chw.insert_axis(Axis(0));
    Ok(nchw)
}

// ----------------- Postprocessing -----------------
// TODO: Does burn have a softmax?
fn softmax(v: &[f32]) -> Vec<f32> {
    if v.is_empty() { return vec![]; }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn topk_indices(values: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_unstable_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap());
    idx.truncate(k);
    idx
}

// ----------------- Inference: Tract -----------------
fn run_tract_inference(model_path: &str, input: &Array4<f32>) -> Result<Vec<f32>> {
    let model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 224, 224)))?
        .into_optimized()?
        .into_runnable()?;

    let data: Vec<f32> = input.iter().copied().collect();
    let input_tensor = TractTensor::from_shape(&[1, 3, 224, 224], &data)?;
    let result = model.run(tvec!(input_tensor.into()))?;

    let output: Vec<f32> = result[0]
        .to_array_view::<f32>()?
        .iter()
        .copied()
        .collect();
    Ok(output)
}

// ----------------- Inference: ONNX Runtime (ORT) -----------------
fn run_ort_inference(model_path: &str, input: &Array4<f32>) -> Result<Vec<f32>> {
    // ORT session must be mutable for .run()
    let mut session: Session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .commit_from_file(model_path)?;

    let data: Vec<f32> = input.iter().copied().collect();
    let shape = [1i64, 3, 224, 224];
    let input_tensor: OrtTensor<f32> = OrtTensor::from_array((shape, data))?;

    let outputs = session.run(ort::inputs![input_tensor])?;
    let out_view = outputs[0].try_extract_array::<f32>()?;
    Ok(out_view.iter().copied().collect())
}

// ----------------- Inference: Burn -----------------
fn run_burn_inference(input: &ndarray::Array4<f32>) -> anyhow::Result<Vec<f32>> {
    type B = burn::backend::ndarray::NdArray<f32>;
    let device = <B as burn::tensor::backend::Backend>::Device::default();

    let flat: Vec<f32> = input.iter().copied().collect();
    let x: burn::tensor::Tensor<B, 4> =
        burn::tensor::Tensor::<B, 1>::from_floats(flat.as_slice(), &device)
            .reshape([1, 3, 224, 224]);

    let weights_path = concat!(env!("OUT_DIR"), "/burn_model/resnet50.mpk"); // This should be created by build.rs
    let model: resnet_model::Model<B> = resnet_model::Model::from_file(weights_path, &device);

    let y = model.forward(x);
    let data = y.into_data();
    let output: Vec<f32> = data
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("Burn tensor to_vec failed: {e:?}"))?;
    Ok(output)
}

// ----------------- main -----------------
fn main() -> Result<()> {
    // --- paths ---
    let img_path = "src/cat.jpg"; // I downloaded this from google images
    let onnx_path = "src/resnet50.onnx"; // This should be created by save_resnet50_onnx.py

    // --- labels ---
    let labels_txt = fs::read_to_string("imagenet_classes.txt")
        .map_err(|e| anyhow!("Failed to read labels: {e}"))?;
    let labels: Vec<String> = labels_txt.lines().map(|s| s.trim().to_string()).collect();

    // --- preprocess ---
    let input = preprocess_imagenet(img_path)?;

    // --- Tract ---
    let t0 = Instant::now();
    let logits_tract = run_tract_inference(onnx_path, &input)?;
    println!("\n[Tract] inference: {:.3?}", t0.elapsed());
    let probs_tract = softmax(&logits_tract);
    let top5_t = topk_indices(&probs_tract, 5);
    println!("Tract top-5:");
    for i in top5_t {
        let label = labels.get(i).map(String::as_str).unwrap_or("<unknown>");
        println!("{:>4}: {:<60}  {:.4}", i, label, probs_tract[i]);
    }

    // --- ORT ---
    let t0 = Instant::now();
    let logits_ort = run_ort_inference(onnx_path, &input)?;
    println!("\n[ORT] inference: {:.3?}", t0.elapsed());
    let probs_ort = softmax(&logits_ort);
    let top5_o = topk_indices(&probs_ort, 5);
    println!("ORT top-5:");
    for i in top5_o {
        let label = labels.get(i).map(String::as_str).unwrap_or("<unknown>");
        println!("{:>4}: {:<60}  {:.4}", i, label, probs_ort[i]);
    }

    // --- Burn ---
    let t0 = Instant::now();
    let burn_output = run_burn_inference(&input)?;
    println!("\n[Burn] inference: {:.3?}", t0.elapsed());
    let probs_burn = softmax(&burn_output);
    let top5_b = topk_indices(&probs_burn, 5);
    println!("Burn top-5:");
    for i in top5_b {
        let label = labels.get(i).map(String::as_str).unwrap_or("<unknown>");
        println!("{:>4}: {:<60}  {:.4}", i, label, probs_burn[i]);
    }

    Ok(())
}
