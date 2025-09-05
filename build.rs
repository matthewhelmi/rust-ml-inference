use burn_import::onnx::ModelGen;

fn main() {
    println!("cargo:rerun-if-changed=src/resnet50.onnx"); // rebuild if model changes

    ModelGen::new()
        .input("src/resnet50.onnx")
        .out_dir("burn_model")
        .embed_states(false)
        .run_from_script();

    println!("ModelGen finished, check OUT_DIR={}", std::env::var("OUT_DIR").unwrap());
}
