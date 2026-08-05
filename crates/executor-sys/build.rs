use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=CUSCO_NATIVE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUSCO_LLAMA_LIB_DIR");
    println!("cargo:rerun-if-changed=../../native/include/cusco_executor.h");
    println!("cargo:rerun-if-changed=../../native/shim/cusco_executor.cpp");
    if let Ok(dir) = env::var("CUSCO_NATIVE_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rerun-if-changed={dir}/libcusco_executor.a");
        println!("cargo:rustc-link-lib=static=cusco_executor");
        if let Ok(llama) = env::var("CUSCO_LLAMA_LIB_DIR") {
            println!("cargo:rustc-link-search=native={llama}");
            println!("cargo:rustc-link-lib=dylib=llama");
            println!("cargo:rustc-link-lib=dylib=ggml");
            println!("cargo:rustc-link-lib=dylib=ggml-base");
            println!("cargo:rustc-link-lib=dylib=ggml-cpu");
        }
        println!("cargo:rustc-link-lib=dylib=stdc++");
    } else {
        let stub = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("stub.c");
        std::fs::write(
            &stub,
            "void cusco_executor_link_stub(void) {}
",
        )
        .unwrap();
        let cc_status = std::process::Command::new("cc")
            .args(["-c", stub.to_str().unwrap(), "-o"])
            .arg(stub.with_extension("o"))
            .status()
            .expect("failed to start cc for executor stub");
        assert!(
            cc_status.success(),
            "cc failed to compile executor stub: {cc_status}"
        );
        let ar_status = std::process::Command::new("ar")
            .args([
                "crs",
                stub.with_file_name("libcusco_executor.a").to_str().unwrap(),
                stub.with_extension("o").to_str().unwrap(),
            ])
            .status()
            .expect("failed to start ar for executor stub");
        assert!(
            ar_status.success(),
            "ar failed to archive executor stub: {ar_status}"
        );
        println!(
            "cargo:rustc-link-search=native={}",
            stub.parent().unwrap().display()
        );
        println!("cargo:rustc-link-lib=static=cusco_executor");
    }
}
