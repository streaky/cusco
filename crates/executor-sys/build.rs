use std::env;
use std::path::PathBuf;

fn executor_abi_version() -> String {
    let header = std::fs::read_to_string("../../native/include/cusco_executor.h")
        .expect("failed to read executor ABI header");
    header
        .lines()
        .find_map(|line| line.strip_prefix("#define CUSCO_EXECUTOR_ABI_VERSION "))
        .and_then(|value| value.strip_suffix('u'))
        .expect("executor ABI header has no CUSCO_EXECUTOR_ABI_VERSION")
        .to_owned()
}
fn main() {
    println!("cargo:rerun-if-env-changed=CUSCO_NATIVE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUSCO_LLAMA_LIB_DIR");
    println!("cargo:rerun-if-changed=../../native/include/cusco_executor.h");
    println!("cargo:rerun-if-changed=../../native/shim/cusco_executor.cpp");
    println!(
        "cargo:rustc-env=CUSCO_EXECUTOR_ABI_VERSION={}",
        executor_abi_version()
    );
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
