use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=CUSCO_NATIVE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUSCO_LLAMA_LIB_DIR");
    if let Ok(dir) = env::var("CUSCO_NATIVE_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
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
        std::process::Command::new("cc")
            .args(["-c", stub.to_str().unwrap(), "-o"])
            .arg(stub.with_extension("o"))
            .status()
            .unwrap();
        std::process::Command::new("ar")
            .args([
                "crs",
                stub.with_file_name("libcusco_executor.a").to_str().unwrap(),
                stub.with_extension("o").to_str().unwrap(),
            ])
            .status()
            .unwrap();
        println!(
            "cargo:rustc-link-search=native={}",
            stub.parent().unwrap().display()
        );
        println!("cargo:rustc-link-lib=static=cusco_executor");
    }
}
