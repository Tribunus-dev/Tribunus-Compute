fn forward(name: &str) {
    let value = std::env::var(name).unwrap_or_else(|_| format!("{name}_MISSING"));
    println!("cargo:rustc-env=TRIBUNUS_{name}={value}");
}

fn main() {
    // Forward git SHA and branch for artifact provenance.
    if std::env::var("VERGEN_GIT_SHA").is_err() {
        if let Ok(out) = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
        {
            let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !sha.is_empty() {
                println!("cargo:rustc-env=VERGEN_GIT_SHA={}", sha);
            }
        }
    }
    if std::env::var("VERGEN_GIT_BRANCH").is_err() {
        if let Ok(out) = std::process::Command::new("git")
            .args(["rev-parse", "--abbrev-ref", "HEAD"])
            .output()
        {
            let branch = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !branch.is_empty() {
                println!("cargo:rustc-env=VERGEN_GIT_BRANCH={}", branch);
            }
        }
    }

    forward("PROFILE");
    forward("OPT_LEVEL");
    forward("TARGET");
    forward("DEBUG");

    // Record RUSTFLAGS
    if let Ok(flags) = std::env::var("RUSTFLAGS") {
        println!("cargo:rustc-env=TRIBUNUS_RUSTFLAGS={}", flags);
    }

    // Record linker if set
    if let Ok(ld) = std::env::var("RUSTC_LINKER") {
        println!("cargo:rustc-env=TRIBUNUS_LINKER={}", ld);
    }

    // Record host info
    println!("cargo:rustc-env=TRIBUNUS_HOST_OS={}", std::env::consts::OS);
    println!(
        "cargo:rustc-env=TRIBUNUS_HOST_ARCH={}",
        std::env::consts::ARCH
    );

    // MLX identity (fixed for this gate - pointing to the new published fork)
    println!("cargo:rustc-env=TRIBUNUS_MLX_IDENTITY=Tribunus-dev/mlx-rs-fork@main");

    // Guard: on non-macOS targets, a CPU backend feature must be explicit.
    #[cfg(all(not(target_os = "macos"), not(feature = "candle-cpu")))]
    compile_error!(
        "On non-macOS targets, feature 'candle-cpu' must be enabled. See Cargo.toml [features]."
    );

    // Compile the ObjC++ Core ML / IOSurface bridge.
    #[cfg(all(target_os = "macos", feature = "mlx-backend"))]
    {
        cc::Build::new()
            .file("src/bridge/coreml_arena.mm")
            .flag("-fobjc-arc")
            .flag("-std=c++17")
            .compile("coreml_arena");
        cc::Build::new()
            .file("src/bridge/coreml_exec.mm")
            .flag("-fobjc-arc")
            .flag("-fblocks")
            .flag("-std=c++17")
            .compile("coreml_exec");
        cc::Build::new()
            .file("src/bridge/coreml_state.mm")
            .flag("-fobjc-arc")
            .flag("-fblocks")
            .flag("-std=c++17")
            .compile("coreml_state");
        cc::Build::new()
            .file("src/bridge/ane_private.mm")
            .flag("-fobjc-arc")
            .flag("-fblocks")
            .flag("-std=c++17")
            .compile("ane_private");
        cc::Build::new()
            .file("src/bridge/ane_weight_dict.mm")
            .flag("-fobjc-arc")
            .flag("-fblocks")
            .flag("-std=c++17")
            .compile("ane_weight_dict");
        // ── Orion ANE runtime ────────────────────────────────────────────
        // C files: compiler passes, graph IR, builders
        let compiler_c_src: &[&str] = &[
            "../../orion-runtime/compiler/graph.c",
            "../../orion-runtime/compiler/builder.c",
            "../../orion-runtime/compiler/topo.c",
            "../../orion-runtime/compiler/patterns.c",
            "../../orion-runtime/compiler/validate.c",
            "../../orion-runtime/compiler/pass_dce.c",
            "../../orion-runtime/compiler/pass_identity.c",
            "../../orion-runtime/compiler/pass_conv_bias.c",
            "../../orion-runtime/compiler/pass_cast.c",
            "../../orion-runtime/compiler/pass_sram.c",
            "../../orion-runtime/compiler/pass_uniform_outputs.c",
            "../../orion-runtime/compiler/pass_ane_validate.c",
            "../../orion-runtime/compiler/pipeline.c",
            "../../orion-runtime/compiler/frontends/gpt2_prefill.c",
            "../../orion-runtime/compiler/frontends/gpt2_decode.c",
            "../../orion-runtime/compiler/frontends/gpt2_final.c",
            "../../orion-runtime/compiler/frontends/classifier_softmax.c",
            "../../orion-runtime/compiler/frontends/lora.c",
        ];
        // ObjC files: core runtime, MIL building, ANE execution
        let objc_src: &[&str] = &[
            "../../orion-runtime/core/ane_runtime.m",
            "../../orion-runtime/core/ane_program_cache.m",
            "../../orion-runtime/core/mil_builder.m",
            "../../orion-runtime/core/iosurface_tensor.m",
            "../../orion-runtime/core/profiler.m",
            "../../orion-runtime/core/bucket.m",
            "../../orion-runtime/core/checkpoint.m",
            "../../orion-runtime/core/model_registry.m",
            "../../orion-runtime/core/kernel.m",
            "../../orion-runtime/core/runtime.m",
            "../../orion-runtime/core/lora_adapter.m",
            "../../orion-runtime/kernels/inference/prefill_ane.m",
            "../../orion-runtime/kernels/inference/decode_ane.m",
            "../../orion-runtime/kernels/inference/decode_cpu.m",
            "../../orion-runtime/kernels/inference/kv_cache.m",
            "../../orion-runtime/compiler/codegen.m",
            "../../orion-runtime/compiler/kernel_adapter.m",
            "../../orion-runtime/compiler/mil_diff.m",
        ];
        let mut orion_build = cc::Build::new();
        orion_build.flag("-fobjc-arc")
            .flag("-O2")
            .flag("-DACCELERATE_NEW_LAPACK")
            .flag("-isysroot")
            .flag("/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
            .include("../../orion-runtime")
            .include("../../orion-runtime/core")
            .include("../../orion-runtime/compiler");
        for f in compiler_c_src {
            orion_build.file(f);
        }
        for f in objc_src {
            orion_build.file(f);
        }
        eprintln!(
            "build.rs: about to compile orion_runtime ({} files)",
            compiler_c_src.len() + objc_src.len()
        );
        orion_build.compile("orion_runtime"); // creates liborion_runtime.a
                                              // The cc crate should emit cargo:rustc-link-lib=static=orion_runtime
                                              // AND cargo:rustc-link-search=native=<out_dir> automatically.
                                              // Framework dependencies for the ObjC runtime:
                                              // Force-load the orion_runtime static library — Apple ld doesn't search
                                              // archives automatically for symbols defined in Rust FFI extern blocks.
        let out_dir = std::env::var("OUT_DIR").unwrap();
        println!("cargo:rustc-link-arg=-Wl,-force_load,{out_dir}/liborion_runtime.a");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=IOSurface");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=CoreVideo");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Cocoa");
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=IOSurface");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=CoreVideo");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Cocoa");
        // Use rustc-link-arg for framework flags — more reliable for test targets
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg=Foundation");
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg=CoreFoundation");
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg=IOSurface");
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg=Accelerate");
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg=CoreML");
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg=CoreVideo");
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg=Metal");
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg=Cocoa");
    }
    eprintln!("build.rs: END of main() — all link directives emitted");
}
