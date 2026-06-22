// build.rs — compiles bridge.cpp against Metal2 host API headers
// On macOS, produces a stub binary. On Linux with tt-metal installed,
// links against the full Metalium runtime.
fn main() {
    // Default: stub only — produces a linkable .o with all symbols defined
    // but returning errors for actual device operations.
    //
    // When TT_METAL_HOME is set and target_os = "linux", compiles against
    // the real Metal2 headers and links tt-metal+ttnn libraries.
    let tt_metal_home = std::env::var("TT_METAL_HOME").ok();
    let is_linux = std::env::var("CARGO_CFG_TARGET_OS").map(|v| v == "linux").unwrap_or(false);

    let mut build = cc::Build::new();
    build.cpp(true);
    build.std("c++20");
    build.file("cpp/bridge.cpp");
    build.include("cpp/");

    if let Some(home) = &tt_metal_home {
        let api = format!("{}/tt_metal/api", home);
        let stl = format!("{}/tt_stl", home);
        let umd_api = format!("{}/tt_metal/third_party/umd/device/api", home);
        let umd_src = format!("{}/tt_metal/third_party/umd/src", home);
        build.include(&api);
        build.include(&stl);
        build.include(&umd_api);
        build.include(&umd_src);
        build.include(&home);
        // Homebrew deps
        for dir in ["/opt/homebrew/include", "/usr/include", "/usr/local/include"] {
            if std::path::Path::new(dir).exists() {
                build.include(dir);
            }
        }
        // real build — need fmt, nlohmann-json, spdlog, magic_enum
        build.define("TENSIX_REAL_MODE", None);
    } else {
        // stub build — define all functions as no-ops
        build.define("TENSIX_STUB_MODE", None);
    }

    build.compile("tensix_ffi");
}
