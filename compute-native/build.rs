fn main() {
    napi_build::setup();

    // The addon is normally loaded into a Node.js process that already exports
    // the N-API symbols. `cargo test` builds a standalone harness instead, so
    // on macOS we keep those symbols as runtime lookups rather than forcing a
    // machine-specific libnode dependency into the build artifacts.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-undefined,dynamic_lookup");
    }
}
