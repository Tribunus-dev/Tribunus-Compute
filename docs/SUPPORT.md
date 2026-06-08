# Support

## Documentation

- [README.md](README.md) — project overview, capabilities, maturity
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — runtime design, process boundaries, thread model
- [STATUS.md](docs/STATUS.md) — capability qualification matrix
- [CONTRIBUTING.md](docs/CONTRIBUTING.md) — development workflow, testing standards
- [SECURITY.md](docs/SECURITY.md) — vulnerability reporting, trust model
- [compatibility.md](docs/compatibility.md) — MLX dependency tuple, fork details
- [RELEASING.md](docs/RELEASING.md) — release process, versioning

## Getting Help

- **Bug reports**: [Open an issue](https://github.com/Tribunus-dev/Tribunus-Compute/issues/new?template=bug_report.md)
- **Security issues**: Email `security@tribunus.io` (do not open a public issue)
- **Questions**: [GitHub Discussions](https://github.com/Tribunus-dev/Tribunus-Compute/discussions) (if enabled) or open a [feature request](https://github.com/Tribunus-dev/Tribunus-Compute/issues/new?template=feature_request.md)

## macOS-Specific Issues

This kernel targets Apple Silicon. Common prerequisites:

- macOS 14.0+ with Xcode 16+
- MLX installed via `brew install mlx`
- Core ML, IOSurface, and Metal frameworks linked by `build.rs`

If you encounter build failures, verify `xcode-select -p` points to the correct Xcode installation and that `brew list mlx` shows the expected headers and libraries.

## Supported Versions

| Version | Status |
|---------|--------|
| 0.1.x | Development (pre-alpha) |

No LTS releases yet. Security fixes target the current minor version.
