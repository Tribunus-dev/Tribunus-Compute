## Description

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation
- [ ] Qualification / testing
- [ ] Build / CI
- [ ] MLX fork update

## Qualification Evidence

For changes to runtime behavior, memory, or performance, paste qualification output or attach receipts:

```
```

## Checklist

- [ ] Code compiles: `cargo check --workspace`
- [ ] Formatting: `cargo fmt --all --check`
- [ ] Clippy clean: `cargo clippy --all-targets -- -D warnings`
- [ ] Tests pass: `cargo test -p tribunus-compute-native`
- [ ] New tests added for new behavior
- [ ] Documentation updated if public API changes
- [ ] Frozen ABIs unchanged (or new version created)
