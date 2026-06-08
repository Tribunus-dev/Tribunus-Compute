# Commercial Licensing — Tribunus Compute Kernel

The Tribunus Compute Kernel is dual-licensed under AGPL-3.0-only and a
separate commercial license. This document explains the commercial option.

## When You Need a Commercial License

AGPL-3.0-only requires that anyone who modifies the kernel and makes it
available to others over a network must make their modifications available
under the same license. A commercial license is necessary when:

- You want to embed the kernel in a **closed-source or proprietary product**
  without releasing your application source code under AGPL-3.0.
- You want to distribute the kernel under terms **other than AGPL-3.0**.
- You operate a **network service** that uses a modified version of the kernel
  and you do not want to make your modifications available under AGPL-3.0.
- You need **warranties, indemnification, or service-level agreements** that
  are not provided by the open-source license.
- You need **broader patent rights** beyond the statutory AGPL patent grant.

## When You Do Not Need a Commercial License

You do not need a commercial license if:

- You use the kernel internally on your own machines **without distributing**
  modified versions to others over a network.
- You distribute your modifications under AGPL-3.0-only, making your source
  code available to all recipients.
- You use the kernel through the unmodified Tribunus platform in the normal
  course — the kernel's AGPL obligations are satisfied by Tribunus itself.

## What a Commercial License Includes

A separately negotiated commercial license from the copyright holder may include:

| Right | AGPL-3.0-only | Commercial |
|---|---|---|
| Use the kernel in any application | Yes (subject to AGPL obligations) | Yes |
| Distribute modified versions without source disclosure | No | Yes |
| Operate a network service with modifications without Corresponding Source | No | Yes |
| Embed the kernel in a proprietary product | No | Yes |
| Warranties and indemnification | No (provided "AS IS") | Subject to negotiation |
| Separately negotiated patent rights beyond AGPL §11 grant | No | Subject to negotiation |
| Support and service-level agreements | No | Subject to negotiation |

## Patent Rights

Under AGPL-3.0-only, every compliant recipient receives a worldwide,
royalty-free patent license covering claims necessarily infringed by the
contributed implementation. You retain ownership of your patents. The
commercial license may grant additional patent rights — for example, covering
use of the kernel in combination with other technology, or patent claims
beyond those necessarily infringed by the contributed code.

## How to Obtain a Commercial License

Contact the copyright holder at **`license@tribunus.io`** with:

1. Your name and organization.
2. A description of the intended use (product, deployment model, scale).
3. Which rights you need (embedding, distribution, warranties, patents, support).
4. Your timeline.

## Frequently Asked Questions

**Q: Can I use the kernel for free in a commercial product?**

Yes, if you comply with AGPL-3.0-only — meaning you distribute your
application's source code under AGPL-3.0 and make it available to all
recipients. If you cannot or will not do that, you need a commercial license.

**Q: Does AGPL-3.0 transfer my patents to Tribunus?**

No. You retain full ownership of your patents. AGPL-3.0 grants compliant
recipients a license to the patent claims necessarily infringed by the
contributed code. It does not assign or transfer the patents themselves.

**Q: What if I only use the kernel through the Tribunus platform?**

Normal use through the Tribunus agent platform (the official distribution)
does not require a separate license. The platform's license covers your use.

**Q: Are the MLX fork and other third-party components covered?**

The MLX compatibility fork (`mlx-rs-fork/`) is used under MIT OR Apache-2.0.
Other Rust dependencies carry permissive licenses (MIT, Apache-2.0, BSD, ISC).
The AGPL-3.0-only applies to the Tribunus-authored portions of the kernel.
The combined work — the kernel binary — is distributed under AGPL-3.0.

**Q: What happens if I contribute code?**

To preserve dual-licensing, substantial external contributions require a
contributor license agreement. See [CONTRIBUTING.md §10](CONTRIBUTING.md#10-license-and-contributor-agreement).
