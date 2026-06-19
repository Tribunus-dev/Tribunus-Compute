//! Paged KV cache arena: physical blocks, COW refcounting, prefix caching,
//! and backend residency mapping.

pub mod refcount;
pub mod backend;
pub mod block;
pub mod prefix;
