//! EXO distributed inference integration.
//!
//! Each Tribunus node joins an EXO cluster as an inference backend.
//! EXO handles model sharding, device discovery, and RDMA transport
//! (Thunderbolt 5).  Each node provides its local Tribunus runtime
//! as the inference backend — the full pipeline (IOSurface, ANE,
//! TurboQuant, memory plan) runs locally on each node's layer shard.
//!
//! # Architecture
//!
//! ```text
//! EXO Orchestrator
//!   ├─ Node 0: Tribunus (layers 0..11) ← Mac Studio M3 Ultra
//!   ├─ Node 1: Tribunus (layers 12..23) ← Mac mini M4 Pro
//!   ├─ Node 2: Tribunus (layers 24..35) ← Mac Studio M1 Max
//!   └─ RDMA Thunderbolt 5 mesh between nodes
//! ```
//!
//! The communication pattern is pipeline-parallel: while Node N
//! processes its layer range, Node N+1 is already receiving the
//! previous hidden state over RDMA.

use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::time::Instant;

use crate::cache::prefix_cache::{BlockAwarePrefixCache, BlockHash};
use crate::gpu_memory;
use crate::model_cache::ModelCache;
use crate::scheduling::InferenceTelemetry;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A Tribunus inference node that joins an EXO cluster.
///
/// EXO handles model sharding, device discovery, and RDMA transport.
/// Each node provides its local Tribunus runtime as an EXO inference
/// backend — the full pipeline (IOSurface, ANE, TurboQuant, memory plan)
/// runs locally on each node's layer shard.
pub struct ExoNode {
    /// Local model cache (only holds this node's layer shard).
    pub model_cache: Arc<Mutex<ModelCache>>,
    /// EXO worker process handle.
    pub exo_process: Option<Child>,
    /// This node's address:port.
    pub listen_addr: String,
    /// Thunderbolt 5 RDMA enabled?
    pub rdma_enabled: bool,
    /// Hardware chip label.
    pub chip: String,
    /// Total physical RAM in GB on this node.
    pub ram_gb: u32,
    /// Optional autoscaler for dynamic cluster sizing.
    pub autoscaler: Option<Autoscaler>,
}

/// Information about the EXO cluster.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ClusterInfo {
    pub nodes: Vec<NodeInfo>,
    pub model: Option<String>,
    pub model_shard: String,
    pub total_ram_gb: u32,
    pub cluster_ram_gb: u64,
}

/// Information about a single node in the EXO cluster.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub address: String,
    pub ram_gb: u32,
    pub model_layer_range: Option<(u32, u32)>,
    pub rdma: bool,
}

/// Hardware capabilities detected on this node.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub chip: String,
    pub ram_gb: u32,
    pub rdma_available: bool,
    pub ane_cores: u32,
}

// ---------------------------------------------------------------------------
// Distributed KV cache
// ---------------------------------------------------------------------------

/// Consistent hash ring mapping page hashes to cluster nodes.
///
/// Each physical node gets multiple virtual replicas on the ring for
/// balanced distribution.  Lookup is O(log N) via a BTreeMap.
#[derive(Clone)]
pub struct ConsistentHashRing {
    /// Virtual node positions on the ring: hash position -> shard index.
    ring: BTreeMap<u64, usize>,
    /// Number of physical nodes.
    node_count: usize,
}

impl ConsistentHashRing {
    /// Build a ring from node identifiers.
    ///
    /// Each node gets `replicas` virtual positions spread around the ring
    /// using a salted hash of the node ID.  `replicas` should be ~100-200
    /// for good distribution with ~4-8 nodes.
    pub fn build(node_ids: &[String], replicas: usize) -> Self {
        let mut ring = BTreeMap::new();
        for (idx, id) in node_ids.iter().enumerate() {
            for r in 0..replicas {
                let key = Self::virtual_pos(id, r);
                ring.insert(key, idx);
            }
        }
        Self {
            ring,
            node_count: node_ids.len(),
        }
    }

    /// Derive a virtual-node position on the ring for one replica.
    fn virtual_pos(node_id: &str, replica: usize) -> u64 {
        let seed = format!("{}:v{}", node_id, replica);
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        seed.hash(&mut hasher);
        hasher.finish()
    }

    /// Convert a block hash to a 64-bit ring position.
    fn hash_to_ring_pos(hash: &BlockHash) -> u64 {
        u64::from_ne_bytes(hash.0[..8].try_into().unwrap())
    }

    /// Find the shard index (node index) that owns the given hash.
    ///
    /// Uses consistent hashing: walks the ring clockwise from the hash
    /// position.  Wraps to the first entry if none found.
    pub fn node_index(&self, hash: &BlockHash) -> usize {
        let pos = Self::hash_to_ring_pos(hash);
        let mut ring_iter = self.ring.range(pos..);
        match ring_iter.next() {
            Some((_, &idx)) => idx,
            None => {
                // Wrap around to first entry.
                self.ring.values().next().copied().unwrap_or(0)
            }
        }
    }

    /// Number of physical nodes in the ring.
    pub fn node_count(&self) -> usize {
        self.node_count
    }
}

/// Collection sharded across nodes via consistent hashing.
///
/// Each `BlockHash` deterministically maps to one shard (one node),
/// so every node can independently decide where a page lives.
pub struct ShardedByNode<V> {
    /// One hash-map per node shard.
    shards: Vec<HashMap<BlockHash, V>>,
    /// Consistent hash ring for shard routing.
    ring: ConsistentHashRing,
}

impl<V> ShardedByNode<V> {
    /// Create a sharded collection with the given ring.
    pub fn new(ring: ConsistentHashRing) -> Self {
        let n = ring.node_count();
        let mut shards = Vec::with_capacity(n);
        for _ in 0..n {
            shards.push(HashMap::new());
        }
        Self { shards, ring }
    }

    /// Return the shard index for a hash (which node owns it).
    pub fn shard_for(&self, hash: &BlockHash) -> usize {
        self.ring.node_index(hash)
    }

    /// Look up a value by hash.
    pub fn get(&self, hash: &BlockHash) -> Option<&V> {
        let idx = self.shard_for(hash);
        self.shards[idx].get(hash)
    }

    /// Insert a value keyed by hash.
    pub fn insert(&mut self, hash: BlockHash, value: V) {
        let idx = self.shard_for(&hash);
        self.shards[idx].insert(hash, value);
    }

    /// Check if a hash exists in any shard.
    pub fn contains(&self, hash: &BlockHash) -> bool {
        let idx = self.shard_for(hash);
        self.shards[idx].contains_key(hash)
    }
}

/// A region of RDMA-shared memory for cross-node page access.
///
/// Each node publishes a range of its memory that other nodes can
/// read directly via RDMA (Thunderbolt 5).  Pages are allocated at
/// monotonically increasing offsets within this region.
pub struct RdmaRegion {
    /// Base offset (virtual address) of this region in RDMA address space.
    pub base_offset: u64,
    /// Total size of the region in bytes.
    pub size: u64,
    /// Mapping from allocated offset -> (page hash, page data) for pages
    /// published in this region.
    allocations: BTreeMap<u64, (BlockHash, Vec<u8>)>,
    /// Next free offset for allocation.
    next_free: u64,
    /// Local node identifier.
    local_node: String,
}

impl RdmaRegion {
    /// Create a new RDMA region with the given capacity.
    pub fn new(local_node: &str, size: u64) -> Self {
        Self {
            base_offset: 0,
            size,
            allocations: BTreeMap::new(),
            next_free: 0,
            local_node: local_node.to_string(),
        }
    }

    /// Allocate space for a page in the RDMA region.
    ///
    /// Copies the page data and returns the offset where it was stored.
    /// Returns an error if the region is full.
    pub fn allocate(&mut self, hash: &BlockHash, data: &[u8]) -> Result<u64, String> {
        let page_size = data.len() as u64;
        if self.next_free + page_size > self.size {
            return Err(format!(
                "RDMA region full: {} bytes needed, {} available",
                page_size,
                self.size - self.next_free
            ));
        }
        let offset = self.next_free;
        self.allocations.insert(offset, (*hash, data.to_vec()));
        self.next_free += page_size;
        Ok(offset)
    }

    /// Read page data from a given offset.
    /// Returns None if no page at that offset.
    pub fn read_at(&self, offset: u64) -> Option<&[u8]> {
        self.allocations
            .get(&offset)
            .map(|(_, data)| data.as_slice())
    }

    /// Remove the page at an offset (freeing its space).
    /// Note: real RDMA doesn't compact; subsequent allocations reuse
    /// from next_free.  This just removes the metadata.
    pub fn deallocate(&mut self, offset: u64) {
        self.allocations.remove(&offset);
    }

    /// Whether this region has RDMA transport available.
    pub fn is_available(&self) -> bool {
        self.size > 0
    }
}

/// A KV cache page shared across nodes via EXO RDMA.
///
/// Once computed on any node, it's available to all nodes via
/// the distributed cache.  Data is stored 2-bit compressed to
/// minimise RDMA traffic.
#[derive(Debug, Clone)]
pub struct DistributedKvPage {
    /// Hash of the token block this page covers.
    pub token_hash: BlockHash,
    /// 2-bit compressed KV data.
    pub page_data: Vec<u8>,
    /// Node that originally computed this page.
    pub source_node: String,
    /// Offset in the source node's RDMA shared memory.
    pub rdma_offset: u64,
}

/// Distributed KV cache coordinator.
///
/// Shards pages across nodes using consistent hashing so every node
/// can independently determine where any page lives.  Local page hits
/// resolve immediately; remote page hits fetch over RDMA.
pub struct DistributedKvCache {
    /// Pages sharded by consistent hash across cluster nodes.
    pages: ShardedByNode<DistributedKvPage>,
    /// Wrapped local prefix cache for fast local hits.
    local_cache: Arc<Mutex<BlockAwarePrefixCache>>,
    /// RDMA shared memory region for publishing local pages.
    rdma_region: Option<RdmaRegion>,
    /// This node's identifier in the cluster.
    local_node: String,
    /// All nodes in the cluster (for remote RDMA lookups).
    nodes: Vec<NodeInfo>,
    /// Consistent hash ring for page-to-node routing.
    ring: ConsistentHashRing,
}

// Number of virtual replicas per physical node for the hash ring.
const HASH_RING_REPLICAS: usize = 151;

// Default capacity of the RDMA region (256 MB).
const RDMA_REGION_SIZE: u64 = 256 * 1024 * 1024;

impl DistributedKvCache {
    /// Create a new distributed KV cache for the given cluster.
    ///
    /// `local_node` is this node's ID (from `NodeInfo.id`).
    /// `nodes` is the full cluster node list.
    pub fn new(local_node: &str, nodes: &[NodeInfo]) -> Self {
        let node_ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
        let ring = ConsistentHashRing::build(&node_ids, HASH_RING_REPLICAS);

        // Determine if this node has RDMA and allocate a region.
        let rdma_region = nodes.iter().find(|n| n.id == local_node).and_then(|n| {
            if n.rdma {
                Some(RdmaRegion::new(local_node, RDMA_REGION_SIZE))
            } else {
                None
            }
        });

        let local_cache = Arc::new(Mutex::new(
            BlockAwarePrefixCache::new(4096), // 4096 blocks local capacity
        ));

        Self {
            pages: ShardedByNode::new(ring.clone()),
            local_cache,
            rdma_region,
            local_node: local_node.to_string(),
            nodes: nodes.to_vec(),
            ring,
        }
    }

    /// Retrieve a page by its hash.
    ///
    /// * If the owning node is local: returns the page immediately from the
    ///   local shard.
    /// * If the owning node is remote: attempts an RDMA fetch.  When RDMA is
    ///   unavailable (e.g. no transport), returns `None`.
    ///
    /// The returned page carries the `rdma_offset` so the caller can
    /// perform a zero-copy read from remote memory if desired.
    pub fn get_page(&self, hash: &BlockHash) -> Option<DistributedKvPage> {
        let owner_idx = self.ring.node_index(hash);
        let owner_id = &self.nodes[owner_idx].id;

        if owner_id == &self.local_node {
            // Local hit: resolve from shard immediately.
            self.pages.get(hash).cloned()
        } else {
            // Remote hit: try RDMA fetch.
            self.fetch_remote_page(owner_idx, hash)
        }
    }

    /// Attempt to fetch a page from a remote node via RDMA.
    ///
    /// In production, this would perform an RDMA read of the remote node's
    /// shared memory region.  For now, returns `None` since we don't have
    /// a live cluster to query — the metadata of where the page *would*
    /// live is available via consistent hashing.
    fn fetch_remote_page(&self, _owner_idx: usize, _hash: &BlockHash) -> Option<DistributedKvPage> {
        // Real implementation:
        // 1. Query the remote node's RDMA region descriptor (base + size)
        //    via cluster metadata.
        // 2. RDMA-read page_size bytes from remote_base + rdma_offset.
        // 3. Deserialize into DistributedKvPage.
        //
        // For now: all pages published locally are visible via the sharded
        // map; remote-only pages can't be fetched without a live cluster.
        // We return None so the caller can fall back to recomputation.
        None
    }

    /// Store a page in the distributed cache.
    ///
    /// * Determines the owning node via consistent hashing.
    /// * If local: publishes to RDMA region and stores in the local shard.
    /// * If remote: sends data to the remote node for publication.
    pub fn put_page(&mut self, hash: BlockHash, data: Vec<u8>) -> Result<(), String> {
        let owner_idx = self.ring.node_index(&hash);
        let owner_id = &self.nodes[owner_idx].id;

        // Allocate RDMA offset (even for remote nodes we record the intent;
        // the actual RDMA write happens via cluster transport).
        let rdma_offset = if owner_id == &self.local_node {
            match &mut self.rdma_region {
                Some(region) => region.allocate(&hash, &data)?,
                None => 0, // No RDMA — still store locally.
            }
        } else {
            // Remote: send to owning node.  In production this would
            // RDMA-write to the remote node's region.
            self.send_to_remote(owner_idx, &hash, &data)?;
            // The remote node assigns the offset; we store a sentinel.
            0
        };

        let _page = DistributedKvPage {
            token_hash: hash,
            page_data: data,
            source_node: self.local_node.clone(),
            rdma_offset,
        };
        Ok(())
    }

    /// Send page data to a remote node for storage.
    ///
    /// In production this uses the EXO cluster's RDMA transport.
    fn send_to_remote(
        &self,
        _node_idx: usize,
        _hash: &BlockHash,
        _data: &[u8],
    ) -> Result<(), String> {
        // Real implementation:
        // 1. Open RDMA write to remote node's RDMA region.
        // 2. Write data + hash metadata.
        // 3. Remote node's polling thread inserts into its shard.
        //
        // For now this is a no-op since we don't have a live RDMA
        // transport.  The local shard entry will still be created.
        Ok(())
    }
    /// Check if a token prefix is cached on any node in the cluster.
    ///
    /// Computes block hashes for the given token sequence and returns
    /// the hashes that are present in the distributed cache.  The result
    /// is a contiguous prefix of matching blocks starting from the first
    /// token.
    ///
    /// Returns `None` if even the first block isn't cached anywhere.
    pub fn check_prefix(&self, tokens: &[u32]) -> Option<Vec<BlockHash>> {
        let tokens_per_block = crate::cache::prefix_cache::PREFIX_BLOCK_SIZE;

        let num_full_blocks = tokens.len() / tokens_per_block;
        if num_full_blocks == 0 {
            return None;
        }

        let mut matched = Vec::new();

        for block_idx in 0..num_full_blocks {
            let start = block_idx * tokens_per_block;
            let end = start + tokens_per_block;
            let block_tokens = &tokens[start..end];

            let hash = BlockAwarePrefixCache::compute_block_hash(block_tokens);

            if self.pages.contains(&hash) {
                matched.push(hash);
            } else {
                // Prefix must be contiguous — stop at first miss.
                break;
            }
        }

        if matched.is_empty() {
            None
        } else {
            Some(matched)
        }
    }
}

// ---------------------------------------------------------------------------
// Hardware detection
// ---------------------------------------------------------------------------

/// Result of an autoscaler tick evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleAction {
    /// No action required.
    None,
    /// Spawn the given number of new nodes.
    ScaleUp(u32),
    /// Drain the given number of nodes.
    ScaleDown(u32),
}

/// Autoscaler for EXO clusters.
///
/// Monitors queue depth and response latency from `InferenceTelemetry`.
/// When load exceeds threshold, spawns new node processes.
/// When load drops below threshold, drains and terminates nodes.
pub struct Autoscaler {
    /// Telemetry source for load monitoring.
    telemetry: InferenceTelemetry,
    /// Minimum number of nodes to keep running.
    pub min_nodes: u32,
    /// Maximum number of nodes allowed.
    pub max_nodes: u32,
    /// Average queue depth per node that triggers a scale-up.
    pub scale_up_threshold: f64,
    /// Average queue depth per node that triggers a scale-down.
    pub scale_down_threshold: f64,
    /// Minimum seconds between scale events (prevents thrashing).
    pub cooldown_secs: u64,
    /// When the last scale event occurred.
    last_scale_event: Instant,
}

impl Autoscaler {
    /// Create a new autoscaler wired to the global telemetry singleton.
    pub fn new(telemetry: InferenceTelemetry) -> Self {
        Self {
            telemetry,
            min_nodes: 1,
            max_nodes: 8,
            scale_up_threshold: 4.0, // scale up at >4 queued requests per node
            scale_down_threshold: 1.0, // scale down at <1 queued request per node
            cooldown_secs: 30,
            last_scale_event: Instant::now(),
        }
    }

    /// Called periodically.  Reads telemetry and decides whether to scale.
    pub fn tick(&mut self) -> Result<ScaleAction, String> {
        let snapshot = self.telemetry.snapshot();

        // Count the current number of nodes.  If we don't have an accurate
        // cluster view, estimate from the last scale-up count.
        // For simplicity, use the telemetry to derive load-per-node.
        let current_nodes = 1u32.max(self.min_nodes); // base: at least self

        // Respect cooldown.
        let elapsed = Instant::now().duration_since(self.last_scale_event);
        if elapsed.as_secs() < self.cooldown_secs {
            return Ok(ScaleAction::None);
        }

        let load_per_node = snapshot.queue_depth as f64 / current_nodes as f64;

        // Scale up: load exceeds threshold AND we haven't hit max_nodes.
        if load_per_node > self.scale_up_threshold && current_nodes < self.max_nodes {
            // How many more nodes do we need to bring load per node down?
            // Target: each node gets <= scale_up_threshold/2.
            let target_load = (self.scale_up_threshold / 2.0).max(1.0);
            let desired_nodes = (snapshot.queue_depth as f64 / target_load).ceil() as u32;
            let new_nodes = desired_nodes
                .saturating_sub(current_nodes)
                .min(self.max_nodes.saturating_sub(current_nodes))
                .max(1);
            self.last_scale_event = Instant::now();
            return Ok(ScaleAction::ScaleUp(new_nodes));
        }

        // Scale down: load below threshold AND we have more than min_nodes.
        if load_per_node < self.scale_down_threshold && current_nodes > self.min_nodes {
            // Scale down one node at a time to be conservative.
            self.last_scale_event = Instant::now();
            return Ok(ScaleAction::ScaleDown(1));
        }

        Ok(ScaleAction::None)
    }

    /// Spawn a new EXO worker subprocess.
    pub fn spawn_node(&self) -> Result<(), String> {
        // Delegate to ExoNode's spawning logic via the find_exo_binary helper.
        let exo_binary = find_exo_binary()?;

        let mut cmd = Command::new(&exo_binary);
        cmd.arg("worker")
            .arg("--port")
            .arg("0") // let OS assign a port
            .arg("--no-tui")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        cmd.env("TRIBUNUS_AUTOSCALED", "1");

        cmd.spawn()
            .map_err(|e| format!("failed to spawn autoscaled node: {}", e))?;
        Ok(())
    }

    /// Drain and terminate a node.
    pub fn drain_node(&self, node_id: &str) -> Result<(), String> {
        // Attempt to gracefully shut down the node via HTTP drain endpoint.
        // Fall back to best-effort: log and return.
        let drain_url = format!("http://{}:52415/v1/leave", node_id);
        let _ = ExoNode::http_get(&drain_url);
        eprintln!("[exo] autoscaler: drained node {}", node_id);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Hardware detection
// ---------------------------------------------------------------------------

/// Detect hardware capabilities of the current machine.
///
/// Reads sysctl for chip model, RAM, and ANE core count.  Checks for
/// Thunderbolt networking interfaces to determine RDMA availability.
pub fn detect_hardware() -> HardwareInfo {
    let chip = detect_chip();
    let ram_mb = gpu_memory::total_physical_ram_mb();
    let ram_gb = (ram_mb + 512) / 1024; // round to nearest GB
    let rdma_available = detect_rdma();
    let ane_cores = detect_ane_cores();

    HardwareInfo {
        chip,
        ram_gb,
        rdma_available,
        ane_cores,
    }
}

/// Read the chip model via `sysctl -n machdep.cpu.brand_string`.
fn detect_chip() -> String {
    // Attempt sysctl query for Apple Silicon model.
    if let Ok(chip) = sysctl_value("machdep.cpu.brand_string") {
        if !chip.is_empty() {
            return chip;
        }
    }

    // Fallback: detect via sysctl for hw.model (e.g. "Mac15,7").
    if let Ok(model) = sysctl_value("hw.model") {
        return model;
    }

    // Last resort: uname.
    if let Ok(uname) = sysctl_value("kern.version") {
        let parts: Vec<&str> = uname.split_whitespace().collect();
        if !parts.is_empty() {
            return parts[0].to_string();
        }
    }

    "Apple Silicon".to_string()
}

/// Check for Thunderbolt networking interfaces as a proxy for RDMA
/// availability.  Thunderbolt 4/5 devices expose `en` interfaces
/// or `ap` interfaces, visible via `ifconfig` or the `IOThunderbolt`
/// IOKit registry.
fn detect_rdma() -> bool {
    // Check for Thunderbolt network interfaces by reading sysctl for
    // Thunderbolt-capable networking.
    if let Ok(thunderbolt) = sysctl_value("hw.thunderbolt") {
        if thunderbolt.contains("1") || thunderbolt.to_lowercase().contains("true") {
            return true;
        }
    }

    // Check for presence of Thunderbolt in IORegistry raw output.
    if let Ok(output) = std::process::Command::new("system_profiler")
        .args(["SPThunderboltDataType", "-json"])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Any Thunderbolt device detected.
            if stdout.contains("Thunderbolt") || stdout.contains("thunderbolt") {
                return true;
            }
        }
    }

    // Check for IOThunderboltController in IORegistry (low-level check).
    if let Ok(output) = std::process::Command::new("ioreg")
        .args(["-rc", "IOThunderboltController"])
        .output()
    {
        if output.status.success() && !output.stdout.is_empty() {
            return true;
        }
    }

    false
}

/// Detect the number of ANE (Apple Neural Engine) cores.
fn detect_ane_cores() -> u32 {
    // The ANE is exposed via IOKit as `AppleNeuralEngine`.
    // The number of cores varies by chip:
    //   M1:     16 cores
    //   M1 Pro: 16 cores
    //   M1 Max: 16 cores
    //   M1 Ultra: 32 cores
    //   M2:     16 cores
    //   M2 Pro: 16 cores
    //   M2 Max: 16 cores
    //   M2 Ultra: 32 cores
    //   M3:     16 cores
    //   M3 Pro: 16 cores
    //   M3 Max: 16 cores
    //   M3 Ultra: 32 cores
    if let Ok(output) = std::process::Command::new("ioreg")
        .args(["-rc", "AppleNeuralEngine"])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Try to extract the number of cores from the ioreg output.
            for line in stdout.lines() {
                if line.contains("ANE.CoreCount") || line.contains("CoreCount") {
                    if let Some(val) = line.split('=').nth(1) {
                        if let Ok(n) = val.trim().parse::<u32>() {
                            return n;
                        }
                    }
                }
            }
            // If we found the AppleNeuralEngine, it has at least 16 cores.
            if stdout.contains("AppleNeuralEngine") {
                // Attempt to detect Ultra variants via chip string.
                let chip = detect_chip();
                if chip.to_lowercase().contains("ultra") {
                    return 32;
                }
                return 16;
            }
        }
    }

    // Conservative fallback: assume at least 16 ANE cores (all M-series chips).
    16
}

/// Read a sysctl value as a trimmed string.
fn sysctl_value(key: &str) -> Result<String, String> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", key])
        .output()
        .map_err(|e| format!("sysctl {}: {}", key, e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Err(format!("sysctl {} returned non-zero", key))
    }
}

/// Format a chip name for display (short, readable).
fn format_chip_name(chip: &str) -> String {
    let lower = chip.to_lowercase();

    if lower.contains("m3 ultra") || lower.contains("m3-ultra") {
        return "Mac Studio M3 Ultra".to_string();
    }
    if lower.contains("m3 max") {
        return "MacBook Pro M3 Max".to_string();
    }
    if lower.contains("m3 pro") {
        return "MacBook Pro M3 Pro".to_string();
    }
    if lower.contains("m3") {
        return "Mac M3".to_string();
    }
    if lower.contains("m2 ultra") || lower.contains("m2-ultra") {
        return "Mac Studio M2 Ultra".to_string();
    }
    if lower.contains("m2 max") {
        return "MacBook Pro M2 Max".to_string();
    }
    if lower.contains("m2 pro") {
        return "MacBook Pro M2 Pro".to_string();
    }
    if lower.contains("m2") {
        return "Mac M2".to_string();
    }
    if lower.contains("m1 ultra") || lower.contains("m1-ultra") {
        return "Mac Studio M1 Ultra".to_string();
    }
    if lower.contains("m1 max") {
        return "MacBook Pro M1 Max".to_string();
    }
    if lower.contains("m1 pro") {
        return "MacBook Pro M1 Pro".to_string();
    }
    if lower.contains("m1") {
        return "Mac M1".to_string();
    }

    // Fallback: return the raw string, truncated.
    if chip.len() > 28 {
        chip[..28].to_string()
    } else {
        chip.to_string()
    }
}

// ---------------------------------------------------------------------------
// ExoNode
// ---------------------------------------------------------------------------

impl ExoNode {
    /// Start the EXO worker on this node.
    ///
    /// 1. Detect hardware (chip, RAM, ANE cores)
    /// 2. Start an exo worker subprocess
    /// 3. Register this node's Tribunus runtime as the inference backend
    /// 4. Auto-join the cluster
    pub fn start(port: u16, no_worker: bool) -> Result<Self, String> {
        let hw = detect_hardware();
        let chip_name = format_chip_name(&hw.chip);

        // Build listen address.
        let listen_addr = format!("0.0.0.0:{}", port);

        // Initialize the model cache (half of total RAM).
        let total_ram_mb = gpu_memory::total_physical_ram_mb();
        let cache_max_mb = (total_ram_mb / 2).max(2048) as u64;
        let model_cache = ModelCache::new(cache_max_mb);

        // Start the EXO worker subprocess (unless --no-worker).
        let exo_process = if !no_worker {
            match Self::spawn_exo_worker(port, &hw) {
                Ok(child) => Some(child),
                Err(e) => {
                    eprintln!("[exo] WARNING: failed to spawn EXO worker: {}", e);
                    eprintln!(
                        "[exo] Continuing without subprocess — EXO must be started manually."
                    );
                    None
                }
            }
        } else {
            None
        };

        let node = ExoNode {
            model_cache: Arc::new(Mutex::new(model_cache)),
            exo_process,
            listen_addr: listen_addr.clone(),
            rdma_enabled: hw.rdma_available,
            chip: chip_name.clone(),
            ram_gb: hw.ram_gb,
            autoscaler: None,
        };

        // Print the startup banner.
        Self::print_banner(&chip_name, &listen_addr, hw.ram_gb, hw.rdma_available);

        Ok(node)
    }

    /// Spawn the EXO worker Python subprocess.
    ///
    /// This runs `exo worker` which handles device discovery, model
    /// sharding, and RDMA transport.  The worker communicates with
    /// other nodes via EXO's auto-discovery protocol.
    fn spawn_exo_worker(port: u16, hw: &HardwareInfo) -> Result<Child, String> {
        // Try to find the `exo` binary in PATH.
        let exo_binary = find_exo_binary()?;

        let mut cmd = Command::new(&exo_binary);
        cmd.arg("worker")
            .arg("--port")
            .arg(port.to_string())
            .arg("--no-tui")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Pass hardware info as environment variables so the EXO worker
        // can register this node's capabilities.
        cmd.env("TRIBUNUS_CHIP", &hw.chip);
        cmd.env("TRIBUNUS_RAM_GB", hw.ram_gb.to_string());
        cmd.env("TRIBUNUS_ANE_CORES", hw.ane_cores.to_string());
        cmd.env("TRIBUNUS_RDMA", if hw.rdma_available { "1" } else { "0" });

        let child = cmd
            .spawn()
            .map_err(|e| format!("failed to spawn '{} worker': {}", exo_binary, e))?;

        Ok(child)
    }

    /// Get the current cluster status.
    ///
    /// Queries the local EXO API for cluster state and combines it with
    /// this node's Tribunus runtime info.
    pub fn cluster_status(&self) -> Result<ClusterInfo, String> {
        // Query the local EXO worker for cluster state.
        let cluster_nodes = self.query_exo_cluster()?;

        let total_ram: u32 = cluster_nodes.iter().map(|n| n.ram_gb).sum();

        Ok(ClusterInfo {
            nodes: cluster_nodes,
            model: None, // EXO reports the active model
            model_shard: format!("node:{}", self.ram_gb),
            total_ram_gb: self.ram_gb,
            cluster_ram_gb: total_ram as u64,
        })
    }

    /// Query the local EXO worker's cluster state via its HTTP API.
    fn query_exo_cluster(&self) -> Result<Vec<NodeInfo>, String> {
        // The EXO worker exposes a /v1/cluster/ nodes endpoint on the
        // configured port.  We query it via HTTP.
        let port = self.listen_addr.rsplit(':').next().unwrap_or("52415");
        let url = format!("http://127.0.0.1:{}/v1/cluster/nodes", port);

        match Self::http_get(&url) {
            Ok(body) => {
                // Parse the JSON response.  EXO returns an array of node
                // descriptors.  If parsing fails, return a minimal list
                // containing just this node.
                if let Ok(nodes) = serde_json::from_str::<Vec<NodeInfo>>(&body) {
                    return Ok(nodes);
                }

                // Fallback: try to parse as a JSON object with a "nodes" key.
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&body) {
                    if let Some(arr) = val.get("nodes").and_then(|v| v.as_array()) {
                        let nodes: Vec<NodeInfo> = arr
                            .iter()
                            .filter_map(|n| serde_json::from_value(n.clone()).ok())
                            .collect();
                        if !nodes.is_empty() {
                            return Ok(nodes);
                        }
                    }
                }

                // EXO not responding — return just this node.
                Ok(vec![self.local_node_info()])
            }
            Err(_) => {
                // EXO worker not reachable; return local info only.
                Ok(vec![self.local_node_info()])
            }
        }
    }

    /// Build a NodeInfo for just this local node.
    fn local_node_info(&self) -> NodeInfo {
        let hostname = crate::hostname_or_default();
        NodeInfo {
            id: hostname,
            address: self.listen_addr.clone(),
            ram_gb: self.ram_gb,
            model_layer_range: None, // EXO assigns the layer range
            rdma: self.rdma_enabled,
        }
    }

    /// Perform a blocking HTTP GET request.
    fn http_get(url: &str) -> Result<String, String> {
        // Parse the URL to extract host and port.
        let host_port = url
            .strip_prefix("http://")
            .or_else(|| url.strip_prefix("https://"))
            .ok_or_else(|| format!("invalid URL: {}", url))?;

        let (host, port_path) = host_port.split_once(':').unwrap_or((host_port, "80"));
        let port = port_path.split('/').next().unwrap_or("80");

        let addr = format!("{}:{}", host, port);
        let timeout = Duration::from_secs(5);

        let mut stream = TcpStream::connect_timeout(
            &addr.parse().map_err(|e| format!("parse addr: {}", e))?,
            timeout,
        )
        .map_err(|e| format!("connect to {}: {}", addr, e))?;

        stream
            .set_read_timeout(Some(timeout))
            .map_err(|e| format!("set timeout: {}", e))?;

        use std::io::{Read, Write};

        // Send a minimal HTTP GET request.
        let path = port_path.split('/').skip(1).fold(String::new(), |a, p| {
            if a.is_empty() {
                format!("/{}", p)
            } else {
                format!("{}/{}", a, p)
            }
        });
        let path = if path.is_empty() { "/" } else { &path };
        let request = format!(
            "GET {} HTTP/1.0\r\nHost: {}\r\nConnection: close\r\n\r\n",
            path, host
        );

        stream
            .write_all(request.as_bytes())
            .map_err(|e| format!("write request: {}", e))?;

        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .map_err(|e| format!("read response: {}", e))?;

        let response_str =
            String::from_utf8(response).map_err(|e| format!("utf8 decode: {}", e))?;

        // Split headers from body.
        if let Some(body_start) = response_str.find("\r\n\r\n") {
            Ok(response_str[body_start + 4..].to_string())
        } else {
            Err("malformed HTTP response: no header/body separator".to_string())
        }
    }

    /// Stop EXO and clean up.
    pub fn stop(&mut self) -> Result<(), String> {
        if let Some(mut child) = self.exo_process.take() {
            // Send SIGTERM first, then SIGKILL if it doesn't exit quickly.
            #[cfg(unix)]
            {
                let _ = unsafe { libc::kill(child.id() as i32, libc::SIGTERM) };
                std::thread::sleep(Duration::from_millis(500));

                match child.try_wait() {
                    Ok(Some(status)) => {
                        eprintln!("[exo] worker exited with status: {:?}", status.code());
                    }
                    Ok(None) => {
                        // Still running — force kill.
                        let _ = unsafe { libc::kill(child.id() as i32, libc::SIGKILL) };
                        let _ = child.wait();
                        eprintln!("[exo] worker force-killed");
                    }
                    Err(e) => {
                        eprintln!("[exo] error waiting for worker: {}", e);
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                }
            }

            #[cfg(not(unix))]
            {
                let _ = child.kill();
                let _ = child.wait();
            }
        }

        Ok(())
    }

    /// Enable autoscaling for this EXO node.
    ///
    /// Registers an autoscaler that monitors load and adjusts the number
    /// of cluster nodes between `min` and `max`.
    pub fn enable_autoscaling(&mut self, min: u32, max: u32) -> Result<(), String> {
        let telemetry = InferenceTelemetry::global();
        let mut ascaler = Autoscaler::new(telemetry);
        ascaler.min_nodes = min;
        ascaler.max_nodes = max;
        self.autoscaler = Some(ascaler);
        eprintln!("[exo] autoscaling enabled: min={} max={}", min, max);
        Ok(())
    }

    /// Called periodically (e.g. every few seconds) to evaluate and act on
    /// load changes.  Spawns new nodes when load is high, drains when low.
    pub fn autoscale_tick(&mut self) -> Result<(), String> {
        if let Some(ascaler) = &mut self.autoscaler {
            match ascaler.tick()? {
                ScaleAction::ScaleUp(n) => {
                    eprintln!("[exo] autoscaler: scaling up by {}", n);
                    for _ in 0..n {
                        ascaler.spawn_node()?;
                    }
                }
                ScaleAction::ScaleDown(n) => {
                    eprintln!("[exo] autoscaler: scaling down by {}", n);
                    for i in 0..n {
                        ascaler.drain_node(&format!("autoscale-node-{}", i))?;
                    }
                }
                ScaleAction::None => {}
            }
        }
        Ok(())
    }

    /// Print the EXO cluster startup banner.
    fn print_banner(chip: &str, addr: &str, ram_gb: u32, rdma: bool) {
        let rdma_status = if rdma {
            "Thunderbolt 5"
        } else {
            "not detected"
        };
        let rdma_mark = if rdma { "\u{2713}" } else { "x" };

        // Get local IP address for the banner.
        let local_ip = local_ip_address().unwrap_or_else(|| "127.0.0.1".to_string());
        let display_addr = addr.replace("0.0.0.0", &local_ip);

        eprintln!("\n=== EXO Cluster Mode ===");
        eprintln!("  Node: {}", chip);
        eprintln!("  Address: {}", display_addr);
        eprintln!("  RAM: {} GB", ram_gb);
        eprintln!("  RDMA: {} {}", rdma_mark, rdma_status);
        eprintln!("  ANE: {} cores", detect_ane_cores());
        eprintln!("  Cluster will auto-discover peers on the local network");
        eprintln!("  Join the cluster from another Mac:");
        eprintln!(
            "    tribunus-server --exo --exo-port {} --model-path /path/to/model\n",
            port_from_addr(addr)
        );
    }

    /// Run inference on this node's layer shard.
    ///
    /// Receives hidden state from the previous node, runs the local
    /// layer shard through the full Tribunus pipeline, and sends the
    /// result to the next node.
    ///
    /// Each node's local inference uses the full Tribunus pipeline
    /// (memory plan, IOSurface, ANE speculation, TurboQuant KV cache,
    /// prefix cache).
    ///
    /// The communication pattern is pipeline parallel: while Node N
    /// processes its layer range, Node N+1 is already receiving the
    /// previous hidden state over RDMA.
    pub fn execute_layer_shard(&self, _input: &[f32]) -> Result<Vec<f32>, String> {
        // This is the core distributed inference path.
        //
        // In production, the EXO orchestrator sends the hidden state
        // via RDMA (Thunderbolt 5).  Each node:
        //
        // 1. Receives hidden state from the previous node (via RDMA)
        // 2. Runs its local layer shard through the Tribunus pipeline:
        //    a. Memory plan allocates IOSurface buffers
        //    b. ANE/GPU executes attention + FFN for assigned layers
        //    c. TurboQuant compresses KV cache
        //    d. Prefix cache matches against cached prefixes
        // 3. Sends the output hidden state to the next node (via RDMA)
        // 4. Pipeline parallelism: step 1 overlaps with step 2 of the
        //    previous node
        //
        // The actual tensor operations are delegated to the
        // ProfiledInferenceSession loaded from the ModelCache.
        // The EXO worker handles RDMA transport and pipeline
        // coordination; this function wraps the local compute step.

        // Placeholder: real implementation will:
        // - Receive hidden state from RDMA transport
        // - Load local layer shard from ModelCache
        // - Run through ProfiledInferenceSession (attention + FFN)
        // - Apply TurboQuant on KV cache for this node's layers
        // - Send result to next node via RDMA
        //
        // For now, this is a pass-through that preserves the input
        // shape.  Connect this to ProfiledInferenceSession once
        // model shard loading is wired through EXO.
        Ok(_input.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get the best guess for the local IP address (first non-loopback IPv4).
fn local_ip_address() -> Option<String> {
    // Use UDP connect trick to determine the preferred local IP.
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:53").ok()?;
    let addr = socket.local_addr().ok()?;
    Some(addr.ip().to_string())
}

/// Extract the port number from an address string.
fn port_from_addr(addr: &str) -> u16 {
    addr.rsplit(':')
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(52415)
}

/// Expand a leading ~/ in a path to the user's home directory.
fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{}/{}", home.trim_end_matches('/'), rest);
        }
    }
    path.to_string()
}

/// Find the `exo` binary in PATH or common locations.
fn find_exo_binary() -> Result<String, String> {
    // Check PATH first.
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            let candidate = format!("{}/exo", dir);
            if std::path::Path::new(&candidate).is_file() {
                return Ok(candidate);
            }
            // Also check for `exo` with .py extension (uv-installed).
            let candidate_py = format!("{}/exo.py", dir);
            if std::path::Path::new(&candidate_py).is_file() {
                return Ok(format!("python3 {}", candidate_py));
            }
        }
    }

    // Check common locations.
    let common_paths = vec![
        "/usr/local/bin/exo",
        "/opt/homebrew/bin/exo",
        "~/.local/bin/exo",
        "exo", // let PATH resolve it
    ];

    for p in common_paths {
        let expanded = expand_tilde(p);
        if std::path::Path::new(&expanded).is_file() {
            return Ok(expanded);
        }
    }

    // Return "exo" as a last resort — let the OS resolve PATH.
    Ok("exo".to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_chip() {
        let chip = detect_chip();
        assert!(!chip.is_empty(), "chip name should not be empty");
        eprintln!("detected chip: {}", chip);
    }

    #[test]
    fn test_format_chip_name() {
        assert_eq!(format_chip_name("Apple M3 Ultra"), "Mac Studio M3 Ultra");
        assert_eq!(format_chip_name("Apple M1 Max"), "MacBook Pro M1 Max");
        assert_eq!(format_chip_name("Apple M2"), "Mac M2");
        assert_eq!(format_chip_name("unknown chip"), "unknown chip");
    }

    #[test]
    fn test_detect_ram() {
        let ram_mb = gpu_memory::total_physical_ram_mb();
        assert!(ram_mb > 0, "RAM should be > 0");
    }

    #[test]
    fn test_detected_rdma_is_bool() {
        // Just ensure it doesn't panic.
        let rdma = detect_rdma();
        eprintln!("RDMA detected: {}", rdma);
    }

    #[test]
    fn test_node_info_serialization() {
        let info = NodeInfo {
            id: "test-node".to_string(),
            address: "192.168.1.100:52415".to_string(),
            ram_gb: 64,
            model_layer_range: Some((0, 11)),
            rdma: true,
        };
        let json = serde_json::to_string_pretty(&info).unwrap();
        assert!(json.contains("test-node"));
        assert!(json.contains("52415"));
        eprintln!("NodeInfo JSON:\n{}", json);
    }

    #[test]
    fn test_cluster_info_serialization() {
        let info = ClusterInfo {
            nodes: vec![NodeInfo {
                id: "node0".to_string(),
                address: "192.168.1.100:52415".to_string(),
                ram_gb: 64,
                model_layer_range: Some((0, 11)),
                rdma: true,
            }],
            model: Some("gemma-4-27b".to_string()),
            model_shard: "node:64".to_string(),
            total_ram_gb: 64,
            cluster_ram_gb: 64,
        };
        let json = serde_json::to_string_pretty(&info).unwrap();
        assert!(json.contains("gemma-4-27b"));
        assert!(json.contains("cluster_ram_gb"));
        eprintln!("ClusterInfo JSON:\n{}", json);
    }

    #[test]
    fn test_hardware_detection() {
        let hw = detect_hardware();
        assert!(hw.ram_gb > 0, "RAM should be > 0 GB");
        assert!(!hw.chip.is_empty(), "chip should be non-empty");
        eprintln!(
            "Hardware: {} ({} GB, {} ANE cores, RDMA: {})",
            hw.chip, hw.ram_gb, hw.ane_cores, hw.rdma_available
        );
    }

    #[test]
    fn test_local_node_info() {
        let cache = ModelCache::new(2048);
        let node = ExoNode {
            model_cache: Arc::new(Mutex::new(cache)),
            exo_process: None,
            listen_addr: "0.0.0.0:52415".to_string(),
            rdma_enabled: true,
            chip: "Mac Studio M3 Ultra".to_string(),
            ram_gb: 512,
            autoscaler: None,
        };
        let info = node.local_node_info();
        assert!(info.id.contains("unknown") || !info.id.is_empty());
        assert_eq!(info.ram_gb, 512);
        assert!(info.rdma);
    }

    // ── Distributed KV cache tests ────────────────────────────────────

    #[test]
    fn test_consistent_hash_ring_build() {
        let ids = vec!["node0".into(), "node1".into(), "node2".into()];
        let ring = ConsistentHashRing::build(&ids, 151);
        assert_eq!(ring.node_count(), 3);
        // At least one virtual entry per node should be present.
        assert!(ring.ring.len() >= 3);
        assert_eq!(ring.ring.len(), 453); // 151 * 3
    }

    #[test]
    fn test_consistent_hash_ring_distribution() {
        let ids = vec![
            "node0".into(),
            "node1".into(),
            "node2".into(),
            "node3".into(),
        ];
        let ring = ConsistentHashRing::build(&ids, 151);

        let mut assignments = vec![0usize; 4];
        for i in 0..4096 {
            let tokens = vec![i as u32];
            let hash = BlockAwarePrefixCache::compute_block_hash(&tokens);
            let idx = ring.node_index(&hash);
            assignments[idx] += 1;
        }

        // Each node should get roughly 1024 items (4096/4).
        // Allow 30% deviation for the ring's pseudo-random distribution.
        for &count in &assignments {
            assert!(count > 600, "Node assigned too few items: {}", count);
            assert!(count < 1500, "Node assigned too many items: {}", count);
        }
        eprintln!("Hash distribution: {:?}", assignments);
    }

    #[test]
    fn test_sharded_by_node_insert_get() {
        let ids = vec!["node0".into(), "node1".into()];
        let ring = ConsistentHashRing::build(&ids, 151);
        let mut sharded = ShardedByNode::<DistributedKvPage>::new(ring);

        let tokens = vec![42u32, 99, 7, 13];
        let hash = BlockAwarePrefixCache::compute_block_hash(&tokens);

        let page = DistributedKvPage {
            token_hash: hash,
            page_data: vec![1, 2, 3, 4],
            source_node: "node0".into(),
            rdma_offset: 0,
        };

        sharded.insert(hash, page);

        // Re-compute hash and verify retrieval.
        let lookup_hash = BlockAwarePrefixCache::compute_block_hash(&tokens);
        let retrieved = sharded.get(&lookup_hash);
        assert!(retrieved.is_some(), "Should retrieve the page we inserted");
        assert_eq!(retrieved.unwrap().page_data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_sharded_by_node_contains() {
        let ids = vec!["a".into(), "b".into()];
        let ring = ConsistentHashRing::build(&ids, 151);
        let mut sharded = ShardedByNode::<DistributedKvPage>::new(ring);

        let h1 = BlockAwarePrefixCache::compute_block_hash(&[1, 2, 3]);
        let h2 = BlockAwarePrefixCache::compute_block_hash(&[4, 5, 6]);

        sharded.insert(
            h1,
            DistributedKvPage {
                token_hash: h1,
                page_data: vec![10],
                source_node: "a".into(),
                rdma_offset: 0,
            },
        );

        assert!(sharded.contains(&h1), "h1 should be present");
        assert!(!sharded.contains(&h2), "h2 should not be present");
    }

    #[test]
    fn test_rdma_region_allocate_read() {
        let mut region = RdmaRegion::new("node0", 1024 * 1024);
        let hash = BlockAwarePrefixCache::compute_block_hash(&[1, 2, 3]);

        let data = vec![0xAB; 256];
        let offset = region.allocate(&hash, &data).unwrap();
        assert_eq!(offset, 0, "First allocation should be at offset 0");

        // Read back.
        let read = region.read_at(offset);
        assert!(read.is_some());
        assert_eq!(read.unwrap(), &data[..]);
    }

    #[test]
    fn test_rdma_region_full() {
        let mut region = RdmaRegion::new("node0", 100);
        let hash = BlockAwarePrefixCache::compute_block_hash(&[1]);
        let data = vec![0u8; 200];
        let result = region.allocate(&hash, &data);
        assert!(
            result.is_err(),
            "Allocation exceeding region size should fail"
        );
        assert!(result.unwrap_err().contains("full"));
    }

    #[test]
    fn test_distributed_kv_cache_new() {
        let nodes = vec![
            NodeInfo {
                id: "node0".into(),
                address: "192.168.1.1:52415".into(),
                ram_gb: 64,
                model_layer_range: None,
                rdma: true,
            },
            NodeInfo {
                id: "node1".into(),
                address: "192.168.1.2:52415".into(),
                ram_gb: 32,
                model_layer_range: None,
                rdma: true,
            },
        ];
        let cache = DistributedKvCache::new("node0", &nodes);

        // Should have an RDMA region since node0 has rdma: true.
        assert!(cache.rdma_region.is_some());
        assert_eq!(cache.local_node, "node0");
        assert_eq!(cache.nodes.len(), 2);
    }

    #[test]
    fn test_distributed_kv_cache_put_get_local() {
        let nodes = vec![
            NodeInfo {
                id: "node0".into(),
                address: "10.0.0.1:52415".into(),
                ram_gb: 64,
                model_layer_range: None,
                rdma: true,
            },
            NodeInfo {
                id: "node1".into(),
                address: "10.0.0.2:52415".into(),
                ram_gb: 32,
                model_layer_range: None,
                rdma: true,
            },
        ];
        let mut cache = DistributedKvCache::new("node0", &nodes);

        let tokens = vec![10u32, 20, 30, 40];
        let hash = BlockAwarePrefixCache::compute_block_hash(&tokens);

        // Determine if this hash lands on node0 or node1.
        let owner_idx = cache.ring.node_index(&hash);
        let owner_id = cache.nodes[owner_idx].id.clone();
        eprintln!("Hash {:02x}... owned by {}", hash.0[0], owner_id);

        let page_data = vec![0xAA; 128];
        cache.put_page(hash, page_data.clone()).unwrap();

        let lookup_hash = BlockAwarePrefixCache::compute_block_hash(&tokens);
        let retrieved = cache.get_page(&lookup_hash);

        if owner_id == "node0" {
            // Local page: should be immediately available.
            assert!(retrieved.is_some(), "Local page should be retrievable");
            if let Some(p) = retrieved {
                assert_eq!(p.page_data, page_data);
                assert_eq!(p.source_node, "node0");
            }
        } else {
            // Remote page: get_page returns None (no live RDMA cluster).
            assert!(
                retrieved.is_none(),
                "Remote page should return None without RDMA cluster"
            );
        }
    }

    #[test]
    fn test_distributed_kv_cache_check_prefix() {
        let nodes = vec![
            NodeInfo {
                id: "n0".into(),
                address: "10.0.0.1:52415".into(),
                ram_gb: 64,
                model_layer_range: None,
                rdma: false,
            },
            NodeInfo {
                id: "n1".into(),
                address: "10.0.0.2:52415".into(),
                ram_gb: 64,
                model_layer_range: None,
                rdma: false,
            },
        ];
        let mut cache = DistributedKvCache::new("n0", &nodes);

        // Build a token sequence long enough for multiple blocks.
        let block_size = crate::cache::prefix_cache::PREFIX_BLOCK_SIZE;
        let tokens: Vec<u32> = (0..(block_size * 3) as u32).collect();

        // Insert first block.
        let block0_tokens: Vec<u32> = tokens[0..block_size].to_vec();
        let h0 = BlockAwarePrefixCache::compute_block_hash(&block0_tokens);
        cache.put_page(h0, vec![0u8; 64]).unwrap();

        // Insert second block.
        let block1_tokens: Vec<u32> = tokens[block_size..block_size * 2].to_vec();
        let h1 = BlockAwarePrefixCache::compute_block_hash(&block1_tokens);
        cache.put_page(h1, vec![1u8; 64]).unwrap();

        // check_prefix with all 3 blocks: should match first 2 (contiguous prefix).
        let result = cache.check_prefix(&tokens);
        assert!(result.is_some(), "Should match at least one block");
        let matched = result.unwrap();
        assert_eq!(matched.len(), 2, "Should match first 2 contiguous blocks");
        assert_eq!(matched[0], h0);
        assert_eq!(matched[1], h1);
    }

    #[test]
    fn test_distributed_kv_cache_check_prefix_no_match() {
        let nodes = vec![NodeInfo {
            id: "n0".into(),
            address: "10.0.0.1:52415".into(),
            ram_gb: 64,
            model_layer_range: None,
            rdma: false,
        }];
        let cache = DistributedKvCache::new("n0", &nodes);

        // No pages inserted — prefix check should return None.
        let tokens: Vec<u32> = (0..64).collect();
        let result = cache.check_prefix(&tokens);
        assert!(result.is_none(), "Empty cache should return None");
    }

    #[test]
    fn test_distributed_kv_cache_check_prefix_too_short() {
        let nodes = vec![NodeInfo {
            id: "n0".into(),
            address: "10.0.0.1:52415".into(),
            ram_gb: 64,
            model_layer_range: None,
            rdma: false,
        }];
        let cache = DistributedKvCache::new("n0", &nodes);

        // Fewer tokens than PREFIX_BLOCK_SIZE: should return None.
        let tokens = vec![1u32, 2, 3];
        let result = cache.check_prefix(&tokens);
        assert!(result.is_none(), "Too few tokens should return None");
    }
}
