use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// Per-IP token bucket.
pub struct TokenBucket {
    pub tokens: AtomicU64,
    pub capacity: u64,
    pub refill_rate: f64,
    pub last_refill: AtomicU64,
}

fn current_time_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl TokenBucket {
    pub fn new(capacity: u64, refill_per_sec: f64) -> Self {
        Self {
            tokens: AtomicU64::new(capacity),
            capacity,
            refill_rate: refill_per_sec,
            last_refill: AtomicU64::new(current_time_secs()),
        }
    }

    /// Attempt to consume `tokens` from the bucket. Returns true if
    /// sufficient capacity was available and the tokens were deducted.
    pub fn try_consume(&self, tokens: u64) -> bool {
        self.refill();
        loop {
            let current = self.tokens.load(Ordering::Acquire);
            if current < tokens {
                return false;
            }
            match self.tokens.compare_exchange(
                current,
                current - tokens,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(_) => continue,
            }
        }
    }

    /// Refill tokens based on elapsed time. At most one thread executes
    /// the refill; all others see the updated token count atomically.
    fn refill(&self) {
        let now = current_time_secs();
        let last = self.last_refill.load(Ordering::Acquire);
        if now <= last {
            return;
        }
        if self
            .last_refill
            .compare_exchange(last, now, Ordering::Release, Ordering::Acquire)
            .is_err()
        {
            return; // another thread claimed the update
        }
        let elapsed = (now - last) as f64;
        let gained = (elapsed * self.refill_rate) as u64;
        if gained > 0 {
            let current = self.tokens.load(Ordering::Acquire);
            let new = self.capacity.min(current.saturating_add(gained));
            self.tokens.store(new, Ordering::Release);
        }
    }
}

/// Global rate limiter (IP -> TokenBucket).
pub struct RateLimiter {
    buckets: RwLock<HashMap<String, Arc<TokenBucket>>>,
    pub default_capacity: u64,
    pub default_refill_rate: f64,
}

impl RateLimiter {
    pub fn new(capacity: u64, refill_per_sec: f64) -> Self {
        Self {
            buckets: RwLock::new(HashMap::new()),
            default_capacity: capacity,
            default_refill_rate: refill_per_sec,
        }
    }

    /// Check whether `ip` is allowed to proceed. Creates a new bucket on
    /// first visit. Returns true when the request may proceed.
    pub async fn check(&self, ip: &str) -> bool {
        let bucket = {
            let read = self.buckets.read().await;
            read.get(ip).cloned()
        };
        let bucket = match bucket {
            Some(b) => b,
            None => {
                let b = Arc::new(TokenBucket::new(
                    self.default_capacity,
                    self.default_refill_rate,
                ));
                self.buckets.write().await.insert(ip.to_string(), b.clone());
                b
            }
        };
        bucket.try_consume(1)
    }

    /// Remove buckets whose last activity was more than 5 minutes ago.
    pub async fn cleanup_stale(&self) {
        const STALE_SECS: u64 = 300;
        let now = current_time_secs();
        let threshold = now.saturating_sub(STALE_SECS);
        let mut write = self.buckets.write().await;
        write.retain(|_, bucket| {
            let last = bucket.last_refill.load(Ordering::Acquire);
            last >= threshold
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_token_bucket_initial_full() {
        let bucket = TokenBucket::new(10, 5.0);
        assert_eq!(bucket.tokens.load(Ordering::Acquire), 10);
    }

    #[test]
    fn test_token_bucket_consume_all() {
        let bucket = TokenBucket::new(3, 1.0);
        assert!(bucket.try_consume(3));
        assert!(!bucket.try_consume(1));
    }

    #[test]
    fn test_token_bucket_consume_partial() {
        let bucket = TokenBucket::new(5, 10.0);
        assert!(bucket.try_consume(3));
        assert!(bucket.try_consume(2));
        assert!(!bucket.try_consume(1));
    }

    #[test]
    fn test_token_bucket_refill_over_time() {
        let bucket = TokenBucket::new(10, 10.0);
        assert!(bucket.try_consume(10));
        assert!(!bucket.try_consume(1));
        // Advance time by ~1 second by manipulating last_refill.
        bucket
            .last_refill
            .store(current_time_secs() - 1, Ordering::Release);
        // Refill should add ~10 tokens (capped at capacity 10).
        bucket.refill();
        assert_eq!(bucket.tokens.load(Ordering::Acquire), 10);
        assert!(bucket.try_consume(1));
    }

    #[test]
    fn test_token_bucket_never_exceeds_capacity() {
        let bucket = TokenBucket::new(5, 100.0);
        bucket
            .last_refill
            .store(current_time_secs() - 60, Ordering::Release);
        bucket.refill();
        assert_eq!(bucket.tokens.load(Ordering::Acquire), 5);
    }

    #[test]
    fn test_token_bucket_concurrent_consume() {
        let bucket = Arc::new(TokenBucket::new(100, 10.0));
        let mut handles = Vec::new();
        for _ in 0..10 {
            let b = bucket.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..10 {
                    assert!(b.try_consume(1));
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(bucket.tokens.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn test_rate_limiter_check_creates_bucket() {
        let limiter = RateLimiter::new(5, 1.0);
        assert!(limiter.check("1.2.3.4").await);
        assert_eq!(limiter.buckets.read().await.len(), 1);
    }

    #[tokio::test]
    async fn test_rate_limiter_exhausts_then_denies() {
        let limiter = RateLimiter::new(2, 1.0);
        assert!(limiter.check("10.0.0.1").await);
        assert!(limiter.check("10.0.0.1").await);
        assert!(!limiter.check("10.0.0.1").await);
    }

    #[tokio::test]
    async fn test_rate_limiter_two_ips_independent() {
        let limiter = RateLimiter::new(1, 1.0);
        assert!(limiter.check("a").await);
        assert!(limiter.check("b").await);
        assert!(!limiter.check("a").await);
    }

    #[tokio::test]
    async fn test_cleanup_stale_removes_old_buckets() {
        let limiter = RateLimiter::new(5, 1.0);
        assert!(limiter.check("stale-ip").await);
        // Tamper the bucket's last_refill to be ancient.
        {
            let read = limiter.buckets.read().await;
            let bucket = read.get("stale-ip").unwrap();
            bucket
                .last_refill
                .store(current_time_secs() - 600, Ordering::Release);
        }
        assert!(limiter.check("fresh-ip").await);
        limiter.cleanup_stale().await;
        assert_eq!(limiter.buckets.read().await.len(), 1);
        assert!(limiter.buckets.read().await.contains_key("fresh-ip"));
    }
}
