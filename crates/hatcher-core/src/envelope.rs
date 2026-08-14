//! Deterministic, digest-backed serialization envelopes.
//!
//! Every observable HAMNS artifact (a node, a mesh step, a pipeline trace) can be
//! sealed into an [`Envelope`]. The digest is taken over the canonical JSON form,
//! which is key-sorted because `serde_json::Map` is a `BTreeMap` in this build.
//! That makes the digest reproducible across processes and machines, so a mesh
//! state can be attested to by the Hatcher control plane.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Codec tag written into every envelope.
pub const CANONICAL_CODEC: &str = "zk-canonical-v1";

/// Schema version for the HAMNS contract layer.
///
/// Bumped to `3` in 1.0.0, when stage records gained quality, latency, error class, and
/// provenance. A digest taken under schema 2 does not compare to one taken under 3, and
/// that is the point of the field.
pub const SCHEMA_VERSION: u32 = 3;

/// A digest-backed wrapper around any serializable payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Envelope<T> {
    pub codec: String,
    pub schema_version: u32,
    pub payload: T,
    pub digest: String,
}

impl<T: Serialize> Envelope<T> {
    /// Seal a payload, computing its canonical digest.
    pub fn seal(payload: T) -> Result<Self, serde_json::Error> {
        let digest = canonical_digest(&payload)?;
        Ok(Self {
            codec: CANONICAL_CODEC.to_string(),
            schema_version: SCHEMA_VERSION,
            payload,
            digest,
        })
    }

    /// Recompute the digest and compare it to the stored one.
    pub fn verify(&self) -> Result<bool, serde_json::Error> {
        Ok(canonical_digest(&self.payload)? == self.digest)
    }
}

/// Hash a value over its canonical (key-sorted) JSON encoding.
pub fn canonical_digest<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    let canonical = serde_json::to_string(&serde_json::to_value(value)?)?;
    let mut hasher = Sha256::new();
    hasher.update(CANONICAL_CODEC.as_bytes());
    hasher.update(b":");
    hasher.update(canonical.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

/// Fold many digests into one commitment, order-sensitive.
pub fn fold_digests<I, S>(digests: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut hasher = Sha256::new();
    hasher.update(b"zk-fold-v1:");
    for digest in digests {
        hasher.update(digest.as_ref().as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
    struct Sample {
        b: u32,
        a: String,
    }

    #[test]
    fn envelope_digest_is_stable_and_verifiable() {
        let sample = Sample {
            b: 7,
            a: "x".into(),
        };
        let sealed = Envelope::seal(sample.clone()).unwrap();
        assert_eq!(sealed.codec, CANONICAL_CODEC);
        assert_eq!(sealed.schema_version, SCHEMA_VERSION);
        assert!(sealed.verify().unwrap());
        assert_eq!(sealed.digest, Envelope::seal(sample).unwrap().digest);
    }

    #[test]
    fn digest_is_field_order_independent() {
        #[derive(Serialize)]
        struct Flipped {
            a: String,
            b: u32,
        }
        let left = canonical_digest(&Sample {
            b: 1,
            a: "y".into(),
        })
        .unwrap();
        let right = canonical_digest(&Flipped {
            a: "y".into(),
            b: 1,
        })
        .unwrap();
        assert_eq!(left, right, "canonical JSON must sort keys");
    }

    #[test]
    fn folding_digests_is_order_sensitive() {
        assert_ne!(fold_digests(["a", "b"]), fold_digests(["b", "a"]));
    }
}
