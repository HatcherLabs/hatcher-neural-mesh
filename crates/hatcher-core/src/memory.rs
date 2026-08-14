//! The long-term memory substrate behind the memory-evolution equation.
//!
//! `M_i(t+1) = M_i(t) + η K_i − δ R_i`
//!
//! `K_i` is not a free parameter: it is the salience of what this agent actually
//! committed to memory on a verified run. `R_i` is what decay removed. Both are
//! measured here, so the equation is fed by observation rather than by a constant.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// One retained fact: what an agent learned, on what task, and how much it mattered.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryRecord {
    pub task_id: String,
    pub agent_id: String,
    pub domain: String,
    /// Salience in `[0, 1]`. Decays over time; contributes to `K_i` when written.
    pub salience: f64,
    /// Whether the verifier accepted the work this memory came from. Unverified
    /// memories are retained at reduced salience so failures are still learnable.
    pub verified: bool,
}

/// A semantic graph of agents, domains, and tasks, plus the retained records.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryGraph {
    pub nodes: Vec<String>,
    pub edges: Vec<(String, String)>,
    pub records: Vec<MemoryRecord>,
}

/// Salience below this is forgotten entirely.
pub const FORGET_THRESHOLD: f64 = 0.02;

impl MemoryGraph {
    pub fn add_node(&mut self, node: impl Into<String>) {
        let node = node.into();
        if !self.nodes.contains(&node) {
            self.nodes.push(node);
        }
    }

    pub fn add_edge(&mut self, from: impl Into<String>, to: impl Into<String>) {
        let from = from.into();
        let to = to.into();
        self.add_node(from.clone());
        self.add_node(to.clone());
        if !self.edges.contains(&(from.clone(), to.clone())) {
            self.edges.push((from, to));
        }
    }

    /// Commit a record and wire it into the semantic graph.
    ///
    /// Returns the salience admitted, which is the run's contribution to `K_i`.
    pub fn remember(&mut self, record: MemoryRecord) -> f64 {
        let salience = if record.verified {
            record.salience.clamp(0.0, 1.0)
        } else {
            // A failed run still teaches something, just less.
            (record.salience * 0.35).clamp(0.0, 1.0)
        };

        self.add_edge(record.agent_id.clone(), record.domain.clone());
        self.add_edge(record.domain.clone(), record.task_id.clone());

        self.records.push(MemoryRecord { salience, ..record });
        salience
    }

    /// Total retained salience for one agent — the observable behind `M_i`.
    pub fn knowledge_for(&self, agent_id: &str) -> f64 {
        self.records
            .iter()
            .filter(|record| record.agent_id == agent_id)
            .map(|record| record.salience)
            .sum()
    }

    /// Retained salience per domain, for specialization routing.
    pub fn domain_mass(&self) -> BTreeMap<String, f64> {
        let mut mass = BTreeMap::new();
        for record in &self.records {
            *mass.entry(record.domain.clone()).or_insert(0.0) += record.salience;
        }
        mass
    }

    /// Age every memory by `rate` and drop what falls below [`FORGET_THRESHOLD`].
    ///
    /// Returns per-agent lost salience: the measured `R_i` for the next update.
    pub fn decay(&mut self, rate: f64) -> BTreeMap<String, f64> {
        let rate = rate.clamp(0.0, 1.0);
        let mut lost: BTreeMap<String, f64> = BTreeMap::new();

        for record in &mut self.records {
            let before = record.salience;
            record.salience = (record.salience * (1.0 - rate)).max(0.0);
            *lost.entry(record.agent_id.clone()).or_insert(0.0) += before - record.salience;
        }

        self.records.retain(|record| {
            if record.salience < FORGET_THRESHOLD {
                *lost.entry(record.agent_id.clone()).or_insert(0.0) += record.salience;
                false
            } else {
                true
            }
        });

        lost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(agent: &str, verified: bool, salience: f64) -> MemoryRecord {
        MemoryRecord {
            task_id: "task-1".into(),
            agent_id: agent.into(),
            domain: "rust".into(),
            salience,
            verified,
        }
    }

    #[test]
    fn memory_graph_tracks_nodes_and_edges() {
        let mut graph = MemoryGraph::default();
        graph.add_edge("signal", "policy");
        assert!(graph.nodes.contains(&"signal".to_string()));
        assert!(graph.nodes.contains(&"policy".to_string()));
        assert!(graph
            .edges
            .contains(&("signal".to_string(), "policy".to_string())));
    }

    #[test]
    fn unverified_memories_are_retained_at_reduced_salience() {
        let mut graph = MemoryGraph::default();
        let verified = graph.remember(record("a", true, 0.8));
        let unverified = graph.remember(record("b", false, 0.8));
        assert!(unverified < verified);
        assert!(unverified > 0.0, "failures must still teach something");
    }

    #[test]
    fn remember_wires_the_semantic_graph() {
        let mut graph = MemoryGraph::default();
        graph.remember(record("coder-1", true, 0.6));
        assert!(graph
            .edges
            .contains(&("coder-1".to_string(), "rust".to_string())));
        assert!(graph
            .edges
            .contains(&("rust".to_string(), "task-1".to_string())));
        assert!((graph.knowledge_for("coder-1") - 0.6).abs() < 1e-12);
        assert_eq!(
            graph.domain_mass().get("rust").copied().map(|m| m > 0.0),
            Some(true)
        );
    }

    #[test]
    fn decay_reports_loss_per_agent_and_forgets_the_faint() {
        let mut graph = MemoryGraph::default();
        graph.remember(record("a", true, 0.5));
        let lost = graph.decay(0.5);
        assert!((lost.get("a").copied().unwrap_or(0.0) - 0.25).abs() < 1e-12);

        graph.records[0].salience = 0.01;
        let lost = graph.decay(0.0);
        assert!(graph.records.is_empty(), "faint memories are dropped");
        assert!(lost.get("a").copied().unwrap_or(0.0) > 0.0);
    }
}
