//! Print the routing benchmark for every scenario.
//!
//! ```bash
//! cargo run -p hatcher-playground --example benchmark
//! ```

use hatcher_playground::{synthetic_recording, Benchmark, Scenario};

fn main() {
    for scenario in [Scenario::rehearsal(), Scenario::stress(), Scenario::frontier()] {
        let report = Benchmark::new().with_scenario(scenario).run();
        println!("{}\n", report.table());
    }

    let replay = synthetic_recording(40, "expert");
    let report = replay.evaluate(
        hatcher_playground::contested_cohort(),
        hatcher_core::RuntimeCalibration::default(),
    );
    println!("{}", report.headline());
}
