use std::path::PathBuf;
use tokio;
use tracing::{info, error, instrument, Level};
use tracing_subscriber;

use muma_tom_rust::config::Config;
use muma_tom_rust::benchmark::runner::BenchmarkRunner;
use muma_tom_rust::error::Result;

#[tokio::main]
#[instrument]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let command = if args.get(2).and_then(|s| s.to_string()) {
        Ok(s.to_string())
    }).unwrap_or_else(|_| String::from(""));

    match command.as_str() {
        "help" => {
            print_help();
            Ok(())
        }
        _ => run_benchmark(command, args.get(3..)).await,
    }
    }
}

fn print_help() {
    println!("MuMA-ToM Rust Implementation - Command Line Interface");
    println!();
    println!("USAGE:");
    println!("  cargo run -- -- <benchmark-path> [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --output <path>     Output directory for results (default: ./results)");
    println!("  --no-eval       Skip evaluation phase, only process questions");
    println!("  --help          Show this help message");
    println!();
    println!("EXAMPLES:");
    println!("  cargo run -- ./data/benchmark");
    println!("  cargo run -- ./data/benchmark --output ./results");
    println!("  cargo run -- ./data/benchmark --no-eval");
    println!();
}

async fn run_benchmark(
    command: &str,
    args: &[String],
) -> Result<()> {
    let mut benchmark_path = "./data/benchmark";
    let mut output_dir = "./results";
    let skip_evaluation = false;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--benchmark" => {
                benchmark_path = args.get(i + 1).cloned().ok_or_else(|_| {
                    Err(MumaTomError::Internal(format!(
                        "Missing benchmark path argument after --benchmark"
                    ))
                })?;
                i += 2;
            }
            "--output" => {
                output_dir = args.get(i + 1).cloned().ok_or_else(|_| {
                    Err(MumaTomError::Internal(format!(
                        "Missing output path argument after --output"
                    ))
                })?;
                i += 2;
            }
            "--no-eval" => {
                skip_evaluation = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_file(false)
        .init();

    info!("Starting MuMA-ToM benchmark evaluation");
    info!("Benchmark path: {}", benchmark_path);
    info!("Output directory: {}", output_dir);
    info!("Skip evaluation: {}", skip_evaluation);

    match Config::from_env() {
        Ok(config) => {
            info!("Configuration loaded successfully");
            match BenchmarkRunner::run(config, &benchmark_path, &output_dir, skip_evaluation).await {
                Ok(()) => {
                    info!("Benchmark completed successfully");
                }
                Err(e) => {
                    error!("Benchmark failed: {}", e);
                }
            }
        }
        Err(e) => {
            error!("Failed to load configuration: {}", e);
        }
    }
    }
}
