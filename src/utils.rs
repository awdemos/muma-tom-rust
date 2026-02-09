use std::time::{Duration, Instant};

pub fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    let millis = duration.subsec_millis();

    if secs > 60 {
        let minutes = secs / 60;
        let seconds = secs % 60;
        format!("{}m {}s", minutes, seconds)
    } else if secs > 0 {
        format!(
            "{}.{:03}s",
            secs,
            duration.subsec_nanos() as f64 / 1_000_000_000
        )
    } else {
        format!("{}.{:03}s", duration.subsec_micros() as f64 / 1_000_000)
    }
}

pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        let gb = bytes as f64 / GB as u64;
        format!("{:.2} GB", gb)
    } else if bytes >= MB {
        let mb = bytes as f64 / MB as u64;
        format!("{:.2} MB", mb)
    } else if bytes >= KB {
        let kb = bytes as f64 / KB as u64;
        format!("{:.2} KB", kb)
    } else {
        format!("{} B", bytes)
    }
}

pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    pub fn elapsed_millis(&self) -> u128 {
        self.start.elapsed().as_millis()
    }

    pub fn elapsed_human_readable(&self) -> String {
        format_duration(self.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration_seconds() {
        let d = Duration::from_secs(65);
        assert_eq!(format_duration(d), "1m 5s");
    }

    #[test]
    fn test_format_duration_milliseconds() {
        let d = Duration::from_millis(2345);
        assert_eq!(format_duration(d), "2.345s");
    }

    #[test]
    fn test_format_duration_microseconds() {
        let d = Duration::from_micros(123456);
        assert_eq!(format_duration(d), "123.456ms");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "10.24 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_timer() {
        let timer = Timer::new();
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        assert!(timer.elapsed_millis() >= 95);
        assert!(timer.elapsed_millis() < 150);
        assert!(!timer.elapsed_human_readable().is_empty());
    }
}
