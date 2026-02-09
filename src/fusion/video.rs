use crate::error::{MumaTomError, Result};
use image::DynamicImage;
use std::path::Path;
use std::time::Duration;

pub struct VideoProcessor {
    pub fps: f32,
    pub max_frames: Option<usize>,
}

impl VideoProcessor {
    pub fn new(fps: f32, max_frames: Option<usize>) -> Self {
        Self { fps, max_frames }
    }

    pub fn extract_frames(&self, video_path: &Path) -> Result<Vec<VideoFrame>> {
        let path_str = video_path.to_str().ok_or_else(|| {
            MumaTomError::VideoProcessing(format!("Invalid path: {:?}", video_path))
        })?;

        if !video_path.exists() {
            return Err(MumaTomError::VideoProcessing(format!(
                "Video file not found: {}",
                path_str
            )));
        }

        let ext = video_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        let frames = match ext.as_str() {
            "mp4" | "mkv" | "avi" | "mov" | "webm" => self.extract_from_video_file(video_path)?,
            _ => {
                return Err(MumaTomError::VideoProcessing(format!(
                    "Unsupported video format: {}",
                    ext
                )));
            }
        };

        Ok(frames)
    }

    fn extract_from_video_file(&self, video_path: &Path) -> Result<Vec<VideoFrame>> {
        use video_rs::{Decoder, Location, Time};

        let mut decoder = Decoder::new(Location::File(video_path.to_path_buf()))
            .map_err(|e| MumaTomError::VideoProcessing(format!("Failed to open video: {}", e)))?;

        let fps = self.fps;
        let frame_interval = Duration::from_secs_f64(1.0 / fps);

        let mut frames = Vec::new();
        let mut timestamp_secs = 0.0;

        loop {
            match decoder.decode_frame() {
                Ok(Some(frame)) => {
                    let timestamp_ms = (timestamp_secs * 1000.0) as u64;

                    let video_frame = VideoFrame {
                        timestamp_ms,
                        frame_number: frames.len(),
                        duration_ms: Some(frame_interval.as_millis() as u64),
                    };

                    frames.push(video_frame);

                    if let Some(max) = self.max_frames {
                        if frames.len() >= max {
                            break;
                        }
                    }

                    timestamp_secs += 1.0 / fps;
                }
                Ok(None) => break,
                Err(e) => {
                    return Err(MumaTomError::VideoProcessing(format!(
                        "Failed to decode frame: {}",
                        e
                    )));
                }
            }
        }

        if frames.is_empty() {
            return Err(MumaTomError::VideoProcessing(
                "No frames extracted from video".to_string(),
            ));
        }

        Ok(frames)
    }

    pub fn preprocess_frame(&self, frame: &DynamicImage) -> Result<DynamicImage> {
        Ok(frame.clone())
    }

    pub fn align_timestamps(&self, timestamps: Vec<u64>) -> Vec<TimestampAlignment> {
        let mut alignments = Vec::new();

        for (i, &ts) in timestamps.iter().enumerate() {
            let duration_ms = if i + 1 < timestamps.len() {
                Some(timestamps[i + 1] - ts)
            } else {
                None
            };

            let alignment = TimestampAlignment {
                frame_index: i,
                timestamp_ms: *ts,
                duration_ms,
                fps: self.fps,
            };

            alignments.push(alignment);
        }

        alignments
    }

    pub fn sample_frames_uniform(&self, frames: &[VideoFrame], sample_count: usize) -> Vec<usize> {
        if frames.len() <= sample_count {
            return (0..frames.len()).collect();
        }

        let step = (frames.len() as f64 / sample_count as f64).ceil() as usize;
        (0..sample_count)
            .map(|i| i * step)
            .filter(|&i| i < frames.len())
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub timestamp_ms: u64,
    pub frame_number: usize,
    pub duration_ms: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct TimestampAlignment {
    pub frame_index: usize,
    pub timestamp_ms: u64,
    pub duration_ms: Option<u64>,
    pub fps: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_processor_creation() {
        let processor = VideoProcessor::new(10.0, Some(100));
        assert_eq!(processor.fps, 10.0);
        assert_eq!(processor.max_frames, Some(100));
    }

    #[test]
    fn test_extract_from_nonexistent_video() {
        let processor = VideoProcessor::new(10.0, None);
        let result = processor.extract_frames(Path::new("/nonexistent/video.mp4"));
        assert!(result.is_err());
    }

    #[test]
    fn test_timestamp_alignment() {
        let processor = VideoProcessor::new(10.0, None);
        let timestamps = vec![0, 100, 200, 300];

        let alignments = processor.align_timestamps(timestamps);

        assert_eq!(alignments.len(), 4);
        assert_eq!(alignments[0].timestamp_ms, 0);
        assert_eq!(alignments[1].timestamp_ms, 100);
        assert_eq!(alignments[2].timestamp_ms, 200);
        assert_eq!(alignments[3].timestamp_ms, 300);
    }

    #[test]
    fn test_sample_frames_uniform() {
        let processor = VideoProcessor::new(10.0, None);
        let frames = (0..100)
            .map(|i| VideoFrame {
                timestamp_ms: i * 100,
                frame_number: i,
                duration_ms: Some(100),
            })
            .collect::<Vec<_>>();

        let sampled = processor.sample_frames_uniform(&frames, 10);

        assert_eq!(sampled.len(), 10);
        assert_eq!(sampled[0], 0);
        assert_eq!(sampled[9], 90);
    }

    #[test]
    fn test_sample_more_than_available() {
        let processor = VideoProcessor::new(10.0, None);
        let frames = vec![
            VideoFrame {
                timestamp_ms: 0,
                frame_number: 0,
                duration_ms: Some(100),
            },
            VideoFrame {
                timestamp_ms: 100,
                frame_number: 1,
                duration_ms: Some(100),
            },
        ];

        let sampled = processor.sample_frames_uniform(&frames, 10);

        assert_eq!(sampled.len(), 2);
        assert_eq!(sampled[0], 0);
        assert_eq!(sampled[1], 1);
    }
}
