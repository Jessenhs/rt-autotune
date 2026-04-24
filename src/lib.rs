use nih_plug::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

const WINDOW_SIZE: usize = 2048;
const HOP_SIZE: usize = 512;
const HALF_BINS: usize = WINDOW_SIZE / 2 + 1;
const RING_SIZE: usize = WINDOW_SIZE * 4;
const YIN_HALF: usize = WINDOW_SIZE / 2;

// Musical scales: which semitones from root are active
const CHROMATIC: [bool; 12] = [true; 12];
const MAJOR: [bool; 12] = [true, false, true, false, true, true, false, true, false, true, false, true];
const MINOR: [bool; 12] = [true, false, true, true, false, true, false, true, true, false, true, false];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
pub enum Scale {
    Chromatic,
    Major,
    Minor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Enum)]
pub enum Key {
    C,
    #[name = "C#"]
    CSharp,
    D,
    #[name = "D#"]
    DSharp,
    E,
    F,
    #[name = "F#"]
    FSharp,
    G,
    #[name = "G#"]
    GSharp,
    A,
    #[name = "A#"]
    ASharp,
    B,
}

#[derive(Params)]
struct AutoTuneParams {
    #[id = "speed"]
    retune_speed: FloatParam,
    #[id = "mix"]
    mix: FloatParam,
    #[id = "key"]
    key: EnumParam<Key>,
    #[id = "scale"]
    scale: EnumParam<Scale>,
}

impl Default for AutoTuneParams {
    fn default() -> Self {
        Self {
            retune_speed: FloatParam::new(
                "Retune Speed",
                50.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 100.0,
                },
            )
            .with_unit(" %"),
            mix: FloatParam::new(
                "Mix",
                100.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 100.0,
                },
            )
            .with_unit(" %"),
            key: EnumParam::new("Key", Key::C),
            scale: EnumParam::new("Scale", Scale::Chromatic),
        }
    }
}

pub struct AutoTune {
    params: Arc<AutoTuneParams>,
    sample_rate: f32,

    // Input circular buffer
    input_ring: Vec<f32>,
    write_pos: usize,

    // Output overlap-add buffer
    output_ring: Vec<f32>,

    // Dry signal delay buffer (aligned with output latency)
    dry_ring: Vec<f32>,

    // Hop counter
    hop_counter: usize,
    total_samples: usize,

    // Phase vocoder state
    prev_input_phases: Vec<f64>,
    prev_output_phases: Vec<f64>,

    // Precomputed Hann window
    window: Vec<f64>,

    // FFT instances (created once in initialize)
    fft: Option<Arc<dyn rustfft::Fft<f64>>>,
    ifft: Option<Arc<dyn rustfft::Fft<f64>>>,

    // Preallocated work buffers (real-time safe - no allocations in process)
    fft_buffer: Vec<Complex<f64>>,
    ifft_buffer: Vec<Complex<f64>>,
    fft_scratch: Vec<Complex<f64>>,
    ana_magnitudes: Vec<f64>,
    ana_frequencies: Vec<f64>,
    syn_magnitudes: Vec<f64>,
    syn_frequencies: Vec<f64>,
    frame_buffer: Vec<f32>,
    yin_diff: Vec<f64>,
    yin_cmndf: Vec<f64>,

    // Smoothed pitch correction ratio
    current_ratio: f64,
}

impl Default for AutoTune {
    fn default() -> Self {
        let window: Vec<f64> = (0..WINDOW_SIZE)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / WINDOW_SIZE as f64).cos()))
            .collect();

        Self {
            params: Arc::new(AutoTuneParams::default()),
            sample_rate: 44100.0,
            input_ring: vec![0.0; RING_SIZE],
            write_pos: 0,
            output_ring: vec![0.0; RING_SIZE],
            dry_ring: vec![0.0; RING_SIZE],
            hop_counter: 0,
            total_samples: 0,
            prev_input_phases: vec![0.0; WINDOW_SIZE],
            prev_output_phases: vec![0.0; WINDOW_SIZE],
            window,
            fft: None,
            ifft: None,
            fft_buffer: vec![Complex::new(0.0, 0.0); WINDOW_SIZE],
            ifft_buffer: vec![Complex::new(0.0, 0.0); WINDOW_SIZE],
            fft_scratch: Vec::new(),
            ana_magnitudes: vec![0.0; WINDOW_SIZE],
            ana_frequencies: vec![0.0; WINDOW_SIZE],
            syn_magnitudes: vec![0.0; WINDOW_SIZE],
            syn_frequencies: vec![0.0; WINDOW_SIZE],
            frame_buffer: vec![0.0; WINDOW_SIZE],
            yin_diff: vec![0.0; YIN_HALF],
            yin_cmndf: vec![0.0; YIN_HALF],
            current_ratio: 1.0,
        }
    }
}

impl AutoTune {
    /// YIN pitch detection. Returns detected frequency in Hz or None.
    fn detect_pitch(&mut self) -> Option<f32> {
        let sr = self.sample_rate as f64;
        let threshold = 0.15;
        let half = YIN_HALF;

        // Step 1: Difference function
        for tau in 1..half {
            let mut sum = 0.0f64;
            for n in 0..half {
                let diff = self.frame_buffer[n] as f64 - self.frame_buffer[n + tau] as f64;
                sum += diff * diff;
            }
            self.yin_diff[tau] = sum;
        }

        // Step 2: Cumulative mean normalized difference function
        self.yin_cmndf[0] = 1.0;
        let mut running_sum = 0.0f64;
        for tau in 1..half {
            running_sum += self.yin_diff[tau];
            if running_sum > 0.0 {
                self.yin_cmndf[tau] = self.yin_diff[tau] * tau as f64 / running_sum;
            } else {
                self.yin_cmndf[tau] = 1.0;
            }
        }

        // Step 3: Absolute threshold - find first dip below threshold
        let min_tau = (sr / 1200.0).max(2.0) as usize;
        let max_tau = (sr / 55.0).min((half - 2) as f64) as usize;

        for tau in min_tau..max_tau {
            if self.yin_cmndf[tau] < threshold && self.yin_cmndf[tau] < self.yin_cmndf[tau - 1] {
                // Step 4: Parabolic interpolation for sub-sample accuracy
                let alpha = self.yin_cmndf[tau - 1];
                let beta = self.yin_cmndf[tau];
                let gamma = self.yin_cmndf[tau + 1];
                let denom = alpha - 2.0 * beta + gamma;
                if denom.abs() > 1e-12 {
                    let refined = tau as f64 + 0.5 * (alpha - gamma) / denom;
                    return Some((sr / refined) as f32);
                }
                return Some((sr / tau as f64) as f32);
            }
        }

        None
    }

    /// Snap frequency to the nearest note in the selected key/scale.
    fn snap_to_scale(&self, freq: f32) -> f32 {
        if freq <= 0.0 || freq > 5000.0 {
            return freq;
        }

        let key = self.params.key.value() as i32;
        let scale = match self.params.scale.value() {
            Scale::Chromatic => &CHROMATIC,
            Scale::Major => &MAJOR,
            Scale::Minor => &MINOR,
        };

        // Frequency to continuous MIDI note number
        let midi = 69.0 + 12.0 * (freq / 440.0).log2();
        let midi_int = midi.round() as i32;

        let mut best_note = midi_int;
        let mut best_dist = f32::MAX;

        for offset in -3..=3i32 {
            let candidate = midi_int + offset;
            let degree = ((candidate - key) % 12 + 12) % 12;
            if scale[degree as usize] {
                let dist = (midi - candidate as f32).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_note = candidate;
                }
            }
        }

        440.0 * 2.0f32.powf((best_note as f32 - 69.0) / 12.0)
    }

    /// Process one STFT frame: pitch detect -> compute correction -> phase vocoder shift.
    fn process_frame(&mut self) {
        let n = WINDOW_SIZE;
        let half = HALF_BINS;
        let hop = HOP_SIZE as f64;
        let sr = self.sample_rate as f64;
        let two_pi = 2.0 * PI;

        // 1. Extract windowed frame from input ring buffer
        let start = (self.write_pos + RING_SIZE - n) % RING_SIZE;
        for i in 0..n {
            self.frame_buffer[i] = self.input_ring[(start + i) % RING_SIZE];
        }

        // 2. Detect pitch and compute correction ratio
        if let Some(detected) = self.detect_pitch() {
            let target = self.snap_to_scale(detected);
            let target_ratio = target as f64 / detected as f64;

            // Smooth correction based on retune speed (0-100%)
            let speed = self.params.retune_speed.value() as f64 / 100.0;
            self.current_ratio += (target_ratio - self.current_ratio) * speed;
        } else {
            // No pitch detected: fade toward unity
            self.current_ratio += (1.0 - self.current_ratio) * 0.05;
        }

        let shift_ratio = self.current_ratio;

        // 3. Window and FFT
        for i in 0..n {
            self.fft_buffer[i] =
                Complex::new(self.frame_buffer[i] as f64 * self.window[i], 0.0);
        }

        if let Some(fft) = &self.fft {
            fft.process_with_scratch(&mut self.fft_buffer, &mut self.fft_scratch);
        }

        // 4. Analysis: magnitude + instantaneous frequency per bin
        let expected_per_bin = two_pi * hop / n as f64;

        for k in 0..half {
            let mag = self.fft_buffer[k].norm();
            let phase = self.fft_buffer[k].arg();

            let delta = phase - self.prev_input_phases[k];
            self.prev_input_phases[k] = phase;

            let expected = k as f64 * expected_per_bin;
            let deviation = wrap_phase(delta - expected);

            // True frequency in Hz for this bin
            self.ana_magnitudes[k] = mag;
            self.ana_frequencies[k] = (expected + deviation) * sr / (two_pi * hop);
        }

        // 5. Pitch shift: remap bins by shift_ratio
        for k in 0..half {
            self.syn_magnitudes[k] = 0.0;
            self.syn_frequencies[k] = k as f64 * sr / n as f64;
        }

        for k in 0..half {
            let new_k = (k as f64 * shift_ratio).round() as usize;
            if new_k < half && self.ana_magnitudes[k] > self.syn_magnitudes[new_k] {
                self.syn_magnitudes[new_k] = self.ana_magnitudes[k];
                self.syn_frequencies[new_k] = self.ana_frequencies[k] * shift_ratio;
            }
        }

        // 6. Synthesis: accumulate output phases and build spectrum
        for k in 0..half {
            let phase_advance = self.syn_frequencies[k] * two_pi * hop / sr;
            self.prev_output_phases[k] += phase_advance;

            self.ifft_buffer[k] =
                Complex::from_polar(self.syn_magnitudes[k], self.prev_output_phases[k]);
        }

        // Conjugate mirror for negative frequencies (real-valued signal)
        for k in 1..n / 2 {
            self.ifft_buffer[n - k] = self.ifft_buffer[k].conj();
        }
        // DC and Nyquist bins are real
        self.ifft_buffer[0] = Complex::new(self.ifft_buffer[0].re, 0.0);
        self.ifft_buffer[n / 2] = Complex::new(self.ifft_buffer[n / 2].re, 0.0);

        // 7. Inverse FFT
        if let Some(ifft) = &self.ifft {
            ifft.process_with_scratch(&mut self.ifft_buffer, &mut self.fft_scratch);
        }

        // 8. Window + normalize + overlap-add to output ring
        let inv_n = 1.0 / n as f64;
        let ola_gain = 2.0 / 3.0; // Normalization for Hann window, 75% overlap

        let out_start = (self.write_pos + RING_SIZE - n) % RING_SIZE;
        for i in 0..n {
            let val = self.ifft_buffer[i].re * inv_n * self.window[i] * ola_gain;
            let pos = (out_start + i) % RING_SIZE;
            self.output_ring[pos] += val as f32;
        }
    }
}

/// Wrap phase to [-PI, PI]
#[inline]
fn wrap_phase(phase: f64) -> f64 {
    phase - (phase / (2.0 * PI)).round() * 2.0 * PI
}

impl Plugin for AutoTune {
    const NAME: &'static str = "RT AutoTune";
    const VENDOR: &'static str = "RT Audio";
    const URL: &'static str = "";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        // Stereo
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        // Mono
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = buffer_config.sample_rate;
        context.set_latency_samples(WINDOW_SIZE as u32);

        // Set up FFT
        let mut planner = FftPlanner::new();
        self.fft = Some(planner.plan_fft_forward(WINDOW_SIZE));
        self.ifft = Some(planner.plan_fft_inverse(WINDOW_SIZE));

        let scratch_len = self
            .fft
            .as_ref()
            .unwrap()
            .get_inplace_scratch_len()
            .max(self.ifft.as_ref().unwrap().get_inplace_scratch_len());
        self.fft_scratch = vec![Complex::new(0.0, 0.0); scratch_len];

        // Reset state
        self.input_ring.fill(0.0);
        self.output_ring.fill(0.0);
        self.dry_ring.fill(0.0);
        self.prev_input_phases.fill(0.0);
        self.prev_output_phases.fill(0.0);
        self.write_pos = 0;
        self.hop_counter = 0;
        self.total_samples = 0;
        self.current_ratio = 1.0;

        true
    }

    fn reset(&mut self) {
        self.input_ring.fill(0.0);
        self.output_ring.fill(0.0);
        self.dry_ring.fill(0.0);
        self.prev_input_phases.fill(0.0);
        self.prev_output_phases.fill(0.0);
        self.write_pos = 0;
        self.hop_counter = 0;
        self.total_samples = 0;
        self.current_ratio = 1.0;
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        for mut channel_samples in buffer.iter_samples() {
            // Read input from first channel
            let input = *channel_samples.get_mut(0).unwrap();

            // Write to input ring and dry delay
            self.input_ring[self.write_pos] = input;
            self.dry_ring[self.write_pos] = input;

            self.write_pos = (self.write_pos + 1) % RING_SIZE;
            self.total_samples += 1;

            // Process a hop when ready
            self.hop_counter += 1;
            if self.hop_counter >= HOP_SIZE && self.total_samples >= WINDOW_SIZE {
                self.hop_counter = 0;
                self.process_frame();
            }

            // Read output (delayed by WINDOW_SIZE for latency alignment)
            let read_pos = (self.write_pos + RING_SIZE - WINDOW_SIZE) % RING_SIZE;
            let wet = self.output_ring[read_pos];
            self.output_ring[read_pos] = 0.0; // Clear for next overlap-add cycle
            let dry = self.dry_ring[read_pos];

            // Dry/wet mix
            let mix = self.params.mix.value() / 100.0;
            let output = dry * (1.0 - mix) + wet * mix;

            // Write output to all channels
            *channel_samples.get_mut(0).unwrap() = output;
            if let Some(right) = channel_samples.get_mut(1) {
                *right = output;
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for AutoTune {
    const CLAP_ID: &'static str = "com.rt-audio.autotune";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Real-time pitch correction");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::PitchShifter,
    ];
}

impl Vst3Plugin for AutoTune {
    const VST3_CLASS_ID: [u8; 16] = *b"RTAutoTune__0001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::PitchShift,
    ];
}

nih_export_clap!(AutoTune);
nih_export_vst3!(AutoTune);
