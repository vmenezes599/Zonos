"""Simple TTS Processing Script for Subprocess Usage"""

import re
import argparse
import logging
import sys
import io
import random

import torch
import torchaudio

from zonos.conditioning import make_cond_dict
from zonos.model import Zonos
from zonos.utils import DEFAULT_DEVICE

# Audio volume configuration
GLOBAL_TARGET_RMS = 0.09


# Set up logger
logger = logging.getLogger("tts_processor")


def log_audio_stats(audio_chunks, label="Audio Chunks", sample_count=3):
    """
    Log audio statistics for debugging - samples a few chunks and logs their stats
    Also analyzes temporal decay within each chunk

    Args:
        audio_chunks: List of audio tensors
        label: Label for the log output
        sample_count: Number of random chunks to sample and log
    """
    pass

    if not audio_chunks:
        logger.info(f"{label}: No chunks to analyze")
        return

    logger.info(f"=== {label} Analysis ===")
    logger.info(f"Total chunks: {len(audio_chunks)}")

    # Sample random chunks or use all if fewer than sample_count
    sample_indices = random.sample(range(len(audio_chunks)), min(sample_count, len(audio_chunks)))

    for _, idx in enumerate(sample_indices):
        chunk = audio_chunks[idx]
        rms = torch.sqrt(torch.mean(chunk**2)).item()
        max_val = torch.max(torch.abs(chunk)).item()
        min_val = torch.min(chunk).item()
        max_abs = torch.max(torch.abs(chunk)).item()
        samples = chunk.shape[-1] if chunk.dim() > 1 else len(chunk)

        logger.info(
            "Chunk %d: RMS=%.6f, Max=%.6f, Min=%.6f, MaxAbs=%.6f, Samples=%d",
            idx + 1,
            rms,
            max_val,
            min_val,
            max_abs,
            samples,
        )

        # Analyze temporal decay within the chunk
        analyze_temporal_decay(chunk, f"Chunk {idx+1}")

    # Overall statistics
    all_rms = [torch.sqrt(torch.mean(chunk**2)).item() for chunk in audio_chunks]
    avg_rms = sum(all_rms) / len(all_rms)
    min_rms = min(all_rms)
    max_rms = max(all_rms)

    logger.info(f"Overall RMS - Avg: {avg_rms:.6f}, Min: {min_rms:.6f}, Max: {max_rms:.6f}")
    logger.info(f"=== End {label} Analysis ===")


def analyze_temporal_decay(audio_chunk, chunk_label):
    """
    Analyze volume decay over time within a single audio chunk

    Args:
        audio_chunk: Single audio tensor
        chunk_label: Label for logging
    """
    if audio_chunk.dim() > 1:
        # For stereo, take the first channel or average
        audio_data = audio_chunk[0] if audio_chunk.shape[0] > 1 else audio_chunk.squeeze()
    else:
        audio_data = audio_chunk

    total_samples = len(audio_data)
    num_segments = 5  # Divide into 5 time segments
    segment_size = total_samples // num_segments

    segment_rms = []
    segment_max = []

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size if i < num_segments - 1 else total_samples
        segment = audio_data[start_idx:end_idx]

        rms = torch.sqrt(torch.mean(segment**2)).item()
        max_val = torch.max(torch.abs(segment)).item()

        segment_rms.append(rms)
        segment_max.append(max_val)

        duration_start = start_idx / 24000  # Assuming 24kHz sample rate
        duration_end = end_idx / 24000

        logger.info(
            f"  {chunk_label} Segment {i + 1} ({duration_start:.2f}s-{duration_end:.2f}s): RMS={rms:.6f}, Max={max_val:.6f}",
        )  # Calculate decay metrics
    if len(segment_rms) > 1:
        rms_decay = (segment_rms[0] - segment_rms[-1]) / segment_rms[0] * 100  # Percentage decay
        max_decay = (segment_max[0] - segment_max[-1]) / segment_max[0] * 100

        logger.info(f"  {chunk_label} RMS Decay: {rms_decay:.1f}% (from {segment_rms[0]:.6f} to {segment_rms[-1]:.6f})")
        logger.info(f"  {chunk_label} Max Decay: {max_decay:.1f}% (from {segment_max[0]:.6f} to {segment_max[-1]:.6f})")


def load_model():
    """Load the TTS model"""
    return Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=str(DEFAULT_DEVICE))


def create_speaker_embedding(model: Zonos, audio_file: str):
    """Create speaker embedding from audio file"""
    with open(audio_file, "rb") as f:
        audio_content = f.read()

    audio_buffer = io.BytesIO(audio_content)
    wav, sample_rate = torchaudio.load(audio_buffer)
    return model.make_speaker_embedding(wav, sample_rate)


def split_text_into_chunks(text: str, max_words: int = 50):
    """Split text into manageable chunks while preserving original punctuation"""
    stripped_text = text.strip()
    if not stripped_text:
        return []

    # Split on sentence boundaries while preserving the original punctuation
    # This regex finds sentence endings (., !, ?) followed by whitespace or end of string
    sentence_pattern = r"([.!?])\s*"
    parts = re.split(sentence_pattern, stripped_text)

    # Reconstruct sentences with their original punctuation
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentence_text = parts[i].strip()
        if sentence_text:  # Skip empty parts
            punctuation = parts[i + 1] if i + 1 < len(parts) else ""
            sentences.append(sentence_text + punctuation)

    # Handle any remaining text without punctuation
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())

    chunks = []
    current_chunk = ""
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current_words + sentence_words > max_words and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_words = sentence_words
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_words += sentence_words

    if current_chunk:
        chunks.append(current_chunk.strip())

    prefixed_chunks = [chunk for chunk in chunks] if chunks else [stripped_text]

    return [chunk for chunk in prefixed_chunks if chunk.strip()]


def generate_audio_chunk(model: Zonos, text: str, **kwargs):
    """Generate audio for a single text chunk"""
    cond_dict = make_cond_dict(text=text, **kwargs)
    print(f"cond_dict: {cond_dict}")
    conditioning = model.prepare_conditioning(cond_dict)
    codes = model.generate(conditioning)
    audio = model.autoencoder.decode(codes).cpu()

    # Ensure 2D tensor
    if audio.dim() > 2:
        audio = audio[0]
    elif audio.dim() == 1:
        audio = audio.unsqueeze(0)

    return audio.float()


def apply_temporal_compensation(audio_chunk, chunk_label, compensation_strength=0.5):
    """
    Apply temporal compensation to counteract volume decay within a chunk

    Args:
        audio_chunk: Single audio tensor
        chunk_label: Label for logging
        compensation_strength: How much to compensate (0.0 = no compensation, 1.0 = full compensation)

    Returns:
        Compensated audio chunk
    """
    if audio_chunk.dim() > 1:
        # For stereo, process each channel
        channels = audio_chunk.shape[0]
        compensated_channels = []

        for ch in range(channels):
            channel_data = audio_chunk[ch]
            compensated_channel = _apply_compensation_to_channel(channel_data, chunk_label, compensation_strength)
            compensated_channels.append(compensated_channel)

        return torch.stack(compensated_channels, dim=0)
    else:
        return _apply_compensation_to_channel(audio_chunk, chunk_label, compensation_strength)


def _apply_compensation_to_channel(audio_data, chunk_label, compensation_strength):
    """Apply compensation to a single audio channel"""
    total_samples = len(audio_data)

    # Create a linear gain ramp that increases over time
    # Start at 1.0, end at (1.0 + compensation_strength)
    gain_start = 1.0
    gain_end = 1.0 + compensation_strength

    # Create linear ramp
    sample_indices = torch.arange(total_samples, dtype=torch.float32)
    gain_ramp = gain_start + (gain_end - gain_start) * (sample_indices / total_samples)

    # Apply the gain ramp
    compensated_audio = audio_data * gain_ramp

    # Log the compensation applied
    logger.info(
        f"  {chunk_label}: Applied temporal compensation ({gain_start:.1f}x to {gain_end:.1f}x)",
    )

    return compensated_audio


def normalize_audio_chunks(audio_chunks, apply_compensation=True, compensation_strength=0.4):
    """Normalize all chunks to consistent volume with optional temporal compensation"""
    if not audio_chunks:
        return audio_chunks

    # Log statistics before normalization
    # (audio_chunks, "BEFORE Normalization")

    # Apply temporal compensation first if enabled
    if apply_compensation:
        logger.info("Applying temporal compensation to counteract volume decay...")
        compensated_chunks = []
        for i, chunk in enumerate(audio_chunks):
            compensated_chunk = apply_temporal_compensation(chunk, f"Chunk {i+1}", compensation_strength)
            compensated_chunks.append(compensated_chunk)

        # Log statistics after compensation but before normalization
        # log_audio_stats(compensated_chunks, "AFTER Compensation (BEFORE Normalization)")
        audio_chunks = compensated_chunks

    logger.info("Normalizing to target RMS: %.6f", GLOBAL_TARGET_RMS)

    normalized_chunks = []

    for i, chunk in enumerate(audio_chunks):
        current_rms = torch.sqrt(torch.mean(chunk**2)).item()
        if current_rms > 0:
            scale_factor = GLOBAL_TARGET_RMS / current_rms
            normalized_chunk = chunk * scale_factor

            # Prevent clipping
            max_val = torch.max(torch.abs(normalized_chunk)).item()
            if max_val > 0.95:
                clip_factor = 0.95 / max_val
                normalized_chunk = normalized_chunk * clip_factor
                logger.info(
                    "Chunk %d: Applied clipping protection (factor: %.3f)",
                    i + 1,
                    clip_factor,
                )

            new_rms = torch.sqrt(torch.mean(normalized_chunk**2)).item()
            logger.info(
                "Chunk %d: RMS %.6f -> %.6f (scale: %.3f)",
                i + 1,
                current_rms,
                new_rms,
                scale_factor,
            )

            normalized_chunks.append(normalized_chunk)
        else:
            logger.warning("Chunk %d: Silent chunk detected", i + 1)
            normalized_chunks.append(chunk)

    # Log statistics after normalization
    # log_audio_stats(normalized_chunks, "AFTER Normalization")

    return normalized_chunks


def concatenate_chunks(audio_chunks, sample_rate):
    """Concatenate audio chunks with silence between them"""
    if len(audio_chunks) == 1:
        return audio_chunks[0]

    silence_samples = int(0.5 * sample_rate)
    silence = torch.zeros(audio_chunks[0].shape[0], silence_samples)

    result = []
    for i, chunk in enumerate(audio_chunks):
        result.append(chunk)
        if i < len(audio_chunks) - 1:
            result.append(silence)

    return torch.cat(result, dim=1)


def resolve_pitch_expressive(expressiveness: float):
    """Resolve pitch variation string to numerical value"""
    expressiveness = min(max(expressiveness, 0), 1)
    pitch_variation_min = 20
    pitch_variation_max = 100
    return pitch_variation_min + (pitch_variation_max - pitch_variation_min) * expressiveness


def resolve_speaking_rate(speaking_rate: float):
    """Resolve speaking rate string to numerical value"""
    speaking_rate = min(max(speaking_rate, 0), 1)
    speaking_rate_min = 10
    speaking_rate_max = 30
    return speaking_rate_min + (speaking_rate_max - speaking_rate_min) * speaking_rate


def process_tts(
    text: str,
    audio_file: str,
    output_file: str,
    seed: int,
    emotion: list[float],
    expressiveness: float,
    speaking_rate: float,
):
    """Main TTS processing function"""
    logger.info("Starting TTS processing")
    logger.info("Text length: %d characters", len(text))
    logger.info("Seed: %d", seed)

    # Split text and generate audio
    logger.info("Splitting text into chunks...")
    chunks = split_text_into_chunks(text)
    if not chunks:
        raise ValueError("No text provided for TTS processing (empty or whitespace-only input)")
    logger.info("Created %d text chunks", len(chunks))

    expressiveness = resolve_pitch_expressive(expressiveness)
    speaking_rate = resolve_speaking_rate(speaking_rate)

    # Load model and create speaker embedding only after validating text
    logger.info("Loading model...")
    torch.manual_seed(seed)
    model = load_model()
    logger.info("Creating speaker embedding...")
    speaker = create_speaker_embedding(model, audio_file)

    logger.info("Generating audio chunks...")
    audio_chunks = []

    model_kwargs = {
        "speaker": speaker,
        "emotion": emotion,
        "pitch_std": expressiveness,
        "speaking_rate": speaking_rate,
        "language": "en-us",
    }

    for i, chunk in enumerate(chunks):
        chunk_preview = chunk[:50] + ("..." if len(chunk) > 50 else "")
        logger.info(f"Processing chunk {i + 1}/{len(chunks)}: '{chunk_preview}'")
        audio = generate_audio_chunk(model, chunk, **model_kwargs)
        audio_chunks.append(audio)

    logger.info("Audio generation completed")

    # Normalize and concatenate
    logger.info("Starting audio normalization...")
    # Apply temporal compensation with moderate strength (0.4 = 40% compensation)
    normalized_chunks = normalize_audio_chunks(audio_chunks, apply_compensation=True, compensation_strength=0.4)

    logger.info("Concatenating audio chunks...")
    final_audio = concatenate_chunks(normalized_chunks, model.autoencoder.sampling_rate)

    # Log final audio stats
    final_rms = torch.sqrt(torch.mean(final_audio**2)).item()
    final_max = torch.max(torch.abs(final_audio)).item()
    final_duration = final_audio.shape[-1] / model.autoencoder.sampling_rate
    logger.info(
        "Final audio - RMS: %.6f, Max: %.6f, Duration: %.2fs",
        final_rms,
        final_max,
        final_duration,
    )

    # Save to file
    logger.info(f"Saving audio to: {output_file}")
    torchaudio.save(output_file, final_audio, model.autoencoder.sampling_rate, format="mp3")
    logger.info("TTS processing completed successfully")
    print(f"Audio saved to: {output_file}")


def main():
    """
    Main function for TTS processing
    """
    parser = argparse.ArgumentParser(description="Simple TTS Processing")
    parser.add_argument("--text", required=True, help="Text to convert to speech")
    parser.add_argument("--audio_file", required=True, help="Reference audio file")
    parser.add_argument("--output_file", required=True, help="Output audio file")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--happiness", type=float, required=True, help="Happiness emotion level (0.0 to 1.0)")
    parser.add_argument("--sadness", type=float, required=True, help="Sadness emotion level (0.0 to 1.0)")
    parser.add_argument("--disgust", type=float, required=True, help="Disgust emotion level (0.0 to 1.0)")
    parser.add_argument("--fear", type=float, required=True, help="Fear emotion level (0.0 to 1.0)")
    parser.add_argument("--surprise", type=float, required=True, help="Surprise emotion level (0.0 to 1.0)")
    parser.add_argument("--anger", type=float, required=True, help="Anger emotion level (0.0 to 1.0)")
    parser.add_argument("--other", type=float, required=True, help="Other emotion level (0.0 to 1.0)")
    parser.add_argument("--neutral", type=float, required=True, help="Neutral emotion level (0.0 to 1.0)")
    parser.add_argument("--expressiveness", type=float, required=True, help="Expressive level (0.0 to 1.0)")
    parser.add_argument("--speaking_rate", type=float, required=True, help="Speaking rate (0.0 to 1.0)")

    args = parser.parse_args()

    # Configure logging to output to both stdout and stderr
    # Info/debug to stdout, warnings/errors to stderr for better subprocess visibility
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.StreamHandler(sys.stderr),
        ],
    )

    try:
        emotion = [
            args.happiness,
            args.sadness,
            args.disgust,
            args.fear,
            args.surprise,
            args.anger,
            args.other,
            args.neutral,
        ]

        process_tts(
            args.text,
            args.audio_file,
            args.output_file,
            args.seed,
            emotion,
            args.expressiveness,
            args.speaking_rate,
        )
        sys.exit(0)
    except FileNotFoundError as e:
        error_msg = f"FileNotFoundError: {e}"
        logger.error(error_msg, exc_info=True)
        print(error_msg, file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        error_msg = f"ValueError: {e}"
        logger.error(error_msg, exc_info=True)
        print(error_msg, file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        # Catch CUDA errors, model loading errors, etc.
        error_msg = f"RuntimeError: {e}"
        logger.error(error_msg, exc_info=True)
        print(error_msg, file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        # Catch file I/O errors, permission errors, etc.
        error_msg = f"OSError: {e}"
        logger.error(error_msg, exc_info=True)
        print(error_msg, file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        error_msg = "Process interrupted by user"
        logger.warning(error_msg)
        print(error_msg, file=sys.stderr)
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:  # pylint: disable=broad-except
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(error_msg, exc_info=True)
        print(f"Unexpected error - {error_msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
