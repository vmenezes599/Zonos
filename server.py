"""Zonos TTS Server"""

import argparse
import io
import os
import logging
import tempfile
import re
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

# Fix Triton cache directory permission issue
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"

# Configure logging to output to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # This ensures output to console
)

from os import getenv
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# Pre-compiled regex patterns for better performance
WHITESPACE_PATTERN = re.compile(r"\s+")
SENTENCE_PATTERN = re.compile(r"[.!?]+")
CLAUSE_PATTERN = re.compile(r"[,;]|\s+(?:and|but|or|yet|so|for|nor)\s+")

# Processing semaphore to allow only one request at a time
_processing_semaphore = asyncio.Semaphore(1)  # Allow only one request at a time


def load_model() -> Zonos:
    """
    Load a fresh model instance for each request.
    This ensures memory is freed after each request.
    """
    logging.info("Loading TTS model...")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    logging.info("TTS model loaded successfully")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app startup and shutdown."""
    # Startup: No need to pre-load model anymore - models are loaded per request
    yield
    # Shutdown: Clear any remaining GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def split_text_into_chunks(text: str, max_words: int = 60) -> list[str]:
    """
    Split text into chunks of maximum specified words while preserving full phrases.
    Prioritizes complete sentences, then clauses, then reduces word count if needed.

    Args:
        text: The input text to split
        max_words: Maximum number of words per chunk

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    # Clean up the text first
    text = re.sub(r"\s+", " ", text.strip())  # Normalize whitespace

    # First try to split by sentences
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text] if text.strip() else []

    chunks = []
    current_chunk = ""
    current_word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)

        # If this single sentence exceeds max_words, try to split by clauses
        if sentence_word_count > max_words:
            # If we have a current chunk, save it first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_word_count = 0

            # Split by clauses (commas, semicolons, conjunctions)
            clauses = re.split(r"[,;]|\s+(?:and|but|or|yet|so|for|nor)\s+", sentence)
            clauses = [c.strip() for c in clauses if c.strip()]

            if not clauses:
                # If no clauses found, force split by words as last resort
                words = sentence.split()
                for i in range(0, len(words), max_words):
                    chunk_words = words[i : i + max_words]
                    chunks.append(" ".join(chunk_words))
                continue

            clause_chunk = ""
            clause_word_count = 0

            for clause in clauses:
                clause_words = clause.split()
                clause_word_count_current = len(clause_words)

                # If even a single clause is too long, force split by words
                if clause_word_count_current > max_words:
                    if clause_chunk:
                        chunks.append(clause_chunk.strip())
                        clause_chunk = ""
                        clause_word_count = 0

                    # Split this clause by words
                    words = clause.split()
                    for i in range(0, len(words), max_words):
                        chunk_words = words[i : i + max_words]
                        chunks.append(" ".join(chunk_words))
                    continue

                if (
                    clause_word_count + clause_word_count_current > max_words
                    and clause_chunk
                ):
                    chunks.append(clause_chunk.strip())
                    clause_chunk = clause
                    clause_word_count = clause_word_count_current
                else:
                    if clause_chunk:
                        clause_chunk += ", " + clause
                    else:
                        clause_chunk = clause
                    clause_word_count += clause_word_count_current

            if clause_chunk:
                chunks.append(clause_chunk.strip())

        else:
            # If adding this sentence would exceed the limit, start a new chunk
            if current_word_count + sentence_word_count > max_words and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_word_count = sentence_word_count
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
                current_word_count += sentence_word_count

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]

    return chunks if chunks else [text.strip()]


def validate_inputs(text: str, audio_content: bytes) -> None:
    """
    Validate input parameters for TTS processing.

    Args:
        text: Input text to validate
        audio_content: Audio file content to validate

    Raises:
        HTTPException: If validation fails
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(text.strip()) < 3:
        raise HTTPException(
            status_code=400, detail="Text must be at least 3 characters long"
        )

    if len(audio_content) == 0:
        raise HTTPException(status_code=400, detail="Audio file cannot be empty")

    if len(audio_content) < 1000:  # Very small audio files are likely invalid
        raise HTTPException(
            status_code=400, detail="Audio file appears to be too small or invalid"
        )


def load_model_and_speaker(audio_content: bytes, filename: str) -> tuple:
    """
    Load the TTS model and create speaker embedding from audio.

    Args:
        audio_content: Raw audio file content
        filename: Name of the audio file for logging

    Returns:
        Tuple of (model, speaker_embedding, sampling_rate)
    """
    logging.info(f"Processing audio file: {filename}")

    model = load_model()

    audio_buffer = io.BytesIO(audio_content)
    wav, sampling_rate = torchaudio.load(audio_buffer)
    speaker = model.make_speaker_embedding(wav, sampling_rate)

    return model, speaker, sampling_rate


def process_single_chunk(model, text: str, speaker, seed: int) -> torch.Tensor:
    """
    Process a single text chunk through the TTS model.

    Args:
        model: The loaded TTS model
        text: Text to process
        speaker: Speaker embedding
        seed: Random seed for generation

    Returns:
        Generated audio tensor
    """
    logging.info("Single chunk detected, using optimized single-pass processing")
    torch.manual_seed(seed)

    cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")
    conditioning = model.prepare_conditioning(cond_dict)
    codes = model.generate(conditioning)
    final_audio = model.autoencoder.decode(codes).cpu()

    # Validate tensor dimensions before accessing
    if final_audio.dim() == 0 or final_audio.shape[0] == 0:
        raise ValueError("Empty tensor generated for single chunk")

    # Remove batch dimension if needed, but keep it 2D (channels, samples)
    if final_audio.dim() > 2:
        # If more than 2D, take first batch item
        final_audio = final_audio[0]
    elif final_audio.dim() == 1:
        # If 1D, add channel dimension
        final_audio = final_audio.unsqueeze(0)
    # If already 2D, keep as is

    # Clean up
    del codes, conditioning, cond_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_audio


def process_multiple_chunks(
    model, text_chunks: list[str], speaker, seed: int
) -> torch.Tensor:
    """
    Process multiple text chunks through the TTS model and concatenate them.

    Args:
        model: The loaded TTS model
        text_chunks: List of text chunks to process
        speaker: Speaker embedding
        seed: Base random seed for generation

    Returns:
        Concatenated audio tensor
    """
    logging.info("Multi-chunk processing with optimized conditioning")

    # Validate chunks are not empty
    valid_chunks = [chunk.strip() for chunk in text_chunks if chunk.strip()]
    if not valid_chunks:
        raise ValueError("No valid text chunks found after filtering empty chunks")

    # Pre-create base conditioning template to reuse speaker embedding and language settings

    # Process each chunk and collect audio tensors
    audio_tensors = []

    for i, chunk in enumerate(valid_chunks):
        logging.info(
            f"Processing chunk {i+1}/{len(valid_chunks)} - Progress: {((i+1)/len(valid_chunks)*100):.1f}%"
        )

        # Set seed for reproducibility with slight variation for each chunk
        torch.manual_seed(
            seed + i * 42
        )  # Use larger increment for better seed separation

        try:
            # Create conditioning efficiently by reusing base structure
            chunk_cond_dict = make_cond_dict(
                text=chunk, speaker=speaker, language="en-us"
            )
            conditioning = model.prepare_conditioning(chunk_cond_dict)

            # Generate audio for this chunk
            codes = model.generate(conditioning)
            chunk_wav = model.autoencoder.decode(codes).cpu()

            # Validate tensor dimensions before accessing
            if chunk_wav.dim() == 0 or chunk_wav.shape[0] == 0:
                logging.warning(f"Empty tensor generated for chunk {i+1}, skipping")
                continue

            # Store the audio tensor for this chunk, ensuring 2D format (channels, samples)
            if chunk_wav.dim() > 2:
                # If more than 2D, take first batch item
                processed_chunk = chunk_wav[0]
            elif chunk_wav.dim() == 1:
                # If 1D, add channel dimension
                processed_chunk = chunk_wav.unsqueeze(0)
            else:
                # Already 2D, keep as is
                processed_chunk = chunk_wav

            audio_tensors.append(processed_chunk)

            # Clean up intermediate variables to save memory
            del codes, conditioning, chunk_cond_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error processing chunk {i+1}: {str(e)}")
            # Continue with next chunk instead of failing completely
            continue

    if not audio_tensors:
        raise ValueError("No audio tensors were successfully generated from any chunks")

    logging.info(f"Audio generation completed for {len(audio_tensors)} chunks.")

    # Concatenate all audio tensors with silence between them
    final_audio = concatenate_audio_tensors(
        audio_tensors, model.autoencoder.sampling_rate
    )

    # Clean up audio tensors list
    del audio_tensors

    return final_audio


def create_audio_response(
    final_audio: torch.Tensor, sampling_rate: int
) -> StreamingResponse:
    """
    Convert audio tensor to streaming response.

    Args:
        final_audio: Generated audio tensor (should be 2D: channels x samples)
        sampling_rate: Audio sampling rate

    Returns:
        StreamingResponse with audio data
    """
    # Validate that tensor has the correct dimensions for torchaudio.save (2D: channels x samples)
    if final_audio.dim() != 2:
        raise ValueError(
            f"Expected 2D tensor for audio output, got {final_audio.dim()}D tensor with shape {final_audio.shape}"
        )

    if final_audio.numel() == 0:
        raise ValueError("Cannot save empty audio tensor")

    # Convert tensor to bytes for streaming
    audio_buffer = io.BytesIO()
    torchaudio.save(
        audio_buffer,
        final_audio,  # Should already be 2D (channels, samples)
        sampling_rate,
        format="wav",
    )
    audio_buffer.seek(0)

    logging.info(f"Sending concatenated audio to client.")

    return StreamingResponse(
        io.BytesIO(audio_buffer.getvalue()),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


def cleanup_memory(local_vars: dict) -> None:
    """
    Comprehensive memory cleanup for GPU and CPU resources.

    Args:
        local_vars: Dictionary of local variables to clean up
    """
    try:
        # Move tensors to CPU before deletion to free GPU memory
        variables_to_cleanup = [
            "codes",
            "final_audio",
            "wav",
            "speaker",
            "conditioning",
            "cond_dict",
            "chunk_cond_dict",
            "base_cond_dict",
        ]

        for var_name in variables_to_cleanup:
            if var_name in local_vars and local_vars[var_name] is not None:
                var_obj = local_vars[var_name]
                if hasattr(var_obj, "cpu"):
                    var_obj = var_obj.cpu()

        # Handle audio_tensors list separately
        if "audio_tensors" in local_vars and local_vars["audio_tensors"] is not None:
            for tensor in local_vars["audio_tensors"]:
                if hasattr(tensor, "cpu"):
                    tensor = tensor.cpu()

        # Delete model and clear references
        if "model" in local_vars and local_vars["model"] is not None:
            # Clear model from GPU memory if possible
            if hasattr(local_vars["model"], "cpu"):
                local_vars["model"].cpu()
            del local_vars["model"]

        # Delete all variables
        for var_name in variables_to_cleanup:
            if var_name in local_vars:
                del local_vars[var_name]

        if "audio_tensors" in local_vars:
            del local_vars["audio_tensors"]

        # Force garbage collection
        import gc

        gc.collect()

        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            # Get memory info before cleanup
            allocated_before = torch.cuda.memory_allocated()
            cached_before = torch.cuda.memory_reserved()

            # Multiple cleanup passes for thorough memory release
            for _ in range(3):  # Multiple passes for thorough cleanup
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Additional cleanup for persistent allocations
            if hasattr(torch.cuda, "reset_accumulated_memory_stats"):
                torch.cuda.reset_accumulated_memory_stats()

            # Get memory info after cleanup
            allocated_after = torch.cuda.memory_allocated()
            cached_after = torch.cuda.memory_reserved()

            memory_freed = (allocated_before - allocated_after) / 1024**2  # MB
            cache_freed = (cached_before - cached_after) / 1024**2  # MB

            logging.info(
                f"GPU memory freed: {memory_freed:.1f}MB allocated, {cache_freed:.1f}MB cached"
            )

        logging.info(f"Memory cleanup completed.")

    except Exception as cleanup_error:
        logging.warning(f"Error during memory cleanup: {cleanup_error}")


def concatenate_audio_tensors(audio_tensors: list, sampling_rate: int) -> torch.Tensor:
    """
    Concatenate multiple audio tensors into a single tensor with smooth silence between chunks.

    Args:
        audio_tensors: List of audio tensors to concatenate
        sampling_rate: The sampling rate of the audio

    Returns:
        Concatenated audio tensor
    """
    if not audio_tensors:
        return torch.empty(0)

    if len(audio_tensors) == 1:
        # Single tensor, should already be 2D
        return audio_tensors[0]

    # Create 0.5 second silence tensor with smooth fade
    silence_duration = 0.5  # seconds
    silence_samples = int(sampling_rate * silence_duration)
    fade_samples = int(sampling_rate * 0.05)  # 50ms fade for smoother transitions

    # All tensors should already be 2D (channels, samples), but let's verify
    processed_tensors = []
    for i, tensor in enumerate(audio_tensors):
        if tensor.dim() != 2:
            raise ValueError(
                f"Expected 2D tensor for audio chunk {i}, got {tensor.dim()}D"
            )
        processed_tensors.append(tensor)

    # Get the number of channels from the first tensor
    num_channels = processed_tensors[0].shape[0]
    silence_tensor = torch.zeros(num_channels, silence_samples)

    # Create fade-out and fade-in windows for smoother transitions
    fade_out = (
        torch.linspace(1.0, 0.0, fade_samples).unsqueeze(0).expand(num_channels, -1)
    )
    fade_in = (
        torch.linspace(0.0, 1.0, fade_samples).unsqueeze(0).expand(num_channels, -1)
    )

    # Concatenate tensors with silence and fades between them
    result_tensors = []
    for i, tensor in enumerate(processed_tensors):
        # Apply fade-out to the end of each chunk (except the last one)
        if i < len(processed_tensors) - 1 and tensor.shape[1] >= fade_samples:
            tensor = tensor.clone()  # Don't modify original
            tensor[:, -fade_samples:] *= fade_out

        result_tensors.append(tensor)

        # Add silence between chunks (but not after the last chunk)
        if i < len(processed_tensors) - 1:
            result_tensors.append(silence_tensor)

    # Concatenate along the time dimension (dim=1)
    concatenated = torch.cat(result_tensors, dim=1)

    # Apply fade-in to the beginning of each chunk after silence
    current_pos = 0
    for i, tensor in enumerate(processed_tensors):
        if i > 0:  # Skip first chunk
            # Move past previous chunk and silence
            current_pos += processed_tensors[i - 1].shape[1] + silence_samples
            # Apply fade-in if there's enough samples
            if current_pos + fade_samples <= concatenated.shape[1]:
                fade_in_section = concatenated[
                    :, current_pos : current_pos + fade_samples
                ].clone()
                concatenated[:, current_pos : current_pos + fade_samples] = (
                    fade_in_section * fade_in
                )
        else:
            current_pos += tensor.shape[1]

    return concatenated


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring"""
    # Check if processing by trying to acquire semaphore without blocking
    is_busy = _processing_semaphore.locked()
    return {"status": "healthy", "processing": is_busy}


@app.get("/tts")
@app.post("/tts")
async def inference_sft(
    text: str = Form(),
    reference_audio_file: UploadFile = File(),
    seed: int = Form(),
):
    """Text-to-Speech Inference with text chunking and concatenation"""

    # Use semaphore for proper async concurrency control
    async with _processing_semaphore:
        # Store local variables for cleanup
        local_vars = {}

        try:
            # Read the uploaded audio file
            audio_content = await reference_audio_file.read()

            # Validate inputs
            validate_inputs(text, audio_content)

            # Load model and create speaker embedding
            model, speaker, sampling_rate = load_model_and_speaker(
                audio_content, reference_audio_file.filename
            )
            local_vars.update({"model": model, "speaker": speaker})

            # Split text into chunks
            text_chunks = split_text_into_chunks(text.strip(), max_words=60)

            if not text_chunks:
                raise HTTPException(
                    status_code=400, detail="Failed to process text into chunks"
                )

            logging.info(f"Split text into {len(text_chunks)} chunks")

            # Log the chunks for debugging
            for i, chunk in enumerate(text_chunks):
                word_count = len(chunk.split())
                logging.info(
                    f"Chunk {i+1}: {word_count} words - '{chunk[:100]}{'...' if len(chunk) > 100 else ''}'"
                )

            # Process chunks based on count
            if len(text_chunks) == 1:
                final_audio = process_single_chunk(model, text_chunks[0], speaker, seed)
            else:
                final_audio = process_multiple_chunks(model, text_chunks, speaker, seed)

            local_vars["final_audio"] = final_audio

            # Create and return audio response
            return create_audio_response(final_audio, model.autoencoder.sampling_rate)

        except HTTPException:
            # Re-raise HTTP exceptions without modification
            raise
        except Exception as e:
            logging.error(f"Error during TTS processing: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"TTS processing failed: {str(e)}"
            ) from e

        finally:
            # Cleanup memory and model
            cleanup_memory(local_vars)
            logging.info(f"Audio generation ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=int, default=8189, help="port to run the server on"
    )
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
