"""Zonos TTS Server"""

import argparse
import io
import os
import logging

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

# Global flag to track if server is processing
is_processing = False

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring"""
    return {"status": "healthy", "processing": is_processing}


@app.get("/tts")
@app.post("/tts")
async def inference_sft(
    text: str = Form(),
    reference_audio_file: UploadFile = File(),
    seed: int = Form(),
):
    """Text-to-Speech Inference"""
    global is_processing

    # Check if already processing
    if is_processing:
        raise HTTPException(
            status_code=429,
            detail="Server is currently processing another request. Please try again later.",
        )

    # Set processing flag
    is_processing = True

    try:
        model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

        # Read the uploaded audio file
        audio_content = await reference_audio_file.read()
        audio_buffer = io.BytesIO(audio_content)

        logging.info(f"Received audio file: {reference_audio_file.filename}")

        wav, sampling_rate = torchaudio.load(audio_buffer)
        speaker = model.make_speaker_embedding(wav, sampling_rate)

        torch.manual_seed(seed)

        cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")
        conditioning = model.prepare_conditioning(cond_dict)

        codes = model.generate(conditioning)

        logging.info(f"Audio generation completed.")

        wavs = model.autoencoder.decode(codes).cpu()

        # Convert tensor to bytes for streaming
        audio_buffer = io.BytesIO()
        torchaudio.save(
            audio_buffer, wavs[0], model.autoencoder.sampling_rate, format="wav"
        )
        audio_buffer.seek(0)

        logging.info(f"Sending audio to client.")

        return StreamingResponse(
            io.BytesIO(audio_buffer.getvalue()),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"},
        )

    except Exception as e:
        logging.error(f"Error during TTS processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS processing failed: {str(e)}")

    finally:
        # Memory cleanup
        try:
            # Move tensors to CPU before deletion to free GPU memory
            if "codes" in locals() and hasattr(codes, "cpu"):
                codes = codes.cpu()
            if "wavs" in locals() and hasattr(wavs, "cpu"):
                wavs = wavs.cpu()
            if "wav" in locals() and hasattr(wav, "cpu"):
                wav = wav.cpu()
            if "speaker" in locals() and hasattr(speaker, "cpu"):
                speaker = speaker.cpu()

            # Delete model and clear references
            if "model" in locals():
                # Clear model from GPU memory if possible
                if hasattr(model, "cpu"):
                    model.cpu()
                del model
            if "codes" in locals():
                del codes
            if "wavs" in locals():
                del wavs
            if "wav" in locals():
                del wav
            if "speaker" in locals():
                del speaker
            if "conditioning" in locals():
                del conditioning
            if "cond_dict" in locals():
                del cond_dict

            # Force garbage collection
            import gc

            gc.collect()

            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                # Get memory info before cleanup
                allocated_before = torch.cuda.memory_allocated()
                cached_before = torch.cuda.memory_reserved()

                # Multiple cleanup passes for thorough memory release
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Force cleanup of all GPU streams
                torch.cuda.empty_cache()

                # Additional cleanup for persistent allocations
                if hasattr(torch.cuda, "reset_accumulated_memory_stats"):
                    torch.cuda.reset_accumulated_memory_stats()

                # Final cleanup pass
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

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

        # Always reset the processing flag
        is_processing = False
        logging.info(f"Audio generation ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=int, default=8189, help="port to run the server on"
    )
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
