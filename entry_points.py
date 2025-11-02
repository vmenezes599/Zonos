import logging
import subprocess
import tempfile
import os
import sys

import asyncio
from fastapi import File, Form, UploadFile, HTTPException, BackgroundTasks, APIRouter
from fastapi.responses import FileResponse

router = APIRouter()


async def process_tts_with_subprocess(
    text: str,
    reference_audio_file: UploadFile,
    seed: int,
    background_tasks: BackgroundTasks,
):
    """Process TTS using subprocess for automatic resource cleanup"""

    # Create temporary files
    temp_audio_input = None
    temp_audio_output = None

    try:
        # Save uploaded audio to temp file
        temp_audio_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        audio_content = await reference_audio_file.read()
        temp_audio_input.write(audio_content)
        temp_audio_input.close()

        # Create temp output file
        temp_audio_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_audio_output.close()

        # Run TTS processor as subprocess
        # Use absolute path to ensure it works regardless of working directory
        tts_processor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tts_processor.py")
        cmd = [
            sys.executable,
            tts_processor_path,
            "--text",
            text,
            "--audio_file",
            temp_audio_input.name,
            "--output_file",
            temp_audio_output.name,
            "--seed",
            str(seed),
        ]

        logging.info(f"Running TTS subprocess: {' '.join(cmd)}")

        # Run subprocess without capturing output so logs appear in console
        # Only capture stderr to handle errors properly
        result = subprocess.run(
            cmd,
            stdout=None,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            logging.error(f"TTS subprocess failed: {result.stderr}")
            raise HTTPException(status_code=500, detail="TTS processing failed")

        # Log any stderr output even on success (warnings, etc.)
        if result.stderr:
            logging.info(f"TTS subprocess stderr: {result.stderr}")

        # Check if output file was created
        if not os.path.exists(temp_audio_output.name):
            raise HTTPException(status_code=500, detail="Output file not created")

        logging.info("TTS subprocess completed successfully")

        # Add cleanup task to background
        background_tasks.add_task(cleanup_temp_files, [temp_audio_input.name, temp_audio_output.name])

        # Return the audio file and clean up temp files automatically
        return FileResponse(
            temp_audio_output.name,
            media_type="audio/mpeg",
            filename="generated_audio.mp3",
        )

    except subprocess.TimeoutExpired:
        logging.error("TTS subprocess timed out")
        cleanup_temp_files(
            [
                temp_audio_input.name if temp_audio_input else None,
                temp_audio_output.name if temp_audio_output else None,
            ]
        )
        raise HTTPException(status_code=504, detail="TTS processing timed out")
    except Exception as e:
        logging.error(f"TTS processing error: {e}")
        cleanup_temp_files(
            [
                temp_audio_input.name if temp_audio_input else None,
                temp_audio_output.name if temp_audio_output else None,
            ]
        )
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


def cleanup_temp_files(file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logging.info(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logging.warning(f"Failed to clean up temp file {file_path}: {e}")


@router.get("/tts")
@router.post("/tts")
async def inference_sft(
    background_tasks: BackgroundTasks,
    text: str = Form(),
    reference_audio_file: UploadFile = File(),
    seed: int = Form(),
):
    """Text-to-Speech Inference endpoint using subprocess"""
    return await asyncio.wait_for(
        process_tts_with_subprocess(text, reference_audio_file, seed, background_tasks),
        timeout=310,
    )
