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
    happiness: float,
    sadness: float,
    disgust: float,
    fear: float,
    surprise: float,
    anger: float,
    other: float,
    neutral: float,
    expressiveness: float,
    speaking_rate: float,
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

        # Run TTS processor as subprocess using module syntax
        # This allows it to work with compiled .pyc files
        cmd = [
            sys.executable,
            "-m",
            "tts_processor",
            "--text",
            text,
            "--audio_file",
            temp_audio_input.name,
            "--output_file",
            temp_audio_output.name,
            "--seed",
            str(seed),
            "--happiness",
            str(happiness),
            "--sadness",
            str(sadness),
            "--disgust",
            str(disgust),
            "--fear",
            str(fear),
            "--surprise",
            str(surprise),
            "--anger",
            str(anger),
            "--other",
            str(other),
            "--neutral",
            str(neutral),
            "--expressiveness",
            str(expressiveness),
            "--speaking_rate",
            str(speaking_rate),
        ]

        logging.info(f"Running TTS subprocess: {' '.join(cmd)}")

        # Run subprocess asynchronously to avoid blocking the event loop
        # This allows health checks and other requests to be processed during TTS generation
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)  # 5 minute timeout
        except asyncio.TimeoutError as e:
            process.kill()
            await process.wait()
            raise subprocess.TimeoutExpired(cmd, 300) from e

        stdout_text = stdout.decode("utf-8") if stdout else ""
        stderr_text = stderr.decode("utf-8") if stderr else ""

        if stdout_text:
            logging.info(f"TTS subprocess stdout: {stdout_text}")

        if process.returncode != 0:
            error_detail = f"TTS subprocess failed with exit code {process.returncode}"
            if stderr_text:
                error_detail += f": {stderr_text.strip()}"
            if stdout_text:
                error_detail += f" | stdout: {stdout_text.strip()}"

            logging.error(error_detail)
            raise HTTPException(status_code=500, detail=error_detail)

        # Log any stderr output even on success (warnings, etc.)
        if stderr_text:
            logging.info(f"TTS subprocess stderr: {stderr_text}")

        # Check if output file was created
        if not os.path.exists(temp_audio_output.name):
            error_detail = f"Output file not created at {temp_audio_output.name}"
            if stderr_text:
                error_detail += f". Subprocess stderr: {stderr_text.strip()}"
            raise HTTPException(status_code=500, detail=error_detail)

        logging.info("TTS subprocess completed successfully")

        # Add cleanup task to background
        background_tasks.add_task(cleanup_temp_files, [temp_audio_input.name, temp_audio_output.name])

        # Return the audio file and clean up temp files automatically
        return FileResponse(
            temp_audio_output.name,
            media_type="audio/mpeg",
            filename="generated_audio.mp3",
        )

    except subprocess.TimeoutExpired as e:
        error_detail = f"TTS subprocess timed out after {e.timeout}s"
        logging.error(error_detail)
        cleanup_temp_files(
            [
                temp_audio_input.name if temp_audio_input else None,
                temp_audio_output.name if temp_audio_output else None,
            ]
        )
        raise HTTPException(status_code=504, detail=error_detail) from e
    except HTTPException:
        # Re-raise HTTPExceptions as-is (already have proper status codes and details)
        cleanup_temp_files(
            [
                temp_audio_input.name if temp_audio_input else None,
                temp_audio_output.name if temp_audio_output else None,
            ]
        )
        raise
    except Exception as e:
        error_detail = f"TTS processing failed: {type(e).__name__}: {str(e)}"
        logging.error(error_detail, exc_info=True)
        cleanup_temp_files(
            [
                temp_audio_input.name if temp_audio_input else None,
                temp_audio_output.name if temp_audio_output else None,
            ]
        )
        raise HTTPException(status_code=500, detail=error_detail) from e


def cleanup_temp_files(file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logging.info(f"Cleaned up temp file: {file_path}")
            except OSError as e:
                logging.warning(f"Failed to clean up temp file {file_path}: {e}")


@router.get("/tts")
@router.post("/tts")
async def inference_sft(
    background_tasks: BackgroundTasks,
    text: str = Form(),
    reference_audio_file: UploadFile = File(),
    seed: int = Form(),
    happiness: float = Form(),
    sadness: float = Form(),
    disgust: float = Form(),
    fear: float = Form(),
    surprise: float = Form(),
    anger: float = Form(),
    other: float = Form(),
    neutral: float = Form(),
    expressiveness: float = Form(),
    speaking_rate: float = Form(),
):
    """Text-to-Speech Inference endpoint using subprocess"""
    try:
        return await asyncio.wait_for(
            process_tts_with_subprocess(
                text,
                reference_audio_file,
                seed,
                happiness,
                sadness,
                disgust,
                fear,
                surprise,
                anger,
                other,
                neutral,
                expressiveness,
                speaking_rate,
                background_tasks,
            ),
            timeout=310,
        )
    except asyncio.TimeoutError as e:
        error_detail = "Request timeout: TTS processing took longer than 310 seconds"
        logging.error(error_detail)
        raise HTTPException(status_code=504, detail=error_detail) from e
