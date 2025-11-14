"""Client script to send text-to-speech requests to a server."""

import argparse
from pathlib import Path
import requests


class Client:
    """Client class for sending text-to-speech requests to a server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9999):
        """
        Initialize the TTS client.

        Args:
            host: Server hostname or IP address
            port: Server port number
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.default_timeout = 300

    def synthesize(
        self,
        text: str,
        reference_audio_path: str | Path,
        seed: int,
        output_path: str | Path | None = None,
        happiness: float = 0.3077,
        sadness: float = 0.0256,
        disgust: float = 0.0256,
        fear: float = 0.0256,
        surprise: float = 0.0256,
        anger: float = 0.0256,
        other: float = 0.2564,
        neutral: float = 0.3077,
        expressiveness: float = 0.5,
        speaking_rate: float = 0.375,
    ) -> bytes | bool | None:
        """
        Synthesize speech from text using a reference audio file.

        Args:
            text: Text to synthesize
            reference_audio_path: Path to reference audio file
            seed: Random seed for generation
            output_path: Optional output path to save the audio file
            happiness: Happiness emotion level (0.0 to 1.0)
            sadness: Sadness emotion level (0.0 to 1.0)
            disgust: Disgust emotion level (0.0 to 1.0)
            fear: Fear emotion level (0.0 to 1.0)
            surprise: Surprise emotion level (0.0 to 1.0)
            anger: Anger emotion level (0.0 to 1.0)
            other: Other emotion level (0.0 to 1.0)
            neutral: Neutral emotion level (0.0 to 1.0)
            expressiveness: Expressive level (0.0 to 1.0)
            speaking_rate: Speaking rate (0.0 to 1.0)

        Returns:
            If output_path is provided, returns True on success, False on failure.
            If output_path is None, returns the audio bytes on success, None on failure.
        """
        # Validate paths using pathlib
        reference_audio = Path(reference_audio_path)

        if not reference_audio.exists():
            print(f"Error: Reference audio file not found: {reference_audio}")
            return False if output_path else None

        if not reference_audio.is_file():
            print(f"Error: Reference audio path is not a file: {reference_audio}")
            return False if output_path else None

        # Create output directory if output_path is provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        url = f"{self.base_url}/tts"

        # Prepare form data and files
        data = {
            "text": text,
            "seed": seed,
            "happiness": happiness,
            "sadness": sadness,
            "disgust": disgust,
            "fear": fear,
            "surprise": surprise,
            "anger": anger,
            "other": other,
            "neutral": neutral,
            "expressiveness": expressiveness,
            "speaking_rate": speaking_rate,
        }

        try:
            # Open and send the audio file
            with open(reference_audio, "rb") as audio_file:
                files = {"reference_audio_file": audio_file}
                response = requests.post(url, data=data, files=files, stream=True, timeout=self.default_timeout)

                if response.status_code == 200:
                    if output_path:
                        # Save to file
                        with open(output_path, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"Audio saved to {output_path}")
                        return True
                    else:
                        # Return audio bytes
                        return response.content
                else:
                    print(f"Error: {response.status_code} - {response.text}")
                    return False if output_path else None

        except (requests.RequestException, IOError) as e:
            print(f"Error during request: {e}")
            return False if output_path else None

    def test_connection(self) -> bool:
        """
        Test connection to the TTS server.

        Returns:
            True if server is reachable, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False


def main():
    """Main function to send a request to the TTS server via command line."""
    parser = argparse.ArgumentParser(description="TTS Client - Send text-to-speech requests")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8189, help="Server port")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument(
        "--reference_audio_path",
        type=str,
        required=True,
        help="Path to reference audio file",
    )
    parser.add_argument("--seed", type=int, required=True, help="Random seed for generation")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for generated audio")
    parser.add_argument("--happiness", type=float, default=0.3077, help="Happiness emotion level (0.0 to 1.0)")
    parser.add_argument("--sadness", type=float, default=0.0256, help="Sadness emotion level (0.0 to 1.0)")
    parser.add_argument("--disgust", type=float, default=0.0256, help="Disgust emotion level (0.0 to 1.0)")
    parser.add_argument("--fear", type=float, default=0.0256, help="Fear emotion level (0.0 to 1.0)")
    parser.add_argument("--surprise", type=float, default=0.0256, help="Surprise emotion level (0.0 to 1.0)")
    parser.add_argument("--anger", type=float, default=0.0256, help="Anger emotion level (0.0 to 1.0)")
    parser.add_argument("--other", type=float, default=0.2564, help="Other emotion level (0.0 to 1.0)")
    parser.add_argument("--neutral", type=float, default=0.3077, help="Neutral emotion level (0.0 to 1.0)")
    parser.add_argument("--expressiveness", type=float, default=0.5, help="Expressive level (0.0 to 1.0)")
    parser.add_argument("--speaking_rate", type=float, default=0.375, help="Speaking rate (0.0 to 1.0)")
    args = parser.parse_args()

    # Use the Client class for CLI functionality
    client = Client(host=args.host, port=args.port)
    success = client.synthesize(
        text=args.text,
        reference_audio_path=args.reference_audio_path,
        seed=args.seed,
        output_path=args.output_path,
        happiness=args.happiness,
        sadness=args.sadness,
        disgust=args.disgust,
        fear=args.fear,
        surprise=args.surprise,
        anger=args.anger,
        other=args.other,
        neutral=args.neutral,
        expressiveness=args.expressiveness,
        speaking_rate=args.speaking_rate,
    )

    if not success:
        exit(1)


if __name__ == "__main__":
    # python client.py --host 127.0.0.1 --port 8190 --reference_audio_path ./assets/exampleaudio.mp3 --text "Testing abcdefg^C--seed 12345
    main()
