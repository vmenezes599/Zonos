"""Client script to send text-to-speech requests to a server."""

import argparse
from pathlib import Path
import requests


class Client:
    """Client class for sending text-to-speech requests to a server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8189):
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
    ) -> bytes | bool | None:
        """
        Synthesize speech from text using a reference audio file.

        Args:
            text: Text to synthesize
            reference_audio_path: Path to reference audio file
            seed: Random seed for generation
            output_path: Optional output path to save the audio file

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
        }

        try:
            # Open and send the audio file
            with open(reference_audio, "rb") as audio_file:
                files = {"reference_audio_file": audio_file}
                response = requests.post(
                    url, data=data, files=files, stream=True, timeout=self.default_timeout
                )

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
    parser = argparse.ArgumentParser(
        description="TTS Client - Send text-to-speech requests"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8189, help="Server port")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument(
        "--reference_audio_path",
        type=str,
        required=True,
        help="Path to reference audio file",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed for generation"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Output path for generated audio"
    )
    args = parser.parse_args()

    # Use the Client class for CLI functionality
    client = Client(host=args.host, port=args.port)
    success = client.synthesize(
        text=args.text,
        reference_audio_path=args.reference_audio_path,
        seed=args.seed,
        output_path=args.output_path,
    )

    if not success:
        exit(1)


if __name__ == "__main__":
    # python client.py --host 127.0.0.1 --port 8190 --reference_audio_path ./assets/exampleaudio.mp3 --text "Testing abcdefg^C--seed 12345
    main()
