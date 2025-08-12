"""Client script to send text-to-speech requests to a server."""

import argparse
import requests


def main():
    """Main function to send a request to the TTS server."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="8189")
    parser.add_argument("--text", type=str)
    parser.add_argument("--reference_audio_path", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/tts"

    # Prepare form data and files
    data = {
        "text": args.text,
        "seed": args.seed,
    }

    # Open and send the audio file
    with open(args.reference_audio_path, "rb") as audio_file:
        files = {"reference_audio_file": audio_file}
        response = requests.post(url, data=data, files=files, stream=True, timeout=10)

        if response.status_code == 200:
            with open(args.output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Audio saved to {args.output_path}")
        else:
            print(f"Error: {response.status_code} - {response.text}")


if __name__ == "__main__":
    main()
