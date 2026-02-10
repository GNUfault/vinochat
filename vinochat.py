import sys
import argparse
import openvino_genai as ov_genai

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)
    parser.add_argument('--device', type=str, default='GPU')
    parser.add_argument('--max-tokens', type=int, default=4096)
    args = parser.parse_args()

    device = args.device.upper()
    if ',' in device:
        device = f"MULTI:{device}"

    print("Loading... Use 'exit' to quit")
    
    try:
        pipe = ov_genai.LLMPipeline(args.model_dir, device)
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = args.max_tokens

        def streamer(token):
            print(token, end='', flush=True)
            return ov_genai.StreamingStatus.RUNNING

        while True:
            prompt = input("> ")
            if prompt.lower() in ['exit', 'quit']:
                break
            if not prompt.strip():
                continue

            print(">> ", end='')
            pipe.generate(prompt, config, streamer)
            print()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
