import openvino_genai as ov_genai
import sys

def main():
    model_path = "gemma-3-4b-ov"
    device = "CPU"
    
    for arg in sys.argv:
        if arg.startswith("--device="):
            device = arg.split("=")[1].upper()

    pipe = ov_genai.VLMPipeline(model_path, device)
    
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 512

    def streamer(subword):
        print(subword, end='', flush=True)
        return False

    print("Loading... Use 'exit' to quit")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'exit':
            break
            
        print(">> ", end='')
        pipe.generate(user_input, generation_config=config, streamer=streamer)
        print() 

if __name__ == "__main__":
    main()
