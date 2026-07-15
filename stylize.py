import sys
from arguments import parse_arguments

def main():
    # Pass live system CLI arguments straight into the custom parsing token scanner
    config = parse_arguments(sys.argv)
    
    print(f"Loaded config for style painting: {config['style_file']}")
    # ... Next Phase: Task B (Image VRAM Loading) will start here ...

if __name__ == "__main__":
    main()