from src.app import FluxApp
from src.config import Config

def main():
    app = FluxApp()
    app.launch(share=True)

if __name__ == "__main__":
    main() 