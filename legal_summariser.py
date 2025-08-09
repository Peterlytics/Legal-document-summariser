import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from legal_summariser.cli import main
if __name__ == "__main__":
    main()
