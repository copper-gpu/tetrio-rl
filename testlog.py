from pathlib import Path
for path in sorted(Path('.').rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts:
        print(path)
