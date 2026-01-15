import sys
from pathlib import Path

def count_words_in_folder(folder_path):
    total_words = 0

    for txt_file in Path(folder_path).glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8")
        words = text.split()
        total_words += len(words)

    return total_words


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    folder = sys.argv[1]

    total = count_words_in_folder(folder)
    print(f"Nombre total de mots dans le corpus : {total}")