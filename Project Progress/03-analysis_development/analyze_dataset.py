# analyze_dataset.py

import csv
from collections import Counter

def main():
    with open("dataset_metadata.csv", newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    print(f"Total repositories: {len(data)}")

    topics = Counter()
    languages = Counter()

    for row in data:
        for topic in row["topics"].split(","):
            if topic.strip():
                topics[topic.strip()] += 1

        for language in row["languages"].split(","):
            if language.strip():
                languages[language.strip()] += 1

    print("\nTop Languages:")
    for lang, count in languages.most_common(10):
        print(f"{lang}: {count}")

    print("\nTop Topics:")
    for topic, count in topics.most_common(10):
        print(f"{topic}: {count}")


if __name__ == "__main__":
    main()
