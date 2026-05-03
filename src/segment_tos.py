import spacy
import json
import os
import argparse

nlp = spacy.load("en_core_web_sm")
MIN_WORDS = 5


def segment_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        sentence = sent.text.strip()
        # Keep only sentences with at least 5 words
        if len(sentence.split()) >= MIN_WORDS:
            sentences.append(sentence)
    return sentences


def main():
    parser = argparse.ArgumentParser(description="Segment ToS text files into JSONL format.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing .txt files")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output .jsonl file")

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, 'w', encoding='utf-8') as out:
        for filename in os.listdir(args.input_folder):
            if filename.endswith(".txt"):
                company = filename.replace(".txt", "")
                filepath = os.path.join(args.input_folder, filename)

                sentences = segment_file(filepath)
                print(f"Processed {company}: {len(sentences)} sentences found.")

                for sentence in sentences:
                    record = {
                        "company": company,
                        "text": sentence,
                        "labels": []  # Empty list for future labeling/prediction
                    }
                    out.write(json.dumps(record) + '\n')

    print(f"\nSegmentation complete. Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()