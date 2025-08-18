import os
import re


def count_lines(lines):
    """Count the number of lines in the input."""
    return len(lines)


def count_words(lines):
    """Count the number of words in the input without punctuation."""
    word_count = 0
    for line in lines:
        # Remove punctuation and split into words
        words = re.findall(r"\b\w+\b", line)
        word_count += len(words)
    return word_count


def five_most_frequent_words(lines):
    """Find the five most frequent words in the input case-insensitively."""
    word_freq = {}
    for line in lines:
        # Remove punctuation and split into words
        words = re.findall(r"\b\w+\b", line.lower())
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    # Sort words by frequency and return the top 5
    sorted_words = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
    return sorted_words[:5]


def print_most_frequent_words(words, output_file=None):
    """Print the most frequent words as word: frequency."""
    output = []
    for word, freq in words:
        output_line = f"{word}: {freq}"
        output.append(output_line)
    return "\n".join(output)


# Check if the file exists
if not os.path.exists("input.txt"):
    # Print an error message if the file does not exist
    print("Error: input.txt does not exist.")
    exit(1)

# Read input.txt
with open("input.txt", "r") as file:
    lines = file.readlines()

num_lines = count_lines(lines)
num_words = count_words(lines)

print(f"Number of lines: {num_lines}")
print(f"Number of words: {num_words}")

most_frequent_words = five_most_frequent_words(lines)
print("Five most frequent words:")
print(print_most_frequent_words(most_frequent_words))

# Save the output to output.txt
with open("output.txt", "w") as file:
    file.write(f"Number of lines: {num_lines}\n")
    file.write(f"Number of words: {num_words}\n")
    file.write("Five most frequent words:\n")
    file.write(print_most_frequent_words(most_frequent_words))
    file.write("\n")
