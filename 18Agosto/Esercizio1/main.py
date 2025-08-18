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
        words = re.findall(r'\b\w+\b', line)
        word_count += len(words)
    return word_count

#Check if the file exists
if not os.path.exists('input.txt'):
    # Print an error message if the file does not exist
    print("Error: input.txt does not exist.")
    exit(1)

# Read input.txt
with open('input.txt', 'r') as file:
    lines = file.readlines()

num_lines = count_lines(lines)
num_words = count_words(lines)

print(f"Number of lines: {num_lines}")
print(f"Number of words: {num_words}")