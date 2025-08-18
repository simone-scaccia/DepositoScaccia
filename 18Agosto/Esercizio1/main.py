#Check if the file exists
import os
if not os.path.exists('input.txt'):
    # Print an error message if the file does not exist
    print("Error: input.txt does not exist.")
    exit(1)

# Read input.txt
with open('input.txt', 'r') as file:
    lines = file.readlines()
