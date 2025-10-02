import re

def clean_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    cleaned_lines = []
    
    for line in lines:
        # Repeatedly remove content within brackets until no more are found
        # (handles potential nested brackets)
        prev_line = ""
        while prev_line != line:
            prev_line = line
            line = re.sub(r'\([^()]*\)', '', line)  # Remove (...)
            line = re.sub(r'\[[^\[\]]*\]', '', line)  # Remove [...]
            line = re.sub(r'\{[^{}]*\}', '', line)  # Remove {...}
        
        # Remove standalone brackets that might be left
        line = re.sub(r'[\(\)\[\]\{\}]', '', line)
        
        # Remove colons and forward slashes
        line = line.replace(':', '')
        line = line.replace('/', '')
        
        # Strip whitespace
        line = line.strip()
        
        # Check if line has at least two words
        words = line.split()
        if len(words) >= 2:
            cleaned_lines.append(line)
    
    # Write cleaned lines to output file
    with open(output_file, 'w', encoding='utf-8') as file:
        for line in cleaned_lines:
            file.write(line + '\n')


# Usage
input_file = 'LazarillodeTormes.txt'
output_file = 'LazarillodeTormes_cleaned.txt'
clean_text(input_file, output_file)

print(f"Cleaned text has been written to {output_file}")