import os

def exact_line_dedup(
    input_files: list[os.PathLike],
    output_directory: os.PathLike,    
):
    counts = {}
    
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                h = hash(line)
                counts[h] = counts.get(h, 0) + 1

    for file in input_files:
        output_file = os.path.join(output_directory, os.path.basename(file))
        with open(file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                h = hash(line)
                if counts[h] == 1:
                    f_out.write(line)
    