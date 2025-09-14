import os
import glob
import argparse
import numpy as np

def mot_to_comp_format(input_path, output_path=None):
    if os.path.isfile(input_path):
        txt_files = [input_path]
        if output_path is None:
            output_path = os.path.dirname(input_path)
    else:
        if output_path is None:
            output_path = input_path
        os.makedirs(output_path, exist_ok=True)
        txt_files = sorted(glob.glob(os.path.join(input_path, "*.txt")))

    for txt_file in txt_files:
        seq_name = os.path.basename(txt_file)
        data = np.loadtxt(txt_file, delimiter=",", dtype=float)

        if data.ndim == 1:
            data = data[None, :]

        fixed_rows = []
        for row in data:
            if len(row) == 10:
                # take first 7 cols, then append 1,1
                new_row = list(row[:7]) + [1, 1]
            elif len(row) == 9:
                # already 9 columns → just force last two to 1,1
                new_row = list(row[:7]) + [1, 1]
            else:
                raise ValueError(f"Unexpected column count {len(row)} in {txt_file}")
            fixed_rows.append(new_row)

        fixed_rows = np.array(fixed_rows, dtype=float)
        save_file = os.path.join(output_path, seq_name)
        np.savetxt(save_file, fixed_rows, fmt="%.6f", delimiter=",")

        print(f"✅ Saved competition-format file: {save_file}")


def make_parser():
    parser = argparse.ArgumentParser(description="Convert MOTChallenge (10-col) files to competition format (9-col).")
    parser.add_argument("--input", required=True, help="Path to input file or folder")
    parser.add_argument("--output", default=None, help="Path to output folder (default: overwrite or same dir)")
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    mot_to_comp_format(args.input, args.output)