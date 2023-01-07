import os
from tqdm import tqdm


def generate_split_file_string(filenames, class_name):
    class_split = list(map(lambda x: os.path.join(class_name, x), filenames))
    return '\n'.join(class_split)


def generate_split_file(basepath, output_file_path):
    classes = os.listdir(basepath)
    output_string = ''
    for c in tqdm(classes):
        filenames = os.listdir(c)
        output_string += generate_split_file_string(filenames, c) + '\n'

    with open(output_file_path, 'w') as split_file:
        split_file.write(output_string)


if __name__ == '__main__':
    BASEPATH = 'split_test/'
    OUTPUT_PATH = 'test.txt'
    generate_split_file(BASEPATH, OUTPUT_PATH)
