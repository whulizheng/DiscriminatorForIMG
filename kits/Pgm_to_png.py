
from PIL import Image
import os
import glob
from tqdm import tqdm


def batch_image(in_dir, out_dir, tag, tech):
    if not os.path.exists(out_dir):
        print(out_dir, 'is not existed.')
        os.mkdir(out_dir)

    if not os.path.exists(in_dir):
        print(in_dir, 'is not existed.')
        return -1
    for files in tqdm(glob.glob(in_dir+'/*')):
        filepath, filename = os.path.split(files)
        index = filename.strip(".pgm")
        out_file = index + "_"+tech+"_"+tag + '.png'
        # print(filepath,',',filename, ',', out_file)
        im = Image.open(files)
        im.save(os.path.join(out_dir, out_file))


def batch_rename(in_dir, out_dir, tag, tech):
    if not os.path.exists(out_dir):
        print(out_dir, 'is not existed.')
        os.mkdir(out_dir)

    if not os.path.exists(in_dir):
        print(in_dir, 'is not existed.')
        return -1
    count = 1
    for files in tqdm(glob.glob(in_dir+'/*')):
        filepath, filename = os.path.split(files)
        out_file = str(count) + "_"+tech+"_"+tag + '.png'
        # print(filepath,',',filename, ',', out_file)
        im = Image.open(files)
        im.save(os.path.join(out_dir, out_file))
        count += 1


if __name__ == '__main__':
    batch_rename(r'C:\Users\WhuLi\Documents\InformationHidingTasks\w',
                 r'C:\Users\WhuLi\Documents\InformationHidingTasks\wave', "1", "w")
