import os
import numpy as np
import librosa

def end_padding(arr, pad, axis=0):
    padding = [(0, 0) for _ in range(axis)] + [(0, pad)]
    arr = np.pad(arr, padding, 'constant')
    return arr

def length_regularize(a, b):
    rank = len(a.shape)
    assert len(b.shape) == rank
    la = a.shape[-1]
    lb = b.shape[-1]
    if la > lb:
        b = end_padding(b, la - lb, rank - 1)
    elif la < lb:
        a = end_padding(a, lb - la, rank - 1)
    return a, b

def process(args):
    files = os.listdir(args.ipath)
    files.sort()
    bgm_files = [f for f in files if '伴奏' in f]
    song_files = [f for f in files if f not in bgm_files]
    if not os.path.exists(args.opath):
        os.makedirs(args.opath)
    for song_file, bgm_file in zip(song_files, bgm_files):
        song, rate = librosa.load(os.path.join(args.ipath, song_file), None, False)
        bgm, rate = librosa.load(os.path.join(args.ipath, bgm_file), None, False)
        song, bgm = length_regularize(song, bgm)
        vocal = song - bgm
        file_splits = os.path.splitext(song_file)
        vocal_file = file_splits[0] + ' (vocal)' + file_splits[1]
        vocal_file = os.path.join(args.opath, vocal_file)
        librosa.output.write_wav(vocal_file, vocal, rate)

def main(argv):
    import argparse
    argp = argparse.ArgumentParser()
    argp.add_argument('ipath')
    argp.add_argument('opath')
    args = argp.parse_args(argv)
    process(args)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
