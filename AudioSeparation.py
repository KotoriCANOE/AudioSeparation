import tensorflow as tf
import numpy as np
import librosa
from scipy.io import wavfile
import os

def bool_argument(argp, name, default):
    argp.add_argument('--' + name, dest=name, action='store_true')
    argp.add_argument('--no-' + name, dest=name, action='store_false')
    eval('argp.set_defaults({}={})'.format(name, 'True' if default else 'False'))

# setup tensorflow and return session
def create_session(graph=None, debug=False):
    # create session
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options,
        allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(graph=graph, config=config)
    if debug:
        from tensorflow.python import debug as tfdbg
        sess = tfdbg.LocalCLIDebugWrapperSession(sess)
    return sess

class AudioSeparation:
    def __init__(self, model_file):
        self.rate = 49152
        self.base_size = 1024
        self.patch_width = 1048576
        self.overlap = 32768
        with tf.Graph().as_default() as g:
            self.graph = g
            self.load(model_file)
            self.sess = create_session()

    def load(self, model_file):
        with open(model_file, "rb") as fd:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fd.read())
        tf.import_graph_def(graph_def, name='')

    def inference(self, src):
        # padding
        length = src.shape[-1]
        padding = (length + self.base_size - 1) // self.base_size * self.base_size - length
        if padding:
            src = np.pad(src, ((0, 0), (0, 0), (0, 0), (0, padding)), 'constant')
        # run session
        fetch = 'Output:0'
        feed_dict = {'Input:0': src}
        dst = self.sess.run(fetch, feed_dict)
        # cropping
        if padding:
            dst = dst[:, :, :, :length]
        return dst
    
    def process_patches(self, audio):
        length = audio.shape[-1]
        max_step = self.patch_width - self.overlap * 2
        splits = (length - self.overlap * 2 + max_step - 1) // max_step
        step = (length - self.overlap * 2 + splits - 1) // splits
        step = step - step % 2
        overlap = (self.patch_width - step + 1) // 2
        # inference
        dsts = []
        for i in range(splits):
            start1 = step * i
            stop1 = None if i == splits - 1 else start1 + self.patch_width
            src = audio[:, :, :, start1:stop1]
            src_length = src.shape[-1]
            dst = self.inference(src)
            start2 = 0 if i == 0 else overlap
            stop2 = None if i == splits - 1 else src_length - overlap
            dst = dst[:, :, :, start2:stop2]
            dsts.append(dst)
        dst = np.concatenate(dsts, -1)
        # return
        return dst

    def process(self, audio):
        audio_origin = audio
        shape = audio.shape
        rank = len(shape)
        # input
        if rank == 1:
            audio = np.stack([audio, audio], 0)
        if rank <= 2:
            audio = np.expand_dims(audio, 0)
        if rank <= 3:
            audio = np.expand_dims(audio, -2)
        # split to patches
        sep0 = self.process_patches(audio)
        # output
        if rank == 1:
            sep0 = np.mean(sep0, -3)
        sep0 = np.reshape(sep0, shape)
        sep1 = audio_origin - sep0
        # return
        return sep0, sep1

    def __call__(self, ifile, ofile):
        if os.path.exists(ofile.format(index=0)):
            print('Output file exists: {}'.format(ofile.format(index=0)))
            return
        audio, rate = librosa.load(ifile, self.rate, False)
        seps = self.process(audio)
        opath = os.path.split(ofile)[0]
        if not os.path.exists(opath):
            os.makedirs(opath)
        for i, sep in enumerate(seps):
            audio_max = max(1.0, np.max(np.abs(sep)))
            sep *= 32767.0 / audio_max
            sep = sep.astype(np.int16)
            wavfile.write(ofile.format(index=i), rate, sep.T)

def run(args):
    model = AudioSeparation(args.model)
    # input
    args.input = os.path.abspath(args.input)
    if not os.path.exists(args.input):
        print('Input not exists: {}'.format(args.input))
        return
    elif os.path.isdir(args.input):
        if args.recursive:
            ifiles = []
            for dirpath, dirnames, filenames in os.walk(args.input):
                ifiles += [os.path.join(dirpath, f) for f in filenames]
        else:
            ifiles = os.listdir(args.input)
            ifiles = [os.path.join(args.input, f) for f in ifiles]
            ifiles = [f for f in ifiles if os.path.isfile(f)]
        filter_ext = ['.wav', '.flac', '.m4a', '.mp3', '.ogg']
        ifiles = [f for f in ifiles if os.path.splitext(f)[1] in filter_ext]
    else:
        ifiles = [os.path.abspath(args.input)]
    # output
    if args.output is None:
        ofiles = [os.path.splitext(f) + '.sep{index}.wav' for f in ifiles]
    elif os.path.isdir(args.input):
        ofiles = [f.replace(args.input, args.output) for f in ifiles]
        ofiles = [os.path.splitext(f)[0] + '.sep{index}.wav' for f in ofiles]
    else:
        ofiles = [os.path.splitext(os.path.split(f)[1])[0] for f in ifiles]
        ofiles = [os.path.join(args.output, f + '.sep{index}.wav') for f in ofiles]
    # process
    for i, o in zip(ifiles, ofiles):
        model(i, o)

def main(argv=None):
    # arguments parsing
    import argparse
    argp = argparse.ArgumentParser()
    # parameters
    argp.add_argument('model')
    argp.add_argument('-i', '--input')
    argp.add_argument('-o', '--output')
    bool_argument(argp, 'recursive', True)
    # parse
    args = argp.parse_args(argv)
    # run
    run(args)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
