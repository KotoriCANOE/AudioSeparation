import numpy as np
import os
from utils import eprint, reset_random
from data import DataSong as Data

class DataPack:
    def __init__(self, config):
        self.output_dir = None
        self.random_seed = None
        self.log_frequency = None
        self.batch_size = None
        self.test = None
        # copy all the properties from config object
        self.config = config
        self.__dict__.update(config.__dict__)

    def initialize(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # set deterministic random seed
        if self.random_seed is not None:
            reset_random(self.random_seed)

    def get_dataset(self):
        self.data = Data(self.config)
        self.epoch_steps = self.data.epoch_steps
        self.max_steps = self.data.max_steps

    def save_batches(self):
        import time
        time_last = time.time()
        data_gen = None
        for step in range(self.max_steps):
            ofile = os.path.join(self.output_dir, 'batch_{:0>8}.npz'.format(step))
            # create data generator from the last existing batch
            if data_gen is None:
                if os.path.exists(ofile):
                    continue
                data_gen = self.data.gen_main(step)
            # generate data
            inputs, labels = next(data_gen)
            # save to output file
            with open(ofile, 'wb') as fd:
                np.savez_compressed(fd, inputs=inputs, labels=labels)
            # logging
            if step % self.log_frequency == 0:
                time_current = time.time()
                duration = time_current - time_last
                time_last = time_current
                epoch = step // self.epoch_steps
                sec_batch = duration / self.log_frequency
                samples_sec = self.batch_size / sec_batch
                data_log = ('epoch {}, step {} ({:.1f} samples/sec, {:.3f} sec/batch)'
                    .format(epoch, step, samples_sec, sec_batch))
                eprint(data_log)

    def __call__(self):
        self.initialize()
        self.get_dataset()
        self.save_batches()

def main(argv=None):
    # arguments parsing
    import argparse
    argp = argparse.ArgumentParser()
    # parameters
    argp.add_argument('dataset')
    argp.add_argument('output_dir')
    argp.add_argument('--num-epochs', type=int, default=1)
    argp.add_argument('--max-steps', type=int)
    argp.add_argument('--random-seed', type=int)
    argp.add_argument('--log-frequency', type=int, default=1000)
    argp.add_argument('--batch-size', type=int)
    # pre-processing parameters
    Data.add_arguments(argp, False)
    # parse
    args = argp.parse_args(argv)
    Data.parse_arguments(args)
    # data pack
    data_pack = DataPack(args)
    data_pack()

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
