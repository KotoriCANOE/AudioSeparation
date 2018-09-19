from abc import ABCMeta, abstractmethod
import numpy as np
import librosa
from scipy import ndimage
import os
import random
from utils import bool_argument, eprint, listdir_files

# ======
# base class

class DataBase:
    __metaclass__ = ABCMeta

    def __init__(self, config):
        self.dataset = None
        self.num_epochs = None
        self.max_steps = None
        self.batch_size = None
        self.val_size = None
        self.packed = None
        self.processes = None
        self.threads = None
        self.prefetch = None
        self.buffer_size = None
        self.shuffle = None
        self.group_size = None
        # copy all the properties from config object
        self.config = config
        self.__dict__.update(config.__dict__)
        # initialize
        self.get_files()

    @staticmethod
    def add_arguments(argp, test=False):
        # base parameters
        bool_argument(argp, 'packed', False)
        bool_argument(argp, 'test', test)
        # pre-processing parameters
        argp.add_argument('--processes', type=int, default=4)
        argp.add_argument('--threads', type=int, default=1)
        argp.add_argument('--prefetch', type=int, default=64)
        argp.add_argument('--buffer-size', type=int, default=256)
        bool_argument(argp, 'shuffle', True)
        # sample parameters
        argp.add_argument('--pp-rate', type=int, default=49152)
        argp.add_argument('--pp-duration', type=float,
            help='0: no slicing, -: fixed slicing, +: random slicing')
        argp.add_argument('--pp-smooth', type=float)
        argp.add_argument('--pp-noise', type=float)
        argp.add_argument('--pp-amplitude', type=int)

    @staticmethod
    def parse_arguments(args):
        def argdefault(name, value):
            if args.__getattribute__(name) is None:
                args.__setattr__(name, value)
        def argchoose(name, cond, tv, fv):
            argdefault(name, tv if cond else fv)
        argchoose('batch_size', args.test, 1, 2)
        argchoose('pp_duration', args.test, 8.0, 8.0)
        argchoose('pp_smooth', args.test, 0, 0)
        argchoose('pp_noise', args.test, 0, 0)
        argchoose('pp_amplitude', args.test, 0, 0)

    def get_files_packed(self):
        data_list = os.listdir(self.dataset)
        data_list = [os.path.join(self.dataset, i) for i in data_list]
        # val set
        if self.val_size is not None:
            self.val_steps = self.val_size // self.batch_size
            assert self.val_steps < len(data_list)
            self.val_size = self.val_steps * self.batch_size
            self.val_set = data_list[:self.val_steps]
            data_list = data_list[self.val_steps:]
            eprint('validation set: {}'.format(self.val_size))
        # main set
        self.epoch_steps = len(data_list)
        self.epoch_size = self.epoch_steps * self.batch_size
        if self.max_steps is None:
            self.max_steps = self.epoch_steps * self.num_epochs
        else:
            self.num_epochs = (self.max_steps + self.epoch_steps - 1) // self.epoch_steps
        self.main_set = data_list

    @abstractmethod
    def get_files_origin(self):
        pass

    def get_files(self):
        if self.packed: # packed dataset
            self.get_files_packed()
        else: # non-packed dataset
            data_list = self.get_files_origin()
            # val set
            if self.val_size is not None:
                assert self.val_size < len(data_list)
                self.val_steps = self.val_size // self.batch_size
                self.val_size = self.val_steps * self.batch_size
                self.val_set = data_list[:self.val_size]
                data_list = data_list[self.val_size:]
                eprint('validation set: {}'.format(self.val_size))
            # main set
            assert self.batch_size <= len(data_list)
            self.epoch_steps = len(data_list) // self.batch_size
            self.epoch_size = self.epoch_steps * self.batch_size
            if self.max_steps is None:
                self.max_steps = self.epoch_steps * self.num_epochs
            else:
                self.num_epochs = (self.max_steps + self.epoch_steps - 1) // self.epoch_steps
            self.main_set = data_list[:self.epoch_size]
        # print
        eprint('main set: {}\nepoch steps: {}\nnum epochs: {}\nmax steps: {}\n'
            .format(self.epoch_size, self.epoch_steps, self.num_epochs, self.max_steps))

    @staticmethod
    def process_sample(input_file, label_file, config):
        # parameters
        sample_rate = config.pp_rate if config.pp_rate > 0 else None
        slice_duration = np.abs(config.pp_duration)
        # slice
        duration = librosa.get_duration(filename=input_file)
        if config.pp_duration > 0 and duration > slice_duration:
            # randomly cropping
            offset = random.uniform(0, duration - slice_duration)
        else:
            offset = 0.0
        # read from file
        input_data, rate = librosa.load(input_file, sr=sample_rate, mono=False,
            offset=offset, duration=slice_duration)
        label_data, rate = librosa.load(label_file, sr=sample_rate, mono=False,
            offset=offset, duration=slice_duration)
        # audio_max = np.maximum(np.max(input_data), np.max(label_data))
        samples = input_data.shape[-1]
        slice_samples = int(slice_duration * rate + 0.5)
        # normalization
        # norm_factor = 1 / audio_max
        # input_data *= norm_factor
        # label_data *= norm_factor
        # wrap padding
        if samples < slice_samples:
            input_data = np.pad(input_data, (0, slice_samples - samples), 'wrap')
            label_data = np.pad(label_data, (0, slice_samples - samples), 'wrap')
        # random data manipulation
        # input_data = DataPP.process(input_data, config)
        # label_data = DataPP.process(label_data, config)
        # return
        return input_data, label_data

    @classmethod
    def extract_batch(cls, batch_set, config):
        from concurrent.futures import ThreadPoolExecutor
        # initialize
        inputs = []
        labels = []
        # load all the data
        if config.threads == 1:
            for input_file, label_file in batch_set:
                _input, _label = cls.process_sample(input_file, label_file, config)
                inputs.append(_input)
                labels.append(_label)
        else:
            with ThreadPoolExecutor(config.threads) as executor:
                futures = []
                for input_file, label_file in batch_set:
                    futures.append(executor.submit(cls.process_sample, input_file, label_file, config))
                # final data
                while len(futures) > 0:
                    _input, _label = futures.pop(0).result()
                    inputs.append(_input)
                    labels.append(_label)
        # stack data to form a batch (NCW)
        inputs = np.stack(inputs)
        labels = np.stack(labels)
        # convert to NCHW format
        inputs = np.expand_dims(inputs, -2)
        labels = np.expand_dims(labels, -2)
        # return
        return inputs, labels

    @classmethod
    def extract_batch_packed(cls, batch_set):
        npz = np.load(batch_set)
        inputs = npz['inputs']
        labels = npz['labels']
        return inputs, labels

    def _gen_batches_packed(self, dataset, epoch_steps, num_epochs=1, start=0):
        _dataset = dataset
        max_steps = epoch_steps * num_epochs
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(self.processes) as executor:
            futures = []
            # loop over epochs
            for epoch in range(start // epoch_steps, num_epochs):
                step_offset = epoch_steps * epoch
                step_start = max(0, start - step_offset)
                step_stop = min(epoch_steps, max_steps - step_offset)
                # loop over steps within an epoch
                for step in range(step_start, step_stop):
                    batch_set = _dataset[step]
                    futures.append(executor.submit(self.extract_batch_packed,
                        batch_set))
                    # yield the data beyond prefetch range
                    while len(futures) >= self.prefetch:
                        yield futures.pop(0).result()
            # yield the remaining data
            for future in futures:
                yield future.result()

    def _gen_batches_origin(self, dataset, epoch_steps, num_epochs=1, start=0,
        shuffle=False):
        _dataset = dataset
        max_steps = epoch_steps * num_epochs
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(self.processes) as executor:
            futures = []
            # loop over epochs
            for epoch in range(start // epoch_steps, num_epochs):
                step_offset = epoch_steps * epoch
                step_start = max(0, start - step_offset)
                step_stop = min(epoch_steps, max_steps - step_offset)
                # random shuffle
                if shuffle and epoch > 0:
                    random.shuffle(_dataset)
                # loop over steps within an epoch
                for step in range(step_start, step_stop):
                    offset = step * self.batch_size
                    upper = min(len(_dataset), offset + self.batch_size)
                    batch_set = _dataset[offset : upper]
                    futures.append(executor.submit(self.extract_batch,
                        batch_set, self.config))
                    # yield the data beyond prefetch range
                    while len(futures) >= self.prefetch:
                        future = futures.pop(0)
                        yield future.result()
            # yield the remaining data
            for future in futures:
                yield future.result()

    def _gen_batches(self, dataset, epoch_steps, num_epochs=1, start=0,
        shuffle=False):
        # packed dataset
        if self.packed:
            return self._gen_batches_packed(dataset, epoch_steps, num_epochs, start)
        else:
            return self._gen_batches_origin(dataset, epoch_steps, num_epochs, start, shuffle)

    def gen_main(self, start=0):
        return self._gen_batches(self.main_set, self.epoch_steps, self.num_epochs,
            start, self.shuffle)

    def gen_val(self, start=0):
        return self._gen_batches(self.val_set, self.val_steps, 1,
            start, False)

class DataPP:
    @classmethod
    def process(cls, data, config):
        # smoothing
        smooth_prob = config.pp_smooth
        smooth_std = 0.75
        if cls.active_prob(smooth_prob):
            smooth_scale = cls.truncate_normal(smooth_std)
            data = ndimage.gaussian_filter1d(data, smooth_scale, truncate=2.0)
        # add noise
        noise_prob = config.pp_noise
        noise_std = 0.01
        noise_smooth_prob = 0.8
        noise_smooth_std = 1.5
        while cls.active_prob(noise_prob):
            # Gaussian noise
            noise_scale = cls.truncate_normal(noise_std)
            noise = np.random.normal(0.0, noise_scale, data.shape)
            # noise smoothing
            if cls.active_prob(noise_smooth_prob):
                smooth_scale = cls.truncate_normal(noise_smooth_std)
                noise = ndimage.gaussian_filter1d(noise, smooth_scale, truncate=2.0)
            # add noise
            data += noise
        # random amplitude
        amplitude = config.pp_amplitude / 10
        if amplitude > 0:
            data *= 0.1 ** np.random.uniform(0, amplitude) # 0~-20 dB
        # return
        return data

    @staticmethod
    def active_prob(prob):
        return np.random.uniform(0, 1) < prob

    @staticmethod
    def truncate_normal(std, mean=0.0, max_rate=4.0):
        max_scale = std * max_rate
        scale = max_scale + 1.0
        while scale > max_scale:
            scale = np.abs(np.random.normal(0.0, std))
        scale += mean
        return scale

# ======
# derived classes

class DataSong(DataBase):
    @classmethod
    def get_duration(cls, file):
        try:
            duration = librosa.get_duration(filename=file)
        except Exception:
            eprint('Failed to read {}'.format(file))
            duration = -1
        return duration

    @classmethod
    def filter_files(cls, files, min_length=None, max_length=None):
        if min_length is None:
            min_length = 0
        if max_length is None:
            max_length = float('inf')
        files = [f for f in files if min_length
            <= cls.get_duration(f) <= max_length]
        return files

    def get_files_origin(self):
        import re
        # get all the audio files
        filter_ext = ['.wav', '.flac', '.m4a', '.mp3']
        data_list = listdir_files(self.dataset, filter_ext=filter_ext)
        # filter files by length
        data_list = self.filter_files(data_list, 2.0, None)
        # list of pairs
        regex = re.compile(r'伴奏|instru|vocal')
        bgm_list = [f for f in data_list if re.findall(regex, f)]
        song_list = [f for f in data_list if f not in bgm_list]
        data_list = list(zip(song_list, bgm_list))
        if len(data_list) < 1024:
            data_list *= (1024 + len(data_list) - 1) // len(data_list)
        # return
        if self.shuffle:
            random.shuffle(data_list)
        return data_list
