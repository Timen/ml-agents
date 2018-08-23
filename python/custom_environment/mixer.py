import numpy as np
import pywt
from pydub import AudioSegment
from scipy import signal
from scipy.signal import find_peaks
import os
import gym
from gym.utils import seeding

def outputSound(filename,sound1,sound2,framerate):

    sound1 = np.array(sound1)
    sound2 = np.array(sound2)
    combined = (sound1 + sound2) / 2
    sound1 = (sound1).astype(np.int16)
    sound2 = (sound2).astype(np.int16)
    combined = (combined).astype(np.int16)

    # Advanced usage, if you have raw audio data:
    combined = AudioSegment(
        # raw audio data (bytes)
        data=combined,

        # 2 byte (16 bit) samples
        sample_width=2,

        # 44.1 kHz frame rate
        frame_rate=framerate,

        # stereo
        channels=1
    )

    combined.export(filename+"_c.wav", format="wav")
    # Advanced usage, if you have raw audio data:
    sound1 = AudioSegment(
        # raw audio data (bytes)
        data=sound1,

        # 2 byte (16 bit) samples
        sample_width=2,

        # 44.1 kHz frame rate
        frame_rate=framerate,

        # stereo
        channels=1
    )

    sound1.export(filename + "_1.wav", format="wav")
    sound2 = AudioSegment(
        # raw audio data (bytes)
        data=sound2,
        # 2 byte (16 bit) samples
        sample_width=2,

        # 44.1 kHz frame rate
        frame_rate=framerate,

        # stereo
        channels=1
    )

    sound2.export(filename + "_2.wav", format="wav")


def get_beat_positions(data, ori_data, fs):
    beat_fs = fs // (len(ori_data) // len(data))
    trigger_val = np.percentile(data, 99)
    cooldown_length = beat_fs // (240 // 60)

    peaks, height = find_peaks(data, height=trigger_val, distance=cooldown_length)
    # print(height)
    # plt.plot(peaks,height["peak_heights"],'ro')
    # plt.plot(data)
    # plt.show()
    # exit()
    return peaks


def compare_beat(pred, anno):
    pred = np.array(pred)
    errors = []
    max_index = len(anno) - 1
    for idx, a in enumerate(anno):
        diff = np.abs(pred - a)
        jdx = np.argmin(diff)
        min_val = np.amin(diff)

        if (idx == 0 or pred[jdx] > a) and not idx == max_index:
            errors.append(float(min_val) / float(anno[idx + 1] - a))
        elif idx == max_index or pred[jdx] <= a:
            errors.append(float(min_val) / float(anno[idx - 1] - a))

    return errors



def information_gain(errors, bins=40):
    hist, bin_edges = np.histogram(errors, bins=bins, range=(-0.5, 0.5), density=True)
    hist = hist / np.sum(hist)

    non_zero_hist = hist[hist != 0]
    hist = np.log2(non_zero_hist)
    hist = hist[np.isfinite(hist)]

    return 1.0 - ((np.sum(non_zero_hist * -hist)) / np.log2(bins))


def beat_error(sound_1_buf, sound_2_buf, fs):
    anno_beats = beat_detector(sound_1_buf, fs)
    pred_beats = beat_detector(sound_2_buf, fs)
    forward_error = compare_beat(pred_beats, anno_beats)
    backward_error = compare_beat(anno_beats, pred_beats)
    mean_forward = np.mean(np.array(np.abs(forward_error)))
    mean_backward = np.mean(np.array(np.abs(backward_error)))
    # i_gain_f = information_gain(forward_error)
    min_error = min(mean_backward, mean_forward)

    return min_error * 2.0


def beat_detector(data, fs):
    cA = []
    cD_sum = []
    levels = 3
    max_decimation = 2 ** (levels - 1)
    for loop in range(0, levels):
        # 1) DWT
        if loop == 0:
            [cA, cD] = pywt.dwt(data, 'db4')
            cD_minlen = len(cD) // max_decimation
            cD_sum = np.zeros(cD_minlen);
        else:
            [cA, cD] = pywt.dwt(cA, 'db4')
        # 2) Filter
        # cD = signal.lfilter([0.01], [1 - 0.99], cD)

        # 5) Decimate for reconstruction later.
        cD = abs(cD[::(2 ** (levels - loop - 1))])
        cD = cD - np.mean(cD)
        # 6) Recombine the signal before ACF
        #    essentially, each level I concatenate 
        #    the detail coefs (i.e. the HPF values)
        #    to the beginning of the array
        cD_sum = cD[0:cD_minlen] + cD_sum;

    # adding in the approximate data as well...
    # cA = signal.lfilter([0.01], [1 - 0.99], cA)
    cA = abs(cA)
    cA = cA - np.mean(cA)
    cD_sum = cA[0:cD_minlen] + cD_sum

    return get_beat_positions(cD_sum, data, fs)


class AudioEnv(gym.Env):
    def __init__(self, file_sound1, file_sound2,seed=1, format="mp3", seconds_per_sample=1, sequences_variance=4, fs=12288,wav_path="/usr/local/share/models/aidj/"):
        self.sound1_offline = AudioSegment.from_file(file_sound1, format=format).set_channels(1).set_frame_rate(fs)
        self.sound2_offline = AudioSegment.from_file(file_sound2, format=format).set_channels(1).set_frame_rate(fs)
        self.sample_width = self.sound1_offline.sample_width
        self.original_frame_rate = self.sound2_offline.frame_rate
        self.seconds_per_sample = seconds_per_sample
        self.ori_sequence_length = int(float(self.original_frame_rate) * seconds_per_sample)
        self.sound1_offline = np.array(self.sound1_offline.get_array_of_samples()).astype(np.float32)
        self.sound2_offline = np.array(self.sound2_offline.get_array_of_samples()).astype(np.float32)
        self.actions = np.array([0.0,0.05, -0.05,-0.008,0.008, 0.001, -0.001, 0.0001, 0.0001])

        self.sequences_variance = sequences_variance
        # self.action_space = gym.spaces.Box(low=-0.05, high=0.05, shape=(1,))
        self.action_space = gym.spaces.Discrete(9)
        self.action_space =  gym.spaces.Box(low=-0.05, high=0.05,shape = [2])
        ii16 = np.iinfo(np.int16)
        self.observation_space = gym.spaces.Box(low=ii16.min, high=ii16.max,shape = [1,fs*seconds_per_sample*2,1])
        self.goal_error = 0.004
        self.seed(seed=seed)
        self.viewer = None
        self.id = str(seed)
        self.wav_path = wav_path
        self.num_done = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def shape_state(self, state):
        return np.reshape(state, self.observation_space.shape)

    def close(self):
        if self.viewer: self.viewer.close()

    def reset(self,train_mode=None):
        if train_mode is not None:
            self.training = not train_mode
        self.phase_offset = 0
        random_shift = int(float(self.original_frame_rate / 2) * self.np_random.uniform(-0.45, 0.45))
        sound1 = self.sound1_offline[self.original_frame_rate:self.ori_sequence_length * (self.sequences_variance + 1)]
        sound2 = self.sound2_offline[self.original_frame_rate + random_shift:self.ori_sequence_length * (
                    self.sequences_variance + 1) + random_shift + self.original_frame_rate]
        self.start_sample_sound1 = self.ori_sequence_length * (self.sequences_variance + 1)
        self.start_sample_sound2 = self.ori_sequence_length * (self.sequences_variance + 1) + random_shift
        self.sound1_buffer = list(sound1)
        self.sound2_buffer = list(sound2)
        self.sequences_ran = 1
        self.tempo_adjust = 0.0
        self.sequence_length = self.ori_sequence_length

        self.min_error = beat_error(self.sound1_offline[:-(self.original_frame_rate + random_shift)],
                                    self.sound2_offline[self.original_frame_rate + random_shift:],
                                    self.original_frame_rate)
        self.ori_error = self.min_error
        state = list(sound1[-self.ori_sequence_length:])
        state.extend(sound2[-self.ori_sequence_length:])
        return self.shape_state(state)


    def tempoChange(self, action):
        self.tempo_adjust = min(0.1, max(self.tempo_adjust + action, -0.1))
        self.sequence_length = int(float(self.ori_sequence_length) * (1.0 + self.tempo_adjust))

    def phaseChange(self, action):
        adjust = action + 1.0
        self.end_sample_sound2 = self.start_sample_sound2 + int(float(self.sequence_length) * adjust)

    def checkMix(self):
        sound1 = self.sound1_offline[
                 self.ori_sequence_length * self.sequences_ran:self.ori_sequence_length * (self.sequences_ran + 1)]
        sound2 = self.sound2_offline[self.start_sample_sound2:self.end_sample_sound2]

        sound2 = signal.resample(sound2, self.ori_sequence_length)
        self.sound1_buffer.extend(sound1)
        self.sound2_buffer.extend(sound2)

        sound_2_buf = self.sound2_buffer[-self.sequences_variance * self.ori_sequence_length:]

        sound_1_buf = self.sound1_buffer[-self.sequences_variance * self.ori_sequence_length:]

        new_error = beat_error(sound_1_buf, sound_2_buf, self.original_frame_rate)

        done = False
        if self.end_sample_sound2 > (
                len(self.sound2_offline) - self.sequence_length) or self.ori_sequence_length * self.sequences_ran > (
                len(self.sound1_offline) - self.ori_sequence_length):
            done = True
            self.num_done += 1
            if self.training and self.num_done % 10 == 0:
                directory = os.path.join(self.wav_path,self.id)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                outputSound(directory+"/step_{}".format(self.num_done), self.sound1_buffer[:32 * self.ori_sequence_length],
                            self.sound2_buffer[:32 * self.ori_sequence_length], self.original_frame_rate)
        if new_error < 0.006:
            reward = 1
        elif new_error < self.min_error * 0.98:
            reward = 1.0 - (new_error / self.ori_error)
            self.min_error = new_error
        else:
            assert new_error <= 1.0, "error out of 0,0.5 range {}".format(new_error)
            reward = -(1 - (1.0 - new_error) / (1.0 - self.min_error))

        # print(sound1_var,combined)
        self.start_sample_sound2 = self.end_sample_sound2


        if self.ori_sequence_length * (self.sequences_ran + 1) > self.sound1_offline.shape[0] or (
                self.end_sample_sound2 + self.ori_sequence_length) > self.sound2_offline.shape[0]:
            self.start_sample_sound1 = 0
            self.start_sample_sound2 = 0
            self.end_sample_sound2 = self.sequence_length
            done = True
        state = list(sound1)
        state.extend(sound2)
        self.sequences_ran += 1
        return reward, done, state

    def step(self, action):

        phaseAdjust = float(action[0]/20.0)
        tempoAjust = float(action[1] / 20.0)
        # action = float(action[0] / 20.0)
        # if isinstance(action, int) or isinstance(action, np.int32) or isinstance(action, np.int64):
        #     adjust = self.actions[action]
        # else:
        #     adjust = action
        self.tempoChange(tempoAjust)
        self.phaseChange(phaseAdjust)

        penalty = 0.0
        # self.adjustMix(action)
        reward, done, state = self.checkMix()
        state = self.shape_state(state)
        if reward == None:
            print(done)
            print(reward)
            print(penalty)
            return state, (reward + penalty), done,{}
        else:
            return state, (reward + penalty), done,{}

    def close(self):
        return