import nussl
import librosa
import time
import json
import museval

import numpy as np
import os

MIXTURE_FILE_PATH = './mixture.wav'
SAMPLE_RATE = 44100
SEP_SOURCE1_PATH = './est_source1.wav'
SEP_SOURCE2_PATH = './est_source2.wav'
NMF_PARAMS = {'num_sources': 2, 'num_iterations': 200, 'mfcc_range': (1, 5)}

source1_path = 'sounds/39914__digifishmusic__katy-sings-laaoooaaa.wav'
source2_path = 'sounds/195138__flcellogrl__cello-tuning.wav'

source1, _ = librosa.core.load(source1_path, sr=SAMPLE_RATE)
source2, _ = librosa.core.load(source2_path, sr=SAMPLE_RATE)

l = min([source1.shape[0], source2.shape[0]])
l = 80000

source1 = source1[40000:40000+l]
source2 = 2.*source2[45000:45000+l]

librosa.output.write_wav('./source1.wav', source1, SAMPLE_RATE)
librosa.output.write_wav('./source2.wav', source2, SAMPLE_RATE)

print("Num of samples in each source: {}".format(source1.shape[0]))

mixture = source1 + source2

librosa.output.write_wav(MIXTURE_FILE_PATH, mixture, SAMPLE_RATE)

print("Running NMF algorithm...")
start_time = time.time()
mixture = nussl.AudioSignal(MIXTURE_FILE_PATH)

nussl_source1 = nussl.AudioSignal('./source1.wav')
nussl_source2 = nussl.AudioSignal('./source2.wav')

nmf_mfcc = nussl.NMF_MFCC(mixture, **NMF_PARAMS)
nmf_mfcc.run()
bg, fg = nmf_mfcc.make_audio_signals()
end_time = time.time()
print("Finished, time taken: {:.2f} secs".format(end_time - start_time))

librosa.output.write_wav(SEP_SOURCE1_PATH, np.squeeze(fg.audio_data), SAMPLE_RATE)
librosa.output.write_wav(SEP_SOURCE2_PATH, np.squeeze(bg.audio_data), SAMPLE_RATE)

reference_sources = np.expand_dims(np.vstack((source1, source2)), axis=2)  # (2, 683008, 1)

# Compute evaluation for both permutations of estimated sources
bss = nussl.evaluation.BSSEvalV4(mixture, [nussl_source1, nussl_source2], [fg, bg])
bss_scores = bss.evaluate()
with open('./eval1.json', 'w+') as json_file:
    json.dump(bss_scores, json_file, sort_keys=True, indent=4)

bss = nussl.evaluation.BSSEvalV4(mixture, [nussl_source2, nussl_source1], [fg, bg])
bss_scores = bss.evaluate()
with open('./eval2.json', 'w+') as json_file:
    json.dump(bss_scores, json_file, sort_keys=True, indent=4)

os.remove(MIXTURE_FILE_PATH)
