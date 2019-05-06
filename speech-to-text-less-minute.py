import os

import io
import subprocess
import time

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

import soundfile as sf
import numpy as np

#Less than one minute and local

# Loads the audio into memory
with io.open(audio_name, 'rb') as audio_file:
    content = audio_file.read()
    audio = types.RecognitionAudio(content=content)

os.system('touch test2.txt')

file_object = open('test.txt', 'w')
file_object.write(str(audio))
file_object.close()

config = types.RecognitionConfig(
    encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code='pt-BR',
    enable_word_time_offsets=True
     )

# Detects speech in the audio file
response = client.recognize(config, audio)

print('Waiting for operation to complete...')
response = operation.result(timeout=90)

file = open("transcricao_biato_2016-04-14_01_3.txt", "w")

for result in response.results:
    alternative = result.alternatives[0]
    print('Transcript: {}'.format(result.alternatives[0].transcript))
    file.write('Transcript: {} \n\n'.format(result.alternatives[0].transcript))

    for word_info in alternative.words:
        word = word_info.word
        start_time = word_info.start_time
        end_time = word_info.end_time
        file.write('Word: {}, start_time: {}, end_time: {} \n'.format(
            word,
            start_time.seconds + start_time.nanos * 1e-9,
            end_time.seconds + end_time.nanos * 1e-9))
        print('Word: {}, start_time: {}, end_time: {}'.format(
            word,
            start_time.seconds + start_time.nanos * 1e-9,
            end_time.seconds + end_time.nanos * 1e-9))

file.close()
