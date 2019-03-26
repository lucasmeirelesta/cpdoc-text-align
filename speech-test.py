import os

import io
import subprocess
import time

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

import soundfile as sf
import numpy as np

duracao_processo_total = time.time()

#Inserir teste de similaridade quando for fazer o match das palavras,
#Escolher métrica de acuracidade
#Otimizar o algoritmo

json_path = '/media/fexu/Tree1/FGV/Projetos/Speech-to-text-CPDOC-b6cbea64d516.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = json_path

audio_path = 'Entrevistas'

audio_name = "Molejo_-_Cilada-fOQ78QpFsds.flac"

audio_path_name = os.path.join(audio_path, audio_name)

y, s = sf.read(audio_path_name)

try:
    y = np.sum(y, axis=1) / 2
except:
    pass

audio_path_name = audio_path_name.split(".")[0]+".flac"
audio_name = audio_name.split(".")[0]+".flac"
sf.write(audio_path_name, y, samplerate=s)

del y

subprocess.call('gsutil cp {} gs://speech-cpdoc/audio-sample/{}'.format(audio_path_name, audio_name), shell=True)

# Instantiates a client
client = speech.SpeechClient()

# More than one minute

duracao_processo_transcricao = time.time()

gs_url = 'gs://speech-cpdoc/audio-sample/{}'.format(audio_name)

operation = client.long_running_recognize(
     audio=speech.types.RecognitionAudio(
         uri=gs_url,
     ),
     config=speech.types.RecognitionConfig(
         encoding='FLAC',
         language_code='pt-BR',
         sample_rate_hertz=s,
         enable_word_time_offsets=True
     ),
 )

print('Waiting for operation to complete...')
response = operation.result()

duracao_processo_transcricao = (time.time() - duracao_processo_transcricao)/60

transcription_path = os.path.join('transcription', "transcricao-"+audio_name.split('.')[0]+".txt")

file = open(transcription_path, "w")

response

for result in response.results:
    alternative = result.alternatives[0]
    file.write('Transcript: {} \n'.format(alternative.transcript))
    file.write('confidence: {} \n\n'.format(alternative.confidence))

    for word_info in alternative.words:
        word = word_info.word
        start_time = word_info.start_time
        end_time = word_info.end_time
        file.write('Word: {}, start_time: {}, end_time: {} \n'.format(
            word,
            start_time.seconds + start_time.nanos * 1e-9,
            end_time.seconds + end_time.nanos * 1e-9))

duracao_processo_total = (time.time() - duracao_processo_total)/60

file.write("\n Duração de todo processo: {} minutos \n".format(np.round(duracao_processo_total,2)))
file.write("Duração do processo de transcrição: {} minutos \n".format(np.round(duracao_processo_transcricao, 2)))

file.close()

subprocess.call('gsutil rm gs://speech-cpdoc/audio-sample/{}'.format(audio_name), shell=True)

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
