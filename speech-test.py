import os
import time
import subprocess
import numpy as np
import soundfile as sf
from google.cloud import speech

time_process_total = time.time()


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
time_process_transcription = time.time()

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

time_process_transcription = (time.time() - time_process_transcription)/60

transcription_path = os.path.join('transcription', "transcricao-"+audio_name.split('.')[0]+".txt")

with open(transcription_path, "w") as file:

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

    time_process_total = (time.time() - time_process_total)/60

    file.write("\n Duração de todo processo: {} minutos \n".format(np.round(time_process_total,2)))
    file.write("Duração do processo de transcrição: {} minutos \n".format(np.round(time_process_transcription, 2)))


subprocess.call('gsutil rm gs://speech-cpdoc/audio-sample/{}'.format(audio_name), shell=True)