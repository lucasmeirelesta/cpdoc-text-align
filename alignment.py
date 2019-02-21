import unidecode
import nwalign3 as nw
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

#import nltk
#nltk.download('stopwords')

file = open("ExemplosAudio_ENTREVISTAS/pho_2303_marcel_biato_2016-06-14_06.vtt", 'r')
transcricao_humana = file.read()

file = open("transcription/transcricao-pho_2303_marcel_biato_2016-06-14_06.txt", 'r')
transcricao_google = file.read()

if 'WEBVTT' in transcricao_humana:
    transcricao_humana = transcricao_humana.replace('WEBVTT', '')

if 'Transcript' in transcricao_google:
    transcricao_google = transcricao_google.replace('Transcript', '')

if 'start_time' in transcricao_google:
    transcricao_google = transcricao_google.replace('start_time', '')

if 'end_time' in transcricao_google:
    transcricao_google = transcricao_google.replace('end_time', '')

if 'Word' in transcricao_google:
    transcricao_google = transcricao_google.replace('Word', '')

if 'confidence' in transcricao_google:
    transcricao_google = transcricao_google.replace('confidence', '')

stopwords_pt = [unidecode.unidecode(i) for i in stopwords.words('portuguese')]

transcricao_google = transcricao_google.lower()
transcricao_google = unidecode.unidecode(transcricao_google)
transcricao_google = " ".join(re.split(r'[^a-zA-Z]*', transcricao_google))
#Sem stop word
transcricao_google = [i for i in transcricao_google.split() if i not in stopwords_pt]
transcricao_google = " ".join(transcricao_google)

transcricao_humana = transcricao_humana.lower()
transcricao_humana = unidecode.unidecode(transcricao_humana)
transcricao_humana = " ".join(re.split(r'[^a-zA-Z]*', transcricao_humana))
#Sem stop word
transcricao_humana = [i for i in transcricao_humana.split() if i not in stopwords_pt]
transcricao_humana = " ".join(transcricao_humana)

#interseção
aux_google = set(transcricao_google.split())
aux_humana = set(transcricao_humana.split())
aux_intersection = aux_google.intersection(aux_humana)
transcricao_humana = [i for i in transcricao_humana.split() if i in aux_intersection]
transcricao_humana = " ".join(transcricao_humana)
transcricao_google = [i for i in transcricao_google.split() if i in aux_intersection]
transcricao_google = " ".join(transcricao_google)

nw_saida = nw.global_align(transcricao_google,transcricao_humana)

print(nw_saida[0])
print(nw_saida[1])

nw_saida = nw.global_align(transcricao_google,transcricao_humana, gap_open=-10, gap_extend=-4)

print(nw_saida[0])
print(nw_saida[1])

transcricao_google = transcricao_google[0:100]
transcricao_humana = transcricao_humana[0:100]

nw_saida = nw.global_align(transcricao_google,transcricao_humana)

print(nw_saida[0])
print(nw_saida[1])

vectorizer_tf_idf = TfidfVectorizer(ngram_range=(1, 2), max_features=50, sublinear_tf=True)

aux_test = transcricao_humana.split()
aux_test.extend(transcricao_google.split())

vector_tf_idf_fitt = vectorizer_tf_idf.fit(aux_test)

transcricao_google = vector_tf_idf_fitt.transform(transcricao_google.split())
transcricao_humana = vector_tf_idf_fitt.transform(transcricao_humana.split())

transcricao_google = transcricao_google.toarray()
transcricao_humana = transcricao_humana.toarray()

return add_dotprod_eucli(question1, question2)
