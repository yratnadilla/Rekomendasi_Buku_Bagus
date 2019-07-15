import numpy as np
import pandas as pd

df = pd.read_csv('books.csv')

df = df.dropna(subset= ['original_title'])
df = df[['book_id', 'original_title', 'authors']]

def mergeCol(i):
    return str(i['authors']) + ' ' + str(i['original_title'])

df['MixedFeatures'] = df.apply(mergeCol, axis= 1)

from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(
    tokenizer= lambda x : x.split(' ')
)

matrixFeature  = model.fit_transform(df['MixedFeatures'])

from sklearn.metrics.pairwise import cosine_similarity
score = cosine_similarity(matrixFeature)

# ==============================
 
AndiBook = 'Catching Fire'
indexBookAndi = df[df['original_title'] == AndiBook].index.values[0]

scoreAndi = list(enumerate(score[indexBookAndi]))

sameBooksAndi = sorted(
    scoreAndi,
    key= lambda x : x[1],
    reverse= True
)

similarBooksAndi=[]
for i in sameBooksAndi:
    if i[1] > 0:
        similarBooksAndi.append(i)

print('Top 5 book recommendation for Andi:')
for i in range(0,5):
    print('-', df['original_title'].iloc[similarBooksAndi[i][0]])


# ==============================

BudiBook = 'Harry Potter and the Chamber of Secrets'
indexBookBudi = df[df['original_title'] == BudiBook].index.values[0]

scoreBudi = list(enumerate(score[indexBookBudi]))

sameBooksBudi = sorted(
    scoreBudi,
    key= lambda x : x[1],
    reverse= True
)

similarBooksBudi=[]
for i in sameBooksBudi:
    if i[1] > 0:
        similarBooksBudi.append(i)

print('Top 5 book recommendation for Budi:')
for i in range(0,5):
    print('-', df['original_title'].iloc[similarBooksBudi[i][0]])

# ==============================

CikoBook = 'Robots and Empire'
indexBookCiko = df[df['original_title'] == CikoBook].index.values[0]

scoreCiko = list(enumerate(score[indexBookCiko]))

sameBooksCiko = sorted(
    scoreCiko,
    key= lambda x : x[1],
    reverse= True
)

similarBooksCiko=[]
for i in sameBooksCiko:
    if i[1] > 0:
        similarBooksCiko.append(i)

print('Top 5 book recommendation for Ciko:')
for i in range(0,5):
    print('-', df['original_title'].iloc[similarBooksCiko[i][0]])

# ==============================

DediBook = 'No god but God: The Origins, Evolution, and Future of Islam'
indexBookDedi = df[df['original_title'] == DediBook].index.values[0]

scoreDedi = list(enumerate(score[indexBookDedi]))

sameBooksDedi = sorted(
    scoreDedi,
    key= lambda x : x[1],
    reverse= True
)

similarBooksDedi=[]
for i in sameBooksDedi:
    if i[1] > 0:
        similarBooksDedi.append(i)

print('Top 5 book recommendation for Dedi:')
for i in range(0,5):
    print('-', df['original_title'].iloc[similarBooksDedi[i][0]])

# ==============================

ElloBook = 'The Story of Doctor Dolittle'
indexBookEllo = df[df['original_title'] == ElloBook].index.values[0]

scoreEllo = list(enumerate(score[indexBookEllo]))

sameBooksEllo = sorted(
    scoreEllo,
    key= lambda x : x[1],
    reverse= True
)

similarBooksEllo=[]
for i in sameBooksEllo:
    if i[1] > 0:
        similarBooksEllo.append(i)

print('Top 5 book recommendation for Ello:')
for i in range(0,5):
    print('-', df['original_title'].iloc[similarBooksEllo[i][0]])

