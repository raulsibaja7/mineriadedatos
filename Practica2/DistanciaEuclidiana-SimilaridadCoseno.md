

```python
import nltk
```


```python
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /home/karisauria/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!





    True




```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
import numpy as np    
from numpy import dot
from numpy.linalg import norm
from IPython.display import display
import csv
import pandas as pd
import math
```


```python
stopwords.fileids()
```




    ['arabic',
     'azerbaijani',
     'danish',
     'dutch',
     'english',
     'finnish',
     'french',
     'german',
     'greek',
     'hungarian',
     'indonesian',
     'italian',
     'kazakh',
     'nepali',
     'norwegian',
     'portuguese',
     'romanian',
     'russian',
     'slovene',
     'spanish',
     'swedish',
     'tajik',
     'turkish']




```python
set(stopwords.words('spanish'))
```




    {'a',
     'al',
     'algo',
     'algunas',
     'algunos',
     'ante',
     'antes',
     'como',
     'con',
     'contra',
     'cual',
     'cuando',
     'de',
     'del',
     'desde',
     'donde',
     'durante',
     'e',
     'el',
     'ella',
     'ellas',
     'ellos',
     'en',
     'entre',
     'era',
     'erais',
     'eran',
     'eras',
     'eres',
     'es',
     'esa',
     'esas',
     'ese',
     'eso',
     'esos',
     'esta',
     'estaba',
     'estabais',
     'estaban',
     'estabas',
     'estad',
     'estada',
     'estadas',
     'estado',
     'estados',
     'estamos',
     'estando',
     'estar',
     'estaremos',
     'estará',
     'estarán',
     'estarás',
     'estaré',
     'estaréis',
     'estaría',
     'estaríais',
     'estaríamos',
     'estarían',
     'estarías',
     'estas',
     'este',
     'estemos',
     'esto',
     'estos',
     'estoy',
     'estuve',
     'estuviera',
     'estuvierais',
     'estuvieran',
     'estuvieras',
     'estuvieron',
     'estuviese',
     'estuvieseis',
     'estuviesen',
     'estuvieses',
     'estuvimos',
     'estuviste',
     'estuvisteis',
     'estuviéramos',
     'estuviésemos',
     'estuvo',
     'está',
     'estábamos',
     'estáis',
     'están',
     'estás',
     'esté',
     'estéis',
     'estén',
     'estés',
     'fue',
     'fuera',
     'fuerais',
     'fueran',
     'fueras',
     'fueron',
     'fuese',
     'fueseis',
     'fuesen',
     'fueses',
     'fui',
     'fuimos',
     'fuiste',
     'fuisteis',
     'fuéramos',
     'fuésemos',
     'ha',
     'habida',
     'habidas',
     'habido',
     'habidos',
     'habiendo',
     'habremos',
     'habrá',
     'habrán',
     'habrás',
     'habré',
     'habréis',
     'habría',
     'habríais',
     'habríamos',
     'habrían',
     'habrías',
     'habéis',
     'había',
     'habíais',
     'habíamos',
     'habían',
     'habías',
     'han',
     'has',
     'hasta',
     'hay',
     'haya',
     'hayamos',
     'hayan',
     'hayas',
     'hayáis',
     'he',
     'hemos',
     'hube',
     'hubiera',
     'hubierais',
     'hubieran',
     'hubieras',
     'hubieron',
     'hubiese',
     'hubieseis',
     'hubiesen',
     'hubieses',
     'hubimos',
     'hubiste',
     'hubisteis',
     'hubiéramos',
     'hubiésemos',
     'hubo',
     'la',
     'las',
     'le',
     'les',
     'lo',
     'los',
     'me',
     'mi',
     'mis',
     'mucho',
     'muchos',
     'muy',
     'más',
     'mí',
     'mía',
     'mías',
     'mío',
     'míos',
     'nada',
     'ni',
     'no',
     'nos',
     'nosotras',
     'nosotros',
     'nuestra',
     'nuestras',
     'nuestro',
     'nuestros',
     'o',
     'os',
     'otra',
     'otras',
     'otro',
     'otros',
     'para',
     'pero',
     'poco',
     'por',
     'porque',
     'que',
     'quien',
     'quienes',
     'qué',
     'se',
     'sea',
     'seamos',
     'sean',
     'seas',
     'sentid',
     'sentida',
     'sentidas',
     'sentido',
     'sentidos',
     'seremos',
     'será',
     'serán',
     'serás',
     'seré',
     'seréis',
     'sería',
     'seríais',
     'seríamos',
     'serían',
     'serías',
     'seáis',
     'siente',
     'sin',
     'sintiendo',
     'sobre',
     'sois',
     'somos',
     'son',
     'soy',
     'su',
     'sus',
     'suya',
     'suyas',
     'suyo',
     'suyos',
     'sí',
     'también',
     'tanto',
     'te',
     'tendremos',
     'tendrá',
     'tendrán',
     'tendrás',
     'tendré',
     'tendréis',
     'tendría',
     'tendríais',
     'tendríamos',
     'tendrían',
     'tendrías',
     'tened',
     'tenemos',
     'tenga',
     'tengamos',
     'tengan',
     'tengas',
     'tengo',
     'tengáis',
     'tenida',
     'tenidas',
     'tenido',
     'tenidos',
     'teniendo',
     'tenéis',
     'tenía',
     'teníais',
     'teníamos',
     'tenían',
     'tenías',
     'ti',
     'tiene',
     'tienen',
     'tienes',
     'todo',
     'todos',
     'tu',
     'tus',
     'tuve',
     'tuviera',
     'tuvierais',
     'tuvieran',
     'tuvieras',
     'tuvieron',
     'tuviese',
     'tuvieseis',
     'tuviesen',
     'tuvieses',
     'tuvimos',
     'tuviste',
     'tuvisteis',
     'tuviéramos',
     'tuviésemos',
     'tuvo',
     'tuya',
     'tuyas',
     'tuyo',
     'tuyos',
     'tú',
     'un',
     'una',
     'uno',
     'unos',
     'vosotras',
     'vosotros',
     'vuestra',
     'vuestras',
     'vuestro',
     'vuestros',
     'y',
     'ya',
     'yo',
     'él',
     'éramos'}




```python
tokenizer = RegexpTokenizer(r'\w+')
```


```python
with open('Tweets.csv', 'r') as file:
    reader = csv.reader(file)
    numero_tweets = 0
    contador = 0
    for row in reader:
          print(f'\nNumero de Tweet: {numero_tweets+1}\n {row[0]}')
          text_01_tokens = tokenizer.tokenize(row[0].lower()) 
          text_01_tokens_wout_stopwords = []
          for word in text_01_tokens:
            if word not in stopwords.words('spanish'): text_01_tokens_wout_stopwords.append(word)
          print(f'\n Tweet sin stopwords: \n {text_01_tokens_wout_stopwords}')
          print(f'Total de palabras en el tweet después de tokenizar: {len(text_01_tokens_wout_stopwords)}')
          x += len(text_01_tokens_wout_stopwords)
          numero_tweets += 1
          contador += 1
print(f'\n\nTotal de Tweets: {numero_tweets}')
```

    
    Numero de Tweet: 1
     Datos concretos sobre el Feminicidio en México, tomados del SESNP. El delito tiene una tendencia a la alza. La solución de López y su Fiscal Carnal: Desaparecerlo del Código Penal. Tenemos un gobierno que prefiere ponerse del lado del delincuente y allanarle el camino.
    
     Tweet sin stopwords: 
     ['datos', 'concretos', 'feminicidio', 'méxico', 'tomados', 'sesnp', 'delito', 'tendencia', 'alza', 'solución', 'lópez', 'fiscal', 'carnal', 'desaparecerlo', 'código', 'penal', 'gobierno', 'prefiere', 'ponerse', 'lado', 'delincuente', 'allanarle', 'camino']
    Total de palabras en el tweet después de tokenizar: 23
    
    Numero de Tweet: 2
     SOCIEDAD El Fiscal General explicó que su propuesta busca facilitar la  investigación por  feminicidio y proteger a las  víctimas.
    
     Tweet sin stopwords: 
     ['sociedad', 'fiscal', 'general', 'explicó', 'propuesta', 'busca', 'facilitar', 'investigación', 'feminicidio', 'proteger', 'víctimas']
    Total de palabras en el tweet después de tokenizar: 11
    
    Numero de Tweet: 3
     Conferencia Presidente Se queda EU con más del 80% de comisiones de remesas: Profeco.
    
     Tweet sin stopwords: 
     ['conferencia', 'presidente', 'queda', 'eu', '80', 'comisiones', 'remesas', 'profeco']
    Total de palabras en el tweet después de tokenizar: 8
    
    Numero de Tweet: 4
     Conferencia Presidente Entrega Fiscalía 2 mil mdp a AMLO para premios de la rifa.
    
     Tweet sin stopwords: 
     ['conferencia', 'presidente', 'entrega', 'fiscalía', '2', 'mil', 'mdp', 'amlo', 'premios', 'rifa']
    Total de palabras en el tweet después de tokenizar: 10
    
    Numero de Tweet: 5
     Durante la  ConferenciaPresidente Alejandro Gertz Manero, titular de la FGR Mexico, entregó 2 mil mdp al Instituto para Devolver al Pueblo lo Robado.
    
     Tweet sin stopwords: 
     ['conferenciapresidente', 'alejandro', 'gertz', 'manero', 'titular', 'fgr', 'mexico', 'entregó', '2', 'mil', 'mdp', 'instituto', 'devolver', 'pueblo', 'robado']
    Total de palabras en el tweet después de tokenizar: 15
    
    Numero de Tweet: 6
     Las bonitas palabras de Emerson Fittipaldi sobre Fernando Alonso: Es un fenómeno muy talentoso que espero vuelva a la Fórmula 1 en 2021
    
     Tweet sin stopwords: 
     ['bonitas', 'palabras', 'emerson', 'fittipaldi', 'fernando', 'alonso', 'fenómeno', 'talentoso', 'espero', 'vuelva', 'fórmula', '1', '2021']
    Total de palabras en el tweet después de tokenizar: 13
    
    Numero de Tweet: 7
     De acuerdo con imágenes obtenidas por la Fiscalía capitalina, la mujer subió a Fátima a un vehículo blanco
    
     Tweet sin stopwords: 
     ['acuerdo', 'imágenes', 'obtenidas', 'fiscalía', 'capitalina', 'mujer', 'subió', 'fátima', 'vehículo', 'blanco']
    Total de palabras en el tweet después de tokenizar: 10
    
    Numero de Tweet: 8
     El presidente López Obrador se comprometió a enfrentar con empresarios y sindicatos el problema de las pensiones de los trabajadores
    
     Tweet sin stopwords: 
     ['presidente', 'lópez', 'obrador', 'comprometió', 'enfrentar', 'empresarios', 'sindicatos', 'problema', 'pensiones', 'trabajadores']
    Total de palabras en el tweet después de tokenizar: 10
    
    Numero de Tweet: 9
     Un estudiante en estado grave tras desalojo de padres de los 43 de Ayotzinapa y manifestantes en Chiapas
    
     Tweet sin stopwords: 
     ['estudiante', 'grave', 'tras', 'desalojo', 'padres', '43', 'ayotzinapa', 'manifestantes', 'chiapas']
    Total de palabras en el tweet después de tokenizar: 9
    
    Numero de Tweet: 10
     El fisco advierte por esquemas de operaciones realizadas entre 2017 y 2019 por más de 339,000 millones de pesos que involucran a 977 contribuyentes
    
     Tweet sin stopwords: 
     ['fisco', 'advierte', 'esquemas', 'operaciones', 'realizadas', '2017', '2019', '339', '000', 'millones', 'pesos', 'involucran', '977', 'contribuyentes']
    Total de palabras en el tweet después de tokenizar: 14
    
    
    Total de Tweets: 10



```python
text_01 = "Datos concretos sobre el Feminicidio en México, tomados del SESNP. El delito tiene una tendencia a la alza. La solución de López y su Fiscal Carnal: Desaparecerlo del Código Penal. Tenemos un gobierno que prefiere ponerse del lado del delincuente y allanarle el camino."
text_02 = "SOCIEDAD El Fiscal General explicó que su propuesta busca facilitar la  investigación por  feminicidio y proteger a las  víctimas."
text_03 = "Conferencia Presidente Se queda EU con más del 80% de comisiones de remesas: Profeco."
text_04 = "Conferencia Presidente Entrega Fiscalía 2 mil mdp a AMLO para premios de la rifa."
text_05 = "Durante la  ConferenciaPresidente Alejandro Gertz Manero, titular de la FGR Mexico, entregó 2 mil mdp al Instituto para Devolver al Pueblo lo Robado."
text_06 = "Las bonitas palabras de Emerson Fittipaldi sobre Fernando Alonso: Es un fenómeno muy talentoso que espero vuelva a la Fórmula 1 en 2021"
text_07 = "De acuerdo con imágenes obtenidas por la Fiscalía capitalina, la mujer subió a Fátima a un vehículo blanco"
text_08 = "El presidente López Obrador se comprometió a enfrentar con empresarios y sindicatos el problema de las pensiones de los trabajadores"
text_09 = "Un estudiante en estado grave tras desalojo de padres de los 43 de Ayotzinapa y manifestantes en Chiapas"
text_10 = "El fisco advierte por esquemas de operaciones realizadas entre 2017 y 2019 por más de 339,000 millones de pesos que involucran a 977 contribuyentes"
```


```python
#ELIMINAMOS STOPWORDS DE LOS TWEETS

tokens1 = tokenizer.tokenize(text_01.lower()) 
stopwords1 = []
for word in tokens1:
    if word not in stopwords.words('spanish'): stopwords1.append(word)

tokens2 = tokenizer.tokenize(text_02.lower()) 
stopwords2 = []
for word in tokens2:
    if word not in stopwords.words('spanish'): stopwords2.append(word)

tokens3 = tokenizer.tokenize(text_03.lower()) 
stopwords3 = []
for word in tokens3:
    if word not in stopwords.words('spanish'): stopwords3.append(word)

tokens4 = tokenizer.tokenize(text_04.lower()) 
stopwords4 = []
for word in tokens4:
    if word not in stopwords.words('spanish'): stopwords4.append(word)

tokens5 = tokenizer.tokenize(text_05.lower()) 
stopwords5 = []
for word in tokens5:
    if word not in stopwords.words('spanish'): stopwords5.append(word)

tokens6 = tokenizer.tokenize(text_06.lower()) 
stopwords6 = []
for word in tokens6:
    if word not in stopwords.words('spanish'): stopwords6.append(word)

tokens7 = tokenizer.tokenize(text_07.lower()) 
stopwords7 = []
for word in tokens7:
    if word not in stopwords.words('spanish'): stopwords7.append(word)

tokens8 = tokenizer.tokenize(text_08.lower()) 
stopwords8 = []
for word in tokens8:
    if word not in stopwords.words('spanish'): stopwords8.append(word)

tokens9 = tokenizer.tokenize(text_09.lower()) 
stopwords9 = []
for word in tokens9:
    if word not in stopwords.words('spanish'): stopwords9.append(word)

tokens10 = tokenizer.tokenize(text_10.lower()) 
stopwords10 = []
for word in tokens10:
    if word not in stopwords.words('spanish'): stopwords10.append(word)
```


```python
dicc_texts = {"1": stopwords1,
              "2": stopwords2,
              "3": stopwords3,
              "4": stopwords4,
              "5": stopwords5,
              "6": stopwords6,
              "7": stopwords7,
              "8": stopwords8,
              "9": stopwords9,
              "10":stopwords10}
```




    {'1': ['datos',
      'concretos',
      'feminicidio',
      'méxico',
      'tomados',
      'sesnp',
      'delito',
      'tendencia',
      'alza',
      'solución',
      'lópez',
      'fiscal',
      'carnal',
      'desaparecerlo',
      'código',
      'penal',
      'gobierno',
      'prefiere',
      'ponerse',
      'lado',
      'delincuente',
      'allanarle',
      'camino'],
     '2': ['sociedad',
      'fiscal',
      'general',
      'explicó',
      'propuesta',
      'busca',
      'facilitar',
      'investigación',
      'feminicidio',
      'proteger',
      'víctimas'],
     '3': ['conferencia',
      'presidente',
      'queda',
      'eu',
      '80',
      'comisiones',
      'remesas',
      'profeco'],
     '4': ['conferencia',
      'presidente',
      'entrega',
      'fiscalía',
      '2',
      'mil',
      'mdp',
      'amlo',
      'premios',
      'rifa'],
     '5': ['conferenciapresidente',
      'alejandro',
      'gertz',
      'manero',
      'titular',
      'fgr',
      'mexico',
      'entregó',
      '2',
      'mil',
      'mdp',
      'instituto',
      'devolver',
      'pueblo',
      'robado'],
     '6': ['bonitas',
      'palabras',
      'emerson',
      'fittipaldi',
      'fernando',
      'alonso',
      'fenómeno',
      'talentoso',
      'espero',
      'vuelva',
      'fórmula',
      '1',
      '2021'],
     '7': ['acuerdo',
      'imágenes',
      'obtenidas',
      'fiscalía',
      'capitalina',
      'mujer',
      'subió',
      'fátima',
      'vehículo',
      'blanco'],
     '8': ['presidente',
      'lópez',
      'obrador',
      'comprometió',
      'enfrentar',
      'empresarios',
      'sindicatos',
      'problema',
      'pensiones',
      'trabajadores'],
     '9': ['estudiante',
      'grave',
      'tras',
      'desalojo',
      'padres',
      '43',
      'ayotzinapa',
      'manifestantes',
      'chiapas'],
     '10': ['fisco',
      'advierte',
      'esquemas',
      'operaciones',
      'realizadas',
      '2017',
      '2019',
      '339',
      '000',
      'millones',
      'pesos',
      'involucran',
      '977',
      'contribuyentes']}




```python
dicc_termns = {}

for text in dicc_texts:
    for word in dicc_texts[text]:
        
        
        if(word in dicc_termns):
            dicc_termns[word] = dicc_termns[word] + 1
            
            
        elif(word not in dicc_termns):        
            dicc_termns[word] = 1
            

print(len(dicc_termns))
dicc_termns
```

    113





    {'datos': 1,
     'concretos': 1,
     'feminicidio': 2,
     'méxico': 1,
     'tomados': 1,
     'sesnp': 1,
     'delito': 1,
     'tendencia': 1,
     'alza': 1,
     'solución': 1,
     'lópez': 2,
     'fiscal': 2,
     'carnal': 1,
     'desaparecerlo': 1,
     'código': 1,
     'penal': 1,
     'gobierno': 1,
     'prefiere': 1,
     'ponerse': 1,
     'lado': 1,
     'delincuente': 1,
     'allanarle': 1,
     'camino': 1,
     'sociedad': 1,
     'general': 1,
     'explicó': 1,
     'propuesta': 1,
     'busca': 1,
     'facilitar': 1,
     'investigación': 1,
     'proteger': 1,
     'víctimas': 1,
     'conferencia': 2,
     'presidente': 3,
     'queda': 1,
     'eu': 1,
     '80': 1,
     'comisiones': 1,
     'remesas': 1,
     'profeco': 1,
     'entrega': 1,
     'fiscalía': 2,
     '2': 2,
     'mil': 2,
     'mdp': 2,
     'amlo': 1,
     'premios': 1,
     'rifa': 1,
     'conferenciapresidente': 1,
     'alejandro': 1,
     'gertz': 1,
     'manero': 1,
     'titular': 1,
     'fgr': 1,
     'mexico': 1,
     'entregó': 1,
     'instituto': 1,
     'devolver': 1,
     'pueblo': 1,
     'robado': 1,
     'bonitas': 1,
     'palabras': 1,
     'emerson': 1,
     'fittipaldi': 1,
     'fernando': 1,
     'alonso': 1,
     'fenómeno': 1,
     'talentoso': 1,
     'espero': 1,
     'vuelva': 1,
     'fórmula': 1,
     '1': 1,
     '2021': 1,
     'acuerdo': 1,
     'imágenes': 1,
     'obtenidas': 1,
     'capitalina': 1,
     'mujer': 1,
     'subió': 1,
     'fátima': 1,
     'vehículo': 1,
     'blanco': 1,
     'obrador': 1,
     'comprometió': 1,
     'enfrentar': 1,
     'empresarios': 1,
     'sindicatos': 1,
     'problema': 1,
     'pensiones': 1,
     'trabajadores': 1,
     'estudiante': 1,
     'grave': 1,
     'tras': 1,
     'desalojo': 1,
     'padres': 1,
     '43': 1,
     'ayotzinapa': 1,
     'manifestantes': 1,
     'chiapas': 1,
     'fisco': 1,
     'advierte': 1,
     'esquemas': 1,
     'operaciones': 1,
     'realizadas': 1,
     '2017': 1,
     '2019': 1,
     '339': 1,
     '000': 1,
     'millones': 1,
     'pesos': 1,
     'involucran': 1,
     '977': 1,
     'contribuyentes': 1}




```python
matrix = np.zeros((len(dicc_texts), len(dicc_termns)))
```


```python
i = 0
j = 0

for word_termns in dicc_termns: 
    for word_texts in dicc_texts:
        if(word_termns in dicc_texts[word_texts]): 
            print(word_termns, "Ubicado en", word_texts)
            
            matrix[j, i] = 1
            
        elif(word_termns not in dicc_texts[word_texts]):
            print(word_termns, "No ubicado en", word_texts)
            
            matrix[j, i] = 0
            
            
        print("se agregó: ", matrix[j,i], "en: ", j, i)
            
        j = j + 1
        
    j = 0
    i = i + 1
```

    datos Ubicado en 1
    se agregó:  1.0 en:  0 0
    datos No ubicado en 2
    se agregó:  0.0 en:  1 0
    datos No ubicado en 3
    se agregó:  0.0 en:  2 0
    datos No ubicado en 4
    se agregó:  0.0 en:  3 0
    datos No ubicado en 5
    se agregó:  0.0 en:  4 0
    datos No ubicado en 6
    se agregó:  0.0 en:  5 0
    datos No ubicado en 7
    se agregó:  0.0 en:  6 0
    datos No ubicado en 8
    se agregó:  0.0 en:  7 0
    datos No ubicado en 9
    se agregó:  0.0 en:  8 0
    datos No ubicado en 10
    se agregó:  0.0 en:  9 0
    concretos Ubicado en 1
    se agregó:  1.0 en:  0 1
    concretos No ubicado en 2
    se agregó:  0.0 en:  1 1
    concretos No ubicado en 3
    se agregó:  0.0 en:  2 1
    concretos No ubicado en 4
    se agregó:  0.0 en:  3 1
    concretos No ubicado en 5
    se agregó:  0.0 en:  4 1
    concretos No ubicado en 6
    se agregó:  0.0 en:  5 1
    concretos No ubicado en 7
    se agregó:  0.0 en:  6 1
    concretos No ubicado en 8
    se agregó:  0.0 en:  7 1
    concretos No ubicado en 9
    se agregó:  0.0 en:  8 1
    concretos No ubicado en 10
    se agregó:  0.0 en:  9 1
    feminicidio Ubicado en 1
    se agregó:  1.0 en:  0 2
    feminicidio Ubicado en 2
    se agregó:  1.0 en:  1 2
    feminicidio No ubicado en 3
    se agregó:  0.0 en:  2 2
    feminicidio No ubicado en 4
    se agregó:  0.0 en:  3 2
    feminicidio No ubicado en 5
    se agregó:  0.0 en:  4 2
    feminicidio No ubicado en 6
    se agregó:  0.0 en:  5 2
    feminicidio No ubicado en 7
    se agregó:  0.0 en:  6 2
    feminicidio No ubicado en 8
    se agregó:  0.0 en:  7 2
    feminicidio No ubicado en 9
    se agregó:  0.0 en:  8 2
    feminicidio No ubicado en 10
    se agregó:  0.0 en:  9 2
    méxico Ubicado en 1
    se agregó:  1.0 en:  0 3
    méxico No ubicado en 2
    se agregó:  0.0 en:  1 3
    méxico No ubicado en 3
    se agregó:  0.0 en:  2 3
    méxico No ubicado en 4
    se agregó:  0.0 en:  3 3
    méxico No ubicado en 5
    se agregó:  0.0 en:  4 3
    méxico No ubicado en 6
    se agregó:  0.0 en:  5 3
    méxico No ubicado en 7
    se agregó:  0.0 en:  6 3
    méxico No ubicado en 8
    se agregó:  0.0 en:  7 3
    méxico No ubicado en 9
    se agregó:  0.0 en:  8 3
    méxico No ubicado en 10
    se agregó:  0.0 en:  9 3
    tomados Ubicado en 1
    se agregó:  1.0 en:  0 4
    tomados No ubicado en 2
    se agregó:  0.0 en:  1 4
    tomados No ubicado en 3
    se agregó:  0.0 en:  2 4
    tomados No ubicado en 4
    se agregó:  0.0 en:  3 4
    tomados No ubicado en 5
    se agregó:  0.0 en:  4 4
    tomados No ubicado en 6
    se agregó:  0.0 en:  5 4
    tomados No ubicado en 7
    se agregó:  0.0 en:  6 4
    tomados No ubicado en 8
    se agregó:  0.0 en:  7 4
    tomados No ubicado en 9
    se agregó:  0.0 en:  8 4
    tomados No ubicado en 10
    se agregó:  0.0 en:  9 4
    sesnp Ubicado en 1
    se agregó:  1.0 en:  0 5
    sesnp No ubicado en 2
    se agregó:  0.0 en:  1 5
    sesnp No ubicado en 3
    se agregó:  0.0 en:  2 5
    sesnp No ubicado en 4
    se agregó:  0.0 en:  3 5
    sesnp No ubicado en 5
    se agregó:  0.0 en:  4 5
    sesnp No ubicado en 6
    se agregó:  0.0 en:  5 5
    sesnp No ubicado en 7
    se agregó:  0.0 en:  6 5
    sesnp No ubicado en 8
    se agregó:  0.0 en:  7 5
    sesnp No ubicado en 9
    se agregó:  0.0 en:  8 5
    sesnp No ubicado en 10
    se agregó:  0.0 en:  9 5
    delito Ubicado en 1
    se agregó:  1.0 en:  0 6
    delito No ubicado en 2
    se agregó:  0.0 en:  1 6
    delito No ubicado en 3
    se agregó:  0.0 en:  2 6
    delito No ubicado en 4
    se agregó:  0.0 en:  3 6
    delito No ubicado en 5
    se agregó:  0.0 en:  4 6
    delito No ubicado en 6
    se agregó:  0.0 en:  5 6
    delito No ubicado en 7
    se agregó:  0.0 en:  6 6
    delito No ubicado en 8
    se agregó:  0.0 en:  7 6
    delito No ubicado en 9
    se agregó:  0.0 en:  8 6
    delito No ubicado en 10
    se agregó:  0.0 en:  9 6
    tendencia Ubicado en 1
    se agregó:  1.0 en:  0 7
    tendencia No ubicado en 2
    se agregó:  0.0 en:  1 7
    tendencia No ubicado en 3
    se agregó:  0.0 en:  2 7
    tendencia No ubicado en 4
    se agregó:  0.0 en:  3 7
    tendencia No ubicado en 5
    se agregó:  0.0 en:  4 7
    tendencia No ubicado en 6
    se agregó:  0.0 en:  5 7
    tendencia No ubicado en 7
    se agregó:  0.0 en:  6 7
    tendencia No ubicado en 8
    se agregó:  0.0 en:  7 7
    tendencia No ubicado en 9
    se agregó:  0.0 en:  8 7
    tendencia No ubicado en 10
    se agregó:  0.0 en:  9 7
    alza Ubicado en 1
    se agregó:  1.0 en:  0 8
    alza No ubicado en 2
    se agregó:  0.0 en:  1 8
    alza No ubicado en 3
    se agregó:  0.0 en:  2 8
    alza No ubicado en 4
    se agregó:  0.0 en:  3 8
    alza No ubicado en 5
    se agregó:  0.0 en:  4 8
    alza No ubicado en 6
    se agregó:  0.0 en:  5 8
    alza No ubicado en 7
    se agregó:  0.0 en:  6 8
    alza No ubicado en 8
    se agregó:  0.0 en:  7 8
    alza No ubicado en 9
    se agregó:  0.0 en:  8 8
    alza No ubicado en 10
    se agregó:  0.0 en:  9 8
    solución Ubicado en 1
    se agregó:  1.0 en:  0 9
    solución No ubicado en 2
    se agregó:  0.0 en:  1 9
    solución No ubicado en 3
    se agregó:  0.0 en:  2 9
    solución No ubicado en 4
    se agregó:  0.0 en:  3 9
    solución No ubicado en 5
    se agregó:  0.0 en:  4 9
    solución No ubicado en 6
    se agregó:  0.0 en:  5 9
    solución No ubicado en 7
    se agregó:  0.0 en:  6 9
    solución No ubicado en 8
    se agregó:  0.0 en:  7 9
    solución No ubicado en 9
    se agregó:  0.0 en:  8 9
    solución No ubicado en 10
    se agregó:  0.0 en:  9 9
    lópez Ubicado en 1
    se agregó:  1.0 en:  0 10
    lópez No ubicado en 2
    se agregó:  0.0 en:  1 10
    lópez No ubicado en 3
    se agregó:  0.0 en:  2 10
    lópez No ubicado en 4
    se agregó:  0.0 en:  3 10
    lópez No ubicado en 5
    se agregó:  0.0 en:  4 10
    lópez No ubicado en 6
    se agregó:  0.0 en:  5 10
    lópez No ubicado en 7
    se agregó:  0.0 en:  6 10
    lópez Ubicado en 8
    se agregó:  1.0 en:  7 10
    lópez No ubicado en 9
    se agregó:  0.0 en:  8 10
    lópez No ubicado en 10
    se agregó:  0.0 en:  9 10
    fiscal Ubicado en 1
    se agregó:  1.0 en:  0 11
    fiscal Ubicado en 2
    se agregó:  1.0 en:  1 11
    fiscal No ubicado en 3
    se agregó:  0.0 en:  2 11
    fiscal No ubicado en 4
    se agregó:  0.0 en:  3 11
    fiscal No ubicado en 5
    se agregó:  0.0 en:  4 11
    fiscal No ubicado en 6
    se agregó:  0.0 en:  5 11
    fiscal No ubicado en 7
    se agregó:  0.0 en:  6 11
    fiscal No ubicado en 8
    se agregó:  0.0 en:  7 11
    fiscal No ubicado en 9
    se agregó:  0.0 en:  8 11
    fiscal No ubicado en 10
    se agregó:  0.0 en:  9 11
    carnal Ubicado en 1
    se agregó:  1.0 en:  0 12
    carnal No ubicado en 2
    se agregó:  0.0 en:  1 12
    carnal No ubicado en 3
    se agregó:  0.0 en:  2 12
    carnal No ubicado en 4
    se agregó:  0.0 en:  3 12
    carnal No ubicado en 5
    se agregó:  0.0 en:  4 12
    carnal No ubicado en 6
    se agregó:  0.0 en:  5 12
    carnal No ubicado en 7
    se agregó:  0.0 en:  6 12
    carnal No ubicado en 8
    se agregó:  0.0 en:  7 12
    carnal No ubicado en 9
    se agregó:  0.0 en:  8 12
    carnal No ubicado en 10
    se agregó:  0.0 en:  9 12
    desaparecerlo Ubicado en 1
    se agregó:  1.0 en:  0 13
    desaparecerlo No ubicado en 2
    se agregó:  0.0 en:  1 13
    desaparecerlo No ubicado en 3
    se agregó:  0.0 en:  2 13
    desaparecerlo No ubicado en 4
    se agregó:  0.0 en:  3 13
    desaparecerlo No ubicado en 5
    se agregó:  0.0 en:  4 13
    desaparecerlo No ubicado en 6
    se agregó:  0.0 en:  5 13
    desaparecerlo No ubicado en 7
    se agregó:  0.0 en:  6 13
    desaparecerlo No ubicado en 8
    se agregó:  0.0 en:  7 13
    desaparecerlo No ubicado en 9
    se agregó:  0.0 en:  8 13
    desaparecerlo No ubicado en 10
    se agregó:  0.0 en:  9 13
    código Ubicado en 1
    se agregó:  1.0 en:  0 14
    código No ubicado en 2
    se agregó:  0.0 en:  1 14
    código No ubicado en 3
    se agregó:  0.0 en:  2 14
    código No ubicado en 4
    se agregó:  0.0 en:  3 14
    código No ubicado en 5
    se agregó:  0.0 en:  4 14
    código No ubicado en 6
    se agregó:  0.0 en:  5 14
    código No ubicado en 7
    se agregó:  0.0 en:  6 14
    código No ubicado en 8
    se agregó:  0.0 en:  7 14
    código No ubicado en 9
    se agregó:  0.0 en:  8 14
    código No ubicado en 10
    se agregó:  0.0 en:  9 14
    penal Ubicado en 1
    se agregó:  1.0 en:  0 15
    penal No ubicado en 2
    se agregó:  0.0 en:  1 15
    penal No ubicado en 3
    se agregó:  0.0 en:  2 15
    penal No ubicado en 4
    se agregó:  0.0 en:  3 15
    penal No ubicado en 5
    se agregó:  0.0 en:  4 15
    penal No ubicado en 6
    se agregó:  0.0 en:  5 15
    penal No ubicado en 7
    se agregó:  0.0 en:  6 15
    penal No ubicado en 8
    se agregó:  0.0 en:  7 15
    penal No ubicado en 9
    se agregó:  0.0 en:  8 15
    penal No ubicado en 10
    se agregó:  0.0 en:  9 15
    gobierno Ubicado en 1
    se agregó:  1.0 en:  0 16
    gobierno No ubicado en 2
    se agregó:  0.0 en:  1 16
    gobierno No ubicado en 3
    se agregó:  0.0 en:  2 16
    gobierno No ubicado en 4
    se agregó:  0.0 en:  3 16
    gobierno No ubicado en 5
    se agregó:  0.0 en:  4 16
    gobierno No ubicado en 6
    se agregó:  0.0 en:  5 16
    gobierno No ubicado en 7
    se agregó:  0.0 en:  6 16
    gobierno No ubicado en 8
    se agregó:  0.0 en:  7 16
    gobierno No ubicado en 9
    se agregó:  0.0 en:  8 16
    gobierno No ubicado en 10
    se agregó:  0.0 en:  9 16
    prefiere Ubicado en 1
    se agregó:  1.0 en:  0 17
    prefiere No ubicado en 2
    se agregó:  0.0 en:  1 17
    prefiere No ubicado en 3
    se agregó:  0.0 en:  2 17
    prefiere No ubicado en 4
    se agregó:  0.0 en:  3 17
    prefiere No ubicado en 5
    se agregó:  0.0 en:  4 17
    prefiere No ubicado en 6
    se agregó:  0.0 en:  5 17
    prefiere No ubicado en 7
    se agregó:  0.0 en:  6 17
    prefiere No ubicado en 8
    se agregó:  0.0 en:  7 17
    prefiere No ubicado en 9
    se agregó:  0.0 en:  8 17
    prefiere No ubicado en 10
    se agregó:  0.0 en:  9 17
    ponerse Ubicado en 1
    se agregó:  1.0 en:  0 18
    ponerse No ubicado en 2
    se agregó:  0.0 en:  1 18
    ponerse No ubicado en 3
    se agregó:  0.0 en:  2 18
    ponerse No ubicado en 4
    se agregó:  0.0 en:  3 18
    ponerse No ubicado en 5
    se agregó:  0.0 en:  4 18
    ponerse No ubicado en 6
    se agregó:  0.0 en:  5 18
    ponerse No ubicado en 7
    se agregó:  0.0 en:  6 18
    ponerse No ubicado en 8
    se agregó:  0.0 en:  7 18
    ponerse No ubicado en 9
    se agregó:  0.0 en:  8 18
    ponerse No ubicado en 10
    se agregó:  0.0 en:  9 18
    lado Ubicado en 1
    se agregó:  1.0 en:  0 19
    lado No ubicado en 2
    se agregó:  0.0 en:  1 19
    lado No ubicado en 3
    se agregó:  0.0 en:  2 19
    lado No ubicado en 4
    se agregó:  0.0 en:  3 19
    lado No ubicado en 5
    se agregó:  0.0 en:  4 19
    lado No ubicado en 6
    se agregó:  0.0 en:  5 19
    lado No ubicado en 7
    se agregó:  0.0 en:  6 19
    lado No ubicado en 8
    se agregó:  0.0 en:  7 19
    lado No ubicado en 9
    se agregó:  0.0 en:  8 19
    lado No ubicado en 10
    se agregó:  0.0 en:  9 19
    delincuente Ubicado en 1
    se agregó:  1.0 en:  0 20
    delincuente No ubicado en 2
    se agregó:  0.0 en:  1 20
    delincuente No ubicado en 3
    se agregó:  0.0 en:  2 20
    delincuente No ubicado en 4
    se agregó:  0.0 en:  3 20
    delincuente No ubicado en 5
    se agregó:  0.0 en:  4 20
    delincuente No ubicado en 6
    se agregó:  0.0 en:  5 20
    delincuente No ubicado en 7
    se agregó:  0.0 en:  6 20
    delincuente No ubicado en 8
    se agregó:  0.0 en:  7 20
    delincuente No ubicado en 9
    se agregó:  0.0 en:  8 20
    delincuente No ubicado en 10
    se agregó:  0.0 en:  9 20
    allanarle Ubicado en 1
    se agregó:  1.0 en:  0 21
    allanarle No ubicado en 2
    se agregó:  0.0 en:  1 21
    allanarle No ubicado en 3
    se agregó:  0.0 en:  2 21
    allanarle No ubicado en 4
    se agregó:  0.0 en:  3 21
    allanarle No ubicado en 5
    se agregó:  0.0 en:  4 21
    allanarle No ubicado en 6
    se agregó:  0.0 en:  5 21
    allanarle No ubicado en 7
    se agregó:  0.0 en:  6 21
    allanarle No ubicado en 8
    se agregó:  0.0 en:  7 21
    allanarle No ubicado en 9
    se agregó:  0.0 en:  8 21
    allanarle No ubicado en 10
    se agregó:  0.0 en:  9 21
    camino Ubicado en 1
    se agregó:  1.0 en:  0 22
    camino No ubicado en 2
    se agregó:  0.0 en:  1 22
    camino No ubicado en 3
    se agregó:  0.0 en:  2 22
    camino No ubicado en 4
    se agregó:  0.0 en:  3 22
    camino No ubicado en 5
    se agregó:  0.0 en:  4 22
    camino No ubicado en 6
    se agregó:  0.0 en:  5 22
    camino No ubicado en 7
    se agregó:  0.0 en:  6 22
    camino No ubicado en 8
    se agregó:  0.0 en:  7 22
    camino No ubicado en 9
    se agregó:  0.0 en:  8 22
    camino No ubicado en 10
    se agregó:  0.0 en:  9 22
    sociedad No ubicado en 1
    se agregó:  0.0 en:  0 23
    sociedad Ubicado en 2
    se agregó:  1.0 en:  1 23
    sociedad No ubicado en 3
    se agregó:  0.0 en:  2 23
    sociedad No ubicado en 4
    se agregó:  0.0 en:  3 23
    sociedad No ubicado en 5
    se agregó:  0.0 en:  4 23
    sociedad No ubicado en 6
    se agregó:  0.0 en:  5 23
    sociedad No ubicado en 7
    se agregó:  0.0 en:  6 23
    sociedad No ubicado en 8
    se agregó:  0.0 en:  7 23
    sociedad No ubicado en 9
    se agregó:  0.0 en:  8 23
    sociedad No ubicado en 10
    se agregó:  0.0 en:  9 23
    general No ubicado en 1
    se agregó:  0.0 en:  0 24
    general Ubicado en 2
    se agregó:  1.0 en:  1 24
    general No ubicado en 3
    se agregó:  0.0 en:  2 24
    general No ubicado en 4
    se agregó:  0.0 en:  3 24
    general No ubicado en 5
    se agregó:  0.0 en:  4 24
    general No ubicado en 6
    se agregó:  0.0 en:  5 24
    general No ubicado en 7
    se agregó:  0.0 en:  6 24
    general No ubicado en 8
    se agregó:  0.0 en:  7 24
    general No ubicado en 9
    se agregó:  0.0 en:  8 24
    general No ubicado en 10
    se agregó:  0.0 en:  9 24
    explicó No ubicado en 1
    se agregó:  0.0 en:  0 25
    explicó Ubicado en 2
    se agregó:  1.0 en:  1 25
    explicó No ubicado en 3
    se agregó:  0.0 en:  2 25
    explicó No ubicado en 4
    se agregó:  0.0 en:  3 25
    explicó No ubicado en 5
    se agregó:  0.0 en:  4 25
    explicó No ubicado en 6
    se agregó:  0.0 en:  5 25
    explicó No ubicado en 7
    se agregó:  0.0 en:  6 25
    explicó No ubicado en 8
    se agregó:  0.0 en:  7 25
    explicó No ubicado en 9
    se agregó:  0.0 en:  8 25
    explicó No ubicado en 10
    se agregó:  0.0 en:  9 25
    propuesta No ubicado en 1
    se agregó:  0.0 en:  0 26
    propuesta Ubicado en 2
    se agregó:  1.0 en:  1 26
    propuesta No ubicado en 3
    se agregó:  0.0 en:  2 26
    propuesta No ubicado en 4
    se agregó:  0.0 en:  3 26
    propuesta No ubicado en 5
    se agregó:  0.0 en:  4 26
    propuesta No ubicado en 6
    se agregó:  0.0 en:  5 26
    propuesta No ubicado en 7
    se agregó:  0.0 en:  6 26
    propuesta No ubicado en 8
    se agregó:  0.0 en:  7 26
    propuesta No ubicado en 9
    se agregó:  0.0 en:  8 26
    propuesta No ubicado en 10
    se agregó:  0.0 en:  9 26
    busca No ubicado en 1
    se agregó:  0.0 en:  0 27
    busca Ubicado en 2
    se agregó:  1.0 en:  1 27
    busca No ubicado en 3
    se agregó:  0.0 en:  2 27
    busca No ubicado en 4
    se agregó:  0.0 en:  3 27
    busca No ubicado en 5
    se agregó:  0.0 en:  4 27
    busca No ubicado en 6
    se agregó:  0.0 en:  5 27
    busca No ubicado en 7
    se agregó:  0.0 en:  6 27
    busca No ubicado en 8
    se agregó:  0.0 en:  7 27
    busca No ubicado en 9
    se agregó:  0.0 en:  8 27
    busca No ubicado en 10
    se agregó:  0.0 en:  9 27
    facilitar No ubicado en 1
    se agregó:  0.0 en:  0 28
    facilitar Ubicado en 2
    se agregó:  1.0 en:  1 28
    facilitar No ubicado en 3
    se agregó:  0.0 en:  2 28
    facilitar No ubicado en 4
    se agregó:  0.0 en:  3 28
    facilitar No ubicado en 5
    se agregó:  0.0 en:  4 28
    facilitar No ubicado en 6
    se agregó:  0.0 en:  5 28
    facilitar No ubicado en 7
    se agregó:  0.0 en:  6 28
    facilitar No ubicado en 8
    se agregó:  0.0 en:  7 28
    facilitar No ubicado en 9
    se agregó:  0.0 en:  8 28
    facilitar No ubicado en 10
    se agregó:  0.0 en:  9 28
    investigación No ubicado en 1
    se agregó:  0.0 en:  0 29
    investigación Ubicado en 2
    se agregó:  1.0 en:  1 29
    investigación No ubicado en 3
    se agregó:  0.0 en:  2 29
    investigación No ubicado en 4
    se agregó:  0.0 en:  3 29
    investigación No ubicado en 5
    se agregó:  0.0 en:  4 29
    investigación No ubicado en 6
    se agregó:  0.0 en:  5 29
    investigación No ubicado en 7
    se agregó:  0.0 en:  6 29
    investigación No ubicado en 8
    se agregó:  0.0 en:  7 29
    investigación No ubicado en 9
    se agregó:  0.0 en:  8 29
    investigación No ubicado en 10
    se agregó:  0.0 en:  9 29
    proteger No ubicado en 1
    se agregó:  0.0 en:  0 30
    proteger Ubicado en 2
    se agregó:  1.0 en:  1 30
    proteger No ubicado en 3
    se agregó:  0.0 en:  2 30
    proteger No ubicado en 4
    se agregó:  0.0 en:  3 30
    proteger No ubicado en 5
    se agregó:  0.0 en:  4 30
    proteger No ubicado en 6
    se agregó:  0.0 en:  5 30
    proteger No ubicado en 7
    se agregó:  0.0 en:  6 30
    proteger No ubicado en 8
    se agregó:  0.0 en:  7 30
    proteger No ubicado en 9
    se agregó:  0.0 en:  8 30
    proteger No ubicado en 10
    se agregó:  0.0 en:  9 30
    víctimas No ubicado en 1
    se agregó:  0.0 en:  0 31
    víctimas Ubicado en 2
    se agregó:  1.0 en:  1 31
    víctimas No ubicado en 3
    se agregó:  0.0 en:  2 31
    víctimas No ubicado en 4
    se agregó:  0.0 en:  3 31
    víctimas No ubicado en 5
    se agregó:  0.0 en:  4 31
    víctimas No ubicado en 6
    se agregó:  0.0 en:  5 31
    víctimas No ubicado en 7
    se agregó:  0.0 en:  6 31
    víctimas No ubicado en 8
    se agregó:  0.0 en:  7 31
    víctimas No ubicado en 9
    se agregó:  0.0 en:  8 31
    víctimas No ubicado en 10
    se agregó:  0.0 en:  9 31
    conferencia No ubicado en 1
    se agregó:  0.0 en:  0 32
    conferencia No ubicado en 2
    se agregó:  0.0 en:  1 32
    conferencia Ubicado en 3
    se agregó:  1.0 en:  2 32
    conferencia Ubicado en 4
    se agregó:  1.0 en:  3 32
    conferencia No ubicado en 5
    se agregó:  0.0 en:  4 32
    conferencia No ubicado en 6
    se agregó:  0.0 en:  5 32
    conferencia No ubicado en 7
    se agregó:  0.0 en:  6 32
    conferencia No ubicado en 8
    se agregó:  0.0 en:  7 32
    conferencia No ubicado en 9
    se agregó:  0.0 en:  8 32
    conferencia No ubicado en 10
    se agregó:  0.0 en:  9 32
    presidente No ubicado en 1
    se agregó:  0.0 en:  0 33
    presidente No ubicado en 2
    se agregó:  0.0 en:  1 33
    presidente Ubicado en 3
    se agregó:  1.0 en:  2 33
    presidente Ubicado en 4
    se agregó:  1.0 en:  3 33
    presidente No ubicado en 5
    se agregó:  0.0 en:  4 33
    presidente No ubicado en 6
    se agregó:  0.0 en:  5 33
    presidente No ubicado en 7
    se agregó:  0.0 en:  6 33
    presidente Ubicado en 8
    se agregó:  1.0 en:  7 33
    presidente No ubicado en 9
    se agregó:  0.0 en:  8 33
    presidente No ubicado en 10
    se agregó:  0.0 en:  9 33
    queda No ubicado en 1
    se agregó:  0.0 en:  0 34
    queda No ubicado en 2
    se agregó:  0.0 en:  1 34
    queda Ubicado en 3
    se agregó:  1.0 en:  2 34
    queda No ubicado en 4
    se agregó:  0.0 en:  3 34
    queda No ubicado en 5
    se agregó:  0.0 en:  4 34
    queda No ubicado en 6
    se agregó:  0.0 en:  5 34
    queda No ubicado en 7
    se agregó:  0.0 en:  6 34
    queda No ubicado en 8
    se agregó:  0.0 en:  7 34
    queda No ubicado en 9
    se agregó:  0.0 en:  8 34
    queda No ubicado en 10
    se agregó:  0.0 en:  9 34
    eu No ubicado en 1
    se agregó:  0.0 en:  0 35
    eu No ubicado en 2
    se agregó:  0.0 en:  1 35
    eu Ubicado en 3
    se agregó:  1.0 en:  2 35
    eu No ubicado en 4
    se agregó:  0.0 en:  3 35
    eu No ubicado en 5
    se agregó:  0.0 en:  4 35
    eu No ubicado en 6
    se agregó:  0.0 en:  5 35
    eu No ubicado en 7
    se agregó:  0.0 en:  6 35
    eu No ubicado en 8
    se agregó:  0.0 en:  7 35
    eu No ubicado en 9
    se agregó:  0.0 en:  8 35
    eu No ubicado en 10
    se agregó:  0.0 en:  9 35
    80 No ubicado en 1
    se agregó:  0.0 en:  0 36
    80 No ubicado en 2
    se agregó:  0.0 en:  1 36
    80 Ubicado en 3
    se agregó:  1.0 en:  2 36
    80 No ubicado en 4
    se agregó:  0.0 en:  3 36
    80 No ubicado en 5
    se agregó:  0.0 en:  4 36
    80 No ubicado en 6
    se agregó:  0.0 en:  5 36
    80 No ubicado en 7
    se agregó:  0.0 en:  6 36
    80 No ubicado en 8
    se agregó:  0.0 en:  7 36
    80 No ubicado en 9
    se agregó:  0.0 en:  8 36
    80 No ubicado en 10
    se agregó:  0.0 en:  9 36
    comisiones No ubicado en 1
    se agregó:  0.0 en:  0 37
    comisiones No ubicado en 2
    se agregó:  0.0 en:  1 37
    comisiones Ubicado en 3
    se agregó:  1.0 en:  2 37
    comisiones No ubicado en 4
    se agregó:  0.0 en:  3 37
    comisiones No ubicado en 5
    se agregó:  0.0 en:  4 37
    comisiones No ubicado en 6
    se agregó:  0.0 en:  5 37
    comisiones No ubicado en 7
    se agregó:  0.0 en:  6 37
    comisiones No ubicado en 8
    se agregó:  0.0 en:  7 37
    comisiones No ubicado en 9
    se agregó:  0.0 en:  8 37
    comisiones No ubicado en 10
    se agregó:  0.0 en:  9 37
    remesas No ubicado en 1
    se agregó:  0.0 en:  0 38
    remesas No ubicado en 2
    se agregó:  0.0 en:  1 38
    remesas Ubicado en 3
    se agregó:  1.0 en:  2 38
    remesas No ubicado en 4
    se agregó:  0.0 en:  3 38
    remesas No ubicado en 5
    se agregó:  0.0 en:  4 38
    remesas No ubicado en 6
    se agregó:  0.0 en:  5 38
    remesas No ubicado en 7
    se agregó:  0.0 en:  6 38
    remesas No ubicado en 8
    se agregó:  0.0 en:  7 38
    remesas No ubicado en 9
    se agregó:  0.0 en:  8 38
    remesas No ubicado en 10
    se agregó:  0.0 en:  9 38
    profeco No ubicado en 1
    se agregó:  0.0 en:  0 39
    profeco No ubicado en 2
    se agregó:  0.0 en:  1 39
    profeco Ubicado en 3
    se agregó:  1.0 en:  2 39
    profeco No ubicado en 4
    se agregó:  0.0 en:  3 39
    profeco No ubicado en 5
    se agregó:  0.0 en:  4 39
    profeco No ubicado en 6
    se agregó:  0.0 en:  5 39
    profeco No ubicado en 7
    se agregó:  0.0 en:  6 39
    profeco No ubicado en 8
    se agregó:  0.0 en:  7 39
    profeco No ubicado en 9
    se agregó:  0.0 en:  8 39
    profeco No ubicado en 10
    se agregó:  0.0 en:  9 39
    entrega No ubicado en 1
    se agregó:  0.0 en:  0 40
    entrega No ubicado en 2
    se agregó:  0.0 en:  1 40
    entrega No ubicado en 3
    se agregó:  0.0 en:  2 40
    entrega Ubicado en 4
    se agregó:  1.0 en:  3 40
    entrega No ubicado en 5
    se agregó:  0.0 en:  4 40
    entrega No ubicado en 6
    se agregó:  0.0 en:  5 40
    entrega No ubicado en 7
    se agregó:  0.0 en:  6 40
    entrega No ubicado en 8
    se agregó:  0.0 en:  7 40
    entrega No ubicado en 9
    se agregó:  0.0 en:  8 40
    entrega No ubicado en 10
    se agregó:  0.0 en:  9 40
    fiscalía No ubicado en 1
    se agregó:  0.0 en:  0 41
    fiscalía No ubicado en 2
    se agregó:  0.0 en:  1 41
    fiscalía No ubicado en 3
    se agregó:  0.0 en:  2 41
    fiscalía Ubicado en 4
    se agregó:  1.0 en:  3 41
    fiscalía No ubicado en 5
    se agregó:  0.0 en:  4 41
    fiscalía No ubicado en 6
    se agregó:  0.0 en:  5 41
    fiscalía Ubicado en 7
    se agregó:  1.0 en:  6 41
    fiscalía No ubicado en 8
    se agregó:  0.0 en:  7 41
    fiscalía No ubicado en 9
    se agregó:  0.0 en:  8 41
    fiscalía No ubicado en 10
    se agregó:  0.0 en:  9 41
    2 No ubicado en 1
    se agregó:  0.0 en:  0 42
    2 No ubicado en 2
    se agregó:  0.0 en:  1 42
    2 No ubicado en 3
    se agregó:  0.0 en:  2 42
    2 Ubicado en 4
    se agregó:  1.0 en:  3 42
    2 Ubicado en 5
    se agregó:  1.0 en:  4 42
    2 No ubicado en 6
    se agregó:  0.0 en:  5 42
    2 No ubicado en 7
    se agregó:  0.0 en:  6 42
    2 No ubicado en 8
    se agregó:  0.0 en:  7 42
    2 No ubicado en 9
    se agregó:  0.0 en:  8 42
    2 No ubicado en 10
    se agregó:  0.0 en:  9 42
    mil No ubicado en 1
    se agregó:  0.0 en:  0 43
    mil No ubicado en 2
    se agregó:  0.0 en:  1 43
    mil No ubicado en 3
    se agregó:  0.0 en:  2 43
    mil Ubicado en 4
    se agregó:  1.0 en:  3 43
    mil Ubicado en 5
    se agregó:  1.0 en:  4 43
    mil No ubicado en 6
    se agregó:  0.0 en:  5 43
    mil No ubicado en 7
    se agregó:  0.0 en:  6 43
    mil No ubicado en 8
    se agregó:  0.0 en:  7 43
    mil No ubicado en 9
    se agregó:  0.0 en:  8 43
    mil No ubicado en 10
    se agregó:  0.0 en:  9 43
    mdp No ubicado en 1
    se agregó:  0.0 en:  0 44
    mdp No ubicado en 2
    se agregó:  0.0 en:  1 44
    mdp No ubicado en 3
    se agregó:  0.0 en:  2 44
    mdp Ubicado en 4
    se agregó:  1.0 en:  3 44
    mdp Ubicado en 5
    se agregó:  1.0 en:  4 44
    mdp No ubicado en 6
    se agregó:  0.0 en:  5 44
    mdp No ubicado en 7
    se agregó:  0.0 en:  6 44
    mdp No ubicado en 8
    se agregó:  0.0 en:  7 44
    mdp No ubicado en 9
    se agregó:  0.0 en:  8 44
    mdp No ubicado en 10
    se agregó:  0.0 en:  9 44
    amlo No ubicado en 1
    se agregó:  0.0 en:  0 45
    amlo No ubicado en 2
    se agregó:  0.0 en:  1 45
    amlo No ubicado en 3
    se agregó:  0.0 en:  2 45
    amlo Ubicado en 4
    se agregó:  1.0 en:  3 45
    amlo No ubicado en 5
    se agregó:  0.0 en:  4 45
    amlo No ubicado en 6
    se agregó:  0.0 en:  5 45
    amlo No ubicado en 7
    se agregó:  0.0 en:  6 45
    amlo No ubicado en 8
    se agregó:  0.0 en:  7 45
    amlo No ubicado en 9
    se agregó:  0.0 en:  8 45
    amlo No ubicado en 10
    se agregó:  0.0 en:  9 45
    premios No ubicado en 1
    se agregó:  0.0 en:  0 46
    premios No ubicado en 2
    se agregó:  0.0 en:  1 46
    premios No ubicado en 3
    se agregó:  0.0 en:  2 46
    premios Ubicado en 4
    se agregó:  1.0 en:  3 46
    premios No ubicado en 5
    se agregó:  0.0 en:  4 46
    premios No ubicado en 6
    se agregó:  0.0 en:  5 46
    premios No ubicado en 7
    se agregó:  0.0 en:  6 46
    premios No ubicado en 8
    se agregó:  0.0 en:  7 46
    premios No ubicado en 9
    se agregó:  0.0 en:  8 46
    premios No ubicado en 10
    se agregó:  0.0 en:  9 46
    rifa No ubicado en 1
    se agregó:  0.0 en:  0 47
    rifa No ubicado en 2
    se agregó:  0.0 en:  1 47
    rifa No ubicado en 3
    se agregó:  0.0 en:  2 47
    rifa Ubicado en 4
    se agregó:  1.0 en:  3 47
    rifa No ubicado en 5
    se agregó:  0.0 en:  4 47
    rifa No ubicado en 6
    se agregó:  0.0 en:  5 47
    rifa No ubicado en 7
    se agregó:  0.0 en:  6 47
    rifa No ubicado en 8
    se agregó:  0.0 en:  7 47
    rifa No ubicado en 9
    se agregó:  0.0 en:  8 47
    rifa No ubicado en 10
    se agregó:  0.0 en:  9 47
    conferenciapresidente No ubicado en 1
    se agregó:  0.0 en:  0 48
    conferenciapresidente No ubicado en 2
    se agregó:  0.0 en:  1 48
    conferenciapresidente No ubicado en 3
    se agregó:  0.0 en:  2 48
    conferenciapresidente No ubicado en 4
    se agregó:  0.0 en:  3 48
    conferenciapresidente Ubicado en 5
    se agregó:  1.0 en:  4 48
    conferenciapresidente No ubicado en 6
    se agregó:  0.0 en:  5 48
    conferenciapresidente No ubicado en 7
    se agregó:  0.0 en:  6 48
    conferenciapresidente No ubicado en 8
    se agregó:  0.0 en:  7 48
    conferenciapresidente No ubicado en 9
    se agregó:  0.0 en:  8 48
    conferenciapresidente No ubicado en 10
    se agregó:  0.0 en:  9 48
    alejandro No ubicado en 1
    se agregó:  0.0 en:  0 49
    alejandro No ubicado en 2
    se agregó:  0.0 en:  1 49
    alejandro No ubicado en 3
    se agregó:  0.0 en:  2 49
    alejandro No ubicado en 4
    se agregó:  0.0 en:  3 49
    alejandro Ubicado en 5
    se agregó:  1.0 en:  4 49
    alejandro No ubicado en 6
    se agregó:  0.0 en:  5 49
    alejandro No ubicado en 7
    se agregó:  0.0 en:  6 49
    alejandro No ubicado en 8
    se agregó:  0.0 en:  7 49
    alejandro No ubicado en 9
    se agregó:  0.0 en:  8 49
    alejandro No ubicado en 10
    se agregó:  0.0 en:  9 49
    gertz No ubicado en 1
    se agregó:  0.0 en:  0 50
    gertz No ubicado en 2
    se agregó:  0.0 en:  1 50
    gertz No ubicado en 3
    se agregó:  0.0 en:  2 50
    gertz No ubicado en 4
    se agregó:  0.0 en:  3 50
    gertz Ubicado en 5
    se agregó:  1.0 en:  4 50
    gertz No ubicado en 6
    se agregó:  0.0 en:  5 50
    gertz No ubicado en 7
    se agregó:  0.0 en:  6 50
    gertz No ubicado en 8
    se agregó:  0.0 en:  7 50
    gertz No ubicado en 9
    se agregó:  0.0 en:  8 50
    gertz No ubicado en 10
    se agregó:  0.0 en:  9 50
    manero No ubicado en 1
    se agregó:  0.0 en:  0 51
    manero No ubicado en 2
    se agregó:  0.0 en:  1 51
    manero No ubicado en 3
    se agregó:  0.0 en:  2 51
    manero No ubicado en 4
    se agregó:  0.0 en:  3 51
    manero Ubicado en 5
    se agregó:  1.0 en:  4 51
    manero No ubicado en 6
    se agregó:  0.0 en:  5 51
    manero No ubicado en 7
    se agregó:  0.0 en:  6 51
    manero No ubicado en 8
    se agregó:  0.0 en:  7 51
    manero No ubicado en 9
    se agregó:  0.0 en:  8 51
    manero No ubicado en 10
    se agregó:  0.0 en:  9 51
    titular No ubicado en 1
    se agregó:  0.0 en:  0 52
    titular No ubicado en 2
    se agregó:  0.0 en:  1 52
    titular No ubicado en 3
    se agregó:  0.0 en:  2 52
    titular No ubicado en 4
    se agregó:  0.0 en:  3 52
    titular Ubicado en 5
    se agregó:  1.0 en:  4 52
    titular No ubicado en 6
    se agregó:  0.0 en:  5 52
    titular No ubicado en 7
    se agregó:  0.0 en:  6 52
    titular No ubicado en 8
    se agregó:  0.0 en:  7 52
    titular No ubicado en 9
    se agregó:  0.0 en:  8 52
    titular No ubicado en 10
    se agregó:  0.0 en:  9 52
    fgr No ubicado en 1
    se agregó:  0.0 en:  0 53
    fgr No ubicado en 2
    se agregó:  0.0 en:  1 53
    fgr No ubicado en 3
    se agregó:  0.0 en:  2 53
    fgr No ubicado en 4
    se agregó:  0.0 en:  3 53
    fgr Ubicado en 5
    se agregó:  1.0 en:  4 53
    fgr No ubicado en 6
    se agregó:  0.0 en:  5 53
    fgr No ubicado en 7
    se agregó:  0.0 en:  6 53
    fgr No ubicado en 8
    se agregó:  0.0 en:  7 53
    fgr No ubicado en 9
    se agregó:  0.0 en:  8 53
    fgr No ubicado en 10
    se agregó:  0.0 en:  9 53
    mexico No ubicado en 1
    se agregó:  0.0 en:  0 54
    mexico No ubicado en 2
    se agregó:  0.0 en:  1 54
    mexico No ubicado en 3
    se agregó:  0.0 en:  2 54
    mexico No ubicado en 4
    se agregó:  0.0 en:  3 54
    mexico Ubicado en 5
    se agregó:  1.0 en:  4 54
    mexico No ubicado en 6
    se agregó:  0.0 en:  5 54
    mexico No ubicado en 7
    se agregó:  0.0 en:  6 54
    mexico No ubicado en 8
    se agregó:  0.0 en:  7 54
    mexico No ubicado en 9
    se agregó:  0.0 en:  8 54
    mexico No ubicado en 10
    se agregó:  0.0 en:  9 54
    entregó No ubicado en 1
    se agregó:  0.0 en:  0 55
    entregó No ubicado en 2
    se agregó:  0.0 en:  1 55
    entregó No ubicado en 3
    se agregó:  0.0 en:  2 55
    entregó No ubicado en 4
    se agregó:  0.0 en:  3 55
    entregó Ubicado en 5
    se agregó:  1.0 en:  4 55
    entregó No ubicado en 6
    se agregó:  0.0 en:  5 55
    entregó No ubicado en 7
    se agregó:  0.0 en:  6 55
    entregó No ubicado en 8
    se agregó:  0.0 en:  7 55
    entregó No ubicado en 9
    se agregó:  0.0 en:  8 55
    entregó No ubicado en 10
    se agregó:  0.0 en:  9 55
    instituto No ubicado en 1
    se agregó:  0.0 en:  0 56
    instituto No ubicado en 2
    se agregó:  0.0 en:  1 56
    instituto No ubicado en 3
    se agregó:  0.0 en:  2 56
    instituto No ubicado en 4
    se agregó:  0.0 en:  3 56
    instituto Ubicado en 5
    se agregó:  1.0 en:  4 56
    instituto No ubicado en 6
    se agregó:  0.0 en:  5 56
    instituto No ubicado en 7
    se agregó:  0.0 en:  6 56
    instituto No ubicado en 8
    se agregó:  0.0 en:  7 56
    instituto No ubicado en 9
    se agregó:  0.0 en:  8 56
    instituto No ubicado en 10
    se agregó:  0.0 en:  9 56
    devolver No ubicado en 1
    se agregó:  0.0 en:  0 57
    devolver No ubicado en 2
    se agregó:  0.0 en:  1 57
    devolver No ubicado en 3
    se agregó:  0.0 en:  2 57
    devolver No ubicado en 4
    se agregó:  0.0 en:  3 57
    devolver Ubicado en 5
    se agregó:  1.0 en:  4 57
    devolver No ubicado en 6
    se agregó:  0.0 en:  5 57
    devolver No ubicado en 7
    se agregó:  0.0 en:  6 57
    devolver No ubicado en 8
    se agregó:  0.0 en:  7 57
    devolver No ubicado en 9
    se agregó:  0.0 en:  8 57
    devolver No ubicado en 10
    se agregó:  0.0 en:  9 57
    pueblo No ubicado en 1
    se agregó:  0.0 en:  0 58
    pueblo No ubicado en 2
    se agregó:  0.0 en:  1 58
    pueblo No ubicado en 3
    se agregó:  0.0 en:  2 58
    pueblo No ubicado en 4
    se agregó:  0.0 en:  3 58
    pueblo Ubicado en 5
    se agregó:  1.0 en:  4 58
    pueblo No ubicado en 6
    se agregó:  0.0 en:  5 58
    pueblo No ubicado en 7
    se agregó:  0.0 en:  6 58
    pueblo No ubicado en 8
    se agregó:  0.0 en:  7 58
    pueblo No ubicado en 9
    se agregó:  0.0 en:  8 58
    pueblo No ubicado en 10
    se agregó:  0.0 en:  9 58
    robado No ubicado en 1
    se agregó:  0.0 en:  0 59
    robado No ubicado en 2
    se agregó:  0.0 en:  1 59
    robado No ubicado en 3
    se agregó:  0.0 en:  2 59
    robado No ubicado en 4
    se agregó:  0.0 en:  3 59
    robado Ubicado en 5
    se agregó:  1.0 en:  4 59
    robado No ubicado en 6
    se agregó:  0.0 en:  5 59
    robado No ubicado en 7
    se agregó:  0.0 en:  6 59
    robado No ubicado en 8
    se agregó:  0.0 en:  7 59
    robado No ubicado en 9
    se agregó:  0.0 en:  8 59
    robado No ubicado en 10
    se agregó:  0.0 en:  9 59
    bonitas No ubicado en 1
    se agregó:  0.0 en:  0 60
    bonitas No ubicado en 2
    se agregó:  0.0 en:  1 60
    bonitas No ubicado en 3
    se agregó:  0.0 en:  2 60
    bonitas No ubicado en 4
    se agregó:  0.0 en:  3 60
    bonitas No ubicado en 5
    se agregó:  0.0 en:  4 60
    bonitas Ubicado en 6
    se agregó:  1.0 en:  5 60
    bonitas No ubicado en 7
    se agregó:  0.0 en:  6 60
    bonitas No ubicado en 8
    se agregó:  0.0 en:  7 60
    bonitas No ubicado en 9
    se agregó:  0.0 en:  8 60
    bonitas No ubicado en 10
    se agregó:  0.0 en:  9 60
    palabras No ubicado en 1
    se agregó:  0.0 en:  0 61
    palabras No ubicado en 2
    se agregó:  0.0 en:  1 61
    palabras No ubicado en 3
    se agregó:  0.0 en:  2 61
    palabras No ubicado en 4
    se agregó:  0.0 en:  3 61
    palabras No ubicado en 5
    se agregó:  0.0 en:  4 61
    palabras Ubicado en 6
    se agregó:  1.0 en:  5 61
    palabras No ubicado en 7
    se agregó:  0.0 en:  6 61
    palabras No ubicado en 8
    se agregó:  0.0 en:  7 61
    palabras No ubicado en 9
    se agregó:  0.0 en:  8 61
    palabras No ubicado en 10
    se agregó:  0.0 en:  9 61
    emerson No ubicado en 1
    se agregó:  0.0 en:  0 62
    emerson No ubicado en 2
    se agregó:  0.0 en:  1 62
    emerson No ubicado en 3
    se agregó:  0.0 en:  2 62
    emerson No ubicado en 4
    se agregó:  0.0 en:  3 62
    emerson No ubicado en 5
    se agregó:  0.0 en:  4 62
    emerson Ubicado en 6
    se agregó:  1.0 en:  5 62
    emerson No ubicado en 7
    se agregó:  0.0 en:  6 62
    emerson No ubicado en 8
    se agregó:  0.0 en:  7 62
    emerson No ubicado en 9
    se agregó:  0.0 en:  8 62
    emerson No ubicado en 10
    se agregó:  0.0 en:  9 62
    fittipaldi No ubicado en 1
    se agregó:  0.0 en:  0 63
    fittipaldi No ubicado en 2
    se agregó:  0.0 en:  1 63
    fittipaldi No ubicado en 3
    se agregó:  0.0 en:  2 63
    fittipaldi No ubicado en 4
    se agregó:  0.0 en:  3 63
    fittipaldi No ubicado en 5
    se agregó:  0.0 en:  4 63
    fittipaldi Ubicado en 6
    se agregó:  1.0 en:  5 63
    fittipaldi No ubicado en 7
    se agregó:  0.0 en:  6 63
    fittipaldi No ubicado en 8
    se agregó:  0.0 en:  7 63
    fittipaldi No ubicado en 9
    se agregó:  0.0 en:  8 63
    fittipaldi No ubicado en 10
    se agregó:  0.0 en:  9 63
    fernando No ubicado en 1
    se agregó:  0.0 en:  0 64
    fernando No ubicado en 2
    se agregó:  0.0 en:  1 64
    fernando No ubicado en 3
    se agregó:  0.0 en:  2 64
    fernando No ubicado en 4
    se agregó:  0.0 en:  3 64
    fernando No ubicado en 5
    se agregó:  0.0 en:  4 64
    fernando Ubicado en 6
    se agregó:  1.0 en:  5 64
    fernando No ubicado en 7
    se agregó:  0.0 en:  6 64
    fernando No ubicado en 8
    se agregó:  0.0 en:  7 64
    fernando No ubicado en 9
    se agregó:  0.0 en:  8 64
    fernando No ubicado en 10
    se agregó:  0.0 en:  9 64
    alonso No ubicado en 1
    se agregó:  0.0 en:  0 65
    alonso No ubicado en 2
    se agregó:  0.0 en:  1 65
    alonso No ubicado en 3
    se agregó:  0.0 en:  2 65
    alonso No ubicado en 4
    se agregó:  0.0 en:  3 65
    alonso No ubicado en 5
    se agregó:  0.0 en:  4 65
    alonso Ubicado en 6
    se agregó:  1.0 en:  5 65
    alonso No ubicado en 7
    se agregó:  0.0 en:  6 65
    alonso No ubicado en 8
    se agregó:  0.0 en:  7 65
    alonso No ubicado en 9
    se agregó:  0.0 en:  8 65
    alonso No ubicado en 10
    se agregó:  0.0 en:  9 65
    fenómeno No ubicado en 1
    se agregó:  0.0 en:  0 66
    fenómeno No ubicado en 2
    se agregó:  0.0 en:  1 66
    fenómeno No ubicado en 3
    se agregó:  0.0 en:  2 66
    fenómeno No ubicado en 4
    se agregó:  0.0 en:  3 66
    fenómeno No ubicado en 5
    se agregó:  0.0 en:  4 66
    fenómeno Ubicado en 6
    se agregó:  1.0 en:  5 66
    fenómeno No ubicado en 7
    se agregó:  0.0 en:  6 66
    fenómeno No ubicado en 8
    se agregó:  0.0 en:  7 66
    fenómeno No ubicado en 9
    se agregó:  0.0 en:  8 66
    fenómeno No ubicado en 10
    se agregó:  0.0 en:  9 66
    talentoso No ubicado en 1
    se agregó:  0.0 en:  0 67
    talentoso No ubicado en 2
    se agregó:  0.0 en:  1 67
    talentoso No ubicado en 3
    se agregó:  0.0 en:  2 67
    talentoso No ubicado en 4
    se agregó:  0.0 en:  3 67
    talentoso No ubicado en 5
    se agregó:  0.0 en:  4 67
    talentoso Ubicado en 6
    se agregó:  1.0 en:  5 67
    talentoso No ubicado en 7
    se agregó:  0.0 en:  6 67
    talentoso No ubicado en 8
    se agregó:  0.0 en:  7 67
    talentoso No ubicado en 9
    se agregó:  0.0 en:  8 67
    talentoso No ubicado en 10
    se agregó:  0.0 en:  9 67
    espero No ubicado en 1
    se agregó:  0.0 en:  0 68
    espero No ubicado en 2
    se agregó:  0.0 en:  1 68
    espero No ubicado en 3
    se agregó:  0.0 en:  2 68
    espero No ubicado en 4
    se agregó:  0.0 en:  3 68
    espero No ubicado en 5
    se agregó:  0.0 en:  4 68
    espero Ubicado en 6
    se agregó:  1.0 en:  5 68
    espero No ubicado en 7
    se agregó:  0.0 en:  6 68
    espero No ubicado en 8
    se agregó:  0.0 en:  7 68
    espero No ubicado en 9
    se agregó:  0.0 en:  8 68
    espero No ubicado en 10
    se agregó:  0.0 en:  9 68
    vuelva No ubicado en 1
    se agregó:  0.0 en:  0 69
    vuelva No ubicado en 2
    se agregó:  0.0 en:  1 69
    vuelva No ubicado en 3
    se agregó:  0.0 en:  2 69
    vuelva No ubicado en 4
    se agregó:  0.0 en:  3 69
    vuelva No ubicado en 5
    se agregó:  0.0 en:  4 69
    vuelva Ubicado en 6
    se agregó:  1.0 en:  5 69
    vuelva No ubicado en 7
    se agregó:  0.0 en:  6 69
    vuelva No ubicado en 8
    se agregó:  0.0 en:  7 69
    vuelva No ubicado en 9
    se agregó:  0.0 en:  8 69
    vuelva No ubicado en 10
    se agregó:  0.0 en:  9 69
    fórmula No ubicado en 1
    se agregó:  0.0 en:  0 70
    fórmula No ubicado en 2
    se agregó:  0.0 en:  1 70
    fórmula No ubicado en 3
    se agregó:  0.0 en:  2 70
    fórmula No ubicado en 4
    se agregó:  0.0 en:  3 70
    fórmula No ubicado en 5
    se agregó:  0.0 en:  4 70
    fórmula Ubicado en 6
    se agregó:  1.0 en:  5 70
    fórmula No ubicado en 7
    se agregó:  0.0 en:  6 70
    fórmula No ubicado en 8
    se agregó:  0.0 en:  7 70
    fórmula No ubicado en 9
    se agregó:  0.0 en:  8 70
    fórmula No ubicado en 10
    se agregó:  0.0 en:  9 70
    1 No ubicado en 1
    se agregó:  0.0 en:  0 71
    1 No ubicado en 2
    se agregó:  0.0 en:  1 71
    1 No ubicado en 3
    se agregó:  0.0 en:  2 71
    1 No ubicado en 4
    se agregó:  0.0 en:  3 71
    1 No ubicado en 5
    se agregó:  0.0 en:  4 71
    1 Ubicado en 6
    se agregó:  1.0 en:  5 71
    1 No ubicado en 7
    se agregó:  0.0 en:  6 71
    1 No ubicado en 8
    se agregó:  0.0 en:  7 71
    1 No ubicado en 9
    se agregó:  0.0 en:  8 71
    1 No ubicado en 10
    se agregó:  0.0 en:  9 71
    2021 No ubicado en 1
    se agregó:  0.0 en:  0 72
    2021 No ubicado en 2
    se agregó:  0.0 en:  1 72
    2021 No ubicado en 3
    se agregó:  0.0 en:  2 72
    2021 No ubicado en 4
    se agregó:  0.0 en:  3 72
    2021 No ubicado en 5
    se agregó:  0.0 en:  4 72
    2021 Ubicado en 6
    se agregó:  1.0 en:  5 72
    2021 No ubicado en 7
    se agregó:  0.0 en:  6 72
    2021 No ubicado en 8
    se agregó:  0.0 en:  7 72
    2021 No ubicado en 9
    se agregó:  0.0 en:  8 72
    2021 No ubicado en 10
    se agregó:  0.0 en:  9 72
    acuerdo No ubicado en 1
    se agregó:  0.0 en:  0 73
    acuerdo No ubicado en 2
    se agregó:  0.0 en:  1 73
    acuerdo No ubicado en 3
    se agregó:  0.0 en:  2 73
    acuerdo No ubicado en 4
    se agregó:  0.0 en:  3 73
    acuerdo No ubicado en 5
    se agregó:  0.0 en:  4 73
    acuerdo No ubicado en 6
    se agregó:  0.0 en:  5 73
    acuerdo Ubicado en 7
    se agregó:  1.0 en:  6 73
    acuerdo No ubicado en 8
    se agregó:  0.0 en:  7 73
    acuerdo No ubicado en 9
    se agregó:  0.0 en:  8 73
    acuerdo No ubicado en 10
    se agregó:  0.0 en:  9 73
    imágenes No ubicado en 1
    se agregó:  0.0 en:  0 74
    imágenes No ubicado en 2
    se agregó:  0.0 en:  1 74
    imágenes No ubicado en 3
    se agregó:  0.0 en:  2 74
    imágenes No ubicado en 4
    se agregó:  0.0 en:  3 74
    imágenes No ubicado en 5
    se agregó:  0.0 en:  4 74
    imágenes No ubicado en 6
    se agregó:  0.0 en:  5 74
    imágenes Ubicado en 7
    se agregó:  1.0 en:  6 74
    imágenes No ubicado en 8
    se agregó:  0.0 en:  7 74
    imágenes No ubicado en 9
    se agregó:  0.0 en:  8 74
    imágenes No ubicado en 10
    se agregó:  0.0 en:  9 74
    obtenidas No ubicado en 1
    se agregó:  0.0 en:  0 75
    obtenidas No ubicado en 2
    se agregó:  0.0 en:  1 75
    obtenidas No ubicado en 3
    se agregó:  0.0 en:  2 75
    obtenidas No ubicado en 4
    se agregó:  0.0 en:  3 75
    obtenidas No ubicado en 5
    se agregó:  0.0 en:  4 75
    obtenidas No ubicado en 6
    se agregó:  0.0 en:  5 75
    obtenidas Ubicado en 7
    se agregó:  1.0 en:  6 75
    obtenidas No ubicado en 8
    se agregó:  0.0 en:  7 75
    obtenidas No ubicado en 9
    se agregó:  0.0 en:  8 75
    obtenidas No ubicado en 10
    se agregó:  0.0 en:  9 75
    capitalina No ubicado en 1
    se agregó:  0.0 en:  0 76
    capitalina No ubicado en 2
    se agregó:  0.0 en:  1 76
    capitalina No ubicado en 3
    se agregó:  0.0 en:  2 76
    capitalina No ubicado en 4
    se agregó:  0.0 en:  3 76
    capitalina No ubicado en 5
    se agregó:  0.0 en:  4 76
    capitalina No ubicado en 6
    se agregó:  0.0 en:  5 76
    capitalina Ubicado en 7
    se agregó:  1.0 en:  6 76
    capitalina No ubicado en 8
    se agregó:  0.0 en:  7 76
    capitalina No ubicado en 9
    se agregó:  0.0 en:  8 76
    capitalina No ubicado en 10
    se agregó:  0.0 en:  9 76
    mujer No ubicado en 1
    se agregó:  0.0 en:  0 77
    mujer No ubicado en 2
    se agregó:  0.0 en:  1 77
    mujer No ubicado en 3
    se agregó:  0.0 en:  2 77
    mujer No ubicado en 4
    se agregó:  0.0 en:  3 77
    mujer No ubicado en 5
    se agregó:  0.0 en:  4 77
    mujer No ubicado en 6
    se agregó:  0.0 en:  5 77
    mujer Ubicado en 7
    se agregó:  1.0 en:  6 77
    mujer No ubicado en 8
    se agregó:  0.0 en:  7 77
    mujer No ubicado en 9
    se agregó:  0.0 en:  8 77
    mujer No ubicado en 10
    se agregó:  0.0 en:  9 77
    subió No ubicado en 1
    se agregó:  0.0 en:  0 78
    subió No ubicado en 2
    se agregó:  0.0 en:  1 78
    subió No ubicado en 3
    se agregó:  0.0 en:  2 78
    subió No ubicado en 4
    se agregó:  0.0 en:  3 78
    subió No ubicado en 5
    se agregó:  0.0 en:  4 78
    subió No ubicado en 6
    se agregó:  0.0 en:  5 78
    subió Ubicado en 7
    se agregó:  1.0 en:  6 78
    subió No ubicado en 8
    se agregó:  0.0 en:  7 78
    subió No ubicado en 9
    se agregó:  0.0 en:  8 78
    subió No ubicado en 10
    se agregó:  0.0 en:  9 78
    fátima No ubicado en 1
    se agregó:  0.0 en:  0 79
    fátima No ubicado en 2
    se agregó:  0.0 en:  1 79
    fátima No ubicado en 3
    se agregó:  0.0 en:  2 79
    fátima No ubicado en 4
    se agregó:  0.0 en:  3 79
    fátima No ubicado en 5
    se agregó:  0.0 en:  4 79
    fátima No ubicado en 6
    se agregó:  0.0 en:  5 79
    fátima Ubicado en 7
    se agregó:  1.0 en:  6 79
    fátima No ubicado en 8
    se agregó:  0.0 en:  7 79
    fátima No ubicado en 9
    se agregó:  0.0 en:  8 79
    fátima No ubicado en 10
    se agregó:  0.0 en:  9 79
    vehículo No ubicado en 1
    se agregó:  0.0 en:  0 80
    vehículo No ubicado en 2
    se agregó:  0.0 en:  1 80
    vehículo No ubicado en 3
    se agregó:  0.0 en:  2 80
    vehículo No ubicado en 4
    se agregó:  0.0 en:  3 80
    vehículo No ubicado en 5
    se agregó:  0.0 en:  4 80
    vehículo No ubicado en 6
    se agregó:  0.0 en:  5 80
    vehículo Ubicado en 7
    se agregó:  1.0 en:  6 80
    vehículo No ubicado en 8
    se agregó:  0.0 en:  7 80
    vehículo No ubicado en 9
    se agregó:  0.0 en:  8 80
    vehículo No ubicado en 10
    se agregó:  0.0 en:  9 80
    blanco No ubicado en 1
    se agregó:  0.0 en:  0 81
    blanco No ubicado en 2
    se agregó:  0.0 en:  1 81
    blanco No ubicado en 3
    se agregó:  0.0 en:  2 81
    blanco No ubicado en 4
    se agregó:  0.0 en:  3 81
    blanco No ubicado en 5
    se agregó:  0.0 en:  4 81
    blanco No ubicado en 6
    se agregó:  0.0 en:  5 81
    blanco Ubicado en 7
    se agregó:  1.0 en:  6 81
    blanco No ubicado en 8
    se agregó:  0.0 en:  7 81
    blanco No ubicado en 9
    se agregó:  0.0 en:  8 81
    blanco No ubicado en 10
    se agregó:  0.0 en:  9 81
    obrador No ubicado en 1
    se agregó:  0.0 en:  0 82
    obrador No ubicado en 2
    se agregó:  0.0 en:  1 82
    obrador No ubicado en 3
    se agregó:  0.0 en:  2 82
    obrador No ubicado en 4
    se agregó:  0.0 en:  3 82
    obrador No ubicado en 5
    se agregó:  0.0 en:  4 82
    obrador No ubicado en 6
    se agregó:  0.0 en:  5 82
    obrador No ubicado en 7
    se agregó:  0.0 en:  6 82
    obrador Ubicado en 8
    se agregó:  1.0 en:  7 82
    obrador No ubicado en 9
    se agregó:  0.0 en:  8 82
    obrador No ubicado en 10
    se agregó:  0.0 en:  9 82
    comprometió No ubicado en 1
    se agregó:  0.0 en:  0 83
    comprometió No ubicado en 2
    se agregó:  0.0 en:  1 83
    comprometió No ubicado en 3
    se agregó:  0.0 en:  2 83
    comprometió No ubicado en 4
    se agregó:  0.0 en:  3 83
    comprometió No ubicado en 5
    se agregó:  0.0 en:  4 83
    comprometió No ubicado en 6
    se agregó:  0.0 en:  5 83
    comprometió No ubicado en 7
    se agregó:  0.0 en:  6 83
    comprometió Ubicado en 8
    se agregó:  1.0 en:  7 83
    comprometió No ubicado en 9
    se agregó:  0.0 en:  8 83
    comprometió No ubicado en 10
    se agregó:  0.0 en:  9 83
    enfrentar No ubicado en 1
    se agregó:  0.0 en:  0 84
    enfrentar No ubicado en 2
    se agregó:  0.0 en:  1 84
    enfrentar No ubicado en 3
    se agregó:  0.0 en:  2 84
    enfrentar No ubicado en 4
    se agregó:  0.0 en:  3 84
    enfrentar No ubicado en 5
    se agregó:  0.0 en:  4 84
    enfrentar No ubicado en 6
    se agregó:  0.0 en:  5 84
    enfrentar No ubicado en 7
    se agregó:  0.0 en:  6 84
    enfrentar Ubicado en 8
    se agregó:  1.0 en:  7 84
    enfrentar No ubicado en 9
    se agregó:  0.0 en:  8 84
    enfrentar No ubicado en 10
    se agregó:  0.0 en:  9 84
    empresarios No ubicado en 1
    se agregó:  0.0 en:  0 85
    empresarios No ubicado en 2
    se agregó:  0.0 en:  1 85
    empresarios No ubicado en 3
    se agregó:  0.0 en:  2 85
    empresarios No ubicado en 4
    se agregó:  0.0 en:  3 85
    empresarios No ubicado en 5
    se agregó:  0.0 en:  4 85
    empresarios No ubicado en 6
    se agregó:  0.0 en:  5 85
    empresarios No ubicado en 7
    se agregó:  0.0 en:  6 85
    empresarios Ubicado en 8
    se agregó:  1.0 en:  7 85
    empresarios No ubicado en 9
    se agregó:  0.0 en:  8 85
    empresarios No ubicado en 10
    se agregó:  0.0 en:  9 85
    sindicatos No ubicado en 1
    se agregó:  0.0 en:  0 86
    sindicatos No ubicado en 2
    se agregó:  0.0 en:  1 86
    sindicatos No ubicado en 3
    se agregó:  0.0 en:  2 86
    sindicatos No ubicado en 4
    se agregó:  0.0 en:  3 86
    sindicatos No ubicado en 5
    se agregó:  0.0 en:  4 86
    sindicatos No ubicado en 6
    se agregó:  0.0 en:  5 86
    sindicatos No ubicado en 7
    se agregó:  0.0 en:  6 86
    sindicatos Ubicado en 8
    se agregó:  1.0 en:  7 86
    sindicatos No ubicado en 9
    se agregó:  0.0 en:  8 86
    sindicatos No ubicado en 10
    se agregó:  0.0 en:  9 86
    problema No ubicado en 1
    se agregó:  0.0 en:  0 87
    problema No ubicado en 2
    se agregó:  0.0 en:  1 87
    problema No ubicado en 3
    se agregó:  0.0 en:  2 87
    problema No ubicado en 4
    se agregó:  0.0 en:  3 87
    problema No ubicado en 5
    se agregó:  0.0 en:  4 87
    problema No ubicado en 6
    se agregó:  0.0 en:  5 87
    problema No ubicado en 7
    se agregó:  0.0 en:  6 87
    problema Ubicado en 8
    se agregó:  1.0 en:  7 87
    problema No ubicado en 9
    se agregó:  0.0 en:  8 87
    problema No ubicado en 10
    se agregó:  0.0 en:  9 87
    pensiones No ubicado en 1
    se agregó:  0.0 en:  0 88
    pensiones No ubicado en 2
    se agregó:  0.0 en:  1 88
    pensiones No ubicado en 3
    se agregó:  0.0 en:  2 88
    pensiones No ubicado en 4
    se agregó:  0.0 en:  3 88
    pensiones No ubicado en 5
    se agregó:  0.0 en:  4 88
    pensiones No ubicado en 6
    se agregó:  0.0 en:  5 88
    pensiones No ubicado en 7
    se agregó:  0.0 en:  6 88
    pensiones Ubicado en 8
    se agregó:  1.0 en:  7 88
    pensiones No ubicado en 9
    se agregó:  0.0 en:  8 88
    pensiones No ubicado en 10
    se agregó:  0.0 en:  9 88
    trabajadores No ubicado en 1
    se agregó:  0.0 en:  0 89
    trabajadores No ubicado en 2
    se agregó:  0.0 en:  1 89
    trabajadores No ubicado en 3
    se agregó:  0.0 en:  2 89
    trabajadores No ubicado en 4
    se agregó:  0.0 en:  3 89
    trabajadores No ubicado en 5
    se agregó:  0.0 en:  4 89
    trabajadores No ubicado en 6
    se agregó:  0.0 en:  5 89
    trabajadores No ubicado en 7
    se agregó:  0.0 en:  6 89
    trabajadores Ubicado en 8
    se agregó:  1.0 en:  7 89
    trabajadores No ubicado en 9
    se agregó:  0.0 en:  8 89
    trabajadores No ubicado en 10
    se agregó:  0.0 en:  9 89
    estudiante No ubicado en 1
    se agregó:  0.0 en:  0 90
    estudiante No ubicado en 2
    se agregó:  0.0 en:  1 90
    estudiante No ubicado en 3
    se agregó:  0.0 en:  2 90
    estudiante No ubicado en 4
    se agregó:  0.0 en:  3 90
    estudiante No ubicado en 5
    se agregó:  0.0 en:  4 90
    estudiante No ubicado en 6
    se agregó:  0.0 en:  5 90
    estudiante No ubicado en 7
    se agregó:  0.0 en:  6 90
    estudiante No ubicado en 8
    se agregó:  0.0 en:  7 90
    estudiante Ubicado en 9
    se agregó:  1.0 en:  8 90
    estudiante No ubicado en 10
    se agregó:  0.0 en:  9 90
    grave No ubicado en 1
    se agregó:  0.0 en:  0 91
    grave No ubicado en 2
    se agregó:  0.0 en:  1 91
    grave No ubicado en 3
    se agregó:  0.0 en:  2 91
    grave No ubicado en 4
    se agregó:  0.0 en:  3 91
    grave No ubicado en 5
    se agregó:  0.0 en:  4 91
    grave No ubicado en 6
    se agregó:  0.0 en:  5 91
    grave No ubicado en 7
    se agregó:  0.0 en:  6 91
    grave No ubicado en 8
    se agregó:  0.0 en:  7 91
    grave Ubicado en 9
    se agregó:  1.0 en:  8 91
    grave No ubicado en 10
    se agregó:  0.0 en:  9 91
    tras No ubicado en 1
    se agregó:  0.0 en:  0 92
    tras No ubicado en 2
    se agregó:  0.0 en:  1 92
    tras No ubicado en 3
    se agregó:  0.0 en:  2 92
    tras No ubicado en 4
    se agregó:  0.0 en:  3 92
    tras No ubicado en 5
    se agregó:  0.0 en:  4 92
    tras No ubicado en 6
    se agregó:  0.0 en:  5 92
    tras No ubicado en 7
    se agregó:  0.0 en:  6 92
    tras No ubicado en 8
    se agregó:  0.0 en:  7 92
    tras Ubicado en 9
    se agregó:  1.0 en:  8 92
    tras No ubicado en 10
    se agregó:  0.0 en:  9 92
    desalojo No ubicado en 1
    se agregó:  0.0 en:  0 93
    desalojo No ubicado en 2
    se agregó:  0.0 en:  1 93
    desalojo No ubicado en 3
    se agregó:  0.0 en:  2 93
    desalojo No ubicado en 4
    se agregó:  0.0 en:  3 93
    desalojo No ubicado en 5
    se agregó:  0.0 en:  4 93
    desalojo No ubicado en 6
    se agregó:  0.0 en:  5 93
    desalojo No ubicado en 7
    se agregó:  0.0 en:  6 93
    desalojo No ubicado en 8
    se agregó:  0.0 en:  7 93
    desalojo Ubicado en 9
    se agregó:  1.0 en:  8 93
    desalojo No ubicado en 10
    se agregó:  0.0 en:  9 93
    padres No ubicado en 1
    se agregó:  0.0 en:  0 94
    padres No ubicado en 2
    se agregó:  0.0 en:  1 94
    padres No ubicado en 3
    se agregó:  0.0 en:  2 94
    padres No ubicado en 4
    se agregó:  0.0 en:  3 94
    padres No ubicado en 5
    se agregó:  0.0 en:  4 94
    padres No ubicado en 6
    se agregó:  0.0 en:  5 94
    padres No ubicado en 7
    se agregó:  0.0 en:  6 94
    padres No ubicado en 8
    se agregó:  0.0 en:  7 94
    padres Ubicado en 9
    se agregó:  1.0 en:  8 94
    padres No ubicado en 10
    se agregó:  0.0 en:  9 94
    43 No ubicado en 1
    se agregó:  0.0 en:  0 95
    43 No ubicado en 2
    se agregó:  0.0 en:  1 95
    43 No ubicado en 3
    se agregó:  0.0 en:  2 95
    43 No ubicado en 4
    se agregó:  0.0 en:  3 95
    43 No ubicado en 5
    se agregó:  0.0 en:  4 95
    43 No ubicado en 6
    se agregó:  0.0 en:  5 95
    43 No ubicado en 7
    se agregó:  0.0 en:  6 95
    43 No ubicado en 8
    se agregó:  0.0 en:  7 95
    43 Ubicado en 9
    se agregó:  1.0 en:  8 95
    43 No ubicado en 10
    se agregó:  0.0 en:  9 95
    ayotzinapa No ubicado en 1
    se agregó:  0.0 en:  0 96
    ayotzinapa No ubicado en 2
    se agregó:  0.0 en:  1 96
    ayotzinapa No ubicado en 3
    se agregó:  0.0 en:  2 96
    ayotzinapa No ubicado en 4
    se agregó:  0.0 en:  3 96
    ayotzinapa No ubicado en 5
    se agregó:  0.0 en:  4 96
    ayotzinapa No ubicado en 6
    se agregó:  0.0 en:  5 96
    ayotzinapa No ubicado en 7
    se agregó:  0.0 en:  6 96
    ayotzinapa No ubicado en 8
    se agregó:  0.0 en:  7 96
    ayotzinapa Ubicado en 9
    se agregó:  1.0 en:  8 96
    ayotzinapa No ubicado en 10
    se agregó:  0.0 en:  9 96
    manifestantes No ubicado en 1
    se agregó:  0.0 en:  0 97
    manifestantes No ubicado en 2
    se agregó:  0.0 en:  1 97
    manifestantes No ubicado en 3
    se agregó:  0.0 en:  2 97
    manifestantes No ubicado en 4
    se agregó:  0.0 en:  3 97
    manifestantes No ubicado en 5
    se agregó:  0.0 en:  4 97
    manifestantes No ubicado en 6
    se agregó:  0.0 en:  5 97
    manifestantes No ubicado en 7
    se agregó:  0.0 en:  6 97
    manifestantes No ubicado en 8
    se agregó:  0.0 en:  7 97
    manifestantes Ubicado en 9
    se agregó:  1.0 en:  8 97
    manifestantes No ubicado en 10
    se agregó:  0.0 en:  9 97
    chiapas No ubicado en 1
    se agregó:  0.0 en:  0 98
    chiapas No ubicado en 2
    se agregó:  0.0 en:  1 98
    chiapas No ubicado en 3
    se agregó:  0.0 en:  2 98
    chiapas No ubicado en 4
    se agregó:  0.0 en:  3 98
    chiapas No ubicado en 5
    se agregó:  0.0 en:  4 98
    chiapas No ubicado en 6
    se agregó:  0.0 en:  5 98
    chiapas No ubicado en 7
    se agregó:  0.0 en:  6 98
    chiapas No ubicado en 8
    se agregó:  0.0 en:  7 98
    chiapas Ubicado en 9
    se agregó:  1.0 en:  8 98
    chiapas No ubicado en 10
    se agregó:  0.0 en:  9 98
    fisco No ubicado en 1
    se agregó:  0.0 en:  0 99
    fisco No ubicado en 2
    se agregó:  0.0 en:  1 99
    fisco No ubicado en 3
    se agregó:  0.0 en:  2 99
    fisco No ubicado en 4
    se agregó:  0.0 en:  3 99
    fisco No ubicado en 5
    se agregó:  0.0 en:  4 99
    fisco No ubicado en 6
    se agregó:  0.0 en:  5 99
    fisco No ubicado en 7
    se agregó:  0.0 en:  6 99
    fisco No ubicado en 8
    se agregó:  0.0 en:  7 99
    fisco No ubicado en 9
    se agregó:  0.0 en:  8 99
    fisco Ubicado en 10
    se agregó:  1.0 en:  9 99
    advierte No ubicado en 1
    se agregó:  0.0 en:  0 100
    advierte No ubicado en 2
    se agregó:  0.0 en:  1 100
    advierte No ubicado en 3
    se agregó:  0.0 en:  2 100
    advierte No ubicado en 4
    se agregó:  0.0 en:  3 100
    advierte No ubicado en 5
    se agregó:  0.0 en:  4 100
    advierte No ubicado en 6
    se agregó:  0.0 en:  5 100
    advierte No ubicado en 7
    se agregó:  0.0 en:  6 100
    advierte No ubicado en 8
    se agregó:  0.0 en:  7 100
    advierte No ubicado en 9
    se agregó:  0.0 en:  8 100
    advierte Ubicado en 10
    se agregó:  1.0 en:  9 100
    esquemas No ubicado en 1
    se agregó:  0.0 en:  0 101
    esquemas No ubicado en 2
    se agregó:  0.0 en:  1 101
    esquemas No ubicado en 3
    se agregó:  0.0 en:  2 101
    esquemas No ubicado en 4
    se agregó:  0.0 en:  3 101
    esquemas No ubicado en 5
    se agregó:  0.0 en:  4 101
    esquemas No ubicado en 6
    se agregó:  0.0 en:  5 101
    esquemas No ubicado en 7
    se agregó:  0.0 en:  6 101
    esquemas No ubicado en 8
    se agregó:  0.0 en:  7 101
    esquemas No ubicado en 9
    se agregó:  0.0 en:  8 101
    esquemas Ubicado en 10
    se agregó:  1.0 en:  9 101
    operaciones No ubicado en 1
    se agregó:  0.0 en:  0 102
    operaciones No ubicado en 2
    se agregó:  0.0 en:  1 102
    operaciones No ubicado en 3
    se agregó:  0.0 en:  2 102
    operaciones No ubicado en 4
    se agregó:  0.0 en:  3 102
    operaciones No ubicado en 5
    se agregó:  0.0 en:  4 102
    operaciones No ubicado en 6
    se agregó:  0.0 en:  5 102
    operaciones No ubicado en 7
    se agregó:  0.0 en:  6 102
    operaciones No ubicado en 8
    se agregó:  0.0 en:  7 102
    operaciones No ubicado en 9
    se agregó:  0.0 en:  8 102
    operaciones Ubicado en 10
    se agregó:  1.0 en:  9 102
    realizadas No ubicado en 1
    se agregó:  0.0 en:  0 103
    realizadas No ubicado en 2
    se agregó:  0.0 en:  1 103
    realizadas No ubicado en 3
    se agregó:  0.0 en:  2 103
    realizadas No ubicado en 4
    se agregó:  0.0 en:  3 103
    realizadas No ubicado en 5
    se agregó:  0.0 en:  4 103
    realizadas No ubicado en 6
    se agregó:  0.0 en:  5 103
    realizadas No ubicado en 7
    se agregó:  0.0 en:  6 103
    realizadas No ubicado en 8
    se agregó:  0.0 en:  7 103
    realizadas No ubicado en 9
    se agregó:  0.0 en:  8 103
    realizadas Ubicado en 10
    se agregó:  1.0 en:  9 103
    2017 No ubicado en 1
    se agregó:  0.0 en:  0 104
    2017 No ubicado en 2
    se agregó:  0.0 en:  1 104
    2017 No ubicado en 3
    se agregó:  0.0 en:  2 104
    2017 No ubicado en 4
    se agregó:  0.0 en:  3 104
    2017 No ubicado en 5
    se agregó:  0.0 en:  4 104
    2017 No ubicado en 6
    se agregó:  0.0 en:  5 104
    2017 No ubicado en 7
    se agregó:  0.0 en:  6 104
    2017 No ubicado en 8
    se agregó:  0.0 en:  7 104
    2017 No ubicado en 9
    se agregó:  0.0 en:  8 104
    2017 Ubicado en 10
    se agregó:  1.0 en:  9 104
    2019 No ubicado en 1
    se agregó:  0.0 en:  0 105
    2019 No ubicado en 2
    se agregó:  0.0 en:  1 105
    2019 No ubicado en 3
    se agregó:  0.0 en:  2 105
    2019 No ubicado en 4
    se agregó:  0.0 en:  3 105
    2019 No ubicado en 5
    se agregó:  0.0 en:  4 105
    2019 No ubicado en 6
    se agregó:  0.0 en:  5 105
    2019 No ubicado en 7
    se agregó:  0.0 en:  6 105
    2019 No ubicado en 8
    se agregó:  0.0 en:  7 105
    2019 No ubicado en 9
    se agregó:  0.0 en:  8 105
    2019 Ubicado en 10
    se agregó:  1.0 en:  9 105
    339 No ubicado en 1
    se agregó:  0.0 en:  0 106
    339 No ubicado en 2
    se agregó:  0.0 en:  1 106
    339 No ubicado en 3
    se agregó:  0.0 en:  2 106
    339 No ubicado en 4
    se agregó:  0.0 en:  3 106
    339 No ubicado en 5
    se agregó:  0.0 en:  4 106
    339 No ubicado en 6
    se agregó:  0.0 en:  5 106
    339 No ubicado en 7
    se agregó:  0.0 en:  6 106
    339 No ubicado en 8
    se agregó:  0.0 en:  7 106
    339 No ubicado en 9
    se agregó:  0.0 en:  8 106
    339 Ubicado en 10
    se agregó:  1.0 en:  9 106
    000 No ubicado en 1
    se agregó:  0.0 en:  0 107
    000 No ubicado en 2
    se agregó:  0.0 en:  1 107
    000 No ubicado en 3
    se agregó:  0.0 en:  2 107
    000 No ubicado en 4
    se agregó:  0.0 en:  3 107
    000 No ubicado en 5
    se agregó:  0.0 en:  4 107
    000 No ubicado en 6
    se agregó:  0.0 en:  5 107
    000 No ubicado en 7
    se agregó:  0.0 en:  6 107
    000 No ubicado en 8
    se agregó:  0.0 en:  7 107
    000 No ubicado en 9
    se agregó:  0.0 en:  8 107
    000 Ubicado en 10
    se agregó:  1.0 en:  9 107
    millones No ubicado en 1
    se agregó:  0.0 en:  0 108
    millones No ubicado en 2
    se agregó:  0.0 en:  1 108
    millones No ubicado en 3
    se agregó:  0.0 en:  2 108
    millones No ubicado en 4
    se agregó:  0.0 en:  3 108
    millones No ubicado en 5
    se agregó:  0.0 en:  4 108
    millones No ubicado en 6
    se agregó:  0.0 en:  5 108
    millones No ubicado en 7
    se agregó:  0.0 en:  6 108
    millones No ubicado en 8
    se agregó:  0.0 en:  7 108
    millones No ubicado en 9
    se agregó:  0.0 en:  8 108
    millones Ubicado en 10
    se agregó:  1.0 en:  9 108
    pesos No ubicado en 1
    se agregó:  0.0 en:  0 109
    pesos No ubicado en 2
    se agregó:  0.0 en:  1 109
    pesos No ubicado en 3
    se agregó:  0.0 en:  2 109
    pesos No ubicado en 4
    se agregó:  0.0 en:  3 109
    pesos No ubicado en 5
    se agregó:  0.0 en:  4 109
    pesos No ubicado en 6
    se agregó:  0.0 en:  5 109
    pesos No ubicado en 7
    se agregó:  0.0 en:  6 109
    pesos No ubicado en 8
    se agregó:  0.0 en:  7 109
    pesos No ubicado en 9
    se agregó:  0.0 en:  8 109
    pesos Ubicado en 10
    se agregó:  1.0 en:  9 109
    involucran No ubicado en 1
    se agregó:  0.0 en:  0 110
    involucran No ubicado en 2
    se agregó:  0.0 en:  1 110
    involucran No ubicado en 3
    se agregó:  0.0 en:  2 110
    involucran No ubicado en 4
    se agregó:  0.0 en:  3 110
    involucran No ubicado en 5
    se agregó:  0.0 en:  4 110
    involucran No ubicado en 6
    se agregó:  0.0 en:  5 110
    involucran No ubicado en 7
    se agregó:  0.0 en:  6 110
    involucran No ubicado en 8
    se agregó:  0.0 en:  7 110
    involucran No ubicado en 9
    se agregó:  0.0 en:  8 110
    involucran Ubicado en 10
    se agregó:  1.0 en:  9 110
    977 No ubicado en 1
    se agregó:  0.0 en:  0 111
    977 No ubicado en 2
    se agregó:  0.0 en:  1 111
    977 No ubicado en 3
    se agregó:  0.0 en:  2 111
    977 No ubicado en 4
    se agregó:  0.0 en:  3 111
    977 No ubicado en 5
    se agregó:  0.0 en:  4 111
    977 No ubicado en 6
    se agregó:  0.0 en:  5 111
    977 No ubicado en 7
    se agregó:  0.0 en:  6 111
    977 No ubicado en 8
    se agregó:  0.0 en:  7 111
    977 No ubicado en 9
    se agregó:  0.0 en:  8 111
    977 Ubicado en 10
    se agregó:  1.0 en:  9 111
    contribuyentes No ubicado en 1
    se agregó:  0.0 en:  0 112
    contribuyentes No ubicado en 2
    se agregó:  0.0 en:  1 112
    contribuyentes No ubicado en 3
    se agregó:  0.0 en:  2 112
    contribuyentes No ubicado en 4
    se agregó:  0.0 en:  3 112
    contribuyentes No ubicado en 5
    se agregó:  0.0 en:  4 112
    contribuyentes No ubicado en 6
    se agregó:  0.0 en:  5 112
    contribuyentes No ubicado en 7
    se agregó:  0.0 en:  6 112
    contribuyentes No ubicado en 8
    se agregó:  0.0 en:  7 112
    contribuyentes No ubicado en 9
    se agregó:  0.0 en:  8 112
    contribuyentes Ubicado en 10
    se agregó:  1.0 en:  9 112



```python
matrix
```


```python
matrix.shape
```


```python
bin_cos_t01_t02 = dot(matrix[0],matrix[1])/(norm(matrix[0])*norm(matrix[1]))
bin_cos_t01_t02
```




    0.12573892269238632




```python
matrix = np.zeros((len(dicc_texts), len(dicc_termns))) 
```


```python
i = 0
j = 0

for word_termns in dicc_termns: 
    for word_texts in dicc_texts: 
        if(word_termns in dicc_texts[word_texts]):
            print(word_termns, "Ubicado en ", word_texts)
            
            matrix[j, i] = dicc_termns[word_termns]
            
        elif(word_termns not in dicc_texts[word_texts]): 
            print(word_termns, "No ubicado en", word_texts)
            
            matrix[j, i] = 0
            
        print("se agregó: ", matrix[j,i], "en: ", j, i)
            
        j = j + 1
        
    j = 0
    i = i + 1
```

    datos Ubicado en  1
    se agregó:  1.0 en:  0 0
    datos No ubicado en 2
    se agregó:  0.0 en:  1 0
    datos No ubicado en 3
    se agregó:  0.0 en:  2 0
    datos No ubicado en 4
    se agregó:  0.0 en:  3 0
    datos No ubicado en 5
    se agregó:  0.0 en:  4 0
    datos No ubicado en 6
    se agregó:  0.0 en:  5 0
    datos No ubicado en 7
    se agregó:  0.0 en:  6 0
    datos No ubicado en 8
    se agregó:  0.0 en:  7 0
    datos No ubicado en 9
    se agregó:  0.0 en:  8 0
    datos No ubicado en 10
    se agregó:  0.0 en:  9 0
    concretos Ubicado en  1
    se agregó:  1.0 en:  0 1
    concretos No ubicado en 2
    se agregó:  0.0 en:  1 1
    concretos No ubicado en 3
    se agregó:  0.0 en:  2 1
    concretos No ubicado en 4
    se agregó:  0.0 en:  3 1
    concretos No ubicado en 5
    se agregó:  0.0 en:  4 1
    concretos No ubicado en 6
    se agregó:  0.0 en:  5 1
    concretos No ubicado en 7
    se agregó:  0.0 en:  6 1
    concretos No ubicado en 8
    se agregó:  0.0 en:  7 1
    concretos No ubicado en 9
    se agregó:  0.0 en:  8 1
    concretos No ubicado en 10
    se agregó:  0.0 en:  9 1
    feminicidio Ubicado en  1
    se agregó:  2.0 en:  0 2
    feminicidio Ubicado en  2
    se agregó:  2.0 en:  1 2
    feminicidio No ubicado en 3
    se agregó:  0.0 en:  2 2
    feminicidio No ubicado en 4
    se agregó:  0.0 en:  3 2
    feminicidio No ubicado en 5
    se agregó:  0.0 en:  4 2
    feminicidio No ubicado en 6
    se agregó:  0.0 en:  5 2
    feminicidio No ubicado en 7
    se agregó:  0.0 en:  6 2
    feminicidio No ubicado en 8
    se agregó:  0.0 en:  7 2
    feminicidio No ubicado en 9
    se agregó:  0.0 en:  8 2
    feminicidio No ubicado en 10
    se agregó:  0.0 en:  9 2
    méxico Ubicado en  1
    se agregó:  1.0 en:  0 3
    méxico No ubicado en 2
    se agregó:  0.0 en:  1 3
    méxico No ubicado en 3
    se agregó:  0.0 en:  2 3
    méxico No ubicado en 4
    se agregó:  0.0 en:  3 3
    méxico No ubicado en 5
    se agregó:  0.0 en:  4 3
    méxico No ubicado en 6
    se agregó:  0.0 en:  5 3
    méxico No ubicado en 7
    se agregó:  0.0 en:  6 3
    méxico No ubicado en 8
    se agregó:  0.0 en:  7 3
    méxico No ubicado en 9
    se agregó:  0.0 en:  8 3
    méxico No ubicado en 10
    se agregó:  0.0 en:  9 3
    tomados Ubicado en  1
    se agregó:  1.0 en:  0 4
    tomados No ubicado en 2
    se agregó:  0.0 en:  1 4
    tomados No ubicado en 3
    se agregó:  0.0 en:  2 4
    tomados No ubicado en 4
    se agregó:  0.0 en:  3 4
    tomados No ubicado en 5
    se agregó:  0.0 en:  4 4
    tomados No ubicado en 6
    se agregó:  0.0 en:  5 4
    tomados No ubicado en 7
    se agregó:  0.0 en:  6 4
    tomados No ubicado en 8
    se agregó:  0.0 en:  7 4
    tomados No ubicado en 9
    se agregó:  0.0 en:  8 4
    tomados No ubicado en 10
    se agregó:  0.0 en:  9 4
    sesnp Ubicado en  1
    se agregó:  1.0 en:  0 5
    sesnp No ubicado en 2
    se agregó:  0.0 en:  1 5
    sesnp No ubicado en 3
    se agregó:  0.0 en:  2 5
    sesnp No ubicado en 4
    se agregó:  0.0 en:  3 5
    sesnp No ubicado en 5
    se agregó:  0.0 en:  4 5
    sesnp No ubicado en 6
    se agregó:  0.0 en:  5 5
    sesnp No ubicado en 7
    se agregó:  0.0 en:  6 5
    sesnp No ubicado en 8
    se agregó:  0.0 en:  7 5
    sesnp No ubicado en 9
    se agregó:  0.0 en:  8 5
    sesnp No ubicado en 10
    se agregó:  0.0 en:  9 5
    delito Ubicado en  1
    se agregó:  1.0 en:  0 6
    delito No ubicado en 2
    se agregó:  0.0 en:  1 6
    delito No ubicado en 3
    se agregó:  0.0 en:  2 6
    delito No ubicado en 4
    se agregó:  0.0 en:  3 6
    delito No ubicado en 5
    se agregó:  0.0 en:  4 6
    delito No ubicado en 6
    se agregó:  0.0 en:  5 6
    delito No ubicado en 7
    se agregó:  0.0 en:  6 6
    delito No ubicado en 8
    se agregó:  0.0 en:  7 6
    delito No ubicado en 9
    se agregó:  0.0 en:  8 6
    delito No ubicado en 10
    se agregó:  0.0 en:  9 6
    tendencia Ubicado en  1
    se agregó:  1.0 en:  0 7
    tendencia No ubicado en 2
    se agregó:  0.0 en:  1 7
    tendencia No ubicado en 3
    se agregó:  0.0 en:  2 7
    tendencia No ubicado en 4
    se agregó:  0.0 en:  3 7
    tendencia No ubicado en 5
    se agregó:  0.0 en:  4 7
    tendencia No ubicado en 6
    se agregó:  0.0 en:  5 7
    tendencia No ubicado en 7
    se agregó:  0.0 en:  6 7
    tendencia No ubicado en 8
    se agregó:  0.0 en:  7 7
    tendencia No ubicado en 9
    se agregó:  0.0 en:  8 7
    tendencia No ubicado en 10
    se agregó:  0.0 en:  9 7
    alza Ubicado en  1
    se agregó:  1.0 en:  0 8
    alza No ubicado en 2
    se agregó:  0.0 en:  1 8
    alza No ubicado en 3
    se agregó:  0.0 en:  2 8
    alza No ubicado en 4
    se agregó:  0.0 en:  3 8
    alza No ubicado en 5
    se agregó:  0.0 en:  4 8
    alza No ubicado en 6
    se agregó:  0.0 en:  5 8
    alza No ubicado en 7
    se agregó:  0.0 en:  6 8
    alza No ubicado en 8
    se agregó:  0.0 en:  7 8
    alza No ubicado en 9
    se agregó:  0.0 en:  8 8
    alza No ubicado en 10
    se agregó:  0.0 en:  9 8
    solución Ubicado en  1
    se agregó:  1.0 en:  0 9
    solución No ubicado en 2
    se agregó:  0.0 en:  1 9
    solución No ubicado en 3
    se agregó:  0.0 en:  2 9
    solución No ubicado en 4
    se agregó:  0.0 en:  3 9
    solución No ubicado en 5
    se agregó:  0.0 en:  4 9
    solución No ubicado en 6
    se agregó:  0.0 en:  5 9
    solución No ubicado en 7
    se agregó:  0.0 en:  6 9
    solución No ubicado en 8
    se agregó:  0.0 en:  7 9
    solución No ubicado en 9
    se agregó:  0.0 en:  8 9
    solución No ubicado en 10
    se agregó:  0.0 en:  9 9
    lópez Ubicado en  1
    se agregó:  2.0 en:  0 10
    lópez No ubicado en 2
    se agregó:  0.0 en:  1 10
    lópez No ubicado en 3
    se agregó:  0.0 en:  2 10
    lópez No ubicado en 4
    se agregó:  0.0 en:  3 10
    lópez No ubicado en 5
    se agregó:  0.0 en:  4 10
    lópez No ubicado en 6
    se agregó:  0.0 en:  5 10
    lópez No ubicado en 7
    se agregó:  0.0 en:  6 10
    lópez Ubicado en  8
    se agregó:  2.0 en:  7 10
    lópez No ubicado en 9
    se agregó:  0.0 en:  8 10
    lópez No ubicado en 10
    se agregó:  0.0 en:  9 10
    fiscal Ubicado en  1
    se agregó:  2.0 en:  0 11
    fiscal Ubicado en  2
    se agregó:  2.0 en:  1 11
    fiscal No ubicado en 3
    se agregó:  0.0 en:  2 11
    fiscal No ubicado en 4
    se agregó:  0.0 en:  3 11
    fiscal No ubicado en 5
    se agregó:  0.0 en:  4 11
    fiscal No ubicado en 6
    se agregó:  0.0 en:  5 11
    fiscal No ubicado en 7
    se agregó:  0.0 en:  6 11
    fiscal No ubicado en 8
    se agregó:  0.0 en:  7 11
    fiscal No ubicado en 9
    se agregó:  0.0 en:  8 11
    fiscal No ubicado en 10
    se agregó:  0.0 en:  9 11
    carnal Ubicado en  1
    se agregó:  1.0 en:  0 12
    carnal No ubicado en 2
    se agregó:  0.0 en:  1 12
    carnal No ubicado en 3
    se agregó:  0.0 en:  2 12
    carnal No ubicado en 4
    se agregó:  0.0 en:  3 12
    carnal No ubicado en 5
    se agregó:  0.0 en:  4 12
    carnal No ubicado en 6
    se agregó:  0.0 en:  5 12
    carnal No ubicado en 7
    se agregó:  0.0 en:  6 12
    carnal No ubicado en 8
    se agregó:  0.0 en:  7 12
    carnal No ubicado en 9
    se agregó:  0.0 en:  8 12
    carnal No ubicado en 10
    se agregó:  0.0 en:  9 12
    desaparecerlo Ubicado en  1
    se agregó:  1.0 en:  0 13
    desaparecerlo No ubicado en 2
    se agregó:  0.0 en:  1 13
    desaparecerlo No ubicado en 3
    se agregó:  0.0 en:  2 13
    desaparecerlo No ubicado en 4
    se agregó:  0.0 en:  3 13
    desaparecerlo No ubicado en 5
    se agregó:  0.0 en:  4 13
    desaparecerlo No ubicado en 6
    se agregó:  0.0 en:  5 13
    desaparecerlo No ubicado en 7
    se agregó:  0.0 en:  6 13
    desaparecerlo No ubicado en 8
    se agregó:  0.0 en:  7 13
    desaparecerlo No ubicado en 9
    se agregó:  0.0 en:  8 13
    desaparecerlo No ubicado en 10
    se agregó:  0.0 en:  9 13
    código Ubicado en  1
    se agregó:  1.0 en:  0 14
    código No ubicado en 2
    se agregó:  0.0 en:  1 14
    código No ubicado en 3
    se agregó:  0.0 en:  2 14
    código No ubicado en 4
    se agregó:  0.0 en:  3 14
    código No ubicado en 5
    se agregó:  0.0 en:  4 14
    código No ubicado en 6
    se agregó:  0.0 en:  5 14
    código No ubicado en 7
    se agregó:  0.0 en:  6 14
    código No ubicado en 8
    se agregó:  0.0 en:  7 14
    código No ubicado en 9
    se agregó:  0.0 en:  8 14
    código No ubicado en 10
    se agregó:  0.0 en:  9 14
    penal Ubicado en  1
    se agregó:  1.0 en:  0 15
    penal No ubicado en 2
    se agregó:  0.0 en:  1 15
    penal No ubicado en 3
    se agregó:  0.0 en:  2 15
    penal No ubicado en 4
    se agregó:  0.0 en:  3 15
    penal No ubicado en 5
    se agregó:  0.0 en:  4 15
    penal No ubicado en 6
    se agregó:  0.0 en:  5 15
    penal No ubicado en 7
    se agregó:  0.0 en:  6 15
    penal No ubicado en 8
    se agregó:  0.0 en:  7 15
    penal No ubicado en 9
    se agregó:  0.0 en:  8 15
    penal No ubicado en 10
    se agregó:  0.0 en:  9 15
    gobierno Ubicado en  1
    se agregó:  1.0 en:  0 16
    gobierno No ubicado en 2
    se agregó:  0.0 en:  1 16
    gobierno No ubicado en 3
    se agregó:  0.0 en:  2 16
    gobierno No ubicado en 4
    se agregó:  0.0 en:  3 16
    gobierno No ubicado en 5
    se agregó:  0.0 en:  4 16
    gobierno No ubicado en 6
    se agregó:  0.0 en:  5 16
    gobierno No ubicado en 7
    se agregó:  0.0 en:  6 16
    gobierno No ubicado en 8
    se agregó:  0.0 en:  7 16
    gobierno No ubicado en 9
    se agregó:  0.0 en:  8 16
    gobierno No ubicado en 10
    se agregó:  0.0 en:  9 16
    prefiere Ubicado en  1
    se agregó:  1.0 en:  0 17
    prefiere No ubicado en 2
    se agregó:  0.0 en:  1 17
    prefiere No ubicado en 3
    se agregó:  0.0 en:  2 17
    prefiere No ubicado en 4
    se agregó:  0.0 en:  3 17
    prefiere No ubicado en 5
    se agregó:  0.0 en:  4 17
    prefiere No ubicado en 6
    se agregó:  0.0 en:  5 17
    prefiere No ubicado en 7
    se agregó:  0.0 en:  6 17
    prefiere No ubicado en 8
    se agregó:  0.0 en:  7 17
    prefiere No ubicado en 9
    se agregó:  0.0 en:  8 17
    prefiere No ubicado en 10
    se agregó:  0.0 en:  9 17
    ponerse Ubicado en  1
    se agregó:  1.0 en:  0 18
    ponerse No ubicado en 2
    se agregó:  0.0 en:  1 18
    ponerse No ubicado en 3
    se agregó:  0.0 en:  2 18
    ponerse No ubicado en 4
    se agregó:  0.0 en:  3 18
    ponerse No ubicado en 5
    se agregó:  0.0 en:  4 18
    ponerse No ubicado en 6
    se agregó:  0.0 en:  5 18
    ponerse No ubicado en 7
    se agregó:  0.0 en:  6 18
    ponerse No ubicado en 8
    se agregó:  0.0 en:  7 18
    ponerse No ubicado en 9
    se agregó:  0.0 en:  8 18
    ponerse No ubicado en 10
    se agregó:  0.0 en:  9 18
    lado Ubicado en  1
    se agregó:  1.0 en:  0 19
    lado No ubicado en 2
    se agregó:  0.0 en:  1 19
    lado No ubicado en 3
    se agregó:  0.0 en:  2 19
    lado No ubicado en 4
    se agregó:  0.0 en:  3 19
    lado No ubicado en 5
    se agregó:  0.0 en:  4 19
    lado No ubicado en 6
    se agregó:  0.0 en:  5 19
    lado No ubicado en 7
    se agregó:  0.0 en:  6 19
    lado No ubicado en 8
    se agregó:  0.0 en:  7 19
    lado No ubicado en 9
    se agregó:  0.0 en:  8 19
    lado No ubicado en 10
    se agregó:  0.0 en:  9 19
    delincuente Ubicado en  1
    se agregó:  1.0 en:  0 20
    delincuente No ubicado en 2
    se agregó:  0.0 en:  1 20
    delincuente No ubicado en 3
    se agregó:  0.0 en:  2 20
    delincuente No ubicado en 4
    se agregó:  0.0 en:  3 20
    delincuente No ubicado en 5
    se agregó:  0.0 en:  4 20
    delincuente No ubicado en 6
    se agregó:  0.0 en:  5 20
    delincuente No ubicado en 7
    se agregó:  0.0 en:  6 20
    delincuente No ubicado en 8
    se agregó:  0.0 en:  7 20
    delincuente No ubicado en 9
    se agregó:  0.0 en:  8 20
    delincuente No ubicado en 10
    se agregó:  0.0 en:  9 20
    allanarle Ubicado en  1
    se agregó:  1.0 en:  0 21
    allanarle No ubicado en 2
    se agregó:  0.0 en:  1 21
    allanarle No ubicado en 3
    se agregó:  0.0 en:  2 21
    allanarle No ubicado en 4
    se agregó:  0.0 en:  3 21
    allanarle No ubicado en 5
    se agregó:  0.0 en:  4 21
    allanarle No ubicado en 6
    se agregó:  0.0 en:  5 21
    allanarle No ubicado en 7
    se agregó:  0.0 en:  6 21
    allanarle No ubicado en 8
    se agregó:  0.0 en:  7 21
    allanarle No ubicado en 9
    se agregó:  0.0 en:  8 21
    allanarle No ubicado en 10
    se agregó:  0.0 en:  9 21
    camino Ubicado en  1
    se agregó:  1.0 en:  0 22
    camino No ubicado en 2
    se agregó:  0.0 en:  1 22
    camino No ubicado en 3
    se agregó:  0.0 en:  2 22
    camino No ubicado en 4
    se agregó:  0.0 en:  3 22
    camino No ubicado en 5
    se agregó:  0.0 en:  4 22
    camino No ubicado en 6
    se agregó:  0.0 en:  5 22
    camino No ubicado en 7
    se agregó:  0.0 en:  6 22
    camino No ubicado en 8
    se agregó:  0.0 en:  7 22
    camino No ubicado en 9
    se agregó:  0.0 en:  8 22
    camino No ubicado en 10
    se agregó:  0.0 en:  9 22
    sociedad No ubicado en 1
    se agregó:  0.0 en:  0 23
    sociedad Ubicado en  2
    se agregó:  1.0 en:  1 23
    sociedad No ubicado en 3
    se agregó:  0.0 en:  2 23
    sociedad No ubicado en 4
    se agregó:  0.0 en:  3 23
    sociedad No ubicado en 5
    se agregó:  0.0 en:  4 23
    sociedad No ubicado en 6
    se agregó:  0.0 en:  5 23
    sociedad No ubicado en 7
    se agregó:  0.0 en:  6 23
    sociedad No ubicado en 8
    se agregó:  0.0 en:  7 23
    sociedad No ubicado en 9
    se agregó:  0.0 en:  8 23
    sociedad No ubicado en 10
    se agregó:  0.0 en:  9 23
    general No ubicado en 1
    se agregó:  0.0 en:  0 24
    general Ubicado en  2
    se agregó:  1.0 en:  1 24
    general No ubicado en 3
    se agregó:  0.0 en:  2 24
    general No ubicado en 4
    se agregó:  0.0 en:  3 24
    general No ubicado en 5
    se agregó:  0.0 en:  4 24
    general No ubicado en 6
    se agregó:  0.0 en:  5 24
    general No ubicado en 7
    se agregó:  0.0 en:  6 24
    general No ubicado en 8
    se agregó:  0.0 en:  7 24
    general No ubicado en 9
    se agregó:  0.0 en:  8 24
    general No ubicado en 10
    se agregó:  0.0 en:  9 24
    explicó No ubicado en 1
    se agregó:  0.0 en:  0 25
    explicó Ubicado en  2
    se agregó:  1.0 en:  1 25
    explicó No ubicado en 3
    se agregó:  0.0 en:  2 25
    explicó No ubicado en 4
    se agregó:  0.0 en:  3 25
    explicó No ubicado en 5
    se agregó:  0.0 en:  4 25
    explicó No ubicado en 6
    se agregó:  0.0 en:  5 25
    explicó No ubicado en 7
    se agregó:  0.0 en:  6 25
    explicó No ubicado en 8
    se agregó:  0.0 en:  7 25
    explicó No ubicado en 9
    se agregó:  0.0 en:  8 25
    explicó No ubicado en 10
    se agregó:  0.0 en:  9 25
    propuesta No ubicado en 1
    se agregó:  0.0 en:  0 26
    propuesta Ubicado en  2
    se agregó:  1.0 en:  1 26
    propuesta No ubicado en 3
    se agregó:  0.0 en:  2 26
    propuesta No ubicado en 4
    se agregó:  0.0 en:  3 26
    propuesta No ubicado en 5
    se agregó:  0.0 en:  4 26
    propuesta No ubicado en 6
    se agregó:  0.0 en:  5 26
    propuesta No ubicado en 7
    se agregó:  0.0 en:  6 26
    propuesta No ubicado en 8
    se agregó:  0.0 en:  7 26
    propuesta No ubicado en 9
    se agregó:  0.0 en:  8 26
    propuesta No ubicado en 10
    se agregó:  0.0 en:  9 26
    busca No ubicado en 1
    se agregó:  0.0 en:  0 27
    busca Ubicado en  2
    se agregó:  1.0 en:  1 27
    busca No ubicado en 3
    se agregó:  0.0 en:  2 27
    busca No ubicado en 4
    se agregó:  0.0 en:  3 27
    busca No ubicado en 5
    se agregó:  0.0 en:  4 27
    busca No ubicado en 6
    se agregó:  0.0 en:  5 27
    busca No ubicado en 7
    se agregó:  0.0 en:  6 27
    busca No ubicado en 8
    se agregó:  0.0 en:  7 27
    busca No ubicado en 9
    se agregó:  0.0 en:  8 27
    busca No ubicado en 10
    se agregó:  0.0 en:  9 27
    facilitar No ubicado en 1
    se agregó:  0.0 en:  0 28
    facilitar Ubicado en  2
    se agregó:  1.0 en:  1 28
    facilitar No ubicado en 3
    se agregó:  0.0 en:  2 28
    facilitar No ubicado en 4
    se agregó:  0.0 en:  3 28
    facilitar No ubicado en 5
    se agregó:  0.0 en:  4 28
    facilitar No ubicado en 6
    se agregó:  0.0 en:  5 28
    facilitar No ubicado en 7
    se agregó:  0.0 en:  6 28
    facilitar No ubicado en 8
    se agregó:  0.0 en:  7 28
    facilitar No ubicado en 9
    se agregó:  0.0 en:  8 28
    facilitar No ubicado en 10
    se agregó:  0.0 en:  9 28
    investigación No ubicado en 1
    se agregó:  0.0 en:  0 29
    investigación Ubicado en  2
    se agregó:  1.0 en:  1 29
    investigación No ubicado en 3
    se agregó:  0.0 en:  2 29
    investigación No ubicado en 4
    se agregó:  0.0 en:  3 29
    investigación No ubicado en 5
    se agregó:  0.0 en:  4 29
    investigación No ubicado en 6
    se agregó:  0.0 en:  5 29
    investigación No ubicado en 7
    se agregó:  0.0 en:  6 29
    investigación No ubicado en 8
    se agregó:  0.0 en:  7 29
    investigación No ubicado en 9
    se agregó:  0.0 en:  8 29
    investigación No ubicado en 10
    se agregó:  0.0 en:  9 29
    proteger No ubicado en 1
    se agregó:  0.0 en:  0 30
    proteger Ubicado en  2
    se agregó:  1.0 en:  1 30
    proteger No ubicado en 3
    se agregó:  0.0 en:  2 30
    proteger No ubicado en 4
    se agregó:  0.0 en:  3 30
    proteger No ubicado en 5
    se agregó:  0.0 en:  4 30
    proteger No ubicado en 6
    se agregó:  0.0 en:  5 30
    proteger No ubicado en 7
    se agregó:  0.0 en:  6 30
    proteger No ubicado en 8
    se agregó:  0.0 en:  7 30
    proteger No ubicado en 9
    se agregó:  0.0 en:  8 30
    proteger No ubicado en 10
    se agregó:  0.0 en:  9 30
    víctimas No ubicado en 1
    se agregó:  0.0 en:  0 31
    víctimas Ubicado en  2
    se agregó:  1.0 en:  1 31
    víctimas No ubicado en 3
    se agregó:  0.0 en:  2 31
    víctimas No ubicado en 4
    se agregó:  0.0 en:  3 31
    víctimas No ubicado en 5
    se agregó:  0.0 en:  4 31
    víctimas No ubicado en 6
    se agregó:  0.0 en:  5 31
    víctimas No ubicado en 7
    se agregó:  0.0 en:  6 31
    víctimas No ubicado en 8
    se agregó:  0.0 en:  7 31
    víctimas No ubicado en 9
    se agregó:  0.0 en:  8 31
    víctimas No ubicado en 10
    se agregó:  0.0 en:  9 31
    conferencia No ubicado en 1
    se agregó:  0.0 en:  0 32
    conferencia No ubicado en 2
    se agregó:  0.0 en:  1 32
    conferencia Ubicado en  3
    se agregó:  2.0 en:  2 32
    conferencia Ubicado en  4
    se agregó:  2.0 en:  3 32
    conferencia No ubicado en 5
    se agregó:  0.0 en:  4 32
    conferencia No ubicado en 6
    se agregó:  0.0 en:  5 32
    conferencia No ubicado en 7
    se agregó:  0.0 en:  6 32
    conferencia No ubicado en 8
    se agregó:  0.0 en:  7 32
    conferencia No ubicado en 9
    se agregó:  0.0 en:  8 32
    conferencia No ubicado en 10
    se agregó:  0.0 en:  9 32
    presidente No ubicado en 1
    se agregó:  0.0 en:  0 33
    presidente No ubicado en 2
    se agregó:  0.0 en:  1 33
    presidente Ubicado en  3
    se agregó:  3.0 en:  2 33
    presidente Ubicado en  4
    se agregó:  3.0 en:  3 33
    presidente No ubicado en 5
    se agregó:  0.0 en:  4 33
    presidente No ubicado en 6
    se agregó:  0.0 en:  5 33
    presidente No ubicado en 7
    se agregó:  0.0 en:  6 33
    presidente Ubicado en  8
    se agregó:  3.0 en:  7 33
    presidente No ubicado en 9
    se agregó:  0.0 en:  8 33
    presidente No ubicado en 10
    se agregó:  0.0 en:  9 33
    queda No ubicado en 1
    se agregó:  0.0 en:  0 34
    queda No ubicado en 2
    se agregó:  0.0 en:  1 34
    queda Ubicado en  3
    se agregó:  1.0 en:  2 34
    queda No ubicado en 4
    se agregó:  0.0 en:  3 34
    queda No ubicado en 5
    se agregó:  0.0 en:  4 34
    queda No ubicado en 6
    se agregó:  0.0 en:  5 34
    queda No ubicado en 7
    se agregó:  0.0 en:  6 34
    queda No ubicado en 8
    se agregó:  0.0 en:  7 34
    queda No ubicado en 9
    se agregó:  0.0 en:  8 34
    queda No ubicado en 10
    se agregó:  0.0 en:  9 34
    eu No ubicado en 1
    se agregó:  0.0 en:  0 35
    eu No ubicado en 2
    se agregó:  0.0 en:  1 35
    eu Ubicado en  3
    se agregó:  1.0 en:  2 35
    eu No ubicado en 4
    se agregó:  0.0 en:  3 35
    eu No ubicado en 5
    se agregó:  0.0 en:  4 35
    eu No ubicado en 6
    se agregó:  0.0 en:  5 35
    eu No ubicado en 7
    se agregó:  0.0 en:  6 35
    eu No ubicado en 8
    se agregó:  0.0 en:  7 35
    eu No ubicado en 9
    se agregó:  0.0 en:  8 35
    eu No ubicado en 10
    se agregó:  0.0 en:  9 35
    80 No ubicado en 1
    se agregó:  0.0 en:  0 36
    80 No ubicado en 2
    se agregó:  0.0 en:  1 36
    80 Ubicado en  3
    se agregó:  1.0 en:  2 36
    80 No ubicado en 4
    se agregó:  0.0 en:  3 36
    80 No ubicado en 5
    se agregó:  0.0 en:  4 36
    80 No ubicado en 6
    se agregó:  0.0 en:  5 36
    80 No ubicado en 7
    se agregó:  0.0 en:  6 36
    80 No ubicado en 8
    se agregó:  0.0 en:  7 36
    80 No ubicado en 9
    se agregó:  0.0 en:  8 36
    80 No ubicado en 10
    se agregó:  0.0 en:  9 36
    comisiones No ubicado en 1
    se agregó:  0.0 en:  0 37
    comisiones No ubicado en 2
    se agregó:  0.0 en:  1 37
    comisiones Ubicado en  3
    se agregó:  1.0 en:  2 37
    comisiones No ubicado en 4
    se agregó:  0.0 en:  3 37
    comisiones No ubicado en 5
    se agregó:  0.0 en:  4 37
    comisiones No ubicado en 6
    se agregó:  0.0 en:  5 37
    comisiones No ubicado en 7
    se agregó:  0.0 en:  6 37
    comisiones No ubicado en 8
    se agregó:  0.0 en:  7 37
    comisiones No ubicado en 9
    se agregó:  0.0 en:  8 37
    comisiones No ubicado en 10
    se agregó:  0.0 en:  9 37
    remesas No ubicado en 1
    se agregó:  0.0 en:  0 38
    remesas No ubicado en 2
    se agregó:  0.0 en:  1 38
    remesas Ubicado en  3
    se agregó:  1.0 en:  2 38
    remesas No ubicado en 4
    se agregó:  0.0 en:  3 38
    remesas No ubicado en 5
    se agregó:  0.0 en:  4 38
    remesas No ubicado en 6
    se agregó:  0.0 en:  5 38
    remesas No ubicado en 7
    se agregó:  0.0 en:  6 38
    remesas No ubicado en 8
    se agregó:  0.0 en:  7 38
    remesas No ubicado en 9
    se agregó:  0.0 en:  8 38
    remesas No ubicado en 10
    se agregó:  0.0 en:  9 38
    profeco No ubicado en 1
    se agregó:  0.0 en:  0 39
    profeco No ubicado en 2
    se agregó:  0.0 en:  1 39
    profeco Ubicado en  3
    se agregó:  1.0 en:  2 39
    profeco No ubicado en 4
    se agregó:  0.0 en:  3 39
    profeco No ubicado en 5
    se agregó:  0.0 en:  4 39
    profeco No ubicado en 6
    se agregó:  0.0 en:  5 39
    profeco No ubicado en 7
    se agregó:  0.0 en:  6 39
    profeco No ubicado en 8
    se agregó:  0.0 en:  7 39
    profeco No ubicado en 9
    se agregó:  0.0 en:  8 39
    profeco No ubicado en 10
    se agregó:  0.0 en:  9 39
    entrega No ubicado en 1
    se agregó:  0.0 en:  0 40
    entrega No ubicado en 2
    se agregó:  0.0 en:  1 40
    entrega No ubicado en 3
    se agregó:  0.0 en:  2 40
    entrega Ubicado en  4
    se agregó:  1.0 en:  3 40
    entrega No ubicado en 5
    se agregó:  0.0 en:  4 40
    entrega No ubicado en 6
    se agregó:  0.0 en:  5 40
    entrega No ubicado en 7
    se agregó:  0.0 en:  6 40
    entrega No ubicado en 8
    se agregó:  0.0 en:  7 40
    entrega No ubicado en 9
    se agregó:  0.0 en:  8 40
    entrega No ubicado en 10
    se agregó:  0.0 en:  9 40
    fiscalía No ubicado en 1
    se agregó:  0.0 en:  0 41
    fiscalía No ubicado en 2
    se agregó:  0.0 en:  1 41
    fiscalía No ubicado en 3
    se agregó:  0.0 en:  2 41
    fiscalía Ubicado en  4
    se agregó:  2.0 en:  3 41
    fiscalía No ubicado en 5
    se agregó:  0.0 en:  4 41
    fiscalía No ubicado en 6
    se agregó:  0.0 en:  5 41
    fiscalía Ubicado en  7
    se agregó:  2.0 en:  6 41
    fiscalía No ubicado en 8
    se agregó:  0.0 en:  7 41
    fiscalía No ubicado en 9
    se agregó:  0.0 en:  8 41
    fiscalía No ubicado en 10
    se agregó:  0.0 en:  9 41
    2 No ubicado en 1
    se agregó:  0.0 en:  0 42
    2 No ubicado en 2
    se agregó:  0.0 en:  1 42
    2 No ubicado en 3
    se agregó:  0.0 en:  2 42
    2 Ubicado en  4
    se agregó:  2.0 en:  3 42
    2 Ubicado en  5
    se agregó:  2.0 en:  4 42
    2 No ubicado en 6
    se agregó:  0.0 en:  5 42
    2 No ubicado en 7
    se agregó:  0.0 en:  6 42
    2 No ubicado en 8
    se agregó:  0.0 en:  7 42
    2 No ubicado en 9
    se agregó:  0.0 en:  8 42
    2 No ubicado en 10
    se agregó:  0.0 en:  9 42
    mil No ubicado en 1
    se agregó:  0.0 en:  0 43
    mil No ubicado en 2
    se agregó:  0.0 en:  1 43
    mil No ubicado en 3
    se agregó:  0.0 en:  2 43
    mil Ubicado en  4
    se agregó:  2.0 en:  3 43
    mil Ubicado en  5
    se agregó:  2.0 en:  4 43
    mil No ubicado en 6
    se agregó:  0.0 en:  5 43
    mil No ubicado en 7
    se agregó:  0.0 en:  6 43
    mil No ubicado en 8
    se agregó:  0.0 en:  7 43
    mil No ubicado en 9
    se agregó:  0.0 en:  8 43
    mil No ubicado en 10
    se agregó:  0.0 en:  9 43
    mdp No ubicado en 1
    se agregó:  0.0 en:  0 44
    mdp No ubicado en 2
    se agregó:  0.0 en:  1 44
    mdp No ubicado en 3
    se agregó:  0.0 en:  2 44
    mdp Ubicado en  4
    se agregó:  2.0 en:  3 44
    mdp Ubicado en  5
    se agregó:  2.0 en:  4 44
    mdp No ubicado en 6
    se agregó:  0.0 en:  5 44
    mdp No ubicado en 7
    se agregó:  0.0 en:  6 44
    mdp No ubicado en 8
    se agregó:  0.0 en:  7 44
    mdp No ubicado en 9
    se agregó:  0.0 en:  8 44
    mdp No ubicado en 10
    se agregó:  0.0 en:  9 44
    amlo No ubicado en 1
    se agregó:  0.0 en:  0 45
    amlo No ubicado en 2
    se agregó:  0.0 en:  1 45
    amlo No ubicado en 3
    se agregó:  0.0 en:  2 45
    amlo Ubicado en  4
    se agregó:  1.0 en:  3 45
    amlo No ubicado en 5
    se agregó:  0.0 en:  4 45
    amlo No ubicado en 6
    se agregó:  0.0 en:  5 45
    amlo No ubicado en 7
    se agregó:  0.0 en:  6 45
    amlo No ubicado en 8
    se agregó:  0.0 en:  7 45
    amlo No ubicado en 9
    se agregó:  0.0 en:  8 45
    amlo No ubicado en 10
    se agregó:  0.0 en:  9 45
    premios No ubicado en 1
    se agregó:  0.0 en:  0 46
    premios No ubicado en 2
    se agregó:  0.0 en:  1 46
    premios No ubicado en 3
    se agregó:  0.0 en:  2 46
    premios Ubicado en  4
    se agregó:  1.0 en:  3 46
    premios No ubicado en 5
    se agregó:  0.0 en:  4 46
    premios No ubicado en 6
    se agregó:  0.0 en:  5 46
    premios No ubicado en 7
    se agregó:  0.0 en:  6 46
    premios No ubicado en 8
    se agregó:  0.0 en:  7 46
    premios No ubicado en 9
    se agregó:  0.0 en:  8 46
    premios No ubicado en 10
    se agregó:  0.0 en:  9 46
    rifa No ubicado en 1
    se agregó:  0.0 en:  0 47
    rifa No ubicado en 2
    se agregó:  0.0 en:  1 47
    rifa No ubicado en 3
    se agregó:  0.0 en:  2 47
    rifa Ubicado en  4
    se agregó:  1.0 en:  3 47
    rifa No ubicado en 5
    se agregó:  0.0 en:  4 47
    rifa No ubicado en 6
    se agregó:  0.0 en:  5 47
    rifa No ubicado en 7
    se agregó:  0.0 en:  6 47
    rifa No ubicado en 8
    se agregó:  0.0 en:  7 47
    rifa No ubicado en 9
    se agregó:  0.0 en:  8 47
    rifa No ubicado en 10
    se agregó:  0.0 en:  9 47
    conferenciapresidente No ubicado en 1
    se agregó:  0.0 en:  0 48
    conferenciapresidente No ubicado en 2
    se agregó:  0.0 en:  1 48
    conferenciapresidente No ubicado en 3
    se agregó:  0.0 en:  2 48
    conferenciapresidente No ubicado en 4
    se agregó:  0.0 en:  3 48
    conferenciapresidente Ubicado en  5
    se agregó:  1.0 en:  4 48
    conferenciapresidente No ubicado en 6
    se agregó:  0.0 en:  5 48
    conferenciapresidente No ubicado en 7
    se agregó:  0.0 en:  6 48
    conferenciapresidente No ubicado en 8
    se agregó:  0.0 en:  7 48
    conferenciapresidente No ubicado en 9
    se agregó:  0.0 en:  8 48
    conferenciapresidente No ubicado en 10
    se agregó:  0.0 en:  9 48
    alejandro No ubicado en 1
    se agregó:  0.0 en:  0 49
    alejandro No ubicado en 2
    se agregó:  0.0 en:  1 49
    alejandro No ubicado en 3
    se agregó:  0.0 en:  2 49
    alejandro No ubicado en 4
    se agregó:  0.0 en:  3 49
    alejandro Ubicado en  5
    se agregó:  1.0 en:  4 49
    alejandro No ubicado en 6
    se agregó:  0.0 en:  5 49
    alejandro No ubicado en 7
    se agregó:  0.0 en:  6 49
    alejandro No ubicado en 8
    se agregó:  0.0 en:  7 49
    alejandro No ubicado en 9
    se agregó:  0.0 en:  8 49
    alejandro No ubicado en 10
    se agregó:  0.0 en:  9 49
    gertz No ubicado en 1
    se agregó:  0.0 en:  0 50
    gertz No ubicado en 2
    se agregó:  0.0 en:  1 50
    gertz No ubicado en 3
    se agregó:  0.0 en:  2 50
    gertz No ubicado en 4
    se agregó:  0.0 en:  3 50
    gertz Ubicado en  5
    se agregó:  1.0 en:  4 50
    gertz No ubicado en 6
    se agregó:  0.0 en:  5 50
    gertz No ubicado en 7
    se agregó:  0.0 en:  6 50
    gertz No ubicado en 8
    se agregó:  0.0 en:  7 50
    gertz No ubicado en 9
    se agregó:  0.0 en:  8 50
    gertz No ubicado en 10
    se agregó:  0.0 en:  9 50
    manero No ubicado en 1
    se agregó:  0.0 en:  0 51
    manero No ubicado en 2
    se agregó:  0.0 en:  1 51
    manero No ubicado en 3
    se agregó:  0.0 en:  2 51
    manero No ubicado en 4
    se agregó:  0.0 en:  3 51
    manero Ubicado en  5
    se agregó:  1.0 en:  4 51
    manero No ubicado en 6
    se agregó:  0.0 en:  5 51
    manero No ubicado en 7
    se agregó:  0.0 en:  6 51
    manero No ubicado en 8
    se agregó:  0.0 en:  7 51
    manero No ubicado en 9
    se agregó:  0.0 en:  8 51
    manero No ubicado en 10
    se agregó:  0.0 en:  9 51
    titular No ubicado en 1
    se agregó:  0.0 en:  0 52
    titular No ubicado en 2
    se agregó:  0.0 en:  1 52
    titular No ubicado en 3
    se agregó:  0.0 en:  2 52
    titular No ubicado en 4
    se agregó:  0.0 en:  3 52
    titular Ubicado en  5
    se agregó:  1.0 en:  4 52
    titular No ubicado en 6
    se agregó:  0.0 en:  5 52
    titular No ubicado en 7
    se agregó:  0.0 en:  6 52
    titular No ubicado en 8
    se agregó:  0.0 en:  7 52
    titular No ubicado en 9
    se agregó:  0.0 en:  8 52
    titular No ubicado en 10
    se agregó:  0.0 en:  9 52
    fgr No ubicado en 1
    se agregó:  0.0 en:  0 53
    fgr No ubicado en 2
    se agregó:  0.0 en:  1 53
    fgr No ubicado en 3
    se agregó:  0.0 en:  2 53
    fgr No ubicado en 4
    se agregó:  0.0 en:  3 53
    fgr Ubicado en  5
    se agregó:  1.0 en:  4 53
    fgr No ubicado en 6
    se agregó:  0.0 en:  5 53
    fgr No ubicado en 7
    se agregó:  0.0 en:  6 53
    fgr No ubicado en 8
    se agregó:  0.0 en:  7 53
    fgr No ubicado en 9
    se agregó:  0.0 en:  8 53
    fgr No ubicado en 10
    se agregó:  0.0 en:  9 53
    mexico No ubicado en 1
    se agregó:  0.0 en:  0 54
    mexico No ubicado en 2
    se agregó:  0.0 en:  1 54
    mexico No ubicado en 3
    se agregó:  0.0 en:  2 54
    mexico No ubicado en 4
    se agregó:  0.0 en:  3 54
    mexico Ubicado en  5
    se agregó:  1.0 en:  4 54
    mexico No ubicado en 6
    se agregó:  0.0 en:  5 54
    mexico No ubicado en 7
    se agregó:  0.0 en:  6 54
    mexico No ubicado en 8
    se agregó:  0.0 en:  7 54
    mexico No ubicado en 9
    se agregó:  0.0 en:  8 54
    mexico No ubicado en 10
    se agregó:  0.0 en:  9 54
    entregó No ubicado en 1
    se agregó:  0.0 en:  0 55
    entregó No ubicado en 2
    se agregó:  0.0 en:  1 55
    entregó No ubicado en 3
    se agregó:  0.0 en:  2 55
    entregó No ubicado en 4
    se agregó:  0.0 en:  3 55
    entregó Ubicado en  5
    se agregó:  1.0 en:  4 55
    entregó No ubicado en 6
    se agregó:  0.0 en:  5 55
    entregó No ubicado en 7
    se agregó:  0.0 en:  6 55
    entregó No ubicado en 8
    se agregó:  0.0 en:  7 55
    entregó No ubicado en 9
    se agregó:  0.0 en:  8 55
    entregó No ubicado en 10
    se agregó:  0.0 en:  9 55
    instituto No ubicado en 1
    se agregó:  0.0 en:  0 56
    instituto No ubicado en 2
    se agregó:  0.0 en:  1 56
    instituto No ubicado en 3
    se agregó:  0.0 en:  2 56
    instituto No ubicado en 4
    se agregó:  0.0 en:  3 56
    instituto Ubicado en  5
    se agregó:  1.0 en:  4 56
    instituto No ubicado en 6
    se agregó:  0.0 en:  5 56
    instituto No ubicado en 7
    se agregó:  0.0 en:  6 56
    instituto No ubicado en 8
    se agregó:  0.0 en:  7 56
    instituto No ubicado en 9
    se agregó:  0.0 en:  8 56
    instituto No ubicado en 10
    se agregó:  0.0 en:  9 56
    devolver No ubicado en 1
    se agregó:  0.0 en:  0 57
    devolver No ubicado en 2
    se agregó:  0.0 en:  1 57
    devolver No ubicado en 3
    se agregó:  0.0 en:  2 57
    devolver No ubicado en 4
    se agregó:  0.0 en:  3 57
    devolver Ubicado en  5
    se agregó:  1.0 en:  4 57
    devolver No ubicado en 6
    se agregó:  0.0 en:  5 57
    devolver No ubicado en 7
    se agregó:  0.0 en:  6 57
    devolver No ubicado en 8
    se agregó:  0.0 en:  7 57
    devolver No ubicado en 9
    se agregó:  0.0 en:  8 57
    devolver No ubicado en 10
    se agregó:  0.0 en:  9 57
    pueblo No ubicado en 1
    se agregó:  0.0 en:  0 58
    pueblo No ubicado en 2
    se agregó:  0.0 en:  1 58
    pueblo No ubicado en 3
    se agregó:  0.0 en:  2 58
    pueblo No ubicado en 4
    se agregó:  0.0 en:  3 58
    pueblo Ubicado en  5
    se agregó:  1.0 en:  4 58
    pueblo No ubicado en 6
    se agregó:  0.0 en:  5 58
    pueblo No ubicado en 7
    se agregó:  0.0 en:  6 58
    pueblo No ubicado en 8
    se agregó:  0.0 en:  7 58
    pueblo No ubicado en 9
    se agregó:  0.0 en:  8 58
    pueblo No ubicado en 10
    se agregó:  0.0 en:  9 58
    robado No ubicado en 1
    se agregó:  0.0 en:  0 59
    robado No ubicado en 2
    se agregó:  0.0 en:  1 59
    robado No ubicado en 3
    se agregó:  0.0 en:  2 59
    robado No ubicado en 4
    se agregó:  0.0 en:  3 59
    robado Ubicado en  5
    se agregó:  1.0 en:  4 59
    robado No ubicado en 6
    se agregó:  0.0 en:  5 59
    robado No ubicado en 7
    se agregó:  0.0 en:  6 59
    robado No ubicado en 8
    se agregó:  0.0 en:  7 59
    robado No ubicado en 9
    se agregó:  0.0 en:  8 59
    robado No ubicado en 10
    se agregó:  0.0 en:  9 59
    bonitas No ubicado en 1
    se agregó:  0.0 en:  0 60
    bonitas No ubicado en 2
    se agregó:  0.0 en:  1 60
    bonitas No ubicado en 3
    se agregó:  0.0 en:  2 60
    bonitas No ubicado en 4
    se agregó:  0.0 en:  3 60
    bonitas No ubicado en 5
    se agregó:  0.0 en:  4 60
    bonitas Ubicado en  6
    se agregó:  1.0 en:  5 60
    bonitas No ubicado en 7
    se agregó:  0.0 en:  6 60
    bonitas No ubicado en 8
    se agregó:  0.0 en:  7 60
    bonitas No ubicado en 9
    se agregó:  0.0 en:  8 60
    bonitas No ubicado en 10
    se agregó:  0.0 en:  9 60
    palabras No ubicado en 1
    se agregó:  0.0 en:  0 61
    palabras No ubicado en 2
    se agregó:  0.0 en:  1 61
    palabras No ubicado en 3
    se agregó:  0.0 en:  2 61
    palabras No ubicado en 4
    se agregó:  0.0 en:  3 61
    palabras No ubicado en 5
    se agregó:  0.0 en:  4 61
    palabras Ubicado en  6
    se agregó:  1.0 en:  5 61
    palabras No ubicado en 7
    se agregó:  0.0 en:  6 61
    palabras No ubicado en 8
    se agregó:  0.0 en:  7 61
    palabras No ubicado en 9
    se agregó:  0.0 en:  8 61
    palabras No ubicado en 10
    se agregó:  0.0 en:  9 61
    emerson No ubicado en 1
    se agregó:  0.0 en:  0 62
    emerson No ubicado en 2
    se agregó:  0.0 en:  1 62
    emerson No ubicado en 3
    se agregó:  0.0 en:  2 62
    emerson No ubicado en 4
    se agregó:  0.0 en:  3 62
    emerson No ubicado en 5
    se agregó:  0.0 en:  4 62
    emerson Ubicado en  6
    se agregó:  1.0 en:  5 62
    emerson No ubicado en 7
    se agregó:  0.0 en:  6 62
    emerson No ubicado en 8
    se agregó:  0.0 en:  7 62
    emerson No ubicado en 9
    se agregó:  0.0 en:  8 62
    emerson No ubicado en 10
    se agregó:  0.0 en:  9 62
    fittipaldi No ubicado en 1
    se agregó:  0.0 en:  0 63
    fittipaldi No ubicado en 2
    se agregó:  0.0 en:  1 63
    fittipaldi No ubicado en 3
    se agregó:  0.0 en:  2 63
    fittipaldi No ubicado en 4
    se agregó:  0.0 en:  3 63
    fittipaldi No ubicado en 5
    se agregó:  0.0 en:  4 63
    fittipaldi Ubicado en  6
    se agregó:  1.0 en:  5 63
    fittipaldi No ubicado en 7
    se agregó:  0.0 en:  6 63
    fittipaldi No ubicado en 8
    se agregó:  0.0 en:  7 63
    fittipaldi No ubicado en 9
    se agregó:  0.0 en:  8 63
    fittipaldi No ubicado en 10
    se agregó:  0.0 en:  9 63
    fernando No ubicado en 1
    se agregó:  0.0 en:  0 64
    fernando No ubicado en 2
    se agregó:  0.0 en:  1 64
    fernando No ubicado en 3
    se agregó:  0.0 en:  2 64
    fernando No ubicado en 4
    se agregó:  0.0 en:  3 64
    fernando No ubicado en 5
    se agregó:  0.0 en:  4 64
    fernando Ubicado en  6
    se agregó:  1.0 en:  5 64
    fernando No ubicado en 7
    se agregó:  0.0 en:  6 64
    fernando No ubicado en 8
    se agregó:  0.0 en:  7 64
    fernando No ubicado en 9
    se agregó:  0.0 en:  8 64
    fernando No ubicado en 10
    se agregó:  0.0 en:  9 64
    alonso No ubicado en 1
    se agregó:  0.0 en:  0 65
    alonso No ubicado en 2
    se agregó:  0.0 en:  1 65
    alonso No ubicado en 3
    se agregó:  0.0 en:  2 65
    alonso No ubicado en 4
    se agregó:  0.0 en:  3 65
    alonso No ubicado en 5
    se agregó:  0.0 en:  4 65
    alonso Ubicado en  6
    se agregó:  1.0 en:  5 65
    alonso No ubicado en 7
    se agregó:  0.0 en:  6 65
    alonso No ubicado en 8
    se agregó:  0.0 en:  7 65
    alonso No ubicado en 9
    se agregó:  0.0 en:  8 65
    alonso No ubicado en 10
    se agregó:  0.0 en:  9 65
    fenómeno No ubicado en 1
    se agregó:  0.0 en:  0 66
    fenómeno No ubicado en 2
    se agregó:  0.0 en:  1 66
    fenómeno No ubicado en 3
    se agregó:  0.0 en:  2 66
    fenómeno No ubicado en 4
    se agregó:  0.0 en:  3 66
    fenómeno No ubicado en 5
    se agregó:  0.0 en:  4 66
    fenómeno Ubicado en  6
    se agregó:  1.0 en:  5 66
    fenómeno No ubicado en 7
    se agregó:  0.0 en:  6 66
    fenómeno No ubicado en 8
    se agregó:  0.0 en:  7 66
    fenómeno No ubicado en 9
    se agregó:  0.0 en:  8 66
    fenómeno No ubicado en 10
    se agregó:  0.0 en:  9 66
    talentoso No ubicado en 1
    se agregó:  0.0 en:  0 67
    talentoso No ubicado en 2
    se agregó:  0.0 en:  1 67
    talentoso No ubicado en 3
    se agregó:  0.0 en:  2 67
    talentoso No ubicado en 4
    se agregó:  0.0 en:  3 67
    talentoso No ubicado en 5
    se agregó:  0.0 en:  4 67
    talentoso Ubicado en  6
    se agregó:  1.0 en:  5 67
    talentoso No ubicado en 7
    se agregó:  0.0 en:  6 67
    talentoso No ubicado en 8
    se agregó:  0.0 en:  7 67
    talentoso No ubicado en 9
    se agregó:  0.0 en:  8 67
    talentoso No ubicado en 10
    se agregó:  0.0 en:  9 67
    espero No ubicado en 1
    se agregó:  0.0 en:  0 68
    espero No ubicado en 2
    se agregó:  0.0 en:  1 68
    espero No ubicado en 3
    se agregó:  0.0 en:  2 68
    espero No ubicado en 4
    se agregó:  0.0 en:  3 68
    espero No ubicado en 5
    se agregó:  0.0 en:  4 68
    espero Ubicado en  6
    se agregó:  1.0 en:  5 68
    espero No ubicado en 7
    se agregó:  0.0 en:  6 68
    espero No ubicado en 8
    se agregó:  0.0 en:  7 68
    espero No ubicado en 9
    se agregó:  0.0 en:  8 68
    espero No ubicado en 10
    se agregó:  0.0 en:  9 68
    vuelva No ubicado en 1
    se agregó:  0.0 en:  0 69
    vuelva No ubicado en 2
    se agregó:  0.0 en:  1 69
    vuelva No ubicado en 3
    se agregó:  0.0 en:  2 69
    vuelva No ubicado en 4
    se agregó:  0.0 en:  3 69
    vuelva No ubicado en 5
    se agregó:  0.0 en:  4 69
    vuelva Ubicado en  6
    se agregó:  1.0 en:  5 69
    vuelva No ubicado en 7
    se agregó:  0.0 en:  6 69
    vuelva No ubicado en 8
    se agregó:  0.0 en:  7 69
    vuelva No ubicado en 9
    se agregó:  0.0 en:  8 69
    vuelva No ubicado en 10
    se agregó:  0.0 en:  9 69
    fórmula No ubicado en 1
    se agregó:  0.0 en:  0 70
    fórmula No ubicado en 2
    se agregó:  0.0 en:  1 70
    fórmula No ubicado en 3
    se agregó:  0.0 en:  2 70
    fórmula No ubicado en 4
    se agregó:  0.0 en:  3 70
    fórmula No ubicado en 5
    se agregó:  0.0 en:  4 70
    fórmula Ubicado en  6
    se agregó:  1.0 en:  5 70
    fórmula No ubicado en 7
    se agregó:  0.0 en:  6 70
    fórmula No ubicado en 8
    se agregó:  0.0 en:  7 70
    fórmula No ubicado en 9
    se agregó:  0.0 en:  8 70
    fórmula No ubicado en 10
    se agregó:  0.0 en:  9 70
    1 No ubicado en 1
    se agregó:  0.0 en:  0 71
    1 No ubicado en 2
    se agregó:  0.0 en:  1 71
    1 No ubicado en 3
    se agregó:  0.0 en:  2 71
    1 No ubicado en 4
    se agregó:  0.0 en:  3 71
    1 No ubicado en 5
    se agregó:  0.0 en:  4 71
    1 Ubicado en  6
    se agregó:  1.0 en:  5 71
    1 No ubicado en 7
    se agregó:  0.0 en:  6 71
    1 No ubicado en 8
    se agregó:  0.0 en:  7 71
    1 No ubicado en 9
    se agregó:  0.0 en:  8 71
    1 No ubicado en 10
    se agregó:  0.0 en:  9 71
    2021 No ubicado en 1
    se agregó:  0.0 en:  0 72
    2021 No ubicado en 2
    se agregó:  0.0 en:  1 72
    2021 No ubicado en 3
    se agregó:  0.0 en:  2 72
    2021 No ubicado en 4
    se agregó:  0.0 en:  3 72
    2021 No ubicado en 5
    se agregó:  0.0 en:  4 72
    2021 Ubicado en  6
    se agregó:  1.0 en:  5 72
    2021 No ubicado en 7
    se agregó:  0.0 en:  6 72
    2021 No ubicado en 8
    se agregó:  0.0 en:  7 72
    2021 No ubicado en 9
    se agregó:  0.0 en:  8 72
    2021 No ubicado en 10
    se agregó:  0.0 en:  9 72
    acuerdo No ubicado en 1
    se agregó:  0.0 en:  0 73
    acuerdo No ubicado en 2
    se agregó:  0.0 en:  1 73
    acuerdo No ubicado en 3
    se agregó:  0.0 en:  2 73
    acuerdo No ubicado en 4
    se agregó:  0.0 en:  3 73
    acuerdo No ubicado en 5
    se agregó:  0.0 en:  4 73
    acuerdo No ubicado en 6
    se agregó:  0.0 en:  5 73
    acuerdo Ubicado en  7
    se agregó:  1.0 en:  6 73
    acuerdo No ubicado en 8
    se agregó:  0.0 en:  7 73
    acuerdo No ubicado en 9
    se agregó:  0.0 en:  8 73
    acuerdo No ubicado en 10
    se agregó:  0.0 en:  9 73
    imágenes No ubicado en 1
    se agregó:  0.0 en:  0 74
    imágenes No ubicado en 2
    se agregó:  0.0 en:  1 74
    imágenes No ubicado en 3
    se agregó:  0.0 en:  2 74
    imágenes No ubicado en 4
    se agregó:  0.0 en:  3 74
    imágenes No ubicado en 5
    se agregó:  0.0 en:  4 74
    imágenes No ubicado en 6
    se agregó:  0.0 en:  5 74
    imágenes Ubicado en  7
    se agregó:  1.0 en:  6 74
    imágenes No ubicado en 8
    se agregó:  0.0 en:  7 74
    imágenes No ubicado en 9
    se agregó:  0.0 en:  8 74
    imágenes No ubicado en 10
    se agregó:  0.0 en:  9 74
    obtenidas No ubicado en 1
    se agregó:  0.0 en:  0 75
    obtenidas No ubicado en 2
    se agregó:  0.0 en:  1 75
    obtenidas No ubicado en 3
    se agregó:  0.0 en:  2 75
    obtenidas No ubicado en 4
    se agregó:  0.0 en:  3 75
    obtenidas No ubicado en 5
    se agregó:  0.0 en:  4 75
    obtenidas No ubicado en 6
    se agregó:  0.0 en:  5 75
    obtenidas Ubicado en  7
    se agregó:  1.0 en:  6 75
    obtenidas No ubicado en 8
    se agregó:  0.0 en:  7 75
    obtenidas No ubicado en 9
    se agregó:  0.0 en:  8 75
    obtenidas No ubicado en 10
    se agregó:  0.0 en:  9 75
    capitalina No ubicado en 1
    se agregó:  0.0 en:  0 76
    capitalina No ubicado en 2
    se agregó:  0.0 en:  1 76
    capitalina No ubicado en 3
    se agregó:  0.0 en:  2 76
    capitalina No ubicado en 4
    se agregó:  0.0 en:  3 76
    capitalina No ubicado en 5
    se agregó:  0.0 en:  4 76
    capitalina No ubicado en 6
    se agregó:  0.0 en:  5 76
    capitalina Ubicado en  7
    se agregó:  1.0 en:  6 76
    capitalina No ubicado en 8
    se agregó:  0.0 en:  7 76
    capitalina No ubicado en 9
    se agregó:  0.0 en:  8 76
    capitalina No ubicado en 10
    se agregó:  0.0 en:  9 76
    mujer No ubicado en 1
    se agregó:  0.0 en:  0 77
    mujer No ubicado en 2
    se agregó:  0.0 en:  1 77
    mujer No ubicado en 3
    se agregó:  0.0 en:  2 77
    mujer No ubicado en 4
    se agregó:  0.0 en:  3 77
    mujer No ubicado en 5
    se agregó:  0.0 en:  4 77
    mujer No ubicado en 6
    se agregó:  0.0 en:  5 77
    mujer Ubicado en  7
    se agregó:  1.0 en:  6 77
    mujer No ubicado en 8
    se agregó:  0.0 en:  7 77
    mujer No ubicado en 9
    se agregó:  0.0 en:  8 77
    mujer No ubicado en 10
    se agregó:  0.0 en:  9 77
    subió No ubicado en 1
    se agregó:  0.0 en:  0 78
    subió No ubicado en 2
    se agregó:  0.0 en:  1 78
    subió No ubicado en 3
    se agregó:  0.0 en:  2 78
    subió No ubicado en 4
    se agregó:  0.0 en:  3 78
    subió No ubicado en 5
    se agregó:  0.0 en:  4 78
    subió No ubicado en 6
    se agregó:  0.0 en:  5 78
    subió Ubicado en  7
    se agregó:  1.0 en:  6 78
    subió No ubicado en 8
    se agregó:  0.0 en:  7 78
    subió No ubicado en 9
    se agregó:  0.0 en:  8 78
    subió No ubicado en 10
    se agregó:  0.0 en:  9 78
    fátima No ubicado en 1
    se agregó:  0.0 en:  0 79
    fátima No ubicado en 2
    se agregó:  0.0 en:  1 79
    fátima No ubicado en 3
    se agregó:  0.0 en:  2 79
    fátima No ubicado en 4
    se agregó:  0.0 en:  3 79
    fátima No ubicado en 5
    se agregó:  0.0 en:  4 79
    fátima No ubicado en 6
    se agregó:  0.0 en:  5 79
    fátima Ubicado en  7
    se agregó:  1.0 en:  6 79
    fátima No ubicado en 8
    se agregó:  0.0 en:  7 79
    fátima No ubicado en 9
    se agregó:  0.0 en:  8 79
    fátima No ubicado en 10
    se agregó:  0.0 en:  9 79
    vehículo No ubicado en 1
    se agregó:  0.0 en:  0 80
    vehículo No ubicado en 2
    se agregó:  0.0 en:  1 80
    vehículo No ubicado en 3
    se agregó:  0.0 en:  2 80
    vehículo No ubicado en 4
    se agregó:  0.0 en:  3 80
    vehículo No ubicado en 5
    se agregó:  0.0 en:  4 80
    vehículo No ubicado en 6
    se agregó:  0.0 en:  5 80
    vehículo Ubicado en  7
    se agregó:  1.0 en:  6 80
    vehículo No ubicado en 8
    se agregó:  0.0 en:  7 80
    vehículo No ubicado en 9
    se agregó:  0.0 en:  8 80
    vehículo No ubicado en 10
    se agregó:  0.0 en:  9 80
    blanco No ubicado en 1
    se agregó:  0.0 en:  0 81
    blanco No ubicado en 2
    se agregó:  0.0 en:  1 81
    blanco No ubicado en 3
    se agregó:  0.0 en:  2 81
    blanco No ubicado en 4
    se agregó:  0.0 en:  3 81
    blanco No ubicado en 5
    se agregó:  0.0 en:  4 81
    blanco No ubicado en 6
    se agregó:  0.0 en:  5 81
    blanco Ubicado en  7
    se agregó:  1.0 en:  6 81
    blanco No ubicado en 8
    se agregó:  0.0 en:  7 81
    blanco No ubicado en 9
    se agregó:  0.0 en:  8 81
    blanco No ubicado en 10
    se agregó:  0.0 en:  9 81
    obrador No ubicado en 1
    se agregó:  0.0 en:  0 82
    obrador No ubicado en 2
    se agregó:  0.0 en:  1 82
    obrador No ubicado en 3
    se agregó:  0.0 en:  2 82
    obrador No ubicado en 4
    se agregó:  0.0 en:  3 82
    obrador No ubicado en 5
    se agregó:  0.0 en:  4 82
    obrador No ubicado en 6
    se agregó:  0.0 en:  5 82
    obrador No ubicado en 7
    se agregó:  0.0 en:  6 82
    obrador Ubicado en  8
    se agregó:  1.0 en:  7 82
    obrador No ubicado en 9
    se agregó:  0.0 en:  8 82
    obrador No ubicado en 10
    se agregó:  0.0 en:  9 82
    comprometió No ubicado en 1
    se agregó:  0.0 en:  0 83
    comprometió No ubicado en 2
    se agregó:  0.0 en:  1 83
    comprometió No ubicado en 3
    se agregó:  0.0 en:  2 83
    comprometió No ubicado en 4
    se agregó:  0.0 en:  3 83
    comprometió No ubicado en 5
    se agregó:  0.0 en:  4 83
    comprometió No ubicado en 6
    se agregó:  0.0 en:  5 83
    comprometió No ubicado en 7
    se agregó:  0.0 en:  6 83
    comprometió Ubicado en  8
    se agregó:  1.0 en:  7 83
    comprometió No ubicado en 9
    se agregó:  0.0 en:  8 83
    comprometió No ubicado en 10
    se agregó:  0.0 en:  9 83
    enfrentar No ubicado en 1
    se agregó:  0.0 en:  0 84
    enfrentar No ubicado en 2
    se agregó:  0.0 en:  1 84
    enfrentar No ubicado en 3
    se agregó:  0.0 en:  2 84
    enfrentar No ubicado en 4
    se agregó:  0.0 en:  3 84
    enfrentar No ubicado en 5
    se agregó:  0.0 en:  4 84
    enfrentar No ubicado en 6
    se agregó:  0.0 en:  5 84
    enfrentar No ubicado en 7
    se agregó:  0.0 en:  6 84
    enfrentar Ubicado en  8
    se agregó:  1.0 en:  7 84
    enfrentar No ubicado en 9
    se agregó:  0.0 en:  8 84
    enfrentar No ubicado en 10
    se agregó:  0.0 en:  9 84
    empresarios No ubicado en 1
    se agregó:  0.0 en:  0 85
    empresarios No ubicado en 2
    se agregó:  0.0 en:  1 85
    empresarios No ubicado en 3
    se agregó:  0.0 en:  2 85
    empresarios No ubicado en 4
    se agregó:  0.0 en:  3 85
    empresarios No ubicado en 5
    se agregó:  0.0 en:  4 85
    empresarios No ubicado en 6
    se agregó:  0.0 en:  5 85
    empresarios No ubicado en 7
    se agregó:  0.0 en:  6 85
    empresarios Ubicado en  8
    se agregó:  1.0 en:  7 85
    empresarios No ubicado en 9
    se agregó:  0.0 en:  8 85
    empresarios No ubicado en 10
    se agregó:  0.0 en:  9 85
    sindicatos No ubicado en 1
    se agregó:  0.0 en:  0 86
    sindicatos No ubicado en 2
    se agregó:  0.0 en:  1 86
    sindicatos No ubicado en 3
    se agregó:  0.0 en:  2 86
    sindicatos No ubicado en 4
    se agregó:  0.0 en:  3 86
    sindicatos No ubicado en 5
    se agregó:  0.0 en:  4 86
    sindicatos No ubicado en 6
    se agregó:  0.0 en:  5 86
    sindicatos No ubicado en 7
    se agregó:  0.0 en:  6 86
    sindicatos Ubicado en  8
    se agregó:  1.0 en:  7 86
    sindicatos No ubicado en 9
    se agregó:  0.0 en:  8 86
    sindicatos No ubicado en 10
    se agregó:  0.0 en:  9 86
    problema No ubicado en 1
    se agregó:  0.0 en:  0 87
    problema No ubicado en 2
    se agregó:  0.0 en:  1 87
    problema No ubicado en 3
    se agregó:  0.0 en:  2 87
    problema No ubicado en 4
    se agregó:  0.0 en:  3 87
    problema No ubicado en 5
    se agregó:  0.0 en:  4 87
    problema No ubicado en 6
    se agregó:  0.0 en:  5 87
    problema No ubicado en 7
    se agregó:  0.0 en:  6 87
    problema Ubicado en  8
    se agregó:  1.0 en:  7 87
    problema No ubicado en 9
    se agregó:  0.0 en:  8 87
    problema No ubicado en 10
    se agregó:  0.0 en:  9 87
    pensiones No ubicado en 1
    se agregó:  0.0 en:  0 88
    pensiones No ubicado en 2
    se agregó:  0.0 en:  1 88
    pensiones No ubicado en 3
    se agregó:  0.0 en:  2 88
    pensiones No ubicado en 4
    se agregó:  0.0 en:  3 88
    pensiones No ubicado en 5
    se agregó:  0.0 en:  4 88
    pensiones No ubicado en 6
    se agregó:  0.0 en:  5 88
    pensiones No ubicado en 7
    se agregó:  0.0 en:  6 88
    pensiones Ubicado en  8
    se agregó:  1.0 en:  7 88
    pensiones No ubicado en 9
    se agregó:  0.0 en:  8 88
    pensiones No ubicado en 10
    se agregó:  0.0 en:  9 88
    trabajadores No ubicado en 1
    se agregó:  0.0 en:  0 89
    trabajadores No ubicado en 2
    se agregó:  0.0 en:  1 89
    trabajadores No ubicado en 3
    se agregó:  0.0 en:  2 89
    trabajadores No ubicado en 4
    se agregó:  0.0 en:  3 89
    trabajadores No ubicado en 5
    se agregó:  0.0 en:  4 89
    trabajadores No ubicado en 6
    se agregó:  0.0 en:  5 89
    trabajadores No ubicado en 7
    se agregó:  0.0 en:  6 89
    trabajadores Ubicado en  8
    se agregó:  1.0 en:  7 89
    trabajadores No ubicado en 9
    se agregó:  0.0 en:  8 89
    trabajadores No ubicado en 10
    se agregó:  0.0 en:  9 89
    estudiante No ubicado en 1
    se agregó:  0.0 en:  0 90
    estudiante No ubicado en 2
    se agregó:  0.0 en:  1 90
    estudiante No ubicado en 3
    se agregó:  0.0 en:  2 90
    estudiante No ubicado en 4
    se agregó:  0.0 en:  3 90
    estudiante No ubicado en 5
    se agregó:  0.0 en:  4 90
    estudiante No ubicado en 6
    se agregó:  0.0 en:  5 90
    estudiante No ubicado en 7
    se agregó:  0.0 en:  6 90
    estudiante No ubicado en 8
    se agregó:  0.0 en:  7 90
    estudiante Ubicado en  9
    se agregó:  1.0 en:  8 90
    estudiante No ubicado en 10
    se agregó:  0.0 en:  9 90
    grave No ubicado en 1
    se agregó:  0.0 en:  0 91
    grave No ubicado en 2
    se agregó:  0.0 en:  1 91
    grave No ubicado en 3
    se agregó:  0.0 en:  2 91
    grave No ubicado en 4
    se agregó:  0.0 en:  3 91
    grave No ubicado en 5
    se agregó:  0.0 en:  4 91
    grave No ubicado en 6
    se agregó:  0.0 en:  5 91
    grave No ubicado en 7
    se agregó:  0.0 en:  6 91
    grave No ubicado en 8
    se agregó:  0.0 en:  7 91
    grave Ubicado en  9
    se agregó:  1.0 en:  8 91
    grave No ubicado en 10
    se agregó:  0.0 en:  9 91
    tras No ubicado en 1
    se agregó:  0.0 en:  0 92
    tras No ubicado en 2
    se agregó:  0.0 en:  1 92
    tras No ubicado en 3
    se agregó:  0.0 en:  2 92
    tras No ubicado en 4
    se agregó:  0.0 en:  3 92
    tras No ubicado en 5
    se agregó:  0.0 en:  4 92
    tras No ubicado en 6
    se agregó:  0.0 en:  5 92
    tras No ubicado en 7
    se agregó:  0.0 en:  6 92
    tras No ubicado en 8
    se agregó:  0.0 en:  7 92
    tras Ubicado en  9
    se agregó:  1.0 en:  8 92
    tras No ubicado en 10
    se agregó:  0.0 en:  9 92
    desalojo No ubicado en 1
    se agregó:  0.0 en:  0 93
    desalojo No ubicado en 2
    se agregó:  0.0 en:  1 93
    desalojo No ubicado en 3
    se agregó:  0.0 en:  2 93
    desalojo No ubicado en 4
    se agregó:  0.0 en:  3 93
    desalojo No ubicado en 5
    se agregó:  0.0 en:  4 93
    desalojo No ubicado en 6
    se agregó:  0.0 en:  5 93
    desalojo No ubicado en 7
    se agregó:  0.0 en:  6 93
    desalojo No ubicado en 8
    se agregó:  0.0 en:  7 93
    desalojo Ubicado en  9
    se agregó:  1.0 en:  8 93
    desalojo No ubicado en 10
    se agregó:  0.0 en:  9 93
    padres No ubicado en 1
    se agregó:  0.0 en:  0 94
    padres No ubicado en 2
    se agregó:  0.0 en:  1 94
    padres No ubicado en 3
    se agregó:  0.0 en:  2 94
    padres No ubicado en 4
    se agregó:  0.0 en:  3 94
    padres No ubicado en 5
    se agregó:  0.0 en:  4 94
    padres No ubicado en 6
    se agregó:  0.0 en:  5 94
    padres No ubicado en 7
    se agregó:  0.0 en:  6 94
    padres No ubicado en 8
    se agregó:  0.0 en:  7 94
    padres Ubicado en  9
    se agregó:  1.0 en:  8 94
    padres No ubicado en 10
    se agregó:  0.0 en:  9 94
    43 No ubicado en 1
    se agregó:  0.0 en:  0 95
    43 No ubicado en 2
    se agregó:  0.0 en:  1 95
    43 No ubicado en 3
    se agregó:  0.0 en:  2 95
    43 No ubicado en 4
    se agregó:  0.0 en:  3 95
    43 No ubicado en 5
    se agregó:  0.0 en:  4 95
    43 No ubicado en 6
    se agregó:  0.0 en:  5 95
    43 No ubicado en 7
    se agregó:  0.0 en:  6 95
    43 No ubicado en 8
    se agregó:  0.0 en:  7 95
    43 Ubicado en  9
    se agregó:  1.0 en:  8 95
    43 No ubicado en 10
    se agregó:  0.0 en:  9 95
    ayotzinapa No ubicado en 1
    se agregó:  0.0 en:  0 96
    ayotzinapa No ubicado en 2
    se agregó:  0.0 en:  1 96
    ayotzinapa No ubicado en 3
    se agregó:  0.0 en:  2 96
    ayotzinapa No ubicado en 4
    se agregó:  0.0 en:  3 96
    ayotzinapa No ubicado en 5
    se agregó:  0.0 en:  4 96
    ayotzinapa No ubicado en 6
    se agregó:  0.0 en:  5 96
    ayotzinapa No ubicado en 7
    se agregó:  0.0 en:  6 96
    ayotzinapa No ubicado en 8
    se agregó:  0.0 en:  7 96
    ayotzinapa Ubicado en  9
    se agregó:  1.0 en:  8 96
    ayotzinapa No ubicado en 10
    se agregó:  0.0 en:  9 96
    manifestantes No ubicado en 1
    se agregó:  0.0 en:  0 97
    manifestantes No ubicado en 2
    se agregó:  0.0 en:  1 97
    manifestantes No ubicado en 3
    se agregó:  0.0 en:  2 97
    manifestantes No ubicado en 4
    se agregó:  0.0 en:  3 97
    manifestantes No ubicado en 5
    se agregó:  0.0 en:  4 97
    manifestantes No ubicado en 6
    se agregó:  0.0 en:  5 97
    manifestantes No ubicado en 7
    se agregó:  0.0 en:  6 97
    manifestantes No ubicado en 8
    se agregó:  0.0 en:  7 97
    manifestantes Ubicado en  9
    se agregó:  1.0 en:  8 97
    manifestantes No ubicado en 10
    se agregó:  0.0 en:  9 97
    chiapas No ubicado en 1
    se agregó:  0.0 en:  0 98
    chiapas No ubicado en 2
    se agregó:  0.0 en:  1 98
    chiapas No ubicado en 3
    se agregó:  0.0 en:  2 98
    chiapas No ubicado en 4
    se agregó:  0.0 en:  3 98
    chiapas No ubicado en 5
    se agregó:  0.0 en:  4 98
    chiapas No ubicado en 6
    se agregó:  0.0 en:  5 98
    chiapas No ubicado en 7
    se agregó:  0.0 en:  6 98
    chiapas No ubicado en 8
    se agregó:  0.0 en:  7 98
    chiapas Ubicado en  9
    se agregó:  1.0 en:  8 98
    chiapas No ubicado en 10
    se agregó:  0.0 en:  9 98
    fisco No ubicado en 1
    se agregó:  0.0 en:  0 99
    fisco No ubicado en 2
    se agregó:  0.0 en:  1 99
    fisco No ubicado en 3
    se agregó:  0.0 en:  2 99
    fisco No ubicado en 4
    se agregó:  0.0 en:  3 99
    fisco No ubicado en 5
    se agregó:  0.0 en:  4 99
    fisco No ubicado en 6
    se agregó:  0.0 en:  5 99
    fisco No ubicado en 7
    se agregó:  0.0 en:  6 99
    fisco No ubicado en 8
    se agregó:  0.0 en:  7 99
    fisco No ubicado en 9
    se agregó:  0.0 en:  8 99
    fisco Ubicado en  10
    se agregó:  1.0 en:  9 99
    advierte No ubicado en 1
    se agregó:  0.0 en:  0 100
    advierte No ubicado en 2
    se agregó:  0.0 en:  1 100
    advierte No ubicado en 3
    se agregó:  0.0 en:  2 100
    advierte No ubicado en 4
    se agregó:  0.0 en:  3 100
    advierte No ubicado en 5
    se agregó:  0.0 en:  4 100
    advierte No ubicado en 6
    se agregó:  0.0 en:  5 100
    advierte No ubicado en 7
    se agregó:  0.0 en:  6 100
    advierte No ubicado en 8
    se agregó:  0.0 en:  7 100
    advierte No ubicado en 9
    se agregó:  0.0 en:  8 100
    advierte Ubicado en  10
    se agregó:  1.0 en:  9 100
    esquemas No ubicado en 1
    se agregó:  0.0 en:  0 101
    esquemas No ubicado en 2
    se agregó:  0.0 en:  1 101
    esquemas No ubicado en 3
    se agregó:  0.0 en:  2 101
    esquemas No ubicado en 4
    se agregó:  0.0 en:  3 101
    esquemas No ubicado en 5
    se agregó:  0.0 en:  4 101
    esquemas No ubicado en 6
    se agregó:  0.0 en:  5 101
    esquemas No ubicado en 7
    se agregó:  0.0 en:  6 101
    esquemas No ubicado en 8
    se agregó:  0.0 en:  7 101
    esquemas No ubicado en 9
    se agregó:  0.0 en:  8 101
    esquemas Ubicado en  10
    se agregó:  1.0 en:  9 101
    operaciones No ubicado en 1
    se agregó:  0.0 en:  0 102
    operaciones No ubicado en 2
    se agregó:  0.0 en:  1 102
    operaciones No ubicado en 3
    se agregó:  0.0 en:  2 102
    operaciones No ubicado en 4
    se agregó:  0.0 en:  3 102
    operaciones No ubicado en 5
    se agregó:  0.0 en:  4 102
    operaciones No ubicado en 6
    se agregó:  0.0 en:  5 102
    operaciones No ubicado en 7
    se agregó:  0.0 en:  6 102
    operaciones No ubicado en 8
    se agregó:  0.0 en:  7 102
    operaciones No ubicado en 9
    se agregó:  0.0 en:  8 102
    operaciones Ubicado en  10
    se agregó:  1.0 en:  9 102
    realizadas No ubicado en 1
    se agregó:  0.0 en:  0 103
    realizadas No ubicado en 2
    se agregó:  0.0 en:  1 103
    realizadas No ubicado en 3
    se agregó:  0.0 en:  2 103
    realizadas No ubicado en 4
    se agregó:  0.0 en:  3 103
    realizadas No ubicado en 5
    se agregó:  0.0 en:  4 103
    realizadas No ubicado en 6
    se agregó:  0.0 en:  5 103
    realizadas No ubicado en 7
    se agregó:  0.0 en:  6 103
    realizadas No ubicado en 8
    se agregó:  0.0 en:  7 103
    realizadas No ubicado en 9
    se agregó:  0.0 en:  8 103
    realizadas Ubicado en  10
    se agregó:  1.0 en:  9 103
    2017 No ubicado en 1
    se agregó:  0.0 en:  0 104
    2017 No ubicado en 2
    se agregó:  0.0 en:  1 104
    2017 No ubicado en 3
    se agregó:  0.0 en:  2 104
    2017 No ubicado en 4
    se agregó:  0.0 en:  3 104
    2017 No ubicado en 5
    se agregó:  0.0 en:  4 104
    2017 No ubicado en 6
    se agregó:  0.0 en:  5 104
    2017 No ubicado en 7
    se agregó:  0.0 en:  6 104
    2017 No ubicado en 8
    se agregó:  0.0 en:  7 104
    2017 No ubicado en 9
    se agregó:  0.0 en:  8 104
    2017 Ubicado en  10
    se agregó:  1.0 en:  9 104
    2019 No ubicado en 1
    se agregó:  0.0 en:  0 105
    2019 No ubicado en 2
    se agregó:  0.0 en:  1 105
    2019 No ubicado en 3
    se agregó:  0.0 en:  2 105
    2019 No ubicado en 4
    se agregó:  0.0 en:  3 105
    2019 No ubicado en 5
    se agregó:  0.0 en:  4 105
    2019 No ubicado en 6
    se agregó:  0.0 en:  5 105
    2019 No ubicado en 7
    se agregó:  0.0 en:  6 105
    2019 No ubicado en 8
    se agregó:  0.0 en:  7 105
    2019 No ubicado en 9
    se agregó:  0.0 en:  8 105
    2019 Ubicado en  10
    se agregó:  1.0 en:  9 105
    339 No ubicado en 1
    se agregó:  0.0 en:  0 106
    339 No ubicado en 2
    se agregó:  0.0 en:  1 106
    339 No ubicado en 3
    se agregó:  0.0 en:  2 106
    339 No ubicado en 4
    se agregó:  0.0 en:  3 106
    339 No ubicado en 5
    se agregó:  0.0 en:  4 106
    339 No ubicado en 6
    se agregó:  0.0 en:  5 106
    339 No ubicado en 7
    se agregó:  0.0 en:  6 106
    339 No ubicado en 8
    se agregó:  0.0 en:  7 106
    339 No ubicado en 9
    se agregó:  0.0 en:  8 106
    339 Ubicado en  10
    se agregó:  1.0 en:  9 106
    000 No ubicado en 1
    se agregó:  0.0 en:  0 107
    000 No ubicado en 2
    se agregó:  0.0 en:  1 107
    000 No ubicado en 3
    se agregó:  0.0 en:  2 107
    000 No ubicado en 4
    se agregó:  0.0 en:  3 107
    000 No ubicado en 5
    se agregó:  0.0 en:  4 107
    000 No ubicado en 6
    se agregó:  0.0 en:  5 107
    000 No ubicado en 7
    se agregó:  0.0 en:  6 107
    000 No ubicado en 8
    se agregó:  0.0 en:  7 107
    000 No ubicado en 9
    se agregó:  0.0 en:  8 107
    000 Ubicado en  10
    se agregó:  1.0 en:  9 107
    millones No ubicado en 1
    se agregó:  0.0 en:  0 108
    millones No ubicado en 2
    se agregó:  0.0 en:  1 108
    millones No ubicado en 3
    se agregó:  0.0 en:  2 108
    millones No ubicado en 4
    se agregó:  0.0 en:  3 108
    millones No ubicado en 5
    se agregó:  0.0 en:  4 108
    millones No ubicado en 6
    se agregó:  0.0 en:  5 108
    millones No ubicado en 7
    se agregó:  0.0 en:  6 108
    millones No ubicado en 8
    se agregó:  0.0 en:  7 108
    millones No ubicado en 9
    se agregó:  0.0 en:  8 108
    millones Ubicado en  10
    se agregó:  1.0 en:  9 108
    pesos No ubicado en 1
    se agregó:  0.0 en:  0 109
    pesos No ubicado en 2
    se agregó:  0.0 en:  1 109
    pesos No ubicado en 3
    se agregó:  0.0 en:  2 109
    pesos No ubicado en 4
    se agregó:  0.0 en:  3 109
    pesos No ubicado en 5
    se agregó:  0.0 en:  4 109
    pesos No ubicado en 6
    se agregó:  0.0 en:  5 109
    pesos No ubicado en 7
    se agregó:  0.0 en:  6 109
    pesos No ubicado en 8
    se agregó:  0.0 en:  7 109
    pesos No ubicado en 9
    se agregó:  0.0 en:  8 109
    pesos Ubicado en  10
    se agregó:  1.0 en:  9 109
    involucran No ubicado en 1
    se agregó:  0.0 en:  0 110
    involucran No ubicado en 2
    se agregó:  0.0 en:  1 110
    involucran No ubicado en 3
    se agregó:  0.0 en:  2 110
    involucran No ubicado en 4
    se agregó:  0.0 en:  3 110
    involucran No ubicado en 5
    se agregó:  0.0 en:  4 110
    involucran No ubicado en 6
    se agregó:  0.0 en:  5 110
    involucran No ubicado en 7
    se agregó:  0.0 en:  6 110
    involucran No ubicado en 8
    se agregó:  0.0 en:  7 110
    involucran No ubicado en 9
    se agregó:  0.0 en:  8 110
    involucran Ubicado en  10
    se agregó:  1.0 en:  9 110
    977 No ubicado en 1
    se agregó:  0.0 en:  0 111
    977 No ubicado en 2
    se agregó:  0.0 en:  1 111
    977 No ubicado en 3
    se agregó:  0.0 en:  2 111
    977 No ubicado en 4
    se agregó:  0.0 en:  3 111
    977 No ubicado en 5
    se agregó:  0.0 en:  4 111
    977 No ubicado en 6
    se agregó:  0.0 en:  5 111
    977 No ubicado en 7
    se agregó:  0.0 en:  6 111
    977 No ubicado en 8
    se agregó:  0.0 en:  7 111
    977 No ubicado en 9
    se agregó:  0.0 en:  8 111
    977 Ubicado en  10
    se agregó:  1.0 en:  9 111
    contribuyentes No ubicado en 1
    se agregó:  0.0 en:  0 112
    contribuyentes No ubicado en 2
    se agregó:  0.0 en:  1 112
    contribuyentes No ubicado en 3
    se agregó:  0.0 en:  2 112
    contribuyentes No ubicado en 4
    se agregó:  0.0 en:  3 112
    contribuyentes No ubicado en 5
    se agregó:  0.0 en:  4 112
    contribuyentes No ubicado en 6
    se agregó:  0.0 en:  5 112
    contribuyentes No ubicado en 7
    se agregó:  0.0 en:  6 112
    contribuyentes No ubicado en 8
    se agregó:  0.0 en:  7 112
    contribuyentes No ubicado en 9
    se agregó:  0.0 en:  8 112
    contribuyentes Ubicado en  10
    se agregó:  1.0 en:  9 112



```python
matrix
```




    array([[1., 1., 2., ..., 0., 0., 0.],
           [0., 0., 2., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 1., 1., 1.]])




```python
matrix.shape
```




    (10, 113)




```python
df_cos_t01_t02 = dot(matrix[0],matrix[1])/(norm(matrix[0])*norm(matrix[1]))
df_cos_t01_t02
```




    0.34299717028501764




```python


```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-111-f7e4c34ab58d> in <module>
    ----> 1 print(math.sqrt((matrix[0]-matrix[1])) )
          2 


    TypeError: only size-1 arrays can be converted to Python scalars



```python

```
