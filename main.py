'''
app p/ identificacao de classes CATMAT a partir da descricao do produto licitado
'''

import re
import json
import pickle
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
from y import descri, labels
from intercept import intercepts
from idfs import idfs
from flask import Flask
from flask import request
from flask import render_template
from unicodedata import normalize
from nltk.stem import RSLPStemmer
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(filename = 'catmat.log', 
                    filemode = 'a', 
                    level = logging.DEBUG,
                    format = '%(asctime)s %(message)s')

# cria classe p/ poder plugar IDF no vetorizador
# eh preciso criar essa classe pq o atributo 'idf_'
# eh na verdade uma @property e nao tem um metodo
# 'setter', entao nao da pra fazer 
# 'vectorizer.idf_ = idfs', pois da AttributeError
class MyVectorizer(TfidfVectorizer):
    TfidfVectorizer.idf_ = idfs

# inicializa vetorizador
vectorizer = MyVectorizer(lowercase = False,
                          min_df = 2,
                          norm = 'l2',
                          smooth_idf = True)

# pluga matriz diagonal dos IDFs
vectorizer._tfidf._idf_diag = sp.spdiags(idfs,
                                         diags = 0,
                                         m = len(idfs),
                                         n = len(idfs))

# pluga vocabulario no vetorizador
vocabulary = json.load(open('vocabulary.json', mode = 'rb'))
vectorizer.vocabulary_ = vocabulary
vectorizer.fixed_vocabulary_ = False

# inicializa classificador
clf = linear_model.SGDClassifier(loss = 'modified_huber',
                                 penalty = 'l2',
                                 alpha = 0.0001,
                                 fit_intercept = True,
                                 n_iter = 60,
                                 shuffle = True,
                                 random_state = None,
                                 n_jobs = 4,
                                 learning_rate = 'optimal',
                                 eta0 = 0.0,
                                 power_t = 0.5,
                                 class_weight = None,
                                 warm_start = False)

# pluga coeficientes no classificador
store = pd.HDFStore('coefficients.h5')
coefs = np.array([np.array(store['row' + str(i)]).T[0] for i in range(560)])
store.close()
clf.coef_ = coefs

# pluga classes no classificador
clf.classes_ = labels

# pluga interceptos no classificador
clf.intercept_ = intercepts

print "ready"

app = Flask(__name__)

def pre_process(description):
    '''
    pre-processa a descricao
    '''

    # compila regex de caracteres nao-especiais
    vanilla = u'[^\u0041-\u005A \
                  \u0061-\u007A \
                  \u00C0-\u00D6 \
                  \u00D8-\u00F6 \
                  \u00F8-\u00FF \
                  \u0100-\u017F \
                  \u0020]'
    regex = re.compile(vanilla)

    # poe tudo em minusculas
    description = description.encode('utf8').decode('utf8')
    lowercased = description.lower()

    # remove caracteres especiais e numeros
    regexed = regex.sub(' ', lowercased)

    # separa palavras
    tokenized = regexed.split()

    # passa o que esta no plural p/ singular
    st = RSLPStemmer()
    singularized = [st.apply_rule(token, 0) for token in tokenized]

    # remove palavras c/ menos de 2 caracteres
    # e mescla palavras novamente
    remerged = ''
    for word in singularized:
        if len(word) > 1:
            remerged += word + ' '

    return remerged

def classify(description):
    '''
    identificar as tres classes CATMAT
    mais provaveis para a descricao
    '''

    # pre-processa
    pre_processed = pre_process(description)

    # vetoriza
    tfidf = vectorizer.transform([pre_processed])

    # estima probabilidades
    probs = clf.predict_proba(tfidf)

    # pega indices das tres classes de maior probabilidade
    top3indices = probs[0].argsort()[-3:][::-1]

    # pega codigos e descricoes das tres classes de maior probabilidade
    top3 = []
    for i in top3indices:
        top3.append(tuple((clf.classes_[i], descri[int(clf.classes_[i])], probs[0][i])))

    return top3

@app.route('/', methods = ['POST', 'GET'])
def hello():
    '''
    retorna pagina principal
    '''
    if request.method == 'POST':
        top3 = classify(request.form['descricao'])
        print top3
        return render_template('index.html',
                               output = True,
                               descricao = request.form['descricao'],
                               classe1 = top3[0][0][1:],
                               label1 = unicode(top3[0][1], 'utf8'),
                               prob1 = round(top3[0][2], 2),
                               classe2 = top3[1][0][1:],
                               label2 = unicode(top3[1][1], 'utf8'),
                               prob2 = round(top3[1][2], 2),
                               classe3 = top3[2][0][1:],
                               label3 = unicode(top3[2][1], 'utf8'),
                               prob3 = round(top3[2][2], 2)
                              )
    return render_template('index.html')

if __name__ == "__main__":
    app.debug = True
    app.run(host = "0.0.0.0")