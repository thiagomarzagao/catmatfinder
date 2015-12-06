# CATMATfinder
==============

Este aplicativo sugere classes CATMAT baseado na descrição do produto licitado (por exemplo, a descrição do produto conforme consta do edital de licitação).

Para usar o aplicativo é preciso ter instalados Python 2.7 e os pacotes Flask, NLTK e scikit-learn. É preciso ainda baixar os seguintes "pickles" e salvá-los na mesma pasta do aplicativo:

clf.pkl          # Contém o classificador, já treinado (8GB).

tfidf_maker.pkl  # Contém o vetorizador das descrições (64.4MB).

labels.pkl       # Contém os nomes das classes CATMAT (65KB).

É preciso ter pelo menos 16GB de RAM.
