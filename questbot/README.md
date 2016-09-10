# Génération de textes et chatbot

Dans ce dossier, je regroupe les différentes explorations autoir de la modélisation, la compréhension et la génération de texte.

Le but ultime est de disposer d'une méthode de génération automatique d'arbre de dialogues à choix multiples, outil fréquemment utiliser dans le jeu, vidéo en particulier.

## Dataset

Les 2 plus intéressant dataset que j'ai trouvé en langue francaises sont :
 - les newsfrancaises compilées sur 2 ans : 
   - wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.fr.shuffled.gz
   - wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.fr.shuffled.gz
 - le dump de wikipedia  : wget http://dumps.wikimedia.org/frwiki/latest/frwiki-latest-pages-articles.xml.bz2
Ainsi qu'une base privée et confidentiel de textes fournis par un auteur de SF français.

## Liste des Techniques

### Chaines de markov

Les bots a base de chaines de markov apprennent juste la probabilité d'un caractères à partir des n précédents. C'est facile a implémenter et relativement puissant par contre ca ne capture aucune composante sémantique. 

Voir markov.py pour une implémentation type.

### Word embedding

### Latent Dirichlet Association

### RNN et LSTM

### Sequence to sequence


 


