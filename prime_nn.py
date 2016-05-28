import numpy as np
import tensorflow as tf





# pas la methode la plus rapide pour savoir si un nombre est premier mais la plus courte en code
# d'autant plus qu'elle permet de sauvegarder les résultats

def eratosthene(sample_size = 50000) :
  # le crible d'eratostrucmescouilles commence avec des zeros partout
  erth = np.zeros(sample_size)
  # pour tous les entiers entre 0 et n/2
  for i in range(sample_size/2):
    # pour tous les multiples de i, on met un "1" dans le tableau
    # qui n'est donc pas un nombre premier
    if i>1 :
      erth[range(sample_size)[i*2::i]] = 1
  return erth

def is_prime(i,crible):
  return 1-crible[i]

## return un tableau de bit de l'encodage de i en binaire
def binary_encode(i, num_digits = 32):
  return np.array([ i >> d & 1 for d in range(num_digits)])


####################################################
## init
crible = eratosthene()

## maintenant, le réseau de neurone
### on construit l'entrainement



