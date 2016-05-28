#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

SAMPLE_SIZE	    = 50000
N_DIGITS	    = 32
N_ROUND		    = 200
BATCH_SIZE	    = 128



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

## Comment initialiser les reseaux de neurones ?
## l'eternel debat. Moi perso, c'est à la gueule
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


####################################################
## init
print("Making primes")
crible = eratosthene(SAMPLE_SIZE)

## maintenant, le réseau de neurone
### on construit l'entrainement

# On construit les données d'entrainement
# en binarisant tous les entiers
X = np.array(map(binary_encode,range(SAMPLE_SIZE)))

# Verifier 
# X.shape


# et on indique dans un autre tableau
# ce qu'est sensé répondre le réseau de neurone
# aka si un nombre est premier ou pas 

pritn("set up")
Y = np.array([ is_prime(i,crible) for i in range(SAMPLE_SIZE)])


# OK. Maintenant, construisons notre réseau de neurone avec tensorflow
# le truc de google

# Combien de neurones ?
# disons 100.
# on pourra toujours faire du Random Grid search 
NUM_HIDDEN = 100

# Ca c'est du formalisme de tensorflow
# pour construie le reseau.  ON s'en fout
# c'est l'input et l'output
# en entrée on a 32 float et en sortie 1 seul
# Pourquoi des float ?
# parce que le reseau va apprendre des probabilité
#  et non pas des certitudes
I = tf.placeholder("float", [None, N_DIGITS])
O = tf.placeholder("float", [None, 1])

# w_h represente les poids des connexions neuronales. C'est ca qui va etre appris
w_h = init_weights([N_DIGITS, NUM_HIDDEN])
# w_o est la proba d'etre un nombre premier
w_o = init_weights([NUM_HIDDEN, 1])

# et la partie interessante donc, ou l'on 
# fabrique la structure du reseau
# de neurone que l'on pense la plsu adaptée pour notre probleme
# ensuite, on le laissera grandir tout seul
def model(X, w_h, w_o):
  # la base : la sortie du réseau est 
  # l'entree une fois passé dans le reseau de neurone  
  # c'est une simple multiplication de matrices
  z=tf.matmul(X, w_h)
  # ensuite, comme c'est un reseau de neurone
  # on colle l'équivalent informatique d'une neurone qui s'ctive
  # ici une fonction qui s'appelle relu. Il y en a pleins d'autres possible mais 
  # celle la est cool
  h = tf.nn.relu(z)
  # et enfin, la probabilité d'etre en nb premier en sortie
  # est fonction de comment les neurones des couches cachées se sont activées.
  # encore une fois une simple matrice de poids
  return tf.matmul(h,w_o)
  
# voici notre bebe cerveau. C'est lui
# appelons le Ness :

ness = model(I, w_h,w_o)


# maintenant ness va apprendre mais 
# pour ca il lui faut un moyen de lui indiquer 
# s il s'est trompé et de combien
# on appelel ca une fonction de cout
# ici on prend une softmax cross entropy
# qui est standard poru ce genre de chose

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(ness, O))

# et enfin, on lui dit que pour corriger ses erreurs
# il doit faire une descente de gradient
# c'est a dire modifier son cerveau de maniere a 
# atteindre le minimum de la fonction de cout 
# c'est aussi relativement standard
# je ne connais pas d'autres méthodes en fait

train = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

# allez c'est partie.
# on a tout préparé, on lance l'entrainement

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for c_round in range(N_ROUND):
      for start in range(0, len(X), BATCH_SIZE):
	end = start + BATCH_SIZE
	sess.run(train, feed_dict={X: X[start:end],Y: Y[start:end]})
    print(c_round, np.mean(np.argmax(O, axis=1) == sess.run(predict_op, feed_dict={X: X, Y: Y})))
      


