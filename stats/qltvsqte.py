#!/usr/bin/env python
# -*- coding: utf-8 -*-


### La question est : une methode bourrine ( faire beaucoup d'essai qui ont peu de chance de succees)
### vaut elle mieux qu'une methode subtile ( peu d'essais mais bien qualibrés )
## voici du code pour repondre


import numpy as np
import pylab as plt
import math

f=np.math.factorial

## methode de qualité, subtile
pa = .5
na = 100

## Methode de quantité, bourrine
pb = .001
nb = 1E06


## Methode formel : loi de bernoulli 
## probabilité de k succes si n tirage de proba p
def binomial(n,p,k):
 if k>n:
  return 0
 Cb = f(n)/(f(k) * f(n-k))
 P  = Cb*p**k*(1-p)**(n-k)
 return P


K=np.arange(0,200)
pka=map(lambda k:binomial(na,pa,k),K)

### Et la c'est la merde parce que 
### La ligne suivante est impossible a calculer en un temsp raisonnable
### et comme j'ai pas envie de me casser le cul avec des maths 
### j'ai pas de solutions
# pkb=map(lambda k:binomial(nb,pb,k),K)


## On part sur une méthode de simulation numérique
### Commencer avec peu de round ou la partie graphique va exploser
nround = 1000
ka = np.random.binomial(na,pa,nround)

## ON verifie graphiquement que la simulation numérique est proche de la théorie
plt.hist(ka,bins=len(ka))
plt.plot([nround*x for x in pka], c='g')
### Normalement les courbes se superposent.
### on peut avoir confiance dans la methode donc 
plt.show()

## On augmente le nb de round pour la qualité de la simulation
nround=1E7
kb = np.random.binomial(nb,pb,nround)
ka = np.random.binomial(na,pa,nround)

plt.hist(ka,bins=100,label="nombre de succes avec la methode subtile")
plt.hist(kb,bins=100,label="nombre de succes avec la methode bourrine")
plt.legend()
plt.show()


result = 0+(ka>kb).sum()
print(" La methode subtile a fonctionné %d sur %d fois plus que la méthode bourrin")%(result,nround)
print kb.min()
print ka.max()


## note final
## dans notre cas, on aurait sans doute pu 
## approximer les lois binomiales
## par une gaussienne
## vu que y a assez d'essais
