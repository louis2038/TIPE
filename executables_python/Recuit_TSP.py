

import math,copy,sys,random,time 
import numpy as np
import matplotlib.pyplot as plt
import pygame
SIZEx = 500
SIZEy = 500
random.seed(15268)
pygame.init()
pygame.display.set_caption("TSP")

screen = pygame.display.set_mode((SIZEx,SIZEy))


class Graphe:
    def __init__(self,graphe,T = -1, alpha = -1, stop_t = -1):
        self.graphe =graphe
        self.start = math.sqrt(7) if T == -1 else T 
        self.alpha = 0.995 if alpha == -1 else alpha
        self.stop = 0.00000001 if stop_t == -1 else stop_t
        
    def arcExist(self,i,j):
        if self.graphe[i][j]!=None:
            return True
    def longueur(self,seq):
        return len(seq)
    def distance(self,seq):
        if self.longueur(seq)>1:
            cout=0
            for i in range(self.longueur(seq)-1):
                if self.arcExist(seq[i],seq[i+1]):
                    cout+=self.graphe[seq[i]][seq[i+1]]
            
            return cout
        else:
            return None
    def proba(self,dist_parcours,dist_p):
        return math.exp(-abs(dist_p-dist_parcours)/self.start)
    def permutation(self,ordre):
        global M
        d  = longueur(x,y,ordre)
        d0 = d+1
        it = 1
        while d < d0 :
            it += 1
            d0 = d
            for i in range(0,len(ordre)-1) :
                for j in range(i+2,len(ordre)):
                    r = ordre[i:j].copy()
                    r.reverse()
                    ordre2 = ordre[:i] + r + ordre[j:]
                    t = longueur(x,y,ordre2)
                    if t < d :
                        d = t
                        ordre = ordre2
        return ordre
    def recuit(self,parcours):
        bestSolution=copy.copy(parcours)
        bestCout=self.distance(parcours)
        bestSolution.append(bestSolution[0])
        print("parcours initial")
        
        print(bestSolution)
        print(bestCout)
        affiche(bestSolution)
        ct = 0
        it = 0
        while self.start>self.stop:
            it += 1
            ct += 1
            if ct >100000:
                print("nbs d'ite", it)
                print(self.start)
                affiche(p)
                ct = 0
            p=copy.copy(parcours)
            j=random.randint(0,len(p)-1)
            i=random.randint(0,j)
            p[i:j]=reversed(p[i:j])
            
            if self.distance(p)<bestCout:
                bestSolution=copy.copy(p)
                bestCout=self.distance(p)
                parcours=copy.copy(p)
                bestSolution.append(bestSolution[0])
                
            else:
                if random.random()<self.proba(self.distance(parcours),self.distance(p)):
                    parcours=copy.copy(p)
            
            self.start *= self.alpha
        print(bestSolution)
        print(bestCout)
        print("nbs d'ite", it)
        print('Solution finale')
        affiche(bestSolution)
  

def affiche(Ordre):
    global M
    x = []
    y = []
    print(Ordre)
    screen.fill((0,0,0))
    pygame.draw.circle(screen,(250,250,250),(M[Ordre[0]][0],M[Ordre[0]][1]),5)
    police = pygame.font.SysFont("monospace",20)
    image_text = police.render( str(Ordre[0]),1,(250,250,250),(0,0,0))
    screen.blit(image_text,(M[Ordre[0]][0],M[Ordre[0]][1]))
    for i in range(1,n-1):
        pygame.draw.circle(screen,(250,250,250),(M[Ordre[i]][0],M[Ordre[i]][1]),5)
        image_text = police.render( str(Ordre[i]),1,(250,250,250),(0,0,0))
        screen.blit(image_text,(M[Ordre[i]][0],M[Ordre[i]][1]))
        pygame.draw.line(screen,(250,250,250),(M[Ordre[i-1]][0],M[Ordre[i-1]][1]),(M[Ordre[i]][0],M[Ordre[i]][1]))
    pygame.display.update()
    #pygame.time.wait(5)

def genere_graph(n):
    M = [(random.randint(0,SIZEx),random.randint(0,SIZEy)) for i in range(n)]
    
    H = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            H[i][j] = int(math.sqrt(abs(M[i][0] - M[j][0])**2 + abs(M[i][1] - M[j][1])**2 ))
            H[j][i] = H[i][j]

    return M,H

global M,H
    
n = 40
M,H = genere_graph(n)


# constante convergente : math.sqrt(6),0.99995,1
# T = -1, alpha = -1, stop_t = -1
g=Graphe(H,math.sqrt(6),0.99995,1)
aa = [2, 1, 3, 0, 4, 5, 7, 8, 6,2]
#affiche(aa)
#print(g.distance(aa))
g.recuit([i for i in range(1,n)])
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
