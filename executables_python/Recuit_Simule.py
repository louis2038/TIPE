import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time
import pygame
import copy
SIZEx = 1800
SIZEy = 800
RAYON = 10
COLOR = [(255,0,255),(0,255,255),(0,255,0),(255,255,0),(155,100,0),(255,0,0)]
#random.seed(88)

# hypothèse : graph de la ville est connexe non orienté et pondéré
def inject_data_distance(file,size):
	# file : chemin d'accès au fichier correspondant
	# size : nombre de sommet de la ville.
	f = open(file,"r")
	M = [[] for _ in range(size)]
	
	for i in range(size):
		ligne = f.readline()
		tab_ligne = ligne.split(", ")
		tab_ligne[0] = tab_ligne[0][1:]
		tab_ligne[-1] = tab_ligne[-1].split("]")[0]
		for j in range(size):
			if tab_ligne[j] == "inf":
				M[i].append(float('inf'))
			else:
				M[i].append(float(tab_ligne[j]))
	return M

def inject_data_heuristie(file,size):
	# file : chemin d'accès au fichier correspondant
	# size : nombre de sommet de la ville.
	f = open(file,"r")
	M = [[] for _ in range(size)]
	
	for i in range(size):
		ligne = f.readline()
		tab_ligne = ligne.split(", ")
		tab_ligne[0] = tab_ligne[0][1:] # on enleve le 1 er char
		tab_ligne[-1] = tab_ligne[-1].split("]")[0] # et les derniers char
		for j in range(size):
			M[i].append(float(tab_ligne[j]))
	return M

def inject_data_liste(file,size):
	# file : chemin d'accès au fichier correspondant
	# size : nombre de sommet de la ville.
	f = open(file,"r")
	M = []
	
	ligne = f.readline()
	tab_ligne = ligne.split("), (")
	tab_ligne[0] = tab_ligne[0][2:] # on enleve les 2 premier char
	tab_ligne[-1] = tab_ligne[-1].split(")]")[0] # et les derniers char
	for j in range(size):
		ctt = tab_ligne[j].split(", ")
		M.append( (int(ctt[0]) , int(ctt[1])) )
	return M

def inject_data_color(file,size):
	# file : chemin d'accès au fichier correspondant
	# size : nombre de sommet de la ville.
	f = open(file,"r")
	M = [[] for _ in range(size)]
	
	for i in range(size):
		ligne = f.readline()
		tab_ligne = ligne.split(", ")
		tab_ligne[0] = tab_ligne[0][1:] # on enleve le 1 er char
		tab_ligne[-1] = tab_ligne[-1].split("]")[0] # et les derniers char
		for j in range(size):
			M[i].append(int(tab_ligne[j]))
	return M

def matrice_to_liste(M,m): 
	# M : matrice d'adjacence
	N = [[] for _ in range(m)]
	for i in range(m):
		for j in range(m):
			if M[i][j] != float('inf') and i!=j:
				N[i].append((j,M[i][j])) 
	return N
	# liste d'adjacence:
	#	N = [  [ (sommet suivant le sommet i note j, poids de l'arc i j) , ...]  , ...]
	#		N[i] les sommets adjacents a i

def cmp1(e):
	return e[1]

def cmp2(e):
	return e[3]



class DARP:

	def __init__(self,M,V,C,H,m,s,S,n,D,A,w,t0,tf,alpha):
		# M : matrice d'adjacence de la ville
		# V : [ (x,y) , ...] position des sommets
		# C : matrice d'adjacence contenant la couleur des arretes
		# L : liste d'adjacence de la ville
		# H : matrice heuristique de la ville ( distance a vol d'oiseau entre chaque sommet)
		# m : taille de M
		# s : nombre de bus
		# S : liste des sommets des bus
		# n : nombre de personne
		# D : liste des sommets de départ des passagers indicé par leur numéros
		# A : liste des sommets d'arrivé des passagers indicié par leur numéros
		# w : poids heuristique dans A*
		# t0 : temperature inital
		# tf : temperature final
		# alpha : vitesse de refroidissement
		# P : [   [nombre de passager, [liste des numero des passager] , [parcours du bus decomposer en liste] , [ordre de recuperation des passagers], poids du parcours, [liste des poids du traject]] ... ] liste des bus
		# G : dico G[(sommet i,sommet j)] = (poids du trajet i -> j, liste contenant le parcours de ce trajet) \ on a i<=j !
		
		self.contraste_bus = 0.3
		self.color_bus = None
		self.aff_sommet = False
		self.M = M
		self.V = V 
		self.C = C 
		self.L = matrice_to_liste(M,m)
		self.H = H
		self.m = m
		self.s = s
		self.S = S
		self.n = n
		self.D = D
		self.A = A
		self.w = w
		self.t0 = t0
		self.tf = tf
		self.alpha = alpha
		self.P = [[0,[],[],[],0,[]] for i in range(s)]
		self.G = {}



	def A_star(self,d,a):
		# d : point de départ
		# a : point d'arrivé
		if (d,a) in self.G:
			return self.G[(d,a)]
		else:
			not_visited = [i for i in range(self.m) if i!=d]
			D = [float('inf') if i != d else 0 for i in range(self.m)]
			DH = [float('inf') if i != d else (self.w*self.H[a][d]) for i in range(self.m)] # temps (du depart a i) + distance (de i a arrive) = dh
			P = [ 0 if i != d else 0 for i in range(self.m) ] # on note P[i] = j pour aller en i, il faut passer par j
			pos = d
			poids = 0 #ceci est le poids en temps !!

			while ( not_visited != [] ) and (pos != a): # on arrete quand on est arrive
				
				for adj in self.L[pos]:
					if adj[0] in not_visited: # adj[0] -> sommet adjacent , adj[1] -> poids de l'arc
						# H[a][adj[0]] distance a vol d'oiseau du sommet adjacent pos
						if ( poids + adj[1] ) + (self.w*self.H[a][adj[0]]) < DH[adj[0]]: 	# different de dijtra 
							D[adj[0]] = poids + adj[1]
							DH[adj[0]] = ( poids + adj[1] ) + (self.w*self.H[a][adj[0]])		# |
							P[adj[0]] = pos 								# |
				
				
				# on selectionne le min de DH
				minn_DH = DH[not_visited[0]]
				minn_D = D[not_visited[0]]
				ind_min = not_visited[0]
				for ind in not_visited[1:]:
					if DH[ind] < minn_DH:
						minn_DH = DH[ind]
						minn_D = D[ind]
						ind_min = ind

				poids = D[ind_min] # le poids en temps
				pos = ind_min
				not_visited.remove(pos) # on le marque comme visite


			G = [a]	# parcours du bus final ( trajectoire du bus)
			cursor = a
			while cursor != d: # temps que je ne suis pas a l'arrive
				cursor = P[cursor]
				G.append(cursor)
			G.reverse()

			self.G[(d,a)] = (poids,G.copy())
			tt = G.copy()
			tt.reverse()
			self.G[(a,d)] = (poids,tt)
			return (poids,G)
	
	def generer_graph_a_star_avec_chemin(self,n,D,A,b):
		# n : local ! nombre de passager du bus !
		# D : local
		# A : local
		# b : sommet du bus
		F = np.zeros((2*n+1,2*n+1)) # cout du trajet
		C = [[[] for _ in range(2*n+1)] for _ in range(2*n+1)] # chemin du trajet
		

		# les arcs : bus => passager / passager => bus
		for i in range(n):
			if b == D[i]:
				F[0,2*i +1] = 0
				F[2*i +1,0] = 0
				C[0][2*i +1] = []
				C[2*i +1][0] = []
			else:
				poids,Trajet = DARP.A_star(self,b,D[i])
				F[0,2*i +1] = poids 
				F[2*i +1,0] = poids
				C[0][2*i +1] = Trajet.copy()
				Trajet.reverse()
				C[2*i +1][0] = Trajet.copy() # le chemin a l'invers


		# les arcs : bus => destination des passagers / destination des passagers => bus
		for i in range(n):
			if b == A[i]:
				F[0,2*i +2] = 0 
				F[2*i +2,0] = 0
				C[0][2*i+2] = []
				C[2*i+2][0] = []
			else:
				
				poids,Trajet = DARP.A_star(self,b,A[i])
				
				F[0,2*i +2] = poids
				F[2*i +2,0] = poids 
				C[0][2*i +2] = Trajet.copy()
				Trajet.reverse()
				C[2*i +2][0] = Trajet.copy() # le chemin a l'invers

		# les arcs : passager i => passager j / passager j => passager i
		for i in range(n):
			for j in range(i+1,n): # on evite ainsi de recalculer la meme chose
				if D[i] == D[j]:
					F[2*i +1,2*j +1] = 0 
					F[2*j +1,2*i +1] = 0
					C[2*i +1][2*j +1] = []
					C[2*j +1][2*i +1] = []
				else:
					
					poids,Trajet = DARP.A_star(self,D[i],D[j])
					
					F[2*i +1,2*j +1] = poids
					F[2*j +1,2*i +1] = poids
					C[2*i +1][2*j +1] = Trajet.copy()
					Trajet.reverse()
					C[2*j +1][2*i +1] = Trajet.copy()

		# les arcs : destination i => destination j / destination j => destination i
		for i in range(n):
			for j in range(i+1,n): # on evite ainsi de recalculer la meme chose
				if A[i] == A[j]:
					F[2*i +2,2*j +2] = 0 
					F[2*j +2,2*i +2] = 0
					C[2*i +2][2*j +2] = []
					C[2*j +2][2*i +2] = []
				else:
					
					poids,Trajet = DARP.A_star(self,A[i],A[j])
					
					F[2*i +2,2*j +2] = poids
					F[2*j +2,2*i +2] = poids
					C[2*i +2][2*j +2] = Trajet.copy()
					Trajet.reverse()
					C[2*j +2][2*i +2] = Trajet.copy()

		# les arcs : passager i => destination j / destination j => passager i
		for i in range(n):
			for j in range(n): # on evite ainsi de recalculer la meme chose
				if D[i] == A[j]:
					F[2*i +1,2*j +2] = 0 
					F[2*j +2,2*i +1] = 0
					C[2*i +1][2*j +2] = []
					C[2*j +2][2*i +1] = []
				else:
					
					poids,Trajet = DARP.A_star(self,D[i],A[j])
					
					F[2*i +1,2*j +2] = poids
					F[2*j +2,2*i +1] = poids
					C[2*i +1][2*j +2] = Trajet.copy()
					Trajet.reverse()
					C[2*j +2][2*i +1] = Trajet.copy()

		return (F,C)

	def bus1_chemin_optimise1(M,d,n):
		# M (numpy) : matrice tsp
		# d : sommet de depart du bus
		# n : nombre de passager
		a = []

		L = []#// liste des chemins a k-1
		L1 = [] # liste temporaire
		L2 = []

		# structure des liste L1 et L2
		# chaque élement sont des listes : [ 'sommet d'arrivée' , 'poids du chemin' , 'etat du chemin' , 'chemin' (liste de sommet) ]


		switchL = True
		Gen = [ (m+1)%2 for m in range(2*n)]

		for k in range(1,2*n+1):
			# parcours de longueur k
			# A chaque étape on switch de liste entre L1 et L2

			if k == 1: # initialisation
				for f in range(n):
					# on modifie l'etat du chemin
					a = Gen.copy()
					a[2*f] = 0
					a[2*f +1] = 1
					L1.append([2*f+1, M[d,2*f+1] , a ,[0, 2*(f+1) -1] ].copy())
					#fin de l'etape, on change de liste
					switchL = True
			else:
				L=[]
				# gestion des listes temporaires
				if switchL:
					L = L1.copy()
					L2.clear()
					switchL = False
				else:
					L = L2.copy()
					L1.clear()
					switchL = True

				for p in range(2*n):
					# on regarde tous les sommets d'arrivées possibles.

					# On doit trouver le chemin minimum de taille k et qui arrive au sommet p

					J = []
					for j in L:# on regarde les élements de notre liste temporaire

						if j[2][p] == 1: # on regarde si les sommets d'arrivés sont potentiellements accessibles

							if p % 2 == 0: # ce sommet est un client
								# on change l'etat de notre chemin
								a = j[2].copy()
								a[p] = 0
								a[p+1] = 1
								J.append([ p+1, j[1] + M[j[0],p+1] ,a, j[3] + [p+1] ].copy())

							else: # ce sommet est une destination
								# on change l'etat de notre chemin
								a = j[2].copy()
								a[p] = 0
								J.append([ p+1, j[1] + M[j[0],p+1] ,a, j[3] + [p+1] ].copy())

					if J!=[]:
						jf = min(J,key=cmp1)
						# je selectionne la liste qui a un poids minimum
						if switchL :
							L1.append(jf)
						else:
							L2.append(jf)



			"""
			print("===========================")
			print("L1 : ")
			for i in L1:
				print(i)
			print("===========================")
			print("L2 : ")
			for i in L2:
				print(i)
			print("===========================")
			"""

		if switchL :
			m = min(L1,key=cmp1)
		else:
			m = min(L2,key=cmp1)
		return (m[1],m[3]) # ( poids , chemin )
		# a la fin, on prend le chemin qui a un poids minimum.
	
	def multi_bus_attribution_naive(self):
		global Per
		Per = [] # [ [ [num des passager] , ...(pour les s bus)  ] ... (combinaison) ]
		
		def generer_permu(n,s,i,permut):
			
			if i == n:
				Per.append(permut)
				return
			for b in range(s):
				tmp = copy.deepcopy(permut)
				tmp[b].append(i)
				generer_permu(n,s,i+1,tmp)
		generer_permu(self.n,self.s,0,[[] for _ in range(self.s)])
		
		start = True
		debit_max = 0
		
		
				
		for perm in Per: # on test tout les permutations
			som_deb = 0
			tmp_P = [[0,[],[],[],0,[]] for i in range(s)]
			for k in range(s): # pour les k bus
				pt = [] # parcours temporaire
				pt_m = []
				
				d = []
				a = []
				nbs_pa = 0
				ordre = []
				for pa in perm[k]: # on itere les passager dans le bus k
					d.append(D[pa]) # on recupere les sommets de départ / arrive de c'est passager 
					a.append(A[pa])
					ordre.append(pa)
					nbs_pa += 1
			
				if nbs_pa == 0:
					tmp_P[k][0] = nbs_pa
					tmp_P[k][1] = []
					tmp_P[k][2] = [self.S[k]]
					tmp_P[k][3] = []
					tmp_P[k][4] = 0
					tmp_P[k][5] = []
				else:
					# OPTI
					R,T = DARP.generer_graph_a_star_avec_chemin(self,nbs_pa,d,a,self.S[k]) # on genere les graphs associés							
					(poids , pt) = DARP.bus1_chemin_optimise1(R,0,nbs_pa)
					parcours = []
					
					ok = [False for _ in range(self.n)]
					
					for g in range(1,len(pt)):
						xx = ordre[(pt[g]-1)//2]
						if ok[xx]:
							parcours.append((xx*2)+1)
						else:
							parcours.append(xx*2)
							ok[xx] = True
					
					som_deb += nbs_pa / poids				
					pt_s_d =[pt[0]]
					taille_pt = len(pt)
					prec = pt[0]
					for o in range(1,taille_pt):
						if prec != pt[o]:
							pt_s_d.append(pt[o])
						prec = pt[o]
					
					pt_m.clear()
					pt_m = [T[pt_s_d[0]][pt_s_d[1]]]
					for o in range(1,taille_pt -1):
						if pt_m == []:
							pt_m.append(T[pt_s_d[o]][pt_s_d[o+1]])
						else:
							pt_m.append(T[pt_s_d[o]][pt_s_d[o+1]][1:])
					Poidsl = []
					for gg in range(len(pt)-1):
						Poidsl.append(R[pt[gg]][pt[gg+1]])
					
					tmp_P[k][0] = nbs_pa
					tmp_P[k][1] = perm[k].copy()
					tmp_P[k][2] = pt_m.copy()
					tmp_P[k][3] = parcours.copy() 
					tmp_P[k][4] = poids
					tmp_P[k][5] = Poidsl.copy()
			if start or (som_deb > debit_max):
				start = False
				debit_max = som_deb
				self.P = copy.deepcopy(tmp_P)
			
	def multi_bus_aleatoire(self):
		global Per
		Per = [[[] for _ in range(self.s)]] # [ [ [num des passager] , ...(pour les s bus)  ] ... (combinaison) ]
		
		cmp_b_vide = self.s
		b_vide = {}
		for i in range(self.s):
			b_vide[i] = True
		for p in range(self.n):
			while True:
				bdp = random.randint(0,self.s -1)
				if cmp_b_vide <= 0 or b_vide[bdp]:
					cmp_b_vide -= 1
					Per[0][bdp].append(p)
					break
		print(Per)
		start = True
		debit_max = 0
		
		
				
		for perm in Per: # on test tout les permutations
			som_deb = 0
			tmp_P = [[0,[],[],[],0,[]] for i in range(s)]
			for k in range(s): # pour les k bus
				pt = [] # parcours temporaire
				pt_m = []
				
				d = []
				a = []
				nbs_pa = 0
				ordre = []
				for pa in perm[k]: # on itere les passager dans le bus k
					d.append(D[pa]) # on recupere les sommets de départ / arrive de c'est passager 
					a.append(A[pa])
					ordre.append(pa)
					nbs_pa += 1
			
				if nbs_pa == 0:
					tmp_P[k][0] = nbs_pa
					tmp_P[k][1] = []
					tmp_P[k][2] = [self.S[k]]
					tmp_P[k][3] = []
					tmp_P[k][4] = 0
					tmp_P[k][5] = []
				else:
					# OPTI
					R,T = DARP.generer_graph_a_star_avec_chemin(self,nbs_pa,d,a,self.S[k]) # on genere les graphs associés							
					(poids , pt) = DARP.bus1_chemin_optimise1(R,0,nbs_pa)
					parcours = []
					
					ok = [False for _ in range(self.n)]
					
					for g in range(1,len(pt)):
						xx = ordre[(pt[g]-1)//2]
						if ok[xx]:
							parcours.append((xx*2)+1)
						else:
							parcours.append(xx*2)
							ok[xx] = True
					
					som_deb += nbs_pa / poids				
					pt_s_d =[pt[0]]
					taille_pt = len(pt)
					prec = pt[0]
					for o in range(1,taille_pt):
						if prec != pt[o]:
							pt_s_d.append(pt[o])
						prec = pt[o]
					
					pt_m.clear()
					pt_m = [T[pt_s_d[0]][pt_s_d[1]]]
					for o in range(1,taille_pt -1):
						if pt_m == []:
							pt_m.append(T[pt_s_d[o]][pt_s_d[o+1]])
						else:
							pt_m.append(T[pt_s_d[o]][pt_s_d[o+1]][1:])
					Poidsl = []
					for gg in range(len(pt)-1):
						Poidsl.append(R[pt[gg]][pt[gg+1]])
					
					tmp_P[k][0] = nbs_pa
					tmp_P[k][1] = perm[k].copy()
					tmp_P[k][2] = pt_m.copy()
					tmp_P[k][3] = parcours.copy() 
					tmp_P[k][4] = poids
					tmp_P[k][5] = Poidsl.copy()
			if start or (som_deb > debit_max):
				start = False
				debit_max = som_deb
				self.P = copy.deepcopy(tmp_P)

	def affiche(self):
		global P2
		P2 = [[] for _ in range(self.s)]
		for b in range(self.s):
			for el in self.P[b][2]:
				for ell in el:
					P2[b].append(ell)
		#def assombrissement(color,x):
		#	moy = ( color[0] + color[1] + color[2] ) / 3
		#	return (color[0]+int((moy - color[0])*x),color[1]+int((moy - color[1])*x),color[2]+int((moy - color[2])*x))

		def rendu(screen,bg,Color_bus,Color_sommet,prio,sel_arc):
			police = pygame.font.SysFont("monospace",20)
			screen.blit(bg,(0,0))

			image_text = police.render( str(prio),1,Color_bus[prio],(0,0,0))
			screen.blit(image_text,(10,10))
				
		

			for i in range(1,self.m):
				for j in range(i):
					if self.C[i][j] != 0:
						pygame.draw.line(screen,(100,100,100),self.V[i],self.V[j],width=self.C[i][j])

			Color_bus_bis = Color_bus.copy()
			for ll in range(self.s):
				if ll != prio:
					Color_bus_bis[ll] = eclairsisement(Color_bus[ll],self.contraste_bus)
			

			for b in range(self.s): # on itere les bus
				if b != prio:
					for i in range(len(P2[b])-1):
						sd = P2[b][i] # sommet de depart
						sa = P2[b][i+1] # sommet d'arrive
						pygame.draw.line(screen,Color_bus_bis[b],self.V[sd],self.V[sa],width=self.C[sd][sa])	

			for i in range(len(P2[prio])-1):
				if i != sel_arc:
					sd = P2[prio][i] # sommet de depart
					sa = P2[prio][i+1] # sommet d'arrive
					pygame.draw.line(screen,Color_bus_bis[prio],self.V[sd],self.V[sa],width=self.C[sd][sa])			
			if sel_arc != -1:
				sd = P2[prio][sel_arc] # sommet de depart
				sa = P2[prio][sel_arc+1] # sommet d'arrive
				pygame.draw.line(screen,(0,0,0),self.V[sd],self.V[sa],width=self.C[sd][sa])
			
			cpt_sommet = 0
			for som in self.V:
				if self.aff_sommet:
				#pygame.draw.circle(screen,(100,100,100),som,RAYON,width=1) 
					
					image_text = police.render( str(cpt_sommet),1,(255,255,255))
					screen.blit(image_text,(som[0]-6,som[1]-10))
				image_text = police.render( Aff_sommet[cpt_sommet],0.5,Color_sommet[cpt_sommet],(0,0,0))
				screen.blit(image_text,(som[0]-4,som[1]-7))
				cpt_sommet += 1

			pygame.display.update()

		def eclairsisement(color,x): # assombrit le vert
			c = [0,0,0]
			if color[0] > color[2]:
				maxx = 0
			else:
				maxx = 2
			c[maxx] = (230 - color[maxx])
			for j in range(3):
				if j!=maxx:
					c[j] = -(color[j])


			return (color[0]+int(x*c[0]),color[1]+int(x*c[1]),color[2]+int(x*c[2]))

		global SIZE,RAYON,COLOR
		pygame.init()
		pygame.display.set_caption("TIPE")

		screen = pygame.display.set_mode((SIZEx,SIZEy))
		bg = pygame.image.load("map_gre2.png")
		
		running = True
		if self.color_bus == None:
			Color_bus = [(random.randint(0,230),random.randint(0,230),random.randint(0,230)) for _ in range(s)]
		else:
			Color_bus = self.color_bus
		Color_sommet = [(0,0,0) for _ in range(m)] # couleur de chaque sommet

		Aff_sommet = ["" for _ in range(m)]
		for i in range(self.s):
			Aff_sommet[self.S[i]] += ("b" + str(i) + " ")
			Color_sommet[self.S[i]] = (200,200,200)  # <--- couleur des bus !

		for i in range(self.n):
			Aff_sommet[self.D[i]] += ("d" + str(i) + " ")
			Aff_sommet[self.A[i]] += ("a" + str(i) + " ")
			for kk in range(self.s):
				if self.D[i] in P2[kk]: # ce sommet est deservie par le bus kk
				
					Color_sommet[self.D[i]] = Color_bus[kk]
					Color_sommet[self.A[i]] = Color_bus[kk]

		rendu(screen,bg,Color_bus,Color_sommet,0,-1)
		select = 0
		sel_arc = 0
		before_color_bus = Color_bus[0]
		taille_coresp = []
		for hh in range(self.s):
			taille_coresp.append(len(P2[hh]))

		while running:
			for event in pygame.event.get():

				if event.type == pygame.QUIT:
					running = False
				if event.type == pygame.KEYDOWN:
					
					if event.key == 1073741903: # fleche de droite
						sel_arc = 0
						if select < (self.s-1):
							for i in range(self.n):
								if i in self.P[select][1]:  
									Color_sommet[self.D[i]] = before_color_bus
								if i in self.P[select][1]: 
									Color_sommet[self.A[i]] = before_color_bus
							Color_bus[select] = before_color_bus # on remet la bonne couleur
							select += 1
							before_color_bus = Color_bus[select] # on change la nouvelle
							Color_bus[select] = (0,255,0)
							for i in range(self.n):
								if i in self.P[select][1]:
									Color_sommet[self.D[i]] = (0,255,0)
								if i in self.P[select][1]:
									Color_sommet[self.A[i]] = (0,255,0)
							rendu(screen,bg,Color_bus,Color_sommet,select,-1)
					if event.key == 1073741904: # fleche de gauche
						sel_arc = 0
						if select > 0:
							for i in range(self.n):
								if i in self.P[select][1]: 
									Color_sommet[self.D[i]] = before_color_bus
								if i in self.P[select][1]: 
									Color_sommet[self.A[i]] = before_color_bus
							Color_bus[select] = before_color_bus # on remet la bonne couleur
							select -= 1
							before_color_bus = Color_bus[select] # on change la nouvelle
							Color_bus[select] = (0,255,0)
							for i in range(self.n):
								if i in self.P[select][1]:
									Color_sommet[self.D[i]] = (0,255,0)
								if i in self.P[select][1]:
									Color_sommet[self.A[i]] = (0,255,0)
							rendu(screen,bg,Color_bus,Color_sommet,select,-1)
					if event.key == 1073741906: # fleche du haut
						if sel_arc < (taille_coresp[select]-2):
							rendu(screen,bg,Color_bus,Color_sommet,select,-1)
							sel_arc += 1
							rendu(screen,bg,Color_bus,Color_sommet,select,sel_arc)
					if event.key == 1073741905: # fleche du bas
						if sel_arc > 0:
							rendu(screen,bg,Color_bus,Color_sommet,select,-1)
							sel_arc -= 1
							rendu(screen,bg,Color_bus,Color_sommet,select,sel_arc)

	def multi_bus_attribution_recui_simule(self):
		global Ordre,Bu,First,Ppb,Tdb

		def affiche_ordre(Ordre,First):
			for b in range(self.s):
				print()
				print("bus :"+str(b))
				state = First[b]
				while state != None:
					L = Ordre[b][state]
					print(state,L)
					state = L[1][0]
				print()

		def proba(delta_deb,temperature):
			return math.exp(-abs(delta_deb)/temperature)

		def condition(bpp,bd,tps_enl,tps_raj,temperature):
			# bpp : bus de la personne a prendre ( qu'on retire du bus )
			# bd : nouveau bus de la personne a prendre
			# tps_enl : temps du bus ou l'on va enlever un passager
			# tps_raj : temps du bus ou l'on va rajouter un passager
			
			# delta_deb = nv_deb - ancien_deb
			delta_deb = ((Ppb[bpp]-1)/tps_enl) + ((Ppb[bd]+1)/tps_raj) - (Ppb[bd]/Tdb[bd]) - (Ppb[bpp]/Tdb[bpp])

			if delta_deb > 0: # la nv solution est meilleur !
				return True
			else:
				if random.random() < proba(delta_deb,temperature):
					return True
				else:
					return False

		def condition2(bpp,bd,tps_nv_b_enl,tps_nv_b_raj):
			return True

		def trfm_ordre_to_p():
			P2 = [[0,[],[],[],0,[]] for i in range(s)]
			for p in range(self.n): # je rajoute la liste des personnes pris par les bus
				P2[Bu[p]][1].append(p)
			
			for b in range(self.s): 
				P2[b][4] = Tdb[b]

			for b in range(self.s):
				P2[b][0] = Ppb[b]
				state = First[b]
				while state != None:
					L = Ordre[b][state]
					P2[b][2].append(L[0][1])
					P2[b][3].append(state)
					P2[b][5].append(L[0][2])
					state = L[1][0]
			return P2

		

		def trfm_etat_to_sommet(etat):
			if etat % 2 == 0:# depart
				return self.D[etat//2]
			else: # arrive
				return self.A[etat//2]

		def voisinage(temperature):
			#on selectionne la personne a prendre
			Bnv = set() # bus avec un nbs de passager sup a 2 
			for b in range(self.s):
				if Ppb[b] >= 2:
					Bnv.add(b)
			P_enl = []
			taille_P_enl = 0
			for p in range(self.n):
				if Bu[p] in Bnv:
					P_enl.append(p)
					taille_P_enl += 1

			pp = P_enl[random.randint(0,taille_P_enl-1)]
			#pp = 2
			bpp = Bu[pp] #  bus de la personne a prendre ( qu'on retire du bus )
			# pp * 2 point de depart de la pp
			bd = random.randint(0,self.s -2 ) # choix du nouveau bus de la personne a prendre
			if bd == bpp: # histoire que ça serve a quel que chose
				bd += 1
			# ====
			tps_nv_b_enl = 0
			tps_nv_b_raj = 0

			# ===========================================
			# on enleve la personne dans le bus selectionné
			L = Ordre[bpp][pp*2]
			avant_d_enl,ch_d_avant_enl,poids_d_av_enl = L[0]
			apres_d_enl,ch_d_apres_enl,poids_d_ap_enl = L[1]
			
			L = Ordre[bpp][pp*2+1]
			avant_a_enl,ch_a_avant_enl,poids_a_av_enl = L[0]
			apres_a_enl,ch_a_apres_enl,poids_a_ap_enl = L[1]
			
			if apres_d_enl == 2*pp +1 or avant_a_enl == 2*pp:
				cote_a_cote_enl = True
			else:
				cote_a_cote_enl = False
			#print("cote a cote enl",cote_a_cote_enl)

			if cote_a_cote_enl:
				if avant_d_enl == None and apres_a_enl == None:
					tps_nv_b_enl = 0
					# il y a rien
				elif avant_d_enl == None: # avant il y a le bus
					# rien
					poids_a_enl,Trajet_a_enl = DARP.A_star(self,self.S[bpp],trfm_etat_to_sommet(apres_a_enl))
					tps_nv_b_enl = Tdb[bpp] - (Ordre[bpp][pp*2][0][2] + Ordre[bpp][pp*2][1][2] + Ordre[bpp][pp*2 +1][1][2]) + poids_a_enl
				elif apres_a_enl == None: #apres il y a rien
					poids_d_enl,Trajet_d_enl = (0,[])
					# rien
					tps_nv_b_enl = Tdb[bpp] - (Ordre[bpp][pp*2][1][2] + Ordre[bpp][pp*2][0][2])
				else:
					poids_d_enl,Trajet_d_enl = DARP.A_star(self,trfm_etat_to_sommet(avant_d_enl),trfm_etat_to_sommet(apres_a_enl))
					tps_nv_b_enl = Tdb[bpp] - (Ordre[bpp][pp*2][0][2] + Ordre[bpp][pp*2][1][2] + Ordre[bpp][pp*2 +1][1][2]) + poids_d_enl
			else:
				if avant_d_enl == None and apres_a_enl == None:
					poids_d_enl,Trajet_d_enl = DARP.A_star(self,self.S[bpp],trfm_etat_to_sommet(apres_d_enl))
					poids_a_enl,Trajet_a_enl = (0,[])
					tps_nv_b_enl = Tdb[bpp] - (Ordre[bpp][pp*2][0][2] + Ordre[bpp][pp*2][1][2] + Ordre[bpp][pp*2+1][0][2]) + poids_d_enl
				elif avant_d_enl == None: # avant il y a le bus
					poids_d_enl,Trajet_d_enl = DARP.A_star(self,self.S[bpp],trfm_etat_to_sommet(apres_d_enl))
					poids_a_enl,Trajet_a_enl = DARP.A_star(self,trfm_etat_to_sommet(avant_a_enl),trfm_etat_to_sommet(apres_a_enl))
					tps_nv_b_enl = Tdb[bpp] - (Ordre[bpp][pp*2][0][2] + Ordre[bpp][pp*2][1][2] + Ordre[bpp][pp*2+1][0][2] + Ordre[bpp][pp*2+1][1][2]) + poids_d_enl + poids_a_enl
				elif apres_a_enl == None: #apres il y a rien
					poids_d_enl,Trajet_d_enl = DARP.A_star(self,trfm_etat_to_sommet(avant_d_enl),trfm_etat_to_sommet(apres_d_enl))
					poids_a_enl,Trajet_a_enl = (0,[])
					tps_nv_b_enl = Tdb[bpp] - (Ordre[bpp][pp*2][0][2] + Ordre[bpp][pp*2][1][2] + Ordre[bpp][pp*2+1][0][2]) + poids_d_enl
				else:
					poids_d_enl,Trajet_d_enl = DARP.A_star(self,trfm_etat_to_sommet(avant_d_enl),trfm_etat_to_sommet(apres_d_enl))
					poids_a_enl,Trajet_a_enl = DARP.A_star(self,trfm_etat_to_sommet(avant_a_enl),trfm_etat_to_sommet(apres_a_enl))
					tps_nv_b_enl = Tdb[bpp] - (Ordre[bpp][pp*2][0][2] + Ordre[bpp][pp*2][1][2] + Ordre[bpp][pp*2+1][0][2] + Ordre[bpp][pp*2+1][1][2]) + poids_d_enl + poids_a_enl

			# ========================================================================
			# poids et trajet du rajout de la personne (dans l'autre bus)


			# variable de stockage du min 
			# ce sont des etats !! minn !
			minn = float('inf')		
			poids_m_d_ava = 0
			poids_m_d_pro = 0
			poids_m_a_ava = 0
			poids_m_a_pro = 0
			avant_d_raj = 0
			avant_a_raj = 0
			apres_d_raj = 0
			apres_a_raj = 0
			ch_avant_d_raj = []
			ch_avant_a_raj = []
			ch_apres_d_raj = []
			ch_apres_a_raj = []
			min_cote_a_cote = True

			#init !!!!!
			etat_d_ava = None
			etat_d_pro = pp*2 +1 # arrie, init cote a cote
			etat_a_ava = pp*2 #init cote a cote 
			etat_a_pro = First[bd]
			while True != None: # on insere la ou c'est le mieux

				cote_a_cote = True
				while True:
					
					#print("etat_d_ava : "+str(etat_d_ava))
					#print("etat_d_pro : "+str(etat_d_pro))
					#print("etat_a_ava : "+str(etat_a_ava))
					#print("etat_a_pro : "+str(etat_a_pro))
					
					if cote_a_cote:
						if etat_d_ava == None and etat_a_pro == None:
							poids1,Trajet_raj_d1 = DARP.A_star(self,self.S[bd],self.D[pp])
							poids2,Trajet_raj_d2 = DARP.A_star(self,self.D[pp],self.A[pp]) 
							poids3,Trajet_raj_a1 = (0,[])
							tps_nv_b_raj = (poids1 + poids2)
						if etat_d_ava == None: #avant etat_d_ava il y a le bus
							poids1,Trajet_raj_d1 = DARP.A_star(self,self.S[bd],self.D[pp])
							poids2,Trajet_raj_d2 = DARP.A_star(self,self.D[pp],self.A[pp]) 
							poids3,Trajet_raj_a1 = DARP.A_star(self,self.A[pp],trfm_etat_to_sommet(etat_a_pro))
							tps_nv_b_raj = Tdb[bd] - (Ordre[bd][etat_a_pro][0][2]) + poids1 + poids2 + poids3

						elif etat_a_pro == None: # tout après il y a rien
							poids1,Trajet_raj_d1 = DARP.A_star(self,trfm_etat_to_sommet(etat_d_ava),self.D[pp])
							poids2,Trajet_raj_d2 = DARP.A_star(self,self.D[pp],self.A[pp]) 
							poids3,Trajet_raj_a1 = (0,[])
							tps_nv_b_raj = Tdb[bd] + poids1 + poids2

						else:
							poids1,Trajet_raj_d1 = DARP.A_star(self,trfm_etat_to_sommet(etat_d_ava),self.D[pp])
							poids2,Trajet_raj_d2 = DARP.A_star(self,self.D[pp],self.A[pp])
							poids3,Trajet_raj_a1 = DARP.A_star(self,self.A[pp],trfm_etat_to_sommet(etat_a_pro))
							tps_nv_b_raj = Tdb[bd] - (Ordre[bd][etat_d_ava][1][2]) + poids1 + poids2 + poids3

						if tps_nv_b_raj < minn: # on selectionne le best
							minn = tps_nv_b_raj
							avant_d_raj = etat_d_ava
							apres_d_raj = pp*2 +1
							avant_a_raj = pp*2								
							apres_a_raj = etat_a_pro
							poids_m_d_ava = poids1
							poids_m_d_pro = poids2
							poids_m_a_ava = poids2
							poids_m_a_pro = poids3
							ch_avant_d_raj = Trajet_raj_d1
							ch_apres_d_raj = Trajet_raj_d2
							ch_avant_a_raj = Trajet_raj_d2							
							ch_apres_a_raj = Trajet_raj_a1
							min_cote_a_cote = True
						if etat_a_pro == None: # si on est cote a cote est que le prochaine est None : break
							break
						etat_a_ava = etat_a_pro
						etat_a_pro = Ordre[bd][etat_a_pro][1][0]
						etat_d_pro = etat_a_ava
						# attention a bien modifier le depart

						cote_a_cote = False
					else:
						if etat_a_pro == None and etat_d_ava == None: # tout après il y a rien et tout avant il y a le bus
							poids_raj_d1,Trajet_raj_d1 = DARP.A_star(self,self.S[bd],self.D[pp])
							poids_raj_d2,Trajet_raj_d2 = DARP.A_star(self,self.D[pp],trfm_etat_to_sommet(etat_d_pro))
							poids_raj_a1,Trajet_raj_a1 = DARP.A_star(self,trfm_etat_to_sommet(etat_a_ava),self.A[pp])
							poids_raj_a2,Trajet_raj_a2 = (0,[])
							tps_nv_b_raj = Tdb[bd] - Ordre[bd][etat_d_pro][0][2] + poids_raj_d1 + poids_raj_d2 + poids_raj_a1
							
						elif etat_d_ava == None: # avant il y a le bus
							poids_raj_d1,Trajet_raj_d1 = DARP.A_star(self,self.S[bd],self.D[pp])
							poids_raj_d2,Trajet_raj_d2 = DARP.A_star(self,self.D[pp],trfm_etat_to_sommet(etat_d_pro))			
							poids_raj_a1,Trajet_raj_a1 = DARP.A_star(self,trfm_etat_to_sommet(etat_a_ava),self.A[pp])
							poids_raj_a2,Trajet_raj_a2 = DARP.A_star(self,self.A[pp],trfm_etat_to_sommet(etat_a_pro))
							tps_nv_b_raj = Tdb[bd] - ( Ordre[bd][etat_d_pro][0][2] + Ordre[bd][etat_a_pro][0][2] ) + poids_raj_d1 + poids_raj_d2 + poids_raj_a1 + poids_raj_a2

						elif etat_a_pro == None: # tout après il y a rien
							poids_raj_d1,Trajet_raj_d1 = DARP.A_star(self,trfm_etat_to_sommet(etat_d_ava),self.D[pp])
							poids_raj_d2,Trajet_raj_d2 = DARP.A_star(self,self.D[pp],trfm_etat_to_sommet(etat_d_pro))
							poids_raj_a1,Trajet_raj_a1 = DARP.A_star(self,trfm_etat_to_sommet(etat_a_ava),self.A[pp])
							poids_raj_a2,Trajet_raj_a2 = (0,[])
							tps_nv_b_raj = Tdb[bd] - ( Ordre[bd][etat_d_ava][1][2] ) + poids_raj_d1 + poids_raj_d2 + poids_raj_a1
						else:
							poids_raj_d1,Trajet_raj_d1 = DARP.A_star(self,trfm_etat_to_sommet(etat_d_ava),self.D[pp])
							poids_raj_d2,Trajet_raj_d2 = DARP.A_star(self,self.D[pp],trfm_etat_to_sommet(etat_d_pro))						
							poids_raj_a1,Trajet_raj_a1 = DARP.A_star(self,trfm_etat_to_sommet(etat_a_ava),self.A[pp])
							poids_raj_a2,Trajet_raj_a2 = DARP.A_star(self,self.A[pp],trfm_etat_to_sommet(etat_a_pro))
							tps_nv_b_raj = Tdb[bd] - ( Ordre[bd][etat_d_ava][1][2] + Ordre[bd][etat_a_pro][0][2]) + poids_raj_d1 + poids_raj_d2 + poids_raj_a1 + poids_raj_a2

						if tps_nv_b_raj < minn:
							minn = tps_nv_b_raj
							avant_d_raj = etat_d_ava
							apres_d_raj = etat_d_pro
							avant_a_raj = etat_a_ava								
							apres_a_raj = etat_a_pro
							poids_m_d_ava = poids_raj_d1
							poids_m_d_pro = poids_raj_d2
							poids_m_a_ava = poids_raj_a1
							poids_m_a_pro = poids_raj_a2
							ch_avant_d_raj = Trajet_raj_d1
							ch_apres_d_raj = Trajet_raj_d2
							ch_avant_a_raj = Trajet_raj_a1							
							ch_apres_a_raj = Trajet_raj_a2
							min_cote_a_cote = False
						if etat_a_pro == None: # on fait un tour de plus
							break
						etat_a_ava = etat_a_pro
						etat_a_pro = Ordre[bd][etat_a_pro][1][0]
				
			
				if cote_a_cote and etat_a_pro == None: # le prochain d est l'arrive
					break
				etat_d_ava = etat_d_pro
				etat_d_pro = pp*2 +1 #ok
				etat_a_ava = pp*2	
				etat_a_pro = Ordre[bd][etat_d_ava][1][0]
				
				
			
			if condition(bpp,bd,tps_nv_b_enl,tps_nv_b_raj,temperature):
				# =======================================================
				# on enleve la personne de bpp
				
				# on enregistre les nouveaux temps et nbs de personne
				Tdb[bpp] = tps_nv_b_enl
				Tdb[bd] = tps_nv_b_raj
				Ppb[bpp] += -1
				Ppb[bd] += 1
				Bu[pp] = bd

				if cote_a_cote_enl:
					if avant_d_enl == None and apres_a_enl == None:				
						First[bpp] = -1
						Ordre[bpp].clear() # plus rien

					elif avant_d_enl == None: # avant il y a le bus		
						First[bpp] = apres_a_enl
						Ordre[bpp][apres_a_enl] = [(None,Trajet_a_enl,poids_a_enl),Ordre[bpp][apres_a_enl][1]]
		
					elif apres_a_enl == None: # apres il y a rie,	
						Ordre[bpp][avant_d_enl] = [Ordre[bpp][avant_d_enl][0],(None,None,None)]
				
					else:	
						Ordre[bpp][avant_d_enl] = [Ordre[bpp][avant_d_enl][0],(apres_a_enl,Trajet_d_enl,poids_d_enl)]
						Ordre[bpp][apres_a_enl] = [(avant_d_enl,Trajet_d_enl,poids_d_enl),Ordre[bpp][apres_a_enl][1]]
						
				else:	
					if avant_d_enl == None and apres_a_enl == None:
						# depart
						First[bpp] = apres_d_enl
						Ordre[bpp][apres_d_enl] = [(None,Trajet_d_enl,poids_d_enl),Ordre[bpp][apres_d_enl][1]]
						# arrive
						Ordre[bpp][avant_a_enl] = [Ordre[bpp][avant_a_enl][0],(None,None,None)]

					elif avant_d_enl == None: # avant il y a le bus
						# depart
						First[bpp] = apres_d_enl
						Ordre[bpp][apres_d_enl] = [(None,Trajet_d_enl,poids_d_enl),Ordre[bpp][apres_d_enl][1]]
						# arrive
						Ordre[bpp][avant_a_enl] = [Ordre[bpp][avant_a_enl][0],(apres_a_enl,Trajet_a_enl,poids_a_enl)]
						Ordre[bpp][apres_a_enl] = [(avant_a_enl,Trajet_a_enl,poids_a_enl),Ordre[bpp][apres_a_enl][1]]
					elif apres_a_enl == None: # apres il y a rien
						# depart
						Ordre[bpp][avant_d_enl] = [Ordre[bpp][avant_d_enl][0],(apres_d_enl,Trajet_d_enl,poids_d_enl)]
						Ordre[bpp][apres_d_enl] = [(avant_d_enl,Trajet_d_enl,poids_d_enl),Ordre[bpp][apres_d_enl][1]]
						# arrive
						Ordre[bpp][avant_a_enl] = [Ordre[bpp][avant_a_enl][0],(None,None,None)]
					else:
						# depart
						Ordre[bpp][avant_d_enl] = [Ordre[bpp][avant_d_enl][0],(apres_d_enl,Trajet_d_enl,poids_d_enl)]
						Ordre[bpp][apres_d_enl] = [(avant_d_enl,Trajet_d_enl,poids_d_enl),Ordre[bpp][apres_d_enl][1]]
						# arrive
						Ordre[bpp][avant_a_enl] = [Ordre[bpp][avant_a_enl][0],(apres_a_enl,Trajet_a_enl,poids_a_enl)]
						Ordre[bpp][apres_a_enl] = [(avant_a_enl,Trajet_a_enl,poids_a_enl),Ordre[bpp][apres_a_enl][1]]
				# =======================================================
				# on rajoute la personne dans le bus bd
				#print("best : ")
				#print(avant_d_raj) 
				#print(apres_d_raj) 
				#print(avant_a_raj) 								
				#print(apres_a_raj)
				if min_cote_a_cote:
					if apres_a_raj == None and avant_d_raj == None: # tout après il y a rien et tout avant il y a le bus
						First[bd] = pp*2 # le 1er c'est le depart
						# rien
						Ordre[bd][pp*2] = [(None,ch_avant_d_raj,poids_m_d_ava),(apres_d_raj,ch_apres_d_raj,poids_m_d_pro)]
						
						Ordre[bd][pp*2 +1] = [(avant_a_raj,ch_avant_a_raj,poids_m_a_ava),(None,None,None)]
						#rien	
						
					elif avant_d_raj == None: # avant il y a le bus
						First[bd] = pp*2 # le 1er c'est le depart
						# rien
						Ordre[bd][pp*2] = [(None,ch_avant_d_raj,poids_m_d_ava),(apres_d_raj,ch_apres_d_raj,poids_m_d_pro)]
						
						Ordre[bd][pp*2 +1] = [(avant_a_raj,ch_avant_a_raj,poids_m_a_ava),(apres_a_raj,ch_apres_a_raj,poids_m_a_pro)]
						Ordre[bd][apres_a_raj] = [(2*pp+1,ch_apres_a_raj,poids_m_a_pro),Ordre[bd][apres_a_raj][1]]
					
					elif apres_a_raj == None: # tout après il y a rien
						Ordre[bd][avant_d_raj] = [Ordre[bd][avant_d_raj][0],(pp*2,ch_avant_d_raj,poids_m_d_ava)]
						Ordre[bd][pp*2] = [(avant_d_raj,ch_avant_d_raj,poids_m_d_ava),(apres_d_raj,ch_apres_d_raj,poids_m_d_pro)]
						
						Ordre[bd][pp*2 +1] = [(avant_a_raj,ch_avant_a_raj,poids_m_a_ava),(None,None,None)]
						#rien
					else:			
						Ordre[bd][avant_d_raj] = [Ordre[bd][avant_d_raj][0],(pp*2,ch_avant_d_raj,poids_m_d_ava)]
						Ordre[bd][pp*2] = [(avant_d_raj,ch_avant_d_raj,poids_m_d_ava),(apres_d_raj,ch_apres_d_raj,poids_m_d_pro)]
						
						Ordre[bd][pp*2 +1] = [(avant_a_raj,ch_avant_a_raj,poids_m_a_ava),(apres_a_raj,ch_apres_a_raj,poids_m_a_pro)]
						Ordre[bd][apres_a_raj] = [(2*pp+1,ch_apres_a_raj,poids_m_a_pro),Ordre[bd][apres_a_raj][1]]
				else:
					if apres_a_raj == None and avant_d_raj == None: # tout après il y a rien et tout avant il y a le bus
						First[bd] = pp*2 # le 1er c'est le depart
						# rien
						Ordre[bd][pp*2] = [(None,ch_avant_d_raj,poids_m_d_ava),(apres_d_raj,ch_apres_d_raj,poids_m_d_pro)]
						Ordre[bd][apres_d_raj] = [(2*pp,ch_apres_d_raj,poids_m_d_pro),Ordre[bd][apres_d_raj][1]]
						# l arrive
						Ordre[bd][avant_a_raj] = [Ordre[bd][avant_a_raj][0],(2*pp+1,ch_avant_a_raj,poids_m_a_ava)]
						Ordre[bd][pp*2 +1] = [(avant_a_raj,ch_avant_a_raj,poids_m_a_ava),(None,None,None)]
						#rien	
						
					elif avant_d_raj == None: # avant il y a le bus
						First[bd] = pp*2 # le 1er c'est le depart
						# rien
						Ordre[bd][pp*2] = [(None,ch_avant_d_raj,poids_m_d_ava),(apres_d_raj,ch_apres_d_raj,poids_m_d_pro)]
						Ordre[bd][apres_d_raj] = [(2*pp,ch_apres_d_raj,poids_m_d_pro),Ordre[bd][apres_d_raj][1]]
						# l arrive
						Ordre[bd][avant_a_raj] = [Ordre[bd][avant_a_raj][0],(2*pp+1,ch_avant_a_raj,poids_m_a_ava)]
						Ordre[bd][pp*2 +1] = [(avant_a_raj,ch_avant_a_raj,poids_m_a_ava),(apres_a_raj,ch_apres_a_raj,poids_m_a_pro)]
						Ordre[bd][apres_a_raj] = [(2*pp+1,ch_apres_a_raj,poids_m_a_pro),Ordre[bd][apres_a_raj][1]]
					
					elif apres_a_raj == None: # tout après il y a rien
						Ordre[bd][avant_d_raj] = [Ordre[bd][avant_d_raj][0],(pp*2,ch_avant_d_raj,poids_m_d_ava)]
						Ordre[bd][pp*2] = [(avant_d_raj,ch_avant_d_raj,poids_m_d_ava),(apres_d_raj,ch_apres_d_raj,poids_m_d_pro)]
						Ordre[bd][apres_d_raj] = [(2*pp,ch_apres_d_raj,poids_m_d_pro),Ordre[bd][apres_d_raj][1]]
						# l arrive
						Ordre[bd][avant_a_raj] = [Ordre[bd][avant_a_raj][0],(2*pp+1,ch_avant_a_raj,poids_m_a_ava)]
						Ordre[bd][pp*2 +1] = [(avant_a_raj,ch_avant_a_raj,poids_m_a_ava),(None,None,None)]
						#rien
					else:			
						Ordre[bd][avant_d_raj] = [Ordre[bd][avant_d_raj][0],(pp*2,ch_avant_d_raj,poids_m_d_ava)]
						Ordre[bd][pp*2] = [(avant_d_raj,ch_avant_d_raj,poids_m_d_ava),(apres_d_raj,ch_apres_d_raj,poids_m_d_pro)]
						Ordre[bd][apres_d_raj] = [(2*pp,ch_apres_d_raj,poids_m_d_pro),Ordre[bd][apres_d_raj][1]]
						# l arrive
						Ordre[bd][avant_a_raj] = [Ordre[bd][avant_a_raj][0],(2*pp+1,ch_avant_a_raj,poids_m_a_ava)]
						Ordre[bd][pp*2 +1] = [(avant_a_raj,ch_avant_a_raj,poids_m_a_ava),(apres_a_raj,ch_apres_a_raj,poids_m_a_pro)]
						Ordre[bd][apres_a_raj] = [(2*pp+1,ch_apres_a_raj,poids_m_a_pro),Ordre[bd][apres_a_raj][1]]


			return 0

		# =============================================================================
		DARP.multi_bus_aleatoire(self) # solution initial


		Ppb = [0 for _ in range(self.s)] # Le nombre de personne par bus
		for b in range(self.s):
			Ppb[b] = self.P[b][0]
		Tdb = [0 for _ in range(self.s)] # c est le temps de parcours de chaque bus
		for b in range(self.s):
			Tdb[b] = self.P[b][4]
			
		
		Ordre = [{} for _ in range(self.s)]
		# Ordre[num_bus][depart paire/arrive impaire personne] = [(depart/arrive,[chemin],poids), (depart/arrive,[chemin],poids)]
		Bu = [0 for _ in range(self.n)] # donne le bus des personnes
		
		for el in self.P:
			print(el)
		print("=====")
		for b in range(self.s):
			for pa in self.P[b][1]:
				Bu[pa] = b

	# P : [   [nombre de passager, [liste des numero des passager] , [parcours du bus] , [ordre de recuperation des passagers], poids du parcours] ... ] liste des bus
		First = [-1 for _ in range(self.s)] #First[bus] = premier sommet a prendre

		#creation de l'ensemble
		for b in range(self.s):
			# connection du 1er a gauche au bus et a droite au reste
			if self.P[b][0] != 0:
				First[b] = self.P[b][3][0]

				Ordre[b][self.P[b][3][0]] = [(None,self.P[b][2][0],self.P[b][5][0]),(self.P[b][3][1],self.P[b][2][1],self.P[b][5][1])] 
				for i in range(1,self.P[b][0]*2 - 1):	
					Ordre[b][self.P[b][3][i]] = [(self.P[b][3][i-1],self.P[b][2][i],self.P[b][5][i]),(self.P[b][3][i+1],self.P[b][2][i+1],self.P[b][5][i+1])] 
				
				Ordre[b][self.P[b][3][-1]] = [(self.P[b][3][-2],self.P[b][2][-1],self.P[b][5][-1]),(None,None,None)]
		# ===========================================================================
		affiche_ordre(Ordre,First)
		
		temperature = self.t0
		ct = 0
		cta = 0
		while temperature>self.tf:
			ct+=1
			cta += 1
			if cta > 10000:
				cta = 0
				print("temperature : ",temperature)
			voisinage(temperature)
			temperature *= self.alpha
		affiche_ordre(Ordre,First)
		print("nbs d'iteration",ct)
		deb = 0
		for b in range(self.s):
			deb+= Ppb[b]/Tdb[b]

		print("debit",deb)
		self.P = copy.deepcopy(trfm_ordre_to_p())

"""
m = len(M)
L = matrice_to_liste(M,m)
d = 5
a = 16
s = 2
S = [4,3]
n = 4
D = [5,0,3,6]
A = [1,7,0,2]

P = multi_bus_attribution_debit(M,H,m,s,S,n,D,A,0.01)
print(P)

affiche(m,V,T,P,s,S,n,D,A)
"""
m = 270
M = inject_data_distance("data_distance.txt", 270)
V = inject_data_liste("data_listesommet.txt",270)
H = inject_data_heuristie("data_heuristie.txt",270)
C = inject_data_color("data_color.txt",270)

s = 2
S = [random.randint(0,m-1) for _ in range(s)]


n = 5
D = [random.randint(0,m-1) for _ in range(n)]
A = [random.randint(0,m-1) for _ in range(n)]


sys = DARP(M,V,C,H,m,s,S,n,D,A,0.01,math.sqrt(6),0.01,0.995)

sys.aff_sommet = False
sys.contraste_bus = 0.5

sys.multi_bus_attribution_recui_simule()



deb = 0
for b in sys.P:
	print(b)
	deb += b[4]
print("TPS : ",deb)
sys.affiche()

