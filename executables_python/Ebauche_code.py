finale = []

	def bus1_chemin_naive(M,n):
		 # [ [ [chemin] , poids du chemin]  , ... ]
		def aux(M,n,pos,Choix,t_choix,Chemin,t_chemin,poids):
			# M numpy : matrice tsp 
			# n : nombre de passager
			# pos : position actuel
			# Choix : liste des possibilité pour le bus 
			# 		| structure : [ (1 si sommet j disponible ) ou (0 si sommet j non disponible)] ou on note  j  l'indice :
			#		|			| on a j pairs sont les sommets des passagers et j impairs sont les sommets des destinations des passagers
			#
			# t_choix : taille de Choix
			# Chemin : liste du chemin actuellement emprunté
			# t_chemin : taille de Chemin
			# poids : poids actuel du chemin
			global finale
			if t_chemin == 2*n +1:
				finale.append([Chemin,poids])
			for i in range(t_choix):
				if Choix[i] == 1:
					if i%2 == 0: # c'est un passager   
						Tmp_choix = Choix.copy()
						Tmp_chemin = Chemin.copy()
						Tmp_choix[i] = 0
						Tmp_choix[i+1] = 1
						Tmp_chemin.append(i+1)
						aux(M,n,i+1,Tmp_choix,t_choix,Tmp_chemin,t_chemin + 1, poids + M[pos,i+1])
					else:
						Tmp_choix = Choix.copy()
						Tmp_chemin = Chemin.copy()
						Tmp_choix[i] = 0
						Tmp_chemin.append(i+1)
						aux(M,n,i+1,Tmp_choix,t_choix,Tmp_chemin,t_chemin + 1, poids + M[pos,i+1])
		aux(M,n,0,[(i+1)%2 for i in range(2*n)] , 2*n , [0] , 1, 0)
		return min(finale,key=cmp1)

	def bus1_chemin_optimise_time_windo(M,d,n,T): # EN COURS
		# M (numpy) : matrice tsp
		# d : sommet de depart du bus
		# n : nombre de passager
		# T : time windows -> [ [ [temps de début de fenetre, tps de fin de fenetre], dure max du trajet ] , ...nombre de passager]
		a = []

		L = []#// liste des chemins a k-1
		L1 = [] # liste temporaire
		L2 = []

		LS = [True for _ in range(n)] # liste des sommet deservie

		Dure = [float('inf') for _ in range(n)] # Dure[i] : temps ou le passager i a était pris !

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
					print(M[d,2*f +1])
					if T[f][0][0] <= M[d,2*f +1] <= T[f][0][1]: # TW correspond !
						# on recup ce passager !
						Dure[f] = M[d,2*f+1]
						a = Gen.copy()
						a[2*f] = 0
						a[2*f +1] = 1
						L1.append([2*f+1, M[d,2*f+1] , a ,[0, 2*(f+1) -1] ].copy())
					else:
						pass


					#fin de l'etape, on change de liste
					switchL = True
			else:
				L=[]
				# gestion des listes temporaires
				if switchL:
					if L1 == []: # plus rien a rechercher !			
						break
					L = L1.copy()
					L2.clear()
					switchL = False
				else:
					if L2 == []: # plus rien a rechercher
						break
					L = L2.copy()
					L1.clear()
					switchL = True
				print("L -> ",L)
				
				for p in range(2*n):
					# on regarde tous les sommets d'arrivées possibles.

					# On doit trouver le chemin minimum de taille k et qui arrive au sommet p

					J = []
					for j in L:# on regarde les élements de notre liste temporaire

						if j[2][p] == 1: # on regarde si les sommets d'arrivés sont potentiellements accessibles

							if p % 2 == 0: # ce sommet est un client
								# on change l'etat de notre chemin

								if T[p//2][0][0] <= ( j[1] + M[j[0],p+1] ) <= T[p//2][0][1]: # TW dispo
									Dure[p//2] = j[1] + M[j[0],p+1]
									a = j[2].copy()
									a[p] = 0
									a[p+1] = 1
									J.append([ p+1, j[1] + M[j[0],p+1] ,a, j[3] + [p+1] ].copy())
								else:
									pass
									
							else: # ce sommet est une destination
								# on change l'etat de notre chemin
								if ( j[1] + M[j[0],p+1] - Dure[p//2] ) <= T[p//2][1]: # TW dispo
									a = j[2].copy()
									a[p] = 0
									J.append([ p+1, j[1] + M[j[0],p+1] ,a, j[3] + [p+1] ].copy())
								else:
									

									pass


					if J!=[]:
						jf = min(J,key=cmp1)
						# je selectionne la liste qui a un poids minimum
						if switchL :
							L1.append(jf)
						else:
							L2.append(jf)
					


			
			print("===========================")
			print("L1 : ")
			for i in L1:
				print(i)
			print("===========================")
			print("L2 : ")
			for i in L2:
				print(i)
			print("===========================")
		

		if L1 == [] and L2 == []:
			Mm = [0,0,0,[]]
		elif L1 == []:
			Mm = min(L2,key=cmp1)
		elif L2 == []:
			Mm = min(L1,key=cmp1)
		else:
			if switchL :
				Mm = min(L1,key=cmp1)
			else:
				Mm = min(L2,key=cmp1)

		return (LS,Mm[1],Mm[3]) # ( poids , chemin )
		# a la fin, on prend le chemin qui a un poids minimum.

def multi_bus_attribution_debit(M,H,m,s,S,n,D,A,w):
		# M : matrice d'adjacence de la ville
		# H : matrice heuristique de la ville ( distance a vol d'oiseau entre chaque sommet)
		# m : taille de M
		# s : nombre de bus
		# S : liste des sommets des bus
		# n : nombre de personne
		# D : liste des sommets de départ des passagers indicé par leur numéros
		# A : liste des sommets d'arrivé des passagers indicié par leur numéros
		# w : poids heuristique dans A*
		L = matrice_to_liste(M,m)
		# P : [ [nombre de passager, [liste des numero des passager] , [parcours du bus] , [ordre de recuperation des passagers], poids du parcours] ... ] liste des bus
		P = [[0,[],[],[],[],0] for i in range(s)]
		Deb = [0 for _ in range(s)]
		pt = [] # parcours temporaire
		pt_m = []
		
		debit_max = 0 # debit max
		num_bus = 0
		for i in range(n): # pour les n passager
			start = True
			for k in range(s): #pour les k bus
				d = []
				a = []
				ordre = []
				for el in P[k][1]: # on itere les passager dans le bus k
					d.append(D[el]) # on recupere les sommets de départ / arrive de c'est passager 
					a.append(A[el])
					ordre.append(el)
				R,T = generer_graph_a_star_avec_chemin(L,H,m,P[k][0] + 1,d + [D[i]] ,a + [A[i]],S[k],w) # on genere les graphs associés
				
				
				(poids , pt) =  bus1_chemin_optimise1(R,0,P[k][0] + 1)
				
				parcours = []
				print(ordre)
				if ordre != []:
					for g in range(1,len(pt)):
						parcours.append(ordre[(pt[g]-1)//2])

				# tmp_deb contient le debit du systeme, c'est la somme des debits des bus
				tmp_deb = (P[k][0] + 1 ) / poids
				#print("tmp deb : "+str(tmp_deb))
				debit_Total = tmp_deb
				for bus in range(s):
					if bus != k:
						debit_Total += Deb[bus]

				if ( debit_Total > debit_max) or start: # on garde le debit le plus grand
					# on enleve les doublons consécutifs
					pt_s_d =[pt[0]]
					taille_pt = len(pt)
					prec = pt[0]
					for o in range(1,taille_pt):
						if prec != pt[o]:
							pt_s_d.append(pt[o])
						prec = pt[o]
					#print(pt_s_d)
					num_bus = k
					debit_max = debit_Total
					debit_de_rajout = tmp_deb
					pt_m.clear()
					pt_m = T[pt_s_d[0]][pt_s_d[1]]
					for o in range(1,taille_pt -1):
						if pt_m == []:
							pt_m += T[pt_s_d[o]][pt_s_d[o+1]]
						else:
							pt_m += T[pt_s_d[o]][pt_s_d[o+1]][1:]
				
				start = False
				

			Deb[num_bus] = debit_de_rajout
			
			P[num_bus][0] += 1
			P[num_bus][1].append(i)
			P[num_bus][2] = pt_m.copy()
			P[num_bus][3] = parcours.copy()
			P[num_bus][4] = poids

		
		# [   [nombre de passager, [liste des numero des passager] , [parcours du bus] , poids du parcours] ... ] liste des bus
		return P

def multi_bus_attribution_naive_time_windo(M,H,m,s,S,n,D,A,w,T): # EN COURS
		# M : matrice d'adjacence de la ville
		# H : matrice heuristique de la ville ( distance a vol d'oiseau entre chaque sommet)
		# m : taille de M
		# s : nombre de bus
		# S : liste des sommets des bus
		# n : nombre de personne
		# D : liste des sommets de départ des passagers indicé par leur numéros
		# A : liste des sommets d'arrivé des passagers indicié par leur numéros
		# w : poids heuristique dans A*
		# T : time windows -> [ [ [temps de début de fenetre, tps de fin de fenetre], dure max du trajet ] , ...nombre de passager]
		def generer_permu(n,s,i,permut):
			global Per
			if i == n:
				Per.append(permut)
				return
			for b in range(s):
				tmp = copy.deepcopy(permut)
				tmp[b].append(i)
				generer_permu(n,s,i+1,tmp)
		generer_permu(n,s,0,[[] for _ in range(s)])
		
		start = True
		debit_max = 0
		L = matrice_to_liste(M,m)
		P = [[0,[],[],0] for i in range(s)] # parcours des bus [ [  nbs_de_passager , [liste des passager (ordre croissant)] , [parcours] , poids ] , ... ]
				
		for perm in Per: # on test tout les permutations
			som_deb = 0
			tmp_P = [[0,[],[],0] for i in range(s)]
			for k in range(s): # pour les k bus
				pt = [] # parcours temporaire
				pt_m = []
				
				d = []
				a = []
				nbs_pa = 0
				for pa in perm[k]: # on itere les passager dans le bus k
					d.append(D[pa]) # on recupere les sommets de départ / arrive de c'est passager 
					a.append(A[pa])
					nbs_pa += 1
			
				if nbs_pa == 0:
					tmp_P[k][0] = nbs_pa
					tmp_P[k][1] = []
					tmp_P[k][2] = [S[k]]
					tmp_P[k][3] = 0
				else:
					R,T = generer_graph_a_star_avec_chemin(L,H,m,nbs_pa,d,a,S[k],w) # on genere les graphs associés							
					(TS,poids , pt) =  bus1_chemin_optimise_time_windo(M,d,n,T)
					som_deb += nbs_pa / poids				
					pt_s_d =[pt[0]]
					taille_pt = len(pt)
					prec = pt[0]
					for o in range(1,taille_pt):
						if prec != pt[o]:
							pt_s_d.append(pt[o])
						prec = pt[o]
					pt_m.clear()
					pt_m = T[pt_s_d[0]][pt_s_d[1]]
					for o in range(1,taille_pt -1):
						if pt_m == []:
							pt_m += T[pt_s_d[o]][pt_s_d[o+1]]
						else:
							pt_m += T[pt_s_d[o]][pt_s_d[o+1]][1:]	
					
					tmp_P[k][0] = nbs_pa
					tmp_P[k][1] = perm[k].copy()
					tmp_P[k][2] = pt_m.copy()
					tmp_P[k][3] = poids
			if start or (som_deb > debit_max):
				start = False
				debit_max = som_deb
				P = copy.deepcopy(tmp_P)
		return P


def map_generer(taille):
	M = [[float('inf') for _ in range(taille)] for _ in range(taille)]
	
	for i in range(taille):
		x = random.randint(0,100)
		if 0 <= x <= 50:
			inter = 3
		elif 50 < x <= 80:
			inter = 4
		elif 80 < x <= 90:
			inter = 5
		else: 
			inter = 6
		for k in range(inter):
			pos = random.randint(0,taille - 1)
			while M[i][pos] != float('inf') or pos == i:
				pos = random.randint(0,taille - 1)
			ran = random.randint(1,100) # poids
			M[i][pos] = ran
			M[pos][i] = ran
			
	return M

def algo_floyd_warshall_avec_chemin(M,m):
	N = [[[k,i] for i in range(m)] for k in range(m)] # matrice des chemins les plus courts pour aller de i (indice de ligne) a j (indice de colonne)
	
	for k in range(m):
		for i in range(m):
			for j in range(m):
				if M[i][j] > ( M[i][k] + M[k][j] ):
					M[i][j] = (M[i][k] + M[k][j])
					N[i][j] = N[i][k] + N[k][j][1:]
			
	return (M,N)


def algo_floyd_warshall(M,m):	
	for k in range(m):
		for i in range(m):
			for j in range(m):
				if M[i][j] > ( M[i][k] + M[k][j] ):
					M[i][j] = (M[i][k] + M[k][j])
					
	return M

def dijtra(L,m,d):
	
	not_visited = [i for i in range(m) if i!=d]
	D = [float('inf') if i != d else 0 for i in range(m)]
	pos = d
	poids = 0

	while not_visited != []:
		for adj in L[pos]:
			if adj[0] in not_visited:
				if poids + adj[1] < D[adj[0]]:
					D[adj[0]] = poids + adj[1]
					
		minn = D[not_visited[0]]
		ind_min = not_visited[0]
		for ind in not_visited[1:]:
			if D[ind] < minn:
				minn = D[ind]
				ind_min = ind
		poids = minn
		pos = ind_min
		not_visited.remove(pos)
	return D

def dijtra_avec_chemin(L,m,d):
	#probleme d'opti pour le trie, plutot faire un trie par insertion !
	not_visited = [i for i in range(m) if i!=d]
	D = [float('inf') if i != d else 0 for i in range(m)]
	P = [ 0 if i != d else 0 for i in range(m) ]
	pos = d
	poids = 0

	while not_visited != []:
		for adj in L[pos]:
			if adj[0] in not_visited: # adj[0] -> sommet adjacent , adj[1] -> poids de l'arc
				if poids + adj[1] < D[adj[0]]:
					D[adj[0]] = poids + adj[1]
					P[adj[0]] = pos


		minn = D[not_visited[0]]
		ind_min = not_visited[0]
		for ind in not_visited[1:]:
			if D[ind] < minn:
				minn = D[ind]
				ind_min = ind
		poids = minn
		pos = ind_min
		not_visited.remove(pos)
	G = [[] for _ in range(m)]
	for i in range(m):
		jj = P[i]
		G[i].append(i)
		while jj != d:
			G[i].append(jj)
			jj = P[jj]
		G[i].append(d)
		G[i].reverse()


	return (D,G)

def generer_graph_aleatoire(n,sup):
	# n : nombre de passager
	# sup : poids maximum
	n = 2*n + 1
	M = np.ones((n,n))
	for i in range(n):
		M[i,i] = 0.
	for i in range(n):
		for j in range(i):
			a = random.randint(0,sup)
			M[i,j] = a
			M[j,i] = a
	return M

def generer_graph_dejaFW(M,m,n,D,A,b):
	#generer le graph finale
	F = np.ones((2*n+1,2*n+1))
	
	F[0,0] = 0

	# les arcs : bus => passager 
	for i in range(n): 
		F[0,2*i +1] = M[b][D[i]] #impair
	# les arcs : bus => destination des passagers
	for i in range(n): 
		F[0,2*i +2] = M[b][A[i]] #pair
	# les arcs : passager => bus
	for i in range(n): 
		F[2*i +1,0] = M[D[i]][b] #impair
	# les arcs : destination des passagers => bus
	for i in range(n):
		F[2*i +2,0] = M[A[i]][b] #pair

	#-------le reste----------
	# passager => passager
	for i in range(n):
		for j in range(n):
			F[2*i +1,2*j +1] = M[D[i]][D[j]]
	# arrive => arrive
	for i in range(n):
		for j in range(n):
			F[2*i +2,2*j +2] = M[A[i]][A[j]]
	# passager => arrive
	for i in range(n):
		for j in range(n):
			F[2*i +1,2*j +2] = M[D[i]][A[j]]
	# arrive => passager
	for i in range(n):
		for j in range(n):
			F[2*i +2,2*j +1] = M[A[i]][D[j]]


	return F

def generer_graph(M,m,n,D,A,b):
	# M : matrice d'adjacence du graph de la ville
	# m : taille de la matrice d'adjancence M
	# n : nombre de passager
	# D : liste des sommets de départ des passagers indicé par leur numéros
	# A : liste des sommets d'arrivé des passagers indicié par leur numéros
	# b : sommet du bus

	# algo du floyd warshall 
	M = algo_floyd_warshall(M,m)

	#generer le graph finale
	F = np.ones((2*n+1,2*n+1))
	
	F[0,0] = 0

	# les arcs : bus => passager 
	for i in range(n): 
		F[0,2*i +1] = M[b][D[i]] #impair
	# les arcs : bus => destination des passagers
	for i in range(n): 
		F[0,2*i +2] = M[b][A[i]] #pair
	# les arcs : passager => bus
	for i in range(n): 
		F[2*i +1,0] = M[D[i]][b] #impair
	# les arcs : destination des passagers => bus
	for i in range(n):
		F[2*i +2,0] = M[A[i]][b] #pair

	#-------le reste----------
	# passager => passager
	for i in range(n):
		for j in range(n):
			F[2*i +1,2*j +1] = M[D[i]][D[j]]
	# arrive => arrive
	for i in range(n):
		for j in range(n):
			F[2*i +2,2*j +2] = M[A[i]][A[j]]
	# passager => arrive
	for i in range(n):
		for j in range(n):
			F[2*i +1,2*j +2] = M[D[i]][A[j]]
	# arrive => passager
	for i in range(n):
		for j in range(n):
			F[2*i +2,2*j +1] = M[A[i]][D[j]]


	return F

def generer_graph_opti(M,m,n,D,A,b):
	# M : matrice d'adjacence du graph de la ville
	# m : taille de la matrice d'adjancence M
	# n : nombre de passager
	# D : liste des sommets de départ des passagers indicé par leur numéros
	# A : liste des sommets d'arrivé des passagers indicié par leur numéros
	# b : sommet du bus
	F = np.ones((2*n+1,2*n+1))
	L = matrice_to_liste(M,m)
	
	J = dijtra(L,m,b)
	F[0,0] = 0
	# les arcs : bus => passager 
	for i in range(n): 
		F[0,2*i +1] = J[D[i]] #impair

	# les arcs : bus => destination des passagers
		F[0,2*i +2] = J[A[i]] #pair

	for i in range(n): 
		J = dijtra(L,m,D[i])# passager => *
		K = dijtra(L,m,A[i])# destination => *
		F[2*i +1,0] = J[b] # passager => bus
		F[2*i +2,0] = K[b] # destination => bus
		for j in range(n):
			# passager => passager
			F[2*i +1,2*j +1] = J[D[j]]
			# destination => passager
			F[2*i +2,2*j +1] = K[D[j]]
			# passager => destination
			F[2*i +1,2*j +2] = J[A[j]]
			# destination => destination
			F[2*i +2,2*j +2] = K[A[j]]
	return F

def generer_graph_opti_avec_chemin(L,m,n,D,A,b):
	# L : liste d'adjacence obtenu en faisant matrice_to_liste(M)
	# m : taille de la matrice d'adjancence M
	# n : nombre de passager
	# D : liste des sommets de départ des passagers indicé par leur numéros
	# A : liste des sommets d'arrivé des passagers indicié par leur numéros
	# b : sommet du bus
	F = np.ones((2*n+1,2*n+1))
	C = [[[] for _ in range(2*n+1)] for _ in range(2*n+1)]
		
	J,U = dijtra_avec_chemin(L,m,b)
	
	F[0,0] = 0
	# les arcs : bus => passager 
	for i in range(n): 
		F[0,2*i +1] = J[D[i]] #impair
		if J[D[i]] == 0:
			C[0][2*i +1] = []
		else:
			C[0][2*i +1] = U[D[i]]
	# les arcs : bus => destination des passagers
		F[0,2*i +2] = J[A[i]] #pair
		if J[A[i]] == 0:
			C[0][2*i+2] = []
		else:
			C[0][2*i+2] = U[A[i]]
	for i in range(n): 
		J,U = dijtra_avec_chemin(L,m,D[i])# passager => *
		K,I = dijtra_avec_chemin(L,m,A[i])# destination => *
		F[2*i +1,0] = J[b] # passager => bus
		if J[b] == 0:
			C[2*i +1][0] = []
		else:
			C[2*i +1][0] = U[b] 
		F[2*i +2,0] = K[b] # destination => bus
		if K[b] == 0:
			C[2*i +2][0] = []
		else:	
			C[2*i +2][0] = I[b]
		for j in range(n):
			# passager => passager
			F[2*i +1,2*j +1] = J[D[j]]
			if J[D[j]] == 0:
				C[2*i +1][2*j +1] = []
			else:
				C[2*i +1][2*j +1] = U[D[j]]
			# destination => passager
			F[2*i +2,2*j +1] = K[D[j]]
			if K[D[j]] == 0:
				C[2*i +2][2*j +1] = []
			else:
				C[2*i +2][2*j +1] = I[D[j]]
			# passager => destination
			F[2*i +1,2*j +2] = J[A[j]]
			if J[A[j]] == 0:
				C[2*i +1][2*j +2] = []
			else:
				C[2*i +1][2*j +2] = U[A[j]]
			# destination => destination
			F[2*i +2,2*j +2] = K[A[j]]
			if K[A[j]] == 0:
				C[2*i +2][2*j +2] = []
			else:
				C[2*i +2][2*j +2] = I[A[j]]
	for i in range(2*n +1):
		C[i][i] = []


	return F,C

def test_complexite(max):
	N = []
	O = []
	X = [i for i in range(2,max)]
	for i in range(2,max):	
		M = generer_graph_aleatoire(i,100)
		"""
		start = time.time()
		bus1_chemin_naive(M,i)
		N.append(time.time() - start)
		"""
		start = time.time()
		bus1_chemin_optimise1(M,0,i)
		O.append(time.time() - start)
	plt.plot(X,O,'--b')
	plt.show()

def multi_bus_attribution(M,m,s,S,n,D,A):
	# M : matrice d'adjacence de la ville
	# m : taille de M
	# s : nombre de bus
	# S : liste des sommets des bus
	# n : nombre de personne
	# D : liste des sommets de départ des passagers indicé par leur numéros
	# A : liste des sommets d'arrivé des passagers indicié par leur numéros 

	(M,N) = algo_floyd_warshall_avec_chemin(M,m)
	P = [[0,[],[],0] for i in range(s)] # parcours des bus [ [  nbs_de_passager , [liste des passager (ordre croissant)] , [parcours] , poids ] , ... ]
	pt = [] # parcours temporaire
	pt_m = []
	poids = 0 # poids temporaire du parcours
	poids_m = 0
	num_bus = 0
	for i in range(n): # pour les n passager
		start = True
		for k in range(s): #pour les k bus
			d = []
			a = []
			mapp={}
			mapp[0] = S[k]
			j = 1
			for el in P[k][1]: # on itere les passager dans le bus k
				d.append(D[el]) # on recupere les sommets de départ / arrive de c'est passager 
				a.append(A[el])
				mapp[j] = D[el]
				j += 1
				mapp[j] = A[el]
				j += 1
			mapp[j] = D[i]
			j+=1
			mapp[j] = A[i]
			#print("bus : " ,k,"nbs pass : ",i)
			R = generer_graph_dejaFW(M,m,P[k][0] + 1 ,d + [D[i]] ,a + [A[i]],S[k]) # on genere les graphs associés
			#print(R)
			(poids , pt) =  bus1_chemin_optimise1(R,0,P[k][0] + 1)
			#print(poids)
			#print(pt)
			if (poids < poids_m) or start:
				poids_m = poids
				pt_m.clear()
				for el in pt:
					pt_m.append(mapp[el])
				num_bus = k
				#print(pt_m)
				#print("select")
			start = False
		P[num_bus][0] += 1
		P[num_bus][1].append(i)
		P[num_bus][2] = pt_m.copy()
		P[num_bus][3] = poids_m 

	J = []
	for l in range(len(P)):
		J.clear()
		for i in range(len(P[l][2]) - 1 ):
			J += N[P[l][2][i]][P[l][2][i+1]]
		prec = J[0]
		F = [prec]
		for k in range(1,len(J)):
			if prec != J[k]:
				F.append(J[k])
			prec = J[k]
		P[l].append(F.copy())
	return P

def multi_bus_attribution_opti(M,m,s,S,n,D,A):
	# M : matrice d'adjacence de la ville
	# m : taille de M
	# s : nombre de bus
	# S : liste des sommets des bus
	# n : nombre de personne
	# D : liste des sommets de départ des passagers indicé par leur numéros
	# A : liste des sommets d'arrivé des passagers indicié par leur numéros 
	L = matrice_to_liste(M,m)
	P = [[0,[],[],0] for i in range(s)] # parcours des bus [ [  nbs_de_passager , [liste des passager (ordre croissant)] , [parcours] , poids ] , ... ]
	pt = [] # parcours temporaire
	pt_m = []
	poids = 0 # poids temporaire du parcours
	poids_m = 0
	num_bus = 0
	for i in range(n): # pour les n passager
		start = True
		for k in range(s): #pour les k bus
			d = []
			a = []
			for el in P[k][1]: # on itere les passager dans le bus k
				d.append(D[el]) # on recupere les sommets de départ / arrive de c'est passager 
				a.append(A[el])
			#print()
			#t1 = time.time()
			R,T = generer_graph_opti_avec_chemin(L,m,P[k][0] + 1 ,d + [D[i]] ,a + [A[i]],S[k]) # on genere les graphs associés
			#print(time.time() - t1)
			#t1= time.time()
			(poids , pt) =  bus1_chemin_optimise1(R,0,P[k][0] + 1)
			#print(time.time() - t1)

			if (poids < poids_m) or start:
				# on enleve les doublons consécutifs
				pt_s_d =[pt[0]]
				taille_pt = len(pt)
				prec = pt[0]

				for o in range(1,taille_pt):
					if prec != pt[o]:
						pt_s_d.append(pt[o])
					prec = pt[o]
				#print(pt_s_d)
				num_bus = k
				poids_m = poids
				pt_m.clear()
				pt_m = T[pt_s_d[0]][pt_s_d[1]]
				for o in range(1,taille_pt -1):
					if pt_m == []:
						pt_m += T[pt_s_d[o]][pt_s_d[o+1]]
					else:
						pt_m += T[pt_s_d[o]][pt_s_d[o+1]][1:]
				#print(pt_m)
				#print("select")
			start = False
		P[num_bus][0] += 1
		P[num_bus][1].append(i)
		P[num_bus][2] = pt_m.copy()
		P[num_bus][3] = poids_m 

	
	# [   [nombre de passager, [liste des numero des passager] , [parcours du bus] , poids du parcours] ... ] liste des bus
	return P

def multi_bus_attribution_prio(M,m,s,S,n,D,A,P):
	# M : matrice d'adjacence de la ville
	# m : taille de M
	# s : nombre de bus
	# S : liste des sommets des bus
	# n : nombre de personne
	# D : liste des sommets de départ des passagers indicé par leur numéros
	# A : liste des sommets d'arrivé des passagers indicié par leur numéros 
	L = matrice_to_liste(M,m)
	P = [[0,[],[],0] for i in range(s)] # parcours des bus [ [  nbs_de_passager , [liste des passager (ordre croissant)] , [parcours] , poids ] , ... ]
	pt = [] # parcours temporaire
	pt_m = []
	poids = 0 # poids temporaire du parcours
	poids_m = 0
	num_bus = 0 
	SCORE = [0 for _ in range(n)] # durée du trajet pour tout les passagers
	for i in range(n): # pour les n passager
		start = True
		for k in range(s): #pour les k bus
			d = []
			a = []
			for el in P[k][1]: # on itere les passager dans le bus k
				d.append(D[el]) # on recupere les sommets de départ / arrive de c'est passager 
				a.append(A[el])
			R,T = generer_graph_opti_avec_chemin(L,m,P[k][0] + 1 ,d + [D[i]] ,a + [A[i]],S[k]) # on genere les graphs associés
			(poids , pt) =  bus1_chemin_optimise1(R,0,P[k][0] + 1)
			print(pt)
			distance = 0
			for j in range(len(pt)-1):
				distance += R[pt[j]][pt[j+1]]
				if pt[j+1] % 2 == 0:
					SCORE[pt[j+1] // 2] = distance
			print(SCORE)

			if (poids < poids_m) or start:
				# on enleve les doublons consécutifs
				pt_s_d =[pt[0]]
				taille_pt = len(pt)
				prec = pt[0]

				for o in range(1,taille_pt):
					if prec != pt[o]:
						pt_s_d.append(pt[o])
					prec = pt[o]
				#print(pt_s_d)
				num_bus = k
				poids_m = poids
				pt_m.clear()
				pt_m = T[pt_s_d[0]][pt_s_d[1]]
				for o in range(1,taille_pt -1):
					if pt_m == []:
						pt_m += T[pt_s_d[o]][pt_s_d[o+1]]
					else:
						pt_m += T[pt_s_d[o]][pt_s_d[o+1]][1:]
				#print(pt_m)
				#print("select")
			start = False
		P[num_bus][0] += 1
		P[num_bus][1].append(i)
		P[num_bus][2] = pt_m.copy()
		P[num_bus][3] = poids_m 

	
	# [   [nombre de passager, [liste des numero des passager] , [parcours du bus] , poids du parcours] ... ] liste des bus
	return P


def bus1_chemin_optimise2(M,d,n): #algo faux !
	L = [1,2] # liste du chemin le plus court 
	for k in range(n-1): 
		# ajout du passager k
		print("============")
		taille_L = len(L)
		minn = M[0][2*k + 3] + M[2*k + 3][L[0]]
		print(minn)
		ind_min = 0
		ind_min2 = 0
		for i in range(taille_L-1):
			print(M[L[i]][2*k + 3] + M[2*k + 3][L[i+1]])
			if ( M[L[i]][2*k + 3] + M[2*k + 3][L[i+1]] ) < minn :
				minn = ( M[L[i]][2*k + 3] + M[2*k + 3][L[i+1]] )
				ind_min = i+1
		print(M[L[-1]][2*k + 3])
		if M[L[-1]][2*k + 3] < minn:
			minn = M[L[-1]][2*k + 3]
			ind_min = taille_L
		L.insert(ind_min,2*k + 3)
		taille_L += 1
		# ajout de la destination
		print("---")
		if ind_min == (taille_L-1):
			miin = M[L[ind_min]][2*k + 4]
		else:
			miin = M[L[ind_min]][2*k + 4] + M[2*k + 4][L[ind_min + 1]]

		for i in range(ind_min,taille_L-1):
			print(M[L[i]][2*k + 4] + M[2*k + 4][L[i+1]])
			if ( M[L[i]][2*k + 4] + M[2*k + 4][L[i+1]] ) < minn :
				minn = ( M[L[i]][2*k + 4] + M[2*k + 4][L[i+1]] )
				ind_min2 = i+1
		print(M[L[-1]][2*k + 4])
		if M[L[-1]][2*k + 4] < minn:
			minn = M[L[-1]][2*k + 4]
			ind_min2 = taille_L 
		L.insert(ind_min2,2*k + 4)
		taille_L += 1

	L.insert(0,0)
	taille_L += 1
	poids = 0
	for i in range(taille_L-1):
		poids += M[L[i]][L[i+1]]

	return (L,poids)

def multi_bus_attribution_correspondance(M,H,m,s,S,n,D,A,max_dist):
	# M : matrice d'adjacence de la ville
	# H : matrice d'adjacence heuristique
	# m : taille de M et de H
	# s : nombre de bus
	# S : liste des sommets des bus
	# n : nombre de personne
	# D : liste des sommets de départ des passagers indicé par leur numéros
	# A : liste des sommets d'arrivé des passagers indicié par leur numéros 

	P = multi_bus_attribution_opti(M,m,s,S,n,D,A)
	tmp = max(P,key=cmp2)
	temps_maxi = tmp[3]
	print(temps_maxi)

	timelane = [[] for _ in range(s)]
	for b in range(s):
		for k in range(len(P[b][2]) - 1):
			repet = M[P[b][2][k]][P[b][2][k+1]]	
			timelane[b].append([P[b][2][k] for _ in range(repet)])
		timelane[b].append(P[b][2][-1])
	for tps in range(temps_maxi):
		exechange_point = [] # [ [bus un , bus deux,...] , sommet d'échange, ... ]
		contact = []
		for b1 in range(s):
			for b2 in range(b1+1,s): # on itere les connections possible entre les bus
				if H[timelane[b1][tps]][timelane[b2][tps]] <= max_dist:
					contact.append({b1,b2})
		for el in contact:
			pass
	