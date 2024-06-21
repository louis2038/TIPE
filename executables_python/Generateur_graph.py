import pygame
SIZEx = 1800
SIZEy = 800
RAYON = 10
COLOR = [(255,0,255),(0,255,255),(0,255,0),(255,255,0),(155,100,0),(255,0,0)]

bg = pygame.image.load("map_gre2.png")

def distance(a,b):
	return ( (a[0] - b[0])**2 + (a[1] - b[1])**2  )**(1/2)

def rendu(screen,event,nbs_sommet,police,L,C):
	global SIZE,RAYON,COLOR,bg
	screen.blit(bg,(0,0)) #reset all entitie

	for i in range(nbs_sommet): # affiche les sommets 
		pygame.draw.circle(screen,(255,255,255),L[i],RAYON,width=1)

		image_text = police.render( str(i),1,(255,255,255))
		screen.blit(image_text,(L[i][0]-6,L[i][1]-10))
	
	for i in range(nbs_sommet):
		for j in range(i):
			if C[i][j] != 0:
				pygame.draw.line(screen,COLOR[C[i][j]-1],L[i],L[j])
			

	pygame.display.update()

def main():
	global SIZE,RAYON,COLOR,bg
	pygame.init()
	pygame.display.set_caption("TIPE")
	screen = pygame.display.set_mode((SIZEx,SIZEy))

	screen.blit(bg,(0,0))

	M = [[]] # temps
	H = [[]] # distance a vole d'oiseau
	C = [[]] # affichage des routes, en couleur 1 , 2 , ... 6

	running = True
	L=[] # position des sommets
	nbs_sommet = 0

	ind_arrete = -1
	police = pygame.font.SysFont("monospace",20)
	pygame.display.update()
	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			if event.type == pygame.MOUSEBUTTONDOWN:
				L.append(event.pos)
				nbs_sommet += 1
				if nbs_sommet == 2:
					M = [[0,float('inf')] , [float('inf'),0]]
					H = [[0,distance(L[0],L[1])] , [distance(L[0],L[1]),0]]
					C = [[0,0] , [0,0]]
				else:
					M.append([float('inf') if i != (nbs_sommet-1) else 0 for i in range(nbs_sommet)])
					H.append([float('inf') if i != (nbs_sommet-1) else 0 for i in range(nbs_sommet)])
					C.append([0 if i != (nbs_sommet-1) else 0 for i in range(nbs_sommet)])
					for i in range(nbs_sommet-1):
						M[i].append(float('inf'))
						C[i].append(0)
						dis = distance(L[i],L[-1])
						H[i].append(dis)
						H[-1][i] = dis
					
				print()
				print(L)
				for el in M:
					print(el)
				print("====")
				for el in H:
					print(el)
				print("====")
				for el in C:
					print(el)
				pygame.draw.circle(screen,(255,255,255),event.pos,RAYON,width=1)
				image_text = police.render( str(nbs_sommet-1),1,(255,255,255))
				screen.blit(image_text,(event.pos[0]-6,event.pos[1]-10))
				pygame.display.update()
			if event.type == pygame.KEYDOWN:
				cc = 0
				print(event)
				if event.key == 109: # touche m enleve le dernier sommet
					del L[-1]
					for i in range(nbs_sommet):
						del H[i][-1]
						del M[i][-1]
						del C[i][-1]
					del C[-1]
					del H[-1]
					del M[-1]
					nbs_sommet -= 1
					rendu(screen,event,nbs_sommet,police,L,C)
				if event.key == 108: # touche l , enleve l'arrete selectionner
					s_d = input("sommet du depart de l'arrete ->")
					s_a = input("sommet de l'arrive de l'arrete ->")
					try:
						d = int(s_d)
						a = int(s_a)
						C[a][d] = 0
						C[d][a] = 0
						M[a][d] = float('inf')
						M[d][a] = float('inf')
						rendu(screen,event,nbs_sommet,police,L,C)
					except:
						print("erreur sythaxe")
				if event.key == 107: # touche k , enleve sommet selectionner
					s_s = input("sommet ->")
					try:
						s = int(s_s)
						del L[s]
						for i in range(nbs_sommet):
							del H[i][s]
							del M[i][s]
							del C[i][s]
						del C[s]
						del H[s]
						del M[s]
						nbs_sommet -= 1
						rendu(screen,event,nbs_sommet,police,L,C)
					except:
						print('erreur sythaxe')
				if event.key == 97: #a
					cc = 1
					vitesse = 30
				if event.key == 122: #z
					cc = 2
					vitesse = 50
				if event.key == 101: # e
					cc = 3
					vitesse = 70
				if event.key == 114: #r
					cc = 4
					vitesse = 90
				if event.key == 116: #t
					cc = 5
					vitesse = 110
				if event.key == 121: #y
					cc = 6
					vitesse = 130
				if cc != 0:
					ppos = pygame.mouse.get_pos()
					for i in range(len(L)):
						dis = distance(ppos,L[i])
						if dis <= RAYON:
							if ind_arrete != (-1):
								dd = distance(L[ind_arrete],L[i])
								tps = dd / vitesse
								M[i][ind_arrete] = tps
								M[ind_arrete][i] = tps
								C[i][ind_arrete] = cc
								C[ind_arrete][i] = cc
								pygame.draw.line(screen,COLOR[cc-1],L[ind_arrete],L[i])
								
								print()
								print(L)
								for el in M:
									print(el)
								print("====")
								for el in H:
									print(el)
								ind_arrete = -1
								pygame.draw.circle(screen,(255,255,255),(10,10),RAYON,width=1)
								pygame.display.update()
							else:
								ind_arrete = i
								pygame.draw.circle(screen,(0,0,0),(10,10),RAYON,width=1)
								pygame.display.update()



main()				



