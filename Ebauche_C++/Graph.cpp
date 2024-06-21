#include "Graph.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

Simulation::Simulation()
{
    
}

vector<int> split(string input)
{
    vector<int> tab;
	istringstream ss(input);
	string token;
	int tmp;

	while(std::getline(ss, token, ' ')) {
	    //cout << "toke: "<<token << endl;
	    tmp = stoi(token);
		tab.push_back(tmp);
	}

	return tab;
}

Arrete::Arrete(int e_id, int e_poids): id(e_id),poids(e_poids)
{
}

Graph::Graph(): nbs_sommet(0),nbs_arc(0),nbs_passager(0),passager(vector<int>() ), destination( vector<int>() ), graph_floyd( vector< vector <Infin_int>>())
{
}

void Graph::charge(string nameofgraph)
{
    string readd;
    ifstream file;
    file.open(nameofgraph);
    string deli1 = ":";
    string token;
    string value;
    int ind;
    vector<int> contant;
    int ll = 0;
    bool debloque_floyd = false;
    while(getline(file,readd))
    {
        ind = readd.find(deli1);
        token = readd.substr(0, ind);
        value = readd.substr(ind+1,readd.length());
        //cout << token<<endl;
        //cout << value << endl;
        if(token.compare("sommet") == 0)
        {
            for(int i=0;i<stoi(value); i++)
            {
                nv_sommet();
                graph_floyd.push_back(vector<Infin_int>());
            }
        }
        if(token.compare("nv_arrete") == 0)
        {
            contant = split(value);
            nv_arrete( contant[0],contant[1],contant[2]);
            contant.clear();
        }
        
        if(token.compare("end") == 0)
        {
            debloque_floyd = false;
        }
        
        if(debloque_floyd)
        {
            cout << value<<endl;
            contant = split(value);
            cout << "size contant "<<contant.size()<<endl;
            ll = 0;
            for(int jj : contant)
            {
                Infin_int mm;
                mm.value = jj;
                graph_floyd[ll].push_back(mm);
                ll++;
            }
            contant.clear();
        }
        
        if(token.compare("floyd") == 0)
        {
            debloque_floyd = true;
        }
        
        
        
        
    }
    for(vector<Infin_int> vvec : graph_floyd)
    {
        for(Infin_int out : vvec)
        {
            cout << out.value << " ";
        }cout << endl;
    }
}


int Arrete::get_id()
{
    return id;
}
int Arrete::get_poids()
{
    return poids;
}


std::vector<Arrete>::iterator Graph::get_begin(int sommet)
{
    return graphset[sommet].begin();
}

std::vector<Arrete>::iterator Graph::get_end(int sommet)
{
    return graphset[sommet].end();
}

std::vector<dij> Graph::algo_dijtra(int sommet_de_depart)
{   
    
    dij d;
    d.select = false;
    d.poids = -1;
    d.provient = sommet_de_depart;
    vector<dij> dijtra_list(nbs_sommet,d);
    
    dijtra_list[sommet_de_depart].poids = 0;// inait
    dijtra_list[sommet_de_depart].select = true;

    int poids_total = 0;
    int sommet_actuel = sommet_de_depart;
    int best_poids;
    int best_sommet;
    for (int k = 0; k < nbs_sommet-1; k++) // pour les changement de sommet actuel;
    {
        //cout << "sommet actuel :"<<sommet_actuel<<endl;

        best_poids = -1;
        best_sommet = -1;
        for (int i = 0; i < (int)graphset[sommet_actuel].size(); i++)// parcour la arc adjacent au sommet actuel
        {
            if (dijtra_list[ graphset[sommet_actuel][i].get_id() ].select == false )// verifie que l'arc ne soit pas select
            {
                //cout<< "arc pointé : "<< graphset[sommet_actuel][i].get_id()<<endl;
                if ( ( poids_total + graphset[sommet_actuel][i].get_poids() < dijtra_list[ graphset[sommet_actuel][i].get_id() ].poids ) || 
                ( dijtra_list[ graphset[sommet_actuel][i].get_id() ].poids == -1 ) )
                {
                    cout << graphset[sommet_actuel][i].get_poids() + poids_total <<endl;
                    dijtra_list[ graphset[sommet_actuel][i].get_id() ].poids = poids_total + graphset[sommet_actuel][i].get_poids();
                    dijtra_list[ graphset[sommet_actuel][i].get_id() ].provient = sommet_actuel;         
                }  
            }      
        }
        // selection du meilleur poids ds la dijtra list
        for (int i = 0; i < nbs_sommet; i++)
        {
            if ( ( dijtra_list[i].select == false ) && ( dijtra_list[i].poids != -1 ) )
            {
                if ( ( best_poids > dijtra_list[i].poids ) || ( best_poids == -1 ) )
                {
                    best_poids = dijtra_list[i].poids;
                    best_sommet = i;
                    //cout << "best poids : "<<best_poids<<endl;
                    //cout << "best sommet : "<<best_sommet<<endl;
                }
            }
        }

        poids_total = best_poids;
        //cout << "poids total : "<< poids_total<<endl;
        if (dijtra_list[best_sommet].poids == -1)   // nouveau, on met de la ou il provient sinon on garde son ancienne provenace
        {
            dijtra_list[best_sommet].provient = sommet_actuel;
        }
        dijtra_list[best_sommet].poids = best_poids;
        dijtra_list[best_sommet].select = true;
        
        
        /*
        cout << "dijtra list : "<<endl;
        for (int m = 0; m < nbs_sommet; m++)
        {       
            cout << "poids : "<<dijtra_list[m].poids << " proviant : " <<dijtra_list[m].provient << " select : "<< dijtra_list[m].select<< endl;
        }
        */
        


        sommet_actuel = best_sommet; // changement de sommet
        
    }
    
    
    return dijtra_list;


}





std::vector<std::vector<Infin_int>> Graph::algo_floyd()
{
    vector< vector<Infin_int> > matrice(nbs_sommet,vector<Infin_int>(nbs_sommet, Infin_int(0,true) ) );
    for (int i = 0; i < nbs_sommet; i++) // init matrice diagonale
    {
        matrice[i][i] = Infin_int(0,false);
    }
    for (int i = 0; i < nbs_sommet; i++) // init matrice poids
    {
        for (int k = 0; k < (int)graphset[i].size(); k++)
        {
            matrice[i][graphset[i][k].get_id()] = Infin_int(graphset[i][k].get_poids() , false);
        }
    }
    
    for (int k = 0; k < nbs_sommet; k++)
    {
        for (int i = 0; i < nbs_sommet; i++)
        {
            for (int j = 0; j < nbs_sommet; j++)
            {
                
                if(matrice[i][j] > ( matrice[i][k] + matrice[k][j] ) )
                {
                    matrice[i][j] = ( matrice[i][k] + matrice[k][j] ); 
                }
            }   
        }
    }
    return matrice;
}


std::vector<Arrete> Graph::get_adj(int sommet)
{
    return graphset[sommet];
}
int Graph::get_nbs_sommet()
{
    return nbs_sommet;
}

int Graph::get_nbs_passager()
{
    return nbs_passager;
}

void Graph::nv_sommet()
{   
    nbs_sommet += 1;
    graphset.push_back(vector<Arrete>());
}

void Graph::nv_arrete(int depart,int arrive,int poids,bool sens_unique)
{   
    graphset[depart].push_back(Arrete(arrive,poids));
    if (sens_unique == false)
    {
        graphset[arrive].push_back(Arrete(depart,poids));
    }
}

void Graph::afficher()
{
    for (int i = 0; i < (int)graphset.size(); i++)
    {    
        cout << i << ":" << " ";
        for (int k = 0; k < (int)graphset[i].size(); k++)
        {
            cout << graphset[i][k].get_id() << " ";
        }
        cout << endl;
    }
}

void Graph::nv_client(int const& position_passager, int const& position_destination)
{
    destination.push_back(position_destination);
    passager.push_back(position_passager);
    nbs_passager++;
}

std::vector< std::vector<int>> Graph::generer_TSP(int sommet_depart)
{
    // bus de depart au autre position !
    cout <<"passa : "<< nbs_passager<<endl;
    int size = nbs_passager*2 +1;
    cout << size << endl;
    std::vector<std::vector<int>> tsp_result(size,std::vector<int>(size,0) );
    /*
    for(int i = 0 ; i< (nbs_passager*2 +1) ;i++)
    {
        tsp_result.push_back(vector<int>())
    }
    */
    
    tsp_result[0][0] = 0;
    
    for(int i = 0; i<(nbs_passager); i++)// iter le nbs de passager
    {
        tsp_result[0][2*i +1] = graph_floyd[sommet_depart][passager[i]].value;
        tsp_result[0][2*i +2] = graph_floyd[sommet_depart][destination[i]].value;
    }
    for(int i = 0; i<nbs_passager; i++)
    {
        tsp_result[2*i +1][0] = graph_floyd[passager[i]][sommet_depart].value;
        tsp_result[2*i +2][0] = graph_floyd[destination[i]][sommet_depart].value;
    }
    
    
    for(int i = 0; i<nbs_passager; i++)// on iter les passager
    {
        for(int k = 0; k<nbs_passager;k++) // les autre passager et destination
        {
            tsp_result[2*i +1][2*k +1] = graph_floyd[passager[i]][passager[k]].value;
            tsp_result[2*i +1][2*k +2] = graph_floyd[passager[i]][destination[k]].value;
        }
    }
    for(int i = 0; i<nbs_passager; i++)// on iter les destination
    {
        for(int k = 0; k<nbs_passager;k++) // les autre passager et destination
        {
            tsp_result[2*i +2][2*k +1] = graph_floyd[destination[i]][passager[k]].value;
            tsp_result[2*i +2][2*k +2] = graph_floyd[destination[i]][destination[k]].value;
        }
    }
    return tsp_result;
    
    
    
}

void Graph::best_way_naif(std::vector<vector<int>>const& graph_tsp,int position,int size_choix,vector<bool> choix,int size_chemin,vector<int> chemin, int poids, vector<vector<int>>& final)
{
    vector<int> tmp_chemin;
    vector<bool> tmp_choix;
    int tmp_poids;
    for( int out : chemin)
    {
        tmp_chemin.push_back(out);
    }
    for( int out : choix)
    {
        tmp_choix.push_back(out);
    }

    bool rien = true;
    for(int i = 0; i<size_choix ; i++)
    {
        if(choix[i] == true)
        {
            
            //cout << "size chemin "<<size_chemin<<endl;
            if(size_chemin == 0  )
            {
                tmp_poids = graph_tsp[0][i+1];
            }else{
                tmp_poids = poids + graph_tsp[position+1][i+1];
            }
            
            rien = false;
            tmp_choix[i] = false;// on verouille notre case
            
            if(i % 2 == 0) // c'est un passager, on deblogue sa destination
            {
                tmp_choix[i+1] = true;
                
            }
            tmp_chemin.push_back(i);
            /*
            cout << "pos "<<position<<endl;
            cout << "select" << i<<endl;;
            

            cout << "tmp chemin " << endl;
            for(int out : tmp_chemin)
            {
                cout << out<< " ";
            }
            cout << endl;
            cout << "tmp choix " << endl;
            for(bool out : tmp_choix)
            {
                cout << out<< " ";
            }
            cout << endl;
            cout << "poids"<<endl;
            cout << poids<< endl;
            cout << "tmp poids"<<endl;
            cout << tmp_poids<< endl;
            cout << endl;
            */
            best_way_naif(graph_tsp,i,size_choix,tmp_choix,(size_chemin+1), tmp_chemin, tmp_poids, final );
            tmp_chemin.clear();
            tmp_choix.clear();
           for( int out : chemin)
            {
                tmp_chemin.push_back(out);
            }
            for( int out : choix)
            {
                tmp_choix.push_back(out);
            }
        }
        
        //cout << "puis"<<endl;
    }
    if(rien == true)
    {
        vector<int> tmprr(size_chemin+1);
        tmprr[0] = poids;
        for(int i = 0; i<size_chemin ; i++)
        {
            tmprr[i+1] = chemin[i];
        }
        final.push_back(tmprr);
        
        
        return;
    }
    
    
    
    
}



