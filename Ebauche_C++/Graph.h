#ifndef __GRAPH_H__
#define __GRAPH_H__

#include <vector>
#include <iterator>

using namespace std;

struct dij
{
    int provient;
    int poids;
    bool select;
};



class Infin_int
{   
public:
    Infin_int() : value(0), infinie(false)
    {
    }
    Infin_int(int e_value, bool e_infinie) : value(e_value), infinie(e_infinie)
    {
    }
    bool infinie;
    int value;
    friend Infin_int operator+(Infin_int const& op1,Infin_int const& op2)
    {
        Infin_int nv;
        if ( ( op1.infinie == true ) || (op2.infinie == true) )
        {
            nv.infinie = true;
        }else{
            nv.value = op1.value + op2.value;
            nv.infinie = false;
        }
        return nv; 
    }

    friend bool operator>(Infin_int const& op1,Infin_int const& op2)
    {
        if (op2.infinie == true)
        {
            return false;
        }else
        {
            if (op1.infinie == true)
            {
                return true;
            }else
            {
                return (op1.value > op2.value);
            }
        }
    }   
    
}; 


class Arrete
{
protected:
    int id;
    int poids;
public:
    Arrete(int e_id, int e_poids);
    int get_id();
    int get_poids();
};



class Graph // gestion de la ville !
{
protected:
    std::vector<std::vector<Arrete> > graphset;
    int nbs_sommet;
    int nbs_arc;
    int nbs_passager;
    std::vector<int> passager;
    std::vector<int> destination;
    std::vector<std::vector<Infin_int>> graph_floyd;
public:
    Graph();
    void charge(string nameofgraph);
    std::vector<Arrete> get_adj(int sommet);
    std::vector<Arrete>::iterator get_begin(int sommet);
    std::vector<Arrete>::iterator get_end(int sommet);
    void nv_sommet();
    int get_nbs_passager();
    
    void nv_client(int const& position_passager, int const& position_destination);
    
    void nv_arrete(int depart,int arrive,int poids,bool sens_unique = false);
    
    void best_way_naif( std::vector<vector<int>>const& graph_tsp,int position,int size_choix,vector<bool> choix,int size_chemin,vector<int> chemin, int poids, vector<vector<int>> &final);
    
    std::vector< std::vector<int>> generer_TSP(int sommet_depart);
    /*
        @sortie:
            sommet d'entre [ici][]
            sommet de sortie [][ici]
            [0][i] : les arretes du bus a tout les autres position_destination
            pour n allant de 0 à (passager)
            [2n +1] : les passager
            [2n +2] : les destination
    */
    
    std::vector<dij> algo_dijtra(int sommet_de_depart); // retourne la table de dijtra
    std::vector< std::vector<Infin_int> > algo_floyd();
    int get_nbs_sommet();
    
    void afficher();
    
     
    
    
};

class Simulation : public Graph
{
    Simulation();
    
  
    
    
    
};






#endif