/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of mpblocks.
 *
 *  mpblocks is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  mpblocks is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with mpblocks.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file   /home/josh/Codes/cpp/builds/mpblocks2/triangulation_Release/src/exp_regularity/main.cpp
 *
 *  @date   Sep 25, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */


#include <algorithm>
#include <Eigen/Dense>
#include <map>
#include "config.h"


struct Simplex
{
    bool isBoundary;         //< true if a boundary simplex
    std::map< int, int > map;  //< vertex pointer -> neighbor pointer
    double               cr;   //< circumradius
    double               V;    //< lebesgue measure, i.e. volume

    Simplex( int ndim )
    {
        reset();
    }

    void reset( )
    {
        isBoundary = false;
        map.clear();
        cr  = 0;
        V   = 0;
    }
};

struct SimplexKey
{
    int     S;      ///< simplex id
    int     idx;    ///< index of flip facet
    double  dr;     ///< change in circumradius
};


void reset( Simplex& s )
{
    s.reset();
}

template <int NDim>
void reset( Eigen::Matrix<double,NDim,1>& v )
{
    v.fill(0);
}


template <class T>
class Store
{
    std::vector< T >   m_storage;
    std::vector< int > m_freeIndices;

    public:
        void init(int size)
        {
            m_storage.clear();
            m_freeIndices.clear();
            m_storage.resize(size);
            m_freeIndices.reserve(size);
            clear();
        }

        void clear()
        {
            m_freeIndices.clear();
            int size = m_storage.size();
            for( size-=1 ; size > 0; size++ )
                m_freeIndices.push_back(size);
        }

        int alloc()
        {
            int idx = m_freeIndices.back();
            m_freeIndices.pop_back();
            return idx;
        }

        void free(int idx)
        {
            reset( m_storage[idx] );
            m_freeIndices.push_back(idx);
        }

        T& deref(int idx)
        {
            return m_storage[idx];
        }

        T& operator[](int idx)
        {
            return m_storage[idx];
        }
};

constexpr int factoral( int n )
{
    return n == 2 ? 2 : n * factoral(n-1);
}


template <int NDim>
void split( Store<Simplex>& S,
            Store< Eigen::Matrix<double,NDim,1> >& V,
            int is, int iv )
{
    Simplex& s = S.deref(is);
    std::map<int,int> new_map;
    for( auto& pair : s.map )
        new_map[pair.first] = S.alloc();
    for( auto& pair : new_map )
    {
        int iv_old = pair.first;
        Simplex& s_new = S.deref(pair.second);
        std::copy_if( new_map.begin(), new_map.end(),
                      std::inserter( s_new.map, s_new.map.end() ),
                      [iv_old]( std::pair<int,int>& pair ){ return pair.first != iv_old; }
                    );
        s_new.map[iv] = pair.second;

        // compute the measure of the simplex
        Eigen::Matrix<double,NDim,NDim> M;
        Eigen::Matrix<double,NDim,1> v0 = V[iv];
        int col = 0;
        for( auto& pair : s.map )
        {
            if( pair.first != iv_old )
            {
              Eigen::Matrix<double,NDim,1> vi = V[pair.second];
                M.col(col++) = vi - v0;
            }
        }
        s_new.V = M.PivFullLUHouseholder().determinant()/factoral(NDim);

    }
}


int main(int argc, char** argv)
{

}







