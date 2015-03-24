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
 *  @file   mpblocks/simplex_tree/Geometry.hpp
 *
 *  @date   Dec 2, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_GEOMETRY_TREE_GEOMETRY_HPP_
#define MPBLOCKS_GEOMETRY_TREE_GEOMETRY_HPP_


namespace mpblocks     {
namespace simplex_tree {

template <class Traits>
const typename Traits::Format
    Geometry<Traits>::sm_n = std::sqrt( Traits::NDim );

template <class Traits>
void Geometry<Traits>::reset()
{
    m_isSet.reset();
}

template <class Traits>
void Geometry<Traits>::refine( Index_t i, const Point_t& v )
{
    m_v.block(0,i,Traits::NDim,1) = v;
    m_isSet[i] = true;
}

template <class Traits>
void Geometry<Traits>::initFaces( )
{

}





}   // namespace simplex_tree
}   // namespace mpblocks












#endif // GEOMETRY_HPP_
