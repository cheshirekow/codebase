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
 *  @file   /home/josh/Codes/cpp/mpblocks2/cuda/include/mpblocks/cuda/polynomial/assign.h
 *
 *  @date   Oct 26, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_ASSIGN_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_ASSIGN_H_

namespace   mpblocks {
namespace       cuda {
namespace polynomial {


template< bool Enable, typename Scalar, class Spec, class Exp, int idx >
struct Assign
{
    __host__ __device__  __forceinline__
    static void step( Polynomial<Scalar,Spec>& p, const Exp& exp ){}
};


template< typename Scalar, class Spec, class Exp, int idx >
struct Assign< true, Scalar, Spec, Exp, idx >
{
    __host__ __device__  __forceinline__
    static void step( Polynomial<Scalar,Spec>& p, const Exp& exp )
    {
        set<idx>(p) = get<idx>(exp);
    }
};

template< typename Scalar, class Spec, class Exp >
__host__ __device__  __forceinline__
void assign( Polynomial<Scalar,Spec>& p, const Exp& exp )
{
    Assign< intlist::contains<Spec,0>::value, Scalar, Spec, Exp, 0 >::step(p,exp);
    Assign< intlist::contains<Spec,1>::value, Scalar, Spec, Exp, 1 >::step(p,exp);
    Assign< intlist::contains<Spec,2>::value, Scalar, Spec, Exp, 2 >::step(p,exp);
    Assign< intlist::contains<Spec,3>::value, Scalar, Spec, Exp, 3 >::step(p,exp);
    Assign< intlist::contains<Spec,4>::value, Scalar, Spec, Exp, 4 >::step(p,exp);
    Assign< intlist::contains<Spec,5>::value, Scalar, Spec, Exp, 5 >::step(p,exp);
    Assign< intlist::contains<Spec,6>::value, Scalar, Spec, Exp, 6 >::step(p,exp);
    Assign< intlist::contains<Spec,7>::value, Scalar, Spec, Exp, 7 >::step(p,exp);
    Assign< intlist::contains<Spec,8>::value, Scalar, Spec, Exp, 8 >::step(p,exp);
    Assign< intlist::contains<Spec,9>::value, Scalar, Spec, Exp, 9 >::step(p,exp);
};







} // polynomial
} // cuda
} // mpblocks










#endif // ASSIGN_H_
