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
 *  @file   include/mpblocks/polynomial/polyval.h
 *
 *  @date   Jan 15, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_POLYVAL_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_POLYVAL_H_

namespace   mpblocks {
namespace       cuda {
namespace polynomial {


namespace polyval_detail
{
    template <class Scalar, class Exp, int idx, int max>
    struct Helper
    {
        __host__ __device__
        static Scalar eval( const Exp& exp, const Scalar& s, Scalar sn  )
        {
            return get<idx>(exp)*sn
                    + Helper<Scalar,Exp,idx+1,max>::eval(exp,s,sn*s);
        }
    };

    template <class Scalar, class Exp, int max>
    struct Helper< Scalar, Exp, max, max>
    {
        __host__ __device__
        static Scalar eval( const Exp& exp, const Scalar& s, Scalar sn  )
        {
            return get<max>(exp)*sn;
        }
    };
}

/// evaluate a polynomial
template <typename Scalar, class Exp, class Spec, typename Scalar2>
__host__ __device__
Scalar polyval( const RValue<Scalar,Exp,Spec>& exp, Scalar2 x )
{
    enum{ max_idx = intlist::max<Spec>::value };
    return polyval_detail::Helper<Scalar,Exp,0,max_idx>::eval(
            static_cast<const Exp&>(exp),Scalar(x),1);
}

} // namespace polynomial
} // namespace cuda
} // namespace mpblocks














#endif // POLYVAL_H_
