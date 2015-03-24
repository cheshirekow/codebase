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
 *  @file   pblocks/cuda/linalg/Product.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG_PRODUCT_H_
#define MPBLOCKS_CUDA_LINALG_PRODUCT_H_

namespace mpblocks {
namespace cuda     {
namespace linalg   {

/// expression template for product of two matrix expressions
template <typename Scalar, class Exp1, class Exp2>
class Product :
    public RValue<Scalar, Product<Scalar,Exp1,Exp2> >
{
    Exp1 const& m_A;
    Exp2 const& m_B;

    public:
        typedef unsigned int Size_t;

        __device__ __host__
        Product( Exp1 const& A, Exp2 const& B ):
            m_A(A),
            m_B(B)
        {
//            assert( A.cols() == B.rows() );
        }

        /// return the size for a vector
        __device__ __host__
        Size_t size() const
        {
            return ( m_A.rows() * m_B.cols() );
        }

        /// return the rows of a matrix expression
        __device__ __host__
        Size_t rows() const
        {
            return m_A.rows();
        }

        /// return the columns of a matrix expression
        __device__ __host__
        Size_t cols() const
        {
            return m_B.cols();
        }

        /// return the evaluated i'th element of a vector expression
        __device__ __host__
        Scalar operator[]( Size_t i ) const
        {
            return 0;
        }

        /// return the evaluated (i,j)'th element of a matrix expression
        __device__ __host__
        Scalar operator()( Size_t i, Size_t j )const
        {
            Scalar r = 0;
            for(int k=0; k < m_A.cols(); k++ )
                r += m_A(i,k) * m_B(k,j);
            return r;
        }
};


template <typename Scalar, class Exp1, class Exp2>
__device__ __host__
Product<Scalar,Exp1,Exp2> operator*(
        RValue<Scalar,Exp1> const& A, RValue<Scalar,Exp2> const& B )
{
    typedef Product<Scalar,Exp1,Exp2> Product_t;
    return Product_t(   static_cast< Exp1 const& >(A),
                        static_cast< Exp2 const& >(B) );
}



} // linalg
} // cuda
} // mpblocks





#endif // PRODUCT_H_
