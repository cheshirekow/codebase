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
 *  @file   pblocks/cuda/linalg2/Product.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG2_PRODUCT_H_
#define MPBLOCKS_CUDA_LINALG2_PRODUCT_H_

namespace mpblocks {
namespace cuda     {
namespace linalg2  {


namespace product  {


/// returns the result of the i,j element of the product
template <typename Scalar,
            Size_t ROWS, Size_t COLS,   Size_t COMMON,
            Size_t i,    Size_t j,      Size_t k,
            class Exp1, class Exp2>
struct Iterator
{
    __device__ __host__
    static Scalar result( const Exp1& A, const Exp2& B )
    {
        return
            ( A.Exp1::template me<i,k>() *
              B.Exp2::template me<k,j>() ) +
              Iterator<Scalar,ROWS,COLS,COMMON,i,j,k+1,Exp1,Exp2>::result(A,B);
    }
};

/// specialization for past the end iterator
template <typename Scalar,
            Size_t ROWS, Size_t COLS,   Size_t COMMON,
            Size_t i,    Size_t j,
            class Exp1, class Exp2>
struct Iterator<Scalar,ROWS,COLS,COMMON,i,j,COMMON,Exp1,Exp2>
{
    __device__ __host__
    static Scalar result( const Exp1& A, const Exp2& B )
    {
        return 0;
    }
};


}

/// expression template for product expressions
template <typename Scalar,
                Size_t ROWS, Size_t COLS, Size_t COMMON,
                class Exp1, class Exp2>
class Product:
    public RValue< Scalar, ROWS, COLS, Product< Scalar, ROWS, COLS, COMMON, Exp1, Exp2> >
{
    Exp1 const& m_A;
    Exp2 const& m_B;

    public:
        __device__ __host__
        Product( Exp1 const& A, Exp2 const& B ):
            m_A(A),
            m_B(B)
        {}

        /// return the evaluated i'th element of a vector expression
        template< Size_t i >
        __device__ __host__
        Scalar ve() const
        {
            if( COLS == 1 )
                return product::
                        Iterator<Scalar,ROWS,COLS,COMMON,i,0,0,Exp1,Exp2>
                        ::result(m_A,m_B);
            else
                return product::
                        Iterator<Scalar,ROWS,COLS,COMMON,0,i,0,Exp1,Exp2>
                        ::result(m_A,m_B);
        }

        /// return the evaluated (j,i)'th element of a matrix expression
        template< Size_t i, Size_t j >
        __device__ __host__
        Scalar me() const
        {
            return product::Iterator<Scalar,ROWS,COLS,COMMON,i,j,0,Exp1,Exp2>
                        ::result(m_A,m_B);
        }
};




template <typename Scalar,
                Size_t ROWS, Size_t COMMON, Size_t COLS,
                class Exp1, class Exp2>
__device__ __host__ inline
Product< Scalar, ROWS, COLS, COMMON, Exp1, Exp2 >
    operator*( RValue<Scalar,ROWS,COMMON,Exp1> const& A,
               RValue<Scalar,COMMON,COLS,Exp2> const& B )
{
    return Product< Scalar, ROWS, COLS, COMMON, Exp1, Exp2 >(
                    static_cast< Exp1 const& >(A),
                    static_cast< Exp2 const& >(B) );
}





} // linalg
} // cuda
} // mpblocks





#endif // PRODUCT_H_
