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
 *  @file   cuda/powersOfTwo.h
 *
 *  @date   Feb 28, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  inline functions for fast math involving powers of two
 *
 *  These functions aren't antything special, it just makes the code a lot
 *  easier to read instead of trying to decode all these bit operations
 */

/**
 *  @defgroup powers_of_two  Math in Powers of Two
 *
 *  Contains inline template functions with meaningful names that use bitwise
 *  operators to do the actual math
 */

#ifndef MPBLOCKS_CUDA_POWERS_OF_TWO_H
#define MPBLOCKS_CUDA_POWERS_OF_TWO_H


#include <climits>


namespace mpblocks {
namespace     cuda {




/// returns true if the parameter is an exact power of two
/**
 *  \ingroup powers_of_two
 *
 * As an example:
 *  \verbatim
    2 : true
    4 : true
    8 : true
   16 : true
    1 : true
    2 : true
    3 : false
    4 : true
    5 : false
    6 : false
    7 : false
    8 : true
\endverbatim
 */
template <typename T>
inline bool isPow2(T x)
{
    return ((x&(x-1))==0);
}




/// returns the smallest power of two that is not less than \p x
/**
 *  \ingroup powers_of_two
 *
 *  If x is a power of two, then it returns x. If it is not a power of two, it
 *  returns the next higher one.
 *
 *  For example:
 *  \verbatim
    x    prev     next
 ------ ------   ------
    0 :     0        0
    1 :     1        1
    2 :     2        2
    3 :     2        4
    4 :     4        4
    5 :     4        8
    6 :     4        8
    7 :     4        8
    8 :     8        8
\endverbatim
 */
template <typename T>
inline T nextPow2( T x )
{
    if (x == 0)
        return 0;
    --x;
    for(unsigned int i=1; i <= sizeof(T)*CHAR_BIT; i++)
        x |= x >> i;
    return ++x;
}




/// returns the largest power of two that is not greater than \p x
/**
 *
 *  If \p x is a power of two, then it returns x, otherwise, it returns the
 *  next lower one
 *
 *  For example:
 *  \verbatim
    x    prev     next
 ------ ------   ------
    0 :     0        0
    1 :     1        1
    2 :     2        2
    3 :     2        4
    4 :     4        4
    5 :     4        8
    6 :     4        8
    7 :     4        8
    8 :     8        8
\endverbatim
 */
template <typename T>
inline T prevPow2( T x )
{
    if(x == 0)
        return 0;

    unsigned int i=0;
    for(;x > 0; x >>= 1)
        i++;
    return 1 << (i-1);
}




/// returns \p x to the power of \p p
/**
 *  \ingroup powers_of_two
 *
 *  For Example
 *  \verbatim
(x,y) :   x^y
----------------
(1,0) :     1
(1,1) :     1
(1,2) :     1
(1,3) :     1
(1,4) :     1
(2,0) :     1
(2,1) :     2
(2,2) :     4
(2,3) :     8
(2,4) :    16
(3,0) :     1
(3,1) :     3
(3,2) :     9
(3,3) :    27
(3,4) :    81
(4,0) :     1
(4,1) :     4
(4,2) :    16
(4,3) :    64
(4,4) :   256
\endverbatim
 */
template <typename T>
inline T intPow( T x, T p)
{
    if (p == 0) return 1;
    if (p == 1) return x;
    return x * intPow<T>(x, p-1);
}




/// returns 2 to the power of \p x (2^x)
/**
 *  \ingroup powers_of_two
 */
template <typename T>
inline T twoPow( T x )
{
    return 0x01 << x;
}




/// if \p x is a power of two then it returns the log of \p x with base 2
/**
 *  \ingroup powers_of_two
 *
 *  In other words if \f$ x = 2^y \f$ then this function returns \f$ y \f$. If
 *  \p x is not in fact a power of two, then the output is unspecified (though
 *  I guess it's similar to prevPow2
 */
template <typename T>
inline T log2( T x )
{
    if(!x)
    {
        return 0;
    }

    else
    {
        T log2x = 0;
        while(  (x & 1) == 0 )
        {
            x >>= 1;
            log2x++;
        }
        return log2x;
    }
}




/// returns x*2 (using bit shift)
/**
 *  \ingroup powers_of_two
 */
template <typename T>
inline T times2( const T& x )
{
    return x << 1;
}




/// returns x/2 (using bit shift), if x is odd then the result is floor(x/2)
/**
 *  \ingroup powers_of_two
 */
template <typename T>
inline T divideBy2( const T& x )
{
    return x >> 1;
}




/// integer divide with round up
/**
 *  \ingroup powers_of_two
 *
 *  returns the smallest integer not less than x/y (i.e. if x/y is a whole
 *  number then it returns x/y, otherwise it returns ceil(x/y) )
 */
template <typename T>
inline T intDivideRoundUp(T x, T y)
{
    return (x+y-1)/y;
}




/// returns true if the number is odd
/**
 *  \ingroup powers_of_two
 */
template <typename T>
inline T isOdd(T x)
{
    return ((x & 0x01) == 0x01);
}



/// returns true if the number is even
/**
 *  \ingroup powers_of_two
 */
template <typename T>
inline T isEven(T x)
{
    return ((x & 0x01) == 0x00);
}




/// returns the value of the specified bit
/**
 *  \ingroup powers_of_two
 */
template <unsigned int I, typename T>
inline T getBit(T x)
{
    return x & (0x01 << I);
}



/// returns x/y if x and y are both powers of two, otherwise the result is
/// undefined
/**
 *  \ingroup powers_of_two
 */
template <typename T>
inline T dividePow2(T x, T y)
{
    return twoPow(log2(x) - log2(y));
}


} // namespace cuda
} // namespace mpblocks


#endif //#ifndef POWERS_OF_TWO_H
