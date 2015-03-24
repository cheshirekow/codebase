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
 *  @file   /home/josh/Codes/cpp/mpblocks2/cuda/include/mpblocks/cuda/polynomial/IntList.h
 *
 *  @date   Oct 24, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_INTLIST_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_INTLIST_H_

#include <iostream>
#include <climits>

#include <mpblocks/cuda/polynomial/Printer.h>

namespace   mpblocks {
namespace       cuda {
namespace polynomial {

namespace intlist
{
    struct Terminal;
}

/// compile time integer list (because cuda doesn't understand variadic
/// templates )
/**
 *  Implemented as a sorted linked list of integers
 */
template< int Head, class Tail = intlist::Terminal >
struct IntList{};

namespace intlist
{
    /// evaluate the length of an integer list
    template < class IntList >
    struct size
    {
        enum{ value = 0 };
    };

    template < int Head, class Tail >
    struct size< IntList<Head,Tail> >
    {
        enum{ value = 1 + size< Tail >::value };
    };

    template < int Head >
    struct size< IntList<Head,Terminal> >
    {
       enum{ value = 1 };
    };




    /// evalutes to true if IntList contains val
    template <class IntList, int val>
    struct contains
    {
        enum{ value = false };
    };

    template <bool B, int val, class Tail>
    struct contains_helper
    {
        enum{ value = contains<Tail,val>::value };
    };

    template <int val, class Tail>
    struct contains_helper<true,val,Tail>
    {
        enum{ value = false };
    };

    template <int Head, class Tail, int val>
    struct contains< IntList<Head,Tail>, val >
    {
        enum{ value = contains_helper<(val < Head),val,Tail>::value };
    };

    template <class Tail, int val>
    struct contains< IntList<val,Tail>, val >
    {
        enum{ value = true };
    };



    /// append IntList2 to the end of IntList1
    template <class IntList1, class IntList2>
    struct join
    {
        typedef void result;
    };

    template <int Head1, class Tail1, class IntList2>
    struct join< IntList<Head1,Tail1>, IntList2 >
    {
        typedef IntList<Head1, typename join<Tail1,IntList2>::result> result;
    };

    template <int Head1, class IntList2>
    struct join< IntList<Head1,Terminal>, IntList2 >
    {
        typedef IntList<Head1, IntList2> result;
    };



    /// return a sublist starting at the specified index and having the
    /// specified length
    template< class IntList, int off, int len >
    struct sublist
    {
        //typedef void result;
    };

    template< int off, int len >
    struct sublist< Terminal, off, len >
    {
        typedef Terminal result;
    };

    template< int Head, class Tail, int off, int len >
    struct sublist< IntList<Head,Tail>, off, len >
    {
        typedef typename sublist< Tail, off-1, len >::result result;
    };

    template< int Head, class Tail, int len >
    struct sublist< IntList<Head,Tail>, 0, len >
    {
        typedef IntList<Head, typename sublist<Tail,0,len-1>::result > result;
    };

    template< int Head, class Tail >
    struct sublist< IntList<Head,Tail>, 0, 0 >
    {
        typedef Terminal result;
    };


    /// return the first len sublist
    template < class IntList, int len >
    struct head
    {
        typedef typename sublist<IntList,0,len>::result result;
    };

    /// return the sublist starting at off and ending at the end
    template < class IntList, int off >
    struct tail
    {
        typedef typename sublist<IntList,off,INT_MAX>::result result;
    };



    /// construct an integer list
    template <int v0>
    struct construct1
    {
        typedef IntList<v0,Terminal> result;
    };

    /// construct an integer list
    template <int v0, int v1>
    struct construct2
    {
        typedef IntList<v0,typename construct1<v1>::result> result;
    };

    /// construct an integer list
    template <int v0, int v1, int v2>
    struct construct3
    {
        typedef IntList<v0,typename construct2<v1,v2>::result> result;
    };

    /// construct an integer list
    template <int v0, int v1, int v2, int v3>
    struct construct4
    {
        typedef IntList<v0,typename construct3<v1,v2,v3>::result> result;
    };


    /// more convenient constructor
    template <int v00,
              int v01=INT_MAX, int v02=INT_MAX, int v03=INT_MAX,
              int v04=INT_MAX, int v05=INT_MAX, int v06=INT_MAX,
              int v07=INT_MAX, int v08=INT_MAX, int v09=INT_MAX,
              int v10=INT_MAX, int v11=INT_MAX, int v12=INT_MAX>
    struct construct
    {
        typedef IntList<v00,
                    typename construct<v01,v02,v03,v04,v05,v06,
                                       v07,v08,v09,v10,v11,v12>::result
                > result;
    };

    template <>
    struct construct< INT_MAX,
                      INT_MAX, INT_MAX, INT_MAX,
                      INT_MAX, INT_MAX, INT_MAX,
                      INT_MAX, INT_MAX, INT_MAX,
                      INT_MAX, INT_MAX, INT_MAX >
    {
        typedef Terminal result;
    };



    template <bool B, class T_if, class T_else >
    struct if_else
    {
        typedef T_else result;
    };

    template< class T_if, class T_else >
    struct if_else<true,T_if,T_else>
    {
        typedef T_if result;
    };

namespace merge_sort_detail
{
    template < class IntList1, class IntList2 >
    struct merge{};

    template < int Head1, class Tail1, int Head2, class Tail2 >
    struct merge< IntList<Head1,Tail1>, IntList<Head2,Tail2> >
    {
        typedef typename if_else< (Head1 < Head2),
                    IntList<Head1, typename merge<Tail1, IntList<Head2,Tail2> >::result>,
                    IntList<Head2, typename merge<Tail2, IntList<Head1,Tail1> >::result>
                    >::result result;
    };

    template < class IntList >
    struct merge< Terminal, IntList >
    {
        typedef IntList result;
    };

    template < class IntList >
    struct merge< IntList, Terminal >
    {
        typedef IntList result;
    };

    template < class IntList >
    struct divide_and_conquer
    {
        enum { split_idx = size<IntList>::value / 2 };

        typedef typename head< IntList, split_idx >::result Left;
        typedef typename tail< IntList, split_idx >::result Right;

        typedef typename divide_and_conquer< Left  >::result LeftSorted;
        typedef typename divide_and_conquer< Right >::result RightSorted;

        typedef typename merge< LeftSorted, RightSorted >::result result;
    };

    template < int Head >
    struct divide_and_conquer< IntList<Head,Terminal> >
    {
        typedef IntList<Head,Terminal> result;
    };

} //< namespace merge_sort_detail


    /// performs merge sort. No seriously, merge sort at compile time.
    /// I'm not even joking.
    template < class IntList >
    struct merge_sort
    {
        typedef typename
                merge_sort_detail::divide_and_conquer<IntList>::result result;
    };

    /// strips duplicates
    template < class IntList, int prev=INT_MAX >
    struct strip_dups{};

    template < int Head, class Tail, int prev >
    struct strip_dups< IntList<Head,Tail>, prev >
    {
        typedef IntList<Head,
                    typename strip_dups<Tail,Head>::result > result;
    };

    template < int Head, class Tail>
    struct strip_dups< IntList<Head,Tail>, Head>
    {
        typedef typename strip_dups<Tail,Head>::result result;
    };

    template < int Head, int prev>
    struct strip_dups< IntList<Head,Terminal>, prev>
    {
        typedef IntList<Head,Terminal> result;
    };

    template < int Head>
    struct strip_dups< IntList<Head,Terminal>, Head>
    {
        typedef Terminal result;
    };


    /// computes sum of an integer with an intlist
    template < int val, class IntList >
    struct pair_sum{};

    template <int val, int Head, class Tail>
    struct pair_sum< val, IntList<Head,Tail> >
    {
        typedef IntList< val+Head,
                    typename pair_sum<val,Tail>::result > result;
    };

    template <int val, int Head>
    struct pair_sum< val, IntList<Head,Terminal> >
    {
        typedef IntList< val+Head, Terminal > result;
    };

    /// computes the unsorted allpairs sum of two two sorted lists
    template < class IntList1, class IntList2 >
    struct allpairs_sum_unsorted{};

    template < int Head, class Tail, class IntList2 >
    struct allpairs_sum_unsorted< IntList<Head,Tail>, IntList2 >
    {
        typedef typename join<
                    typename pair_sum<Head,IntList2>::result,
                    typename allpairs_sum_unsorted<Tail,IntList2>::result >::result
                    result;
    };

    template < int Head, class IntList2 >
    struct allpairs_sum_unsorted< IntList<Head,Terminal>, IntList2 >
    {
        typedef typename pair_sum<Head,IntList2>::result result;
    };

    template < class IntList1, class IntList2 >
    struct allpairs_sum
    {
        typedef typename strip_dups<
                    typename merge_sort<
                        typename allpairs_sum_unsorted<
                            IntList1,IntList2
                        >::result
                    >::result
                >::result result;
    };


    template < class IntList1, class IntList2 >
    struct make_union
    {
        typedef typename strip_dups<
                    typename merge_sort<
                        typename join<
                            IntList1,IntList2
                        >::result
                    >::result
                >::result result;
    };


    /// retrieve the i'th element of the list
    template < class IntList, int i >
    struct get{};

    template < int Head, class Tail, int i >
    struct get< IntList<Head,Tail>, i >
    {
        enum{ value = get<Tail,i-1>::value };
    };

    template < int Head, class Tail >
    struct get< IntList<Head,Tail>, 0 >
    {
        enum{ value = Head };
    };

    /// retrieve the max element of the list (linear without short circut
    /// so it works with non-sorted arrays )
    template < class IntList >
    struct max
    {
        enum{ value = -1 };
    };

    template< int Head, class Tail >
    struct max< IntList<Head,Tail> >
    {
        enum{ value = (Head > max<Tail>::value) ? Head : max<Tail>::value };
    };

    template< int Head >
    struct max< IntList<Head,Terminal> >
    {
        enum{ value = Head };
    };


    template <bool Enable, int i, int j>
    struct enabled_range
    {
        typedef Terminal result;
    };

    template< int i, int j >
    struct enabled_range<true,i,j>
    {
        typedef IntList<i, typename enabled_range<true,i+1,j>::result > result;
    };

    template< int i>
    struct enabled_range<true,i,i>
    {
        typedef IntList<i,Terminal> result;
    };

    /// creates an integer list from i to j (including both i and j)
    template <int i, int j>
    struct range
    {
//        typedef IntList<i, typename range<i+1,j>::result > result;
          typedef typename enabled_range< (i<=j), i,j >::result result;
    };

    template <int i>
    struct range<i,i>
    {
        typedef IntList<i, Terminal> result;
    };



} //< namespace intlist


template< int Head, class Tail>
struct Printer< IntList<Head,Tail> >
{
    static std::ostream& print( std::ostream& out )
    {
        out << Head << ",";
        return Printer<Tail>::print(out);
    }
};

template< int Tail>
struct Printer< IntList<Tail,intlist::Terminal> >
{
    static std::ostream& print( std::ostream& out )
    {
        out << Tail;
        return out;
    }
};




} // polynomial
} // cuda
} // mpblocks
















#endif // INTLIST_H_
