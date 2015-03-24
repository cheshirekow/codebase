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
 *  @file   include/mpblocks/polynomial/Product.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_PRODUCT_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_PRODUCT_H_

namespace   mpblocks {
namespace       cuda {
namespace polynomial {


namespace product_detail
{

struct Null;

/// pairs one index of the first polynomial with one index of the second
/// polynomial
template< int A, int B >
struct IdxPair{};

/// compile time map of output monomial to input monomial pairs that contribute
/// to that coefficient
template< int Key, class Value, class Next=Null >
struct Node{};

/// evaluates ordering on nodes by resolving their keys
template< class A, class B >
struct less_than{};

template< int KeyA, class ValueA, class NextA,
          int KeyB, class ValueB, class NextB  >
struct less_than< Node<KeyA,ValueA,NextA>,
                  Node<KeyB,ValueB,NextB> >
{
    enum{ value = (KeyA < KeyB) };
};

template< int KeyA, class ValueA, class NextA  >
struct less_than< Node<KeyA,ValueA,NextA>, Null >
{
    enum{ value = true };
};

template< int KeyA, class ValueA, class NextA  >
struct less_than< Null, Node<KeyA,ValueA,NextA> >
{
    enum{ value = false };
};


/// if ALessThanB is true, then template recurses, otherwise B is inserted
/// in front of A
template< class NodeA, class NodeB, bool ALessThanB >
struct insert_postTest{};

template< int KeyA, class ValueA, class NextA, class NodeB  >
struct insert_postTest< Node<KeyA,ValueA,NextA>, NodeB, true >
{
    typedef Node<KeyA,ValueA,
                typename insert_postTest<
                        NextA,NodeB, less_than<NextA,NodeB>::value
                    >::result
                > result;
};

template< class NodeA, int KeyB, class ValueB >
struct insert_postTest< NodeA, Node<KeyB,ValueB,Null>, false >
{
    typedef Node<KeyB,ValueB,NodeA> result;
};

/// insert a new key/value pair into the map
template< class Root, class Node >
struct insert
{
    typedef typename insert_postTest<
            Root,Node,less_than<Root,Node>::value >::result result;
};

/// insert a Node into an empty map
template< class Node >
struct insert< Null,Node >
{
    typedef Node result;
};


/// inner loop
template< int Coeff1, class Spec2, class Map >
struct inner_loop{};

template< int Coeff1, int Head, class Tail, class Map >
struct inner_loop<Coeff1, IntList<Head,Tail>, Map >
{
    typedef Node< Coeff1+Head, IdxPair<Coeff1,Head>, Null > NewNode;
    typedef typename insert< Map, NewNode >::result         NewMap;
    typedef typename inner_loop<Coeff1,Tail,NewMap>::result result;
};

template< int Coeff1, int Head, class Map >
struct inner_loop<Coeff1, IntList<Head,intlist::Terminal>, Map >
{
    typedef Node< Coeff1+Head, IdxPair<Coeff1,Head>, Null > NewNode;
    typedef typename insert< Map, NewNode >::result         result;
};

/// outer loop
template< class Spec1, class Spec2, class Map >
struct outer_loop{};

template< int Head, class Tail, class Spec2, class Map >
struct outer_loop< IntList<Head,Tail>, Spec2, Map >
{
    typedef typename inner_loop< Head, Spec2, Map  >::result MapA;
    typedef typename outer_loop< Tail, Spec2, MapA >::result result;
};

template< int Head, class Spec2, class Map >
struct outer_loop< IntList<Head,intlist::Terminal>, Spec2, Map >
{
    typedef typename inner_loop< Head, Spec2, Map  >::result result;
};


/// compute product specification
template< class Spec1, class Spec2 >
struct product_spec
{
    typedef typename outer_loop<Spec1,Spec2,Null>::result result;
};


template < class Spec >
struct SpecPrinter
{
    static void print( std::ostream& out )
    {
        out << "ERR, ";
    }
};

template < >
struct SpecPrinter< Null >
{
    static void print( std::ostream& out )
    {
        out << "FINI";
    }
};

template< int Key, int ValA, int ValB, class Next >
struct SpecPrinter< Node<Key,IdxPair<ValA,ValB>,Next> >
{
    static void print( std::ostream& out )
    {
        out << "(" << Key << "," << ValA << "," << ValB << "), ";
        SpecPrinter< Next >::print(out);
    }
};


template< int Key, int ValA, int ValB, class Next >
std::ostream& operator<<( std::ostream& out,
                    SpecPrinter< Node<Key,IdxPair<ValA,ValB>,Next> > printer)
{
    out << "[";
    printer.print(out);
    out << "]";
    return out;
}




} //< namespace product_detail


/// expression template for sum of two matrix expressions
template <typename Scalar,
            class Exp1, class Spec1,
            class Exp2, class Spec2 >
struct Product :
    public RValue< Scalar, Product<Scalar,Exp1,Spec1,Exp2,Spec2>,
                   typename intlist::allpairs_sum<Spec1,Spec2>::result >
{
    Exp1 const& m_A;
    Exp2 const& m_B;

    __host__ __device__
    Product( Exp1 const& A, Exp2 const& B ):
        m_A(A),
        m_B(B)
    {
    }

    __host__ __device__
    Scalar eval( Scalar x )
    {
        return m_A.eval(x) * m_B.eval(x);
    }
};


namespace product_detail
{

template <int i, typename Scalar, class Exp1, class Exp2, class Spec >
struct Summation{};

template <int i, typename Scalar, class Exp1, class Exp2,
            int Key, int ValA, int ValB, class Next >
struct Summation< i, Scalar, Exp1, Exp2, Node<Key,IdxPair<ValA,ValB>,Next> >
{
    __host__ __device__
    static Scalar do_it( const Exp1& exp1, const Exp2& exp2 )
    {
        return Summation<i,Scalar,Exp1,Exp2,Next>::do_it(exp1,exp2);
    }
};

template <int i, typename Scalar, class Exp1, class Exp2,
            int ValA, int ValB, class Next >
struct Summation< i, Scalar, Exp1, Exp2, Node<i,IdxPair<ValA,ValB>,Next> >
{
    __host__ __device__
    static Scalar do_it( const Exp1& exp1, const Exp2& exp2 )
    {
        return get<ValA>(exp1)*get<ValB>(exp2) +
               Summation<i,Scalar,Exp1,Exp2,Next>::do_it(exp1,exp2);
    }
};

template <int i, typename Scalar, class Exp1, class Exp2,
            int Key, int ValA, int ValB >
struct Summation< i, Scalar, Exp1, Exp2, Node<Key,IdxPair<ValA,ValB>,Null> >
{
    __host__ __device__
    static Scalar do_it( const Exp1& exp1, const Exp2& exp2 )
    {
        return 0;
    }
};

template <int i, typename Scalar, class Exp1, class Exp2,
            int ValA, int ValB >
struct Summation< i, Scalar, Exp1, Exp2, Node<i,IdxPair<ValA,ValB>,Null> >
{
    __host__ __device__
    static Scalar do_it( const Exp1& exp1, const Exp2& exp2 )
    {
        return get<ValA>(exp1)*get<ValB>(exp2);
    }
};

} //< namespace product_detail


template <typename Scalar,
            class Exp1, class Spec1,
            class Exp2, class Spec2 >
struct get_spec< Product<Scalar,Exp1,Spec1,Exp2,Spec2> >
{
    typedef typename intlist::allpairs_sum<Spec1,Spec2>::result result;
};

template <int i, typename Scalar,
            class Exp1, class Spec1,
            class Exp2, class Spec2 >
__host__ __device__
Scalar get( const Product<Scalar,Exp1,Spec1,Exp2,Spec2>& exp )
{
    typedef typename product_detail::product_spec<Spec1,Spec2>::result Spec;
    return product_detail::Summation<i,Scalar,Exp1,Exp2,Spec>::do_it(
            exp.m_A, exp.m_B );
}


template <typename Scalar,
            class Exp1, class Spec1,
            class Exp2, class Spec2 >
__host__ __device__
Product<Scalar,Exp1,Spec1,Exp2,Spec2> operator*(
        RValue<Scalar,Exp1,Spec1> const& A,
        RValue<Scalar,Exp2,Spec2> const& B )
{
    return Product<Scalar,Exp1,Spec1,Exp2,Spec2>(
            static_cast<Exp1 const&>(A),
            static_cast<Exp2 const&>(B));
}




} // polynomial
} // cuda
} // mpblocks





#endif // PRODUCT_H_
