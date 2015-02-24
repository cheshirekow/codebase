/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample implemenets bitonic sort and odd-even merge sort, algorithms
 * belonging to the class of sorting networks. 
 * While generally subefficient on large sequences 
 * compared to algorithms with better asymptotic algorithmic complexity
 * (i.e. merge sort or radix sort), may be the algorithms of choice for sorting
 * batches of short- or mid-sized arrays.
 * Refer to the excellent tutorial by H. W. Lang:
 * http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/indexen.htm
 * 
 * Victor Podlozhnyuk, 07/09/2009
 */

#include <cstdlib>
#include <limits>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <mpblocks/cuda.hpp>
#include <mpblocks/cuda/polynomial.h>

void dispatch_test_kernel(float* g_out);
void print_function_attributes();

using namespace mpblocks::cuda;
using namespace mpblocks::cuda::polynomial;


template< class Exp, class Spec >
struct CudaInput;

template< class Exp, int Head, class Tail >
struct CudaInput< Exp, IntList<Head,Tail> >
{
    static void read( float*& in, Exp& exp )
    {
        set<Head>(exp) = *(in++);
        CudaInput<Exp,Tail>::read(in,exp);
    }
};

template< class Exp >
struct CudaInput< Exp, intlist::Terminal >
{
    static void read( float*& in, Exp& exp ){}
};

template< class Spec, class Exp >
void read( float*& in, Exp& exp )
{
    CudaInput<Exp,Spec>::read(in,exp);
}

struct InFloatStream
{
    float* in;

    InFloatStream(float* in):
        in(in)
    {}
};

template< int max, int i >
struct SturmInput
{
    static void read( InFloatStream& in, SturmSequence<float,max>& sturm )
    {
        in >> get<i>(sturm);
        SturmInput<max,i+1>::read(in, sturm);
    }
};

template< int max >
struct SturmInput<max,max>
{
    static void read( InFloatStream& in, SturmSequence<float,max>& sturm )
    {
        in >> get<max>(sturm);
    }
};




template< class Exp >
InFloatStream& operator>>( InFloatStream& in, Exp& exp )
{
    typedef typename get_spec<Exp>::result Spec;
    read<Spec>( in.in, exp );

    return in;
}

template< int max >
__device__
InFloatStream& operator>>( InFloatStream& in, SturmSequence<float,max>& sturm )
{
    SturmInput<max,0>::read(in,sturm);
    return in;
}

__device__
InFloatStream& operator>>( InFloatStream& in, float& val )
{
    val = *(in.in++);
    return in;
}

__device__
InFloatStream& operator>>( InFloatStream& in, int& val )
{
    val =  *(in.in++);
    return in;
}




////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{

    std::cout << "Starting up CUDA context\n";
    cudaSetDevice(0);

    int dev;
    cudaGetDevice(&dev);
    std::cout << "Current device: " << dev << "\n";
    printDeviceReport(std::cout);

    void* vp_out;
    enum{ numFloats = 200 };
    cudaMalloc(&vp_out,numFloats*sizeof(float));
    cudaMemset(vp_out,0,numFloats*sizeof(float));
    float* g_out = static_cast<float*>(vp_out);
    dispatch_test_kernel(g_out);
    cudaDeviceSynchronize();
    float  g_in[numFloats];
    cudaMemcpy( g_in, g_out, numFloats*sizeof(float), cudaMemcpyDeviceToHost );
    
    InFloatStream istream(g_in);

    typedef intlist::construct<0,1,3>::result SpecA;
    typedef intlist::construct<2,4>::result   SpecB;
    typedef intlist::make_union<SpecA,SpecB>::result   UnionAB;
    typedef intlist::allpairs_sum<SpecA,SpecB>::result ProductAB;
    typedef intlist::construct<0>::result              Scalar;
    typedef intlist::make_union<Scalar,SpecB>::result  ScalarB;

    Polynomial<float, SpecA   > pA(0,1,3);
    Polynomial<float, SpecB   > pB(2,4);
    Polynomial<float, UnionAB > pAplusB;
    Polynomial<float, UnionAB > pAminusB;
    Polynomial<float, ProductAB> pAtimesB;
    Polynomial<float, ScalarB >  p1plusB;
    Polynomial<float, SpecB   >  p3timesB;

    istream >> pA;
    istream >> pB;
    istream >> pAplusB;
    istream >> pAminusB;
    istream >> pAtimesB;
    istream >> p1plusB;
    istream >> p3timesB;

    typedef UnionAB SpecC;
    Polynomial<float, SpecC> pC = pA + pB;
    istream >> pC;

    Polynomial<float, intlist::construct<0,2>::result > pD;
    istream >> pD;

    typedef intlist::construct<0,1,3,5>::result SpecE;
    Polynomial<float, SpecE > pE;
    Polynomial<float, SpecE > pEnormalized;
    Polynomial<float, SpecE > pEnegative;
    float   pEVal;

    istream >> pE;
    istream >> pEnormalized;
    istream >> pEnegative;
    istream >> pEVal;

    typedef intlist::construct<3,5,8>::result pF_spec;
    typedef DerivativeSpec<pF_spec,1>::result d1_pF_spec;
    typedef DerivativeSpec<pF_spec,2>::result d2_pF_spec;
    typedef DerivativeSpec<pF_spec,3>::result d3_pF_spec;
    typedef DerivativeSpec<pF_spec,4>::result d4_pF_spec;

    Polynomial<float, pF_spec > pF;
    Polynomial<float, d1_pF_spec > d1pF;
    Polynomial<float, d2_pF_spec > d2pF;
    Polynomial<float, d3_pF_spec > d3pF;
    Polynomial<float, d4_pF_spec > d4pF;

    istream >> pF >> d1pF >> d2pF >> d3pF >> d4pF;

    typedef intlist::construct<0,1,2>::result       SpecNum;
    typedef intlist::construct<0,1>::result         SpecDen;
    typedef QuotientSpec<SpecNum,SpecDen>::result   SpecQuot;
    typedef ScratchSpec<SpecNum,SpecDen>::result    SpecScratch;
    typedef RemainderSpec<SpecNum,SpecDen>::result  SpecRemainder;
    typedef Polynomial<float, SpecNum >             ExpNum;
    typedef Polynomial<float, SpecDen >             ExpDen;
    typedef Polynomial<float, SpecQuot >            ExpQuot;
    typedef Polynomial<float, SpecScratch >         ExpScratch;
    typedef Polynomial<float, SpecRemainder >       ExpRemainder;
    typedef Quotient<float,ExpNum,SpecNum,ExpDen,SpecDen>   Quotient_t;

    ExpNum Num(-10,-9,1);
    ExpDen Den(1,1);
    ExpQuot q;
    ExpRemainder r;

    istream >> Num >> Den >> q >> r;

    typedef intlist::construct<0,1,2,3,4,5>::result SpecG;
    typedef Polynomial<float,SpecG>                 PolyG;
    typedef SpecG   Spec0;
    typedef PolyG   Poly0;
    typedef typename derivative_detail::DerivativeSpec<true,SpecG>::result Spec1;
    typedef Polynomial<float,Spec1> Poly1;
    typedef sturm2_detail::SturmSpec<SpecG,2>::result Spec2;
    typedef Polynomial<float,Spec2> Poly2;
    typedef sturm2_detail::SturmSpec<SpecG,3>::result Spec3;
    typedef Polynomial<float,Spec3> Poly3;
    typedef sturm2_detail::SturmSpec<SpecG,4>::result Spec4;
    typedef Polynomial<float,Spec4> Poly4;
    typedef sturm2_detail::SturmSpec<SpecG,5>::result Spec5;
    typedef Polynomial<float,Spec5> Poly5;

    PolyG pG;
    Poly0 p0;
    Poly1 p1;
    Poly2 p2;
    Poly3 p3;
    Poly4 p4;
    Poly5 p5;

    SturmSequence2< float, SpecG> sG;

    istream >> pG;
//            >> p0
//            >> p1
//            >> p2
//            >> p3
//            >> p4
//            >> p5;

    float zero_crossings;
    istream >> zero_crossings;

    typedef intlist::construct<0,1,2,3>::result SpecH;
    typedef Polynomial<float,SpecH>             PolyH;
    PolyH pH;
    istream >> pH;
    float nH;
    istream >> nH;

    SturmSequence2< float, SpecH> sH;
    istream >> get<0>(sH)
            >> get<1>(sH)
            >> get<2>(sH)
            >> get<3>(sH);
    float v0a,v0b,v0c,v0d,v1a,v1b,v1c,v1d;
    istream >> v0a >> v0b >> v0c >> v0d >> v1a >> v1b >> v1c >> v1d;

    std::cout << "number of bytes: " << istream.in - g_in << "\n";
    print_function_attributes();

    std::cout
      << "\n     pA : " << pA
      << "\n     pB : " << pB
      << "\n  pA+pB : " << pAplusB
      << "\n  pA-pB : " << pAminusB
      << "\n  pA*pB : " << pAtimesB
      << "\n  1 +pB : " << p1plusB
      << "\n  3 *pB : " << p3timesB
      << "\n     pC : " << pC
      << "\n     pD : " << pD
      << "\n     pE : " << pE
      << "\n   |pE| : " << pEnormalized
      << "\n    -pE : " << pEnegative
      << "\n pE(2.5): " << pEVal
      << "\n     pF : " << pF
      << "\n dpf/ds : " << d1pF
      << "\n d2pf/ds: " << d2pF
      << "\n d4pf/ds: " << d3pF
      << "\n d4pf/ds: " << d4pF
      << "\n    Num : " << Num
      << "\n    Den : " << Den
      << "\n    N/D : " << q
      << "\n  remain: " << r
      << "\n      pG: " << pG
//      << "\n      sG: "
//      << "\n          0: " << p0
//      << "\n          1: " << p1
//      << "\n          2: " << p2
//      << "\n          3: " << p3
//      << "\n          4: " << p4
//      << "\n          5: " << p5
      << "\n   zeros: " << zero_crossings
      << "\n      pH: " << pH
      << "\n   zeros: " << nH
      << "\n      sH: "
      << "\n       0: " << get<0>(sH)
      << "\n       1: " << get<1>(sH)
      << "\n       2: " << get<2>(sH)
      << "\n       3: " << get<3>(sH)
      << "\n    vals: "
      << "\n       0: " << v0a << "    " << v1a
      << "\n       1: " << v0b << "    " << v1b
      << "\n       2: " << v0c << "    " << v1c
      << "\n       3: " << v0d << "    " << v1d
      << "\n"
      << "\n";

    return 0;
}
