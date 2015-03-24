/*
 *  \file   bitonicSort.cu
 *
 *  \date   Sep 3, 2011
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <mpblocks/cuda.hpp>
#include <mpblocks/cuda/polynomial.h>
#include <mpblocks/cuda/polynomial/divide.h>
#include <mpblocks/cuda/polynomial/SturmSequence3.h>


void dispatch_test_kernel(float* g_out);

using namespace mpblocks::cuda;
using namespace mpblocks::cuda::polynomial;



template< class Exp, class Spec >
struct CudaOutput;

template< class Exp, int Head, class Tail >
struct CudaOutput< Exp, IntList<Head,Tail> >
{
    __device__
    static void write( float*& out, const Exp& exp )
    {
        *(out++) = get<Head>(exp);
        CudaOutput<Exp,Tail>::write(out,exp);
    }
};

template< class Exp >
struct CudaOutput< Exp, intlist::Terminal >
{
    __device__
    static void write( float*& out, const Exp& exp ){}
};

template< class Spec, class Exp >
__device__
void write( float*& out, const Exp& exp )
{
    CudaOutput<Exp,Spec>::write(out,exp);
}


struct FloatStream
{
    float* out;

    __device__
    FloatStream(float* out):
        out(out)
    {}
};

template< int max, int i >
struct SturmOutput
{
    __device__
    static void write( FloatStream& out, SturmSequence<float,max>& sturm )
    {
        out << get<i>(sturm);
        SturmOutput<max,i+1>::write(out, sturm);
    }
};

template< int max >
struct SturmOutput<max,max>
{
    __device__
    static void write( FloatStream& out, SturmSequence<float,max>& sturm )
    {
        out << get<max>(sturm);
    }
};











template< class Exp >
__device__
FloatStream& operator<<( FloatStream& out, const Exp& exp )
{
    typedef typename get_spec<Exp>::result Spec;
    write<Spec>( out.out, exp );

    return out;
}

template< int max >
__device__
FloatStream& operator<<( FloatStream& out, SturmSequence<float,max>& sturm )
{
    SturmOutput<max,0>::write(out,sturm);
    return out;
}

__device__
FloatStream& operator<<( FloatStream& out, float val )
{
    *(out.out++) = val;
    return out;
}

__device__
FloatStream& operator<<( FloatStream& out, int val )
{
    *(out.out++) = val;
    return out;
}



/// verify functionality of polynomial library
__global__ void p_test( float* out )
{
    int mem_block = 100;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out += idx*mem_block;

    FloatStream ostream(out);

    typedef intlist::construct<0,1,3>::result SpecA;
    typedef intlist::construct<2,4>::result   SpecB;

    Polynomial<float, SpecA > pA(0,1,3);
    Polynomial<float, SpecB > pB(2,4);

    ostream << pA;
    ostream << pB;
    ostream << (pA + pB);
    ostream << (pA - pB);
    ostream << (pA * pB);
    ostream << 1.0f + pB;
    ostream << 3.0f * pB;

    typedef intlist::construct<0,1,2,3,4>::result SpecC;
    Polynomial<float, SpecC> pC = pA + pB;
    ostream << pC;

    using namespace device_param_key;
    using namespace device_coefficient_key;
    Polynomial<float, intlist::construct<0,2>::result >
        pD = 1.2f * (s^_0) + 1.35f * (s^_2);
    ostream << pD;

    Polynomial<float, intlist::construct<0,1,3,5>::result > pE;
    pE << 0.23, 1.25, 2.3, 0.12;

    ostream << pE;
    ostream << normalized(pE);
    ostream << -pE;
    ostream << polyval( pE, 2.5 );

    typedef intlist::construct<3,5,8>::result pF_spec;
    typedef DerivativeSpec<pF_spec,1>::result d1_pF_spec;
    typedef DerivativeSpec<pF_spec,2>::result d2_pF_spec;
    typedef DerivativeSpec<pF_spec,3>::result d3_pF_spec;
    typedef DerivativeSpec<pF_spec,4>::result d4_pF_spec;

    Polynomial<float, pF_spec > pF(1,1,1);
    Polynomial<float, d1_pF_spec > d1pF = d_dx(pF);
    Polynomial<float, d2_pF_spec > d2pF = d_ds<2>(pF);
    Polynomial<float, d3_pF_spec > d3pF = d_ds<3>(pF);
    Polynomial<float, d4_pF_spec > d4pF = d_ds<4>(pF);

    ostream << pF << d1pF << d2pF << d3pF << d4pF;

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

    divide<float,ExpNum,SpecNum,ExpDen,SpecDen>(Num,Den,q,r);
    ostream << Num << Den << q << r;


    typedef intlist::construct<0,1,2,3,4,5>::result SpecG;
    typedef Polynomial<float,SpecG>                 PolyG;
//    typedef SpecG   Spec0;
//    typedef PolyG   Poly0;
//    typedef typename derivative_detail::DerivativeSpec<true,SpecG>::result Spec1;
//    typedef Polynomial<float,Spec1> Poly1;
//    typedef sturm2_detail::SturmSpec<SpecG,2>::result Spec2;
//    typedef Polynomial<float,Spec2> Poly2;
//    typedef sturm2_detail::SturmSpec<SpecG,3>::result Spec3;
//    typedef Polynomial<float,Spec3> Poly3;
//    typedef sturm2_detail::SturmSpec<SpecG,4>::result Spec4;
//    typedef Polynomial<float,Spec4> Poly4;
//    typedef sturm2_detail::SturmSpec<SpecG,5>::result Spec5;
//    typedef Polynomial<float,Spec5> Poly5;

    PolyG pG( 2, -10, -20, 0, 5, 1 );
    ostream << pG;
//
//    Poly0 p0 = normalized( pG );
////    ostream << p0;
//
//    Poly1 p1 = normalized( d_dx(p0) );
////    ostream << p1;
//
//    Poly2 p2,p2temp;
//    mod(p0,p1,p2temp);
//    p2 = normalized( p2temp );
////    ostream << p2;
//
//    Poly3 p3,p3temp;
//    mod(p1,p2,p3temp);
//    p3 = normalized( p3temp );
////    ostream << p3;
//
//    Poly4 p4,p4temp;
//    mod(p2,p3,p4temp);
//    p4 = normalized( p4temp );
////    ostream << p4;
//
//    Poly5 p5,p5temp;
//    mod(p3,p4,p5temp);
//    p5 = normalized( p5temp );
////    ostream << p5;
//
//    ostream << p0 << p1 << p2 << p3 << p4 << p5;
//
//    int     count0      = 0;
//    float   val0 = polyval(p0,-10);
//    float   val1 = polyval(p1,-10);
//    if( sturm_detail::sgn(val1) )
//    {
//        if( sturm_detail::sgn(val1) != sturm_detail::sgn(val0) )
//            count0++;
//        val0 = val1;
//    }
//
//    val1 = polyval(p2,-10);
//    if( sturm_detail::sgn(val1) )
//    {
//        if( sturm_detail::sgn(val1) != sturm_detail::sgn(val0) )
//            count0++;
//        val0 = val1;
//    }
//
//    val1 = polyval(p3,-10);
//    if( sturm_detail::sgn(val1) )
//    {
//        if( sturm_detail::sgn(val1) != sturm_detail::sgn(val0) )
//            count0++;
//        val0 = val1;
//    }
//
//    val1 = polyval(p4,-10);
//    if( sturm_detail::sgn(val1) )
//    {
//        if( sturm_detail::sgn(val1) != sturm_detail::sgn(val0) )
//            count0++;
//        val0 = val1;
//    }
//
//    val1 = polyval(p5,-10);
//    if( sturm_detail::sgn(val1) )
//    {
//        if( sturm_detail::sgn(val1) != sturm_detail::sgn(val0) )
//            count0++;
//        val0 = val1;
//    }
//
//    ostream << count0;
//
//    int count1 = 0;
//    val0 = polyval(p0,10);
//    val1 = polyval(p1,10);
//    if( sturm_detail::sgn(val1) )
//    {
//        if( sturm_detail::sgn(val1) != sturm_detail::sgn(val0) )
//            count1++;
//        val0 = val1;
//    }
//
//    val1 = polyval(p2,10);
//    if( sturm_detail::sgn(val1) )
//    {
//        if( sturm_detail::sgn(val1) != sturm_detail::sgn(val0) )
//            count1++;
//        val0 = val1;
//    }
//
//    val1 = polyval(p3,10);
//        if( sturm_detail::sgn(val1) )
//    {
//        if( sturm_detail::sgn(val1) != sturm_detail::sgn(val0) )
//            count1++;
//        val0 = val1;
//    }
//
//    val1 = polyval(p4,10);
//    if( sturm_detail::sgn(val1) )
//    {
//        if( sturm_detail::sgn(val1) != sturm_detail::sgn(val0) )
//            count1++;
//        val0 = val1;
//    }
//
//    val1 = polyval(p5,10);
//    if( sturm_detail::sgn(val1) )
//    {
//        if( sturm_detail::sgn(val1) != sturm_detail::sgn(val0) )
//            count1++;
//        val0 = val1;
//    }
//
//    ostream << count1;
    ostream << CountZeros<5>::count_zeros(pG,-10.0f,10.0f);

/// SturmSequence2 has register spilling
//    SturmSequence2< float, SpecG> sG;
////    assign( get<0>(sG), pG );
////    get<0>(sG) = pG;
//
//    construct( sG, pG );
//    ostream << get<0>(sG);
//    ostream << get<1>(sG);
//    ostream << get<2>(sG);
//    ostream << get<3>(sG);
//    ostream << get<4>(sG);
//    ostream << get<5>(sG);
//    ostream << signChanges( sG, -10.0f );
//    ostream << sG.signChanges(-10);
//    ostream << sG.signChanges(-5);
//    ostream << sG.signChanges(-4);
//    ostream << sG.signChanges(-3);
//    ostream << sG.signChanges(-2);
//    ostream << sG.signChanges(-1);
//    ostream << sG.signChanges(0);
//    ostream << sG.signChanges(1);
//    ostream << sG.signChanges(2);
//    ostream << sG.signChanges(5);
//    ostream << sG.signChanges(10);


    typedef intlist::construct<0,1,2,3>::result SpecH;
    typedef Polynomial<float,SpecH>             PolyH;
    PolyH pH(-4.0281f,-6.0294f,-0.0708f,-0.1060f);
    ostream << pH;
    ostream << CountZeros<3>::count_zeros(pH,-100.0f,100.0f);

    SturmSequence2< float, SpecH> sH;
    construct(sH,pH);
    ostream << get<0>(sH)
            << get<1>(sH)
            << get<2>(sH)
            << get<3>(sH);
    ostream << polyval( get<0>(sH), 0.0 )
            << polyval( get<1>(sH), 0.0 )
            << polyval( get<2>(sH), 0.0 )
            << polyval( get<3>(sH), 0.0 )
            << polyval( get<0>(sH), 1.0 )
            << polyval( get<1>(sH), 1.0 )
            << polyval( get<2>(sH), 1.0 )
            << polyval( get<3>(sH), 1.0 );

////    assign( get<0>(sG), pG );
////    get<0>(sG) = pG;

}


void dispatch_test_kernel(float* g_out)
{
    p_test<<<1,1>>>(g_out);
}

void print_function_attributes()
{
    FuncAttributes attr;
    attr.getFrom( p_test );

    std::cout << "p_test attributes: "
              << "\n    binary version : " << attr.binaryVersion
              << "\n       ptx version : " << attr.ptxVersion
              << "\n         const size: " << attr.constSizeBytes
              << "\n         local size: " << attr.localSizeBytes
              << "\n        shared size: " << attr.sharedSizeBytes
              << "\n    num 32bit regs : " << attr.numRegs
              << "\n max threads/block : " << attr.maxThreadsPerBlock
              << "\n";

}
