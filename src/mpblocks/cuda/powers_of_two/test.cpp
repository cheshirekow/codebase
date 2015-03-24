/*
 *  \file   test.cpp
 *
 *  \date   Sep 1, 2011
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */


#include <mpblocks/cuda/powersOfTwo.h>
#include <iostream>

using namespace std;
using namespace mpblocks::cuda;

template <typename T>
void testIsPow2()
{
    for(T x = 2; x < 1024; x*= 2)
    {
        cout.width(5);
        cout << x << " : " << (isPow2(x) ? "true" : "false") << "\n";
    }

    for(T x = 1; x < 20; x++)
    {
        cout.width(5);
        cout << x << " : " << (isPow2(x) ? "true" : "false") << "\n";
    }
}


template <typename T>
void testNearestPow2()
{
    for(T x = 0; x < 20; x++)
    {
        cout.width(5);
        cout << x << " : ";

        cout.width(5);
        cout << prevPow2(x) << "    ";

        cout.width(5);
        cout << nextPow2(x) << "\n";
    }
}


template <typename T>
void testIntPow()
{
    for(T x = 0; x < 5; x++)
    {
        for(T y =0; y < 5; y++)
        {
            cout << "(" << x << "," << y << ") : ";
            cout.width(5);
            cout << intPow(x,y) << "\n";
        }
    }
}


int main(int argc, char** argv)
{
    cout << "Testing isPow2 for unsigned int\n----------------\n";
    testIsPow2<unsigned int>();
    cout << "\nTesting isPow2 for signed int\n----------------\n";
    testIsPow2<int>();

    cout << "Testing prevPow2 and nextPow2 for unsigned int\n----------------\n";
    testNearestPow2<unsigned int>();
    cout << "\nTesting prevPow2 and nextPow2 for signed int\n----------------\n";
    testNearestPow2<int>();

    cout << "Testing intPow for unsigned int\n----------------\n";
    testIntPow<unsigned int>();
    cout << "\nTesting intPow for signed int\n----------------\n";
    testIntPow<int>();


    cout.flush();
    return 0;
}
