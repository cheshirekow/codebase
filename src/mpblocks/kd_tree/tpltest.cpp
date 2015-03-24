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
 *  @file   /home/josh/Codes/cpp/mpblocks2/kdtree/test/tpltest.cpp
 *
 *  @date   Nov 21, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#include <sigc++/sigc++.h>
#include <iostream>


template <class C,
          typename T0,
          T0 (C::*Fn)(void) >
class Breakout0
{
    public:
        typedef Breakout0<C,T0,Fn>   This_t;

    private:
        C* m_obj;
        sigc::signal<void,T0> m_signal;

    public:
        Breakout0(C* obj):
            m_obj(obj)
        {

        }

        void call()
        {
            m_signal( ((*m_obj).*(Fn))() );
        }

        sigc::signal<void,T0> signal()
        {
            return m_signal;
        }

        sigc::slot<void> slot()
        {
            return sigc::mem_fun(*this,&This_t::call);
        }
};

template <class C,
          typename T0,
          typename T1,
          T0 (C::*Fn)(T1) >
class Breakout1
{
    public:
        typedef Breakout1<C,T0,T1,Fn>   This_t;

    private:
        C* m_obj;
        sigc::signal<void,T0> m_signal;

    public:
        Breakout1(C* obj):
            m_obj(obj)
        {

        }

        void call(T1 p1)
        {
            m_signal( ((*m_obj).*(Fn))(p1) );
        }

        sigc::signal<void,T0> signal()
        {
            return m_signal;
        }

        sigc::slot<void,T1> slot()
        {
            return sigc::mem_fun(*this,&This_t::call);
        }
};




template <class C,
          void (C::*Fn)(void) >
class Breakout0<C,void,Fn>
{
    public:
        typedef Breakout0<C,void,Fn>   This_t;

    private:
        C* m_obj;

    public:
        Breakout0(C* obj):
            m_obj(obj)
        {

        }

        void call()
        {
            ((*m_obj).*(Fn))();
        }

        sigc::slot<void> slot()
        {
            return sigc::mem_fun(*this,&This_t::call);
        }
};







class Foo
{
    public:
        void voidFn()
        {
            std::cout << "voidFn" << std::endl;
        }

        int  intFn()
        {
            std::cout << "intFn" << std::endl;
            return 0;
        }

        int  intFn1(int i)
        {
            std::cout << "intFn1: " << i << std::endl;
            return 0;
        }
};



class FooWrap
{
    private:
        Foo m_foo;

    public:
        Breakout0<Foo,void,&Foo::voidFn>     voidFn;
        Breakout0<Foo,int,&Foo::intFn>       intFn;
        Breakout1<Foo,int,int,&Foo::intFn1>  intFn1;

        FooWrap():
            voidFn(&m_foo),
            intFn(&m_foo),
            intFn1(&m_foo)
        {}
};




int main(int argc, char** argv)
{
    FooWrap wrap;
    wrap.voidFn.slot()();
    wrap.intFn.slot()();
    wrap.intFn1.slot()(1);
    return 0;
}
