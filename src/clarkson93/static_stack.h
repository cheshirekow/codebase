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
#ifndef MPBLOCKS_CLARKSON93_STATICSTACK_H_
#define MPBLOCKS_CLARKSON93_STATICSTACK_H_

#include <mpblocks/clarkson93.h>
#include <vector>
#include <type_traits>

namespace   mpblocks {
namespace clarkson93 {



template <typename T, typename SetEnum = void >
class Stack:
    public std::vector<T>
{
    public:
        typedef std::vector<T> Base;
        typedef unsigned int uint;

    private:
        SetEnum m_setBit;

    public:
        Stack(SetEnum setBit = (SetEnum)0):
            m_setBit(setBit)
        {
        }

        ~Stack(){ clear(); }

        void setBit( SetEnum bit )
        {
            m_setBit = bit;
        }

        T  pop()
        {
            assert( Base::size() > 0);
            T returnMe = Base::back();
            returnMe.removeFrom(m_setBit);
            Base::pop_back();
            return returnMe;
        }

        template <class... Args>
        void push( Args&&... args )
        {
            Base::emplace_back( args... );
            Base::back().addTo(m_setBit);
        }

        void clear()
        {
            size_t n = Base::size();
            for(size_t i=0; i < n; i++)
                (*this)[i].removeFrom(m_setBit);

            Base::clear();
        }

        bool isMember( T& obj )
        {
            return obj.isMemberOf( m_setBit );
        }
};





template <typename T, typename SetEnum >
class Stack< T*, SetEnum >:
    public std::vector<T*>
{
    public:
        typedef unsigned int uint;
        typedef std::vector<T*> Base;

    private:
        SetEnum m_setBit;

    public:
        Stack(SetEnum setBit = (SetEnum)0):
            m_setBit(setBit)
        {
        }

        ~Stack(){ clear(); }

        void setBit( SetEnum bit )
        {
            m_setBit = bit;
        }

        T*  pop()
        {
            assert( Base::size() > 0);
            T* returnMe = Base::back();
            returnMe->removeFrom(m_setBit);
            Base::pop_back();
            return returnMe;
        }

        void push(T* obj)
        {
            obj->addTo(m_setBit);
            Base::push_back(obj);
        }

        void clear()
        {
            size_t n = Base::size();
            for(size_t i=0; i < n; i++)
                (*this)[i]->removeFrom(m_setBit);

            Base::clear();
        }

        bool isMember( T* obj )
        {
            return obj->isMemberOf( m_setBit );
        }
};




template <typename T >
struct Stack<T,void>:
    public std::vector<T>
{
    typedef std::vector<T> Base;

    T  pop()
    {
        assert( Base::size() > 0);
        T returnMe = Base::back();
        Base::pop_back();
        return returnMe;
    }
};



} // namespace clarkson93
} // namespace mpblocks







#endif // STATICSTACK_H_
