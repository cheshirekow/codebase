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
 *  @file   include/mpblocks/cuda/linalg2/iostream.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_LINALG2_IOSTREAM_H_
#define MPBLOCKS_CUDA_LINALG2_IOSTREAM_H_

#include <iostream>
#include <iomanip>

namespace mpblocks {
namespace cuda     {
namespace linalg2  {
namespace ostream  {

template <typename Scalar>
struct OutputIterator
{
    virtual ~OutputIterator(){}

    virtual Scalar              value()      =0;
    virtual OutputIterator*     next()       =0;
    virtual bool                finished()   =0;
    virtual bool                rowFinished()=0;
};

/// iterates over elements
template <typename Scalar, Size_t ROWS, Size_t COLS, Size_t i, Size_t j, class Exp>
class ElementIterator :
    public OutputIterator<Scalar>
{
    private:
        Exp const& m_A;

    public:
        ElementIterator( Exp const& A ):
            m_A(A)
        {}

        virtual ~ElementIterator(){}

        virtual Scalar  value()
        {
            return m_A.Exp::template me<i,j>();
        }

        virtual OutputIterator<Scalar>* next()
        {
            OutputIterator<Scalar>* next =
                    new ElementIterator<Scalar,ROWS,COLS,i,j+1,Exp>(m_A);
            delete this;
            return next;
        }

        virtual bool finished()
        {
            return false;
        }

        virtual bool rowFinished()
        {
            return false;
        }
};


/// specialization for last element in a row
template <typename Scalar, Size_t ROWS, Size_t COLS, Size_t i, class Exp>
class ElementIterator<Scalar,ROWS,COLS,i,COLS,Exp> :
    public OutputIterator<Scalar>
{
    private:
        Exp const& m_A;

    public:
        ElementIterator( Exp const& A ):
            m_A(A)
        {}

        virtual ~ElementIterator(){}

        virtual Scalar  value()
        {
            return m_A.Exp::template me<i,COLS-1>();
        }

        virtual OutputIterator<Scalar>* next()
        {
            OutputIterator<Scalar>* next =
                    new ElementIterator<Scalar,ROWS,COLS,i+1,0,Exp>(m_A);
            delete this;
            return next;
        }

        virtual bool finished()
        {
            return false;
        }

        virtual bool rowFinished()
        {
            return true;
        }
};

/// specialization for one past the end of the array
template <typename Scalar, Size_t ROWS, Size_t COLS, class Exp>
class ElementIterator<Scalar,ROWS,COLS,ROWS,0,Exp> :
    public OutputIterator<Scalar>
{
    private:
        Exp const& m_A;

    public:
        ElementIterator( Exp const& A ):
            m_A(A)
        {}

        virtual ~ElementIterator(){}

        virtual Scalar  value()
        {
            return m_A.Exp::template me<ROWS-1,COLS-1>();
        }

        virtual OutputIterator<Scalar>* next()
        {
            OutputIterator<Scalar>* next =
                    new ElementIterator<Scalar,ROWS,COLS,ROWS,0,Exp>(m_A);
            delete this;
            return next;
        }

        virtual bool finished()
        {
            return true;
        }

        virtual bool rowFinished()
        {
            return false;
        }
};

}





template <typename Scalar, Size_t ROWS, Size_t COLS, class Exp>
std::ostream& operator<<(
            std::ostream& out,
            RValue<Scalar,ROWS,COLS,Exp> const& M)
{
    ostream::OutputIterator<Scalar>* iter =
        new ostream::ElementIterator<Scalar,ROWS,COLS,0,0,Exp>(
                                        static_cast<Exp const&>(M) );
    while( !iter->finished() )
    {
        if( iter->rowFinished() )
            out << "\n";
        else
            out << std::setw(8)
                        << std::setiosflags(std::ios::fixed)
                        << std::setprecision(4) << iter->value();
        iter = iter->next();
    }

    delete iter;
    return out;
}






} // linalg
} // cuda
} // mpblocks









#endif // IOSTREAM_H_
