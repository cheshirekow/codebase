/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of cppfontconfig.
 *
 *  cppfontconfig is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  cppfontconfig is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with cppfontconfig.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file   include/cppfontconfig/Matrix.h
 *
 *  \date   Jul 20, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef FONTCONFIG_MATRIX_H_
#define FONTCONFIG_MATRIX_H_

#include <fontconfig/fontconfig.h>

namespace fontconfig
{

/// wraps FcMatrix with it's member methods
/**
 *  Matrix inherits from matrix (lowercase) which is a struct with the same
 *  definition as FcMatrix. Furthermore Matrix does not declare any virtual
 *  members so it is binary compatable with FcMatrix and can be casted
 *  without reserve
 */
struct Matrix:
    public FcMatrix
{
    public:
        /// Default constructor, uninitialized memory
        Matrix(){}

        /// copy constructor
        /**
         *  stack-allocates a new matrix and copies the contents from \p other
         */
        Matrix( const Matrix& other );

        /// assignment operator, copies the values of the matrix in other
        Matrix& operator=( const Matrix& other );

        /// initializes the matrix to be the identity matrix
        void init();

        /// returns true if the two matrices are equal
        bool equal( const Matrix& other ) const;

        /// rotate a matrix
        void rotate(double c, double s);

        /// scale a matrix
        void scale(double sx, double sy);

        /// shear a matrix
        void shear(double sh, double sv);
};

/// multiply matricies
Matrix operator*( const Matrix& a, const Matrix& b );

/// test for equality
bool operator==( const Matrix& a, const Matrix& b );


} // namespace fontconfig

#endif // MATRIX_H_
