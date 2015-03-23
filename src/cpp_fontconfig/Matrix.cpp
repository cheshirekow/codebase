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
 *  @file   src/Matrix.cpp
 *
 *  \date   Jul 20, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <cpp_fontconfig/Matrix.h>
#include <fontconfig/fontconfig.h>
#include <cstring>

namespace fontconfig
{


Matrix::Matrix(const Matrix& other)
{
    memcpy(this,&other,sizeof(Matrix));
}


Matrix& Matrix::operator =(const Matrix& other)
{
    memcpy(this,&other,sizeof(Matrix));
    return *this;
}


void Matrix::init()
{
    xx = yy = 1;
    xy = yx = 0;
}


bool Matrix::equal(const Matrix& other) const
{
    return FcMatrixEqual( this, &other );
}

void Matrix::rotate(double c, double s)
{
    FcMatrixRotate( this, c, s );
}

void Matrix::scale(double sx, double sy)
{
    FcMatrixScale( this, sx, sy );
}

void Matrix::shear(double sh, double sv)
{
    FcMatrixShear( this, sh, sv );
}



Matrix operator*(const Matrix& a, const Matrix& b)
{
    Matrix newMat;
    FcMatrixMultiply( &newMat, &a, &b );
    return newMat;
}

bool operator==(const Matrix& a, const Matrix& b)
{
    return a.equal(b);
}


}// namespace fontconfig
