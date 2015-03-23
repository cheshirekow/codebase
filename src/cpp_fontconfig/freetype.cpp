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
 *  @file   src/freetype.cpp
 *
 *  \date   Jul 23, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <fontconfig/fontconfig.h>
#include <fontconfig/fcfreetype.h>
#include <cpp_fontconfig/freetype.h>
#include <cpp_freetype/cpp_freetype.h>


namespace fontconfig {


Char32_t CharIndex(ft::RefPtr<ft::Face> face, Char32_t ucs4)
{
    return FcFreeTypeCharIndex( face.subvert(), ucs4 );
}

RefPtr<CharSet> GetCharSet(ft::RefPtr<ft::Face> face, RefPtr<Blanks> blanks)
{
    return RefPtr<CharSet>(
            FcFreeTypeCharSet(   face.subvert(),
                                blanks.subvert() ) );
}

RefPtr<CharSet> CharSetAndSpacing(ft::RefPtr<ft::Face> face, RefPtr<Blanks> blanks,
        int* spacing)
{
    return RefPtr<CharSet>(
            FcFreeTypeCharSetAndSpacing(
                                face.subvert(),
                                blanks.subvert(), spacing ) );
}

RefPtr<Pattern> Query(const Char8_t* file, int id, RefPtr<Blanks> blanks,
        int* count)
{
    return RefPtr<Pattern>(
            FcFreeTypeQuery(file, id, blanks.subvert(), count ) );
}

RefPtr<Pattern> QueryFace(const ft::RefPtr<ft::Face>& face,
        const Char8_t* file, int id, RefPtr<Blanks> blanks)
{
    return RefPtr<Pattern>(
            FcFreeTypeQueryFace(
                    face.subvert(),
                    file,
                    id,
                    blanks.subvert() ) );
}



}







