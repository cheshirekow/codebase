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
 *  @file   src/string.cpp
 *
 *  \date   Jul 23, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */


#include <cpp_fontconfig/string.h>
#include <fontconfig/fontconfig.h>







namespace fontconfig {
namespace        str {



Char8_t* copy(const Char8_t* s)
{
    return FcStrCopy( s );
}

Char8_t* copyFilename(const Char8_t* s)
{
    return FcStrCopyFilename( s );
}

Char8_t* plus(const Char8_t* s1, const Char8_t* s2)
{
    return FcStrPlus( s1, s2 );
}

void free(Char8_t* s)
{
    return FcStrFree( s );
}

bool isUpper(Char8_t c)
{
    return (0101 <= (c) && (c) <= 0132);
}

bool isLower(Char8_t c)
{
    return (0141 <= (c) && (c) <= 0172);
}

bool toLower(Char8_t c)
{
    return isUpper(c) ? (c) - 0101 + 0141 : (c);
}

Char8_t* downcase(const Char8_t* s)
{
    return FcStrDowncase(s);
}

int cmpIgnoreCase(const Char8_t* s1, const Char8_t* s2)
{
    return FcStrCmpIgnoreCase(s1,s2);
}

int cmp(const Char8_t* s1, const Char8_t* s2)
{
    return FcStrCmp(s1,s2);
}

const Char8_t* strIgnoreCase(const Char8_t* s1, const Char8_t* s2)
{
    return FcStrStrIgnoreCase(s1,s2);
}

const Char8_t* str(const Char8_t* s1, const Char8_t* s2)
{
    return FcStrStr(s1,s2);
}

int Utf8ToUcs4(const Char8_t* src_orig, Char32_t* dst, int len)
{
    return FcUtf8ToUcs4(src_orig,dst,len);
}

bool Utf8Len(const Char8_t* string, int len, int* nchar, int* wchar)
{
    return FcUtf8Len(string,len,nchar,wchar);
}

int Ucs4ToUtf8(Char32_t ucs4, Char8_t dest[UTF8_MAX_LEN])
{
    return FcUcs4ToUtf8(ucs4,dest);
}

int Utf16ToUcs4(const Char8_t* src_orig, Endian_t endian, Char32_t* dst,
        int len)
{
    return FcUtf16ToUcs4(src_orig,(FcEndian)endian,dst,len);
}

bool Utf16Len(const Char8_t* string, Endian_t endian, int len, int* nchar,
        int* wchar)
{
    return FcUtf16Len(string, (FcEndian)endian, len, nchar, wchar);
}

Char8_t* dirname(const Char8_t* file)
{
    return FcStrDirname(file);
}

Char8_t* basename(const Char8_t* file)
{
    return FcStrBasename(file);
}


}}
