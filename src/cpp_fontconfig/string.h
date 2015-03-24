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
 *  @file   include/cppfontconfig/string.h
 *
 *  \date   Jul 23, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef CPPFONTCONFIG_STRING_H_
#define CPPFONTCONFIG_STRING_H_

#include <cpp_fontconfig/common.h>

namespace fontconfig {

const unsigned int UTF8_MAX_LEN = 6;

namespace        str {


Char8_t*  copy (const Char8_t* s);

Char8_t*  copyFilename (const Char8_t* s);

Char8_t* plus (const Char8_t* s1, const Char8_t* s2);

void free (Char8_t* s);

//  These are ASCII only, suitable only for pattern element names
bool isUpper(Char8_t c);
bool isLower(Char8_t c);
bool toLower(Char8_t c);

Char8_t* downcase (const Char8_t* s);

int cmpIgnoreCase (const Char8_t* s1, const Char8_t* s2);

int cmp (const Char8_t* s1, const Char8_t* s2);

const Char8_t* strIgnoreCase (const Char8_t* s1, const Char8_t* s2);

const Char8_t* str (const Char8_t* s1, const Char8_t* s2);

int Utf8ToUcs4 (  const Char8_t* src_orig,
                  Char32_t*      dst,
                  int            len);

bool Utf8Len (const Char8_t*    string,
               int              len,
               int*             nchar,
               int*             wchar);

int Ucs4ToUtf8 (Char32_t  ucs4,
                Char8_t   dest[UTF8_MAX_LEN]);

int Utf16ToUcs4 (   const Char8_t*  src_orig,
                    Endian_t        endian,
                    Char32_t*       dst,
                    int             len);       //  in bytes

bool Utf16Len ( const Char8_t*  string,
                Endian_t        endian,
                int             len,        //  in bytes
                int*            nchar,
                int*            wchar);

Char8_t* dirname (const Char8_t* file);

Char8_t* basename (const Char8_t* file);



}
}












#endif // STRING_H_
