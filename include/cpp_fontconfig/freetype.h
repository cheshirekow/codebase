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
 *  @file   include/cppfontconfig/freetype.h
 *
 *  \date   Jul 23, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef CPPFONTCONFIG_FREETYPE_H_
#define CPPFONTCONFIG_FREETYPE_H_

#include <fontconfig/fontconfig.h>
#include <cpp_fontconfig/cpp_fontconfig.h>

namespace freetype
{
    template <typename T> class RefPtr;
    class Face;
}

namespace fontconfig {

namespace ft = freetype;

/// map Unicode to glyph id
/**
 *  Maps a Unicode char to a glyph index. This function uses information from
 *  several possible underlying encoding tables to work around broken fonts.
 *  As a result, this function isn't designed to be used in performance
 *  sensitive areas; results from this function are intended to be cached by
 *  higher level functions.
 */
Char32_t CharIndex(ft::RefPtr<ft::Face> face, Char32_t ucs4);

/// compute Unicode coverage
/**
 *  Scans a FreeType face and returns the set of encoded Unicode chars. This
 *  scans several encoding tables to build as complete a list as possible. If
 *  'blanks' is not 0, the glyphs in the font are examined and any blank
 *  glyphs not in 'blanks' are not placed in the returned FcCharSet.
 */
RefPtr<CharSet> GetCharSet(ft::RefPtr<ft::Face> face, RefPtr<Blanks> blanks);

/// compute Unicode coverage and spacing type
/**
 *  Scans a FreeType face and returns the set of encoded Unicode chars. This
 *  scans several encoding tables to build as complete a list as possible. If
 *  'blanks' is not 0, the glyphs in the font are examined and any blank
 *  glyphs not in 'blanks' are not placed in the returned FcCharSet. spacing
 *  receives the computed spacing type of the font, one of FC_MONO for a font
 *  where all glyphs have the same width, FC_DUAL, where the font has glyphs
 *  in precisely two widths, one twice as wide as the other, or FC_PROPORTIONAL
 *  where the font has glyphs of many widths.
 */
RefPtr<CharSet> CharSetAndSpacing(ft::RefPtr<ft::Face>,
                            RefPtr<Blanks> blanks, int *spacing);

/// compute pattern from font file (and index)
/**
 *  Constructs a pattern representing the 'id'th font in 'file'. The number of
 *  fonts in 'file' is returned in 'count'.
 */
RefPtr<Pattern> Query(const Char8_t *file, int id,
                    RefPtr<Blanks> blanks, int *count);

/// compute pattern from FT_Face
/**
 *  Constructs a pattern representing 'face'. 'file' and 'id' are used solely
 *  as data for pattern elements (FC_FILE, FC_INDEX and sometimes FC_FAMILY).
 */
RefPtr<Pattern> QueryFace(const ft::RefPtr<ft::Face>& face,
                    const Char8_t *file, int id,
                    RefPtr<Blanks> blanks);


}














#endif // FREETYPE_H_
