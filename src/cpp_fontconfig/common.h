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
 *  @file   include/cppfontconfig/common.h
 *
 *  \date   Jul 19, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef CPPFONTCONFIG_COMMON_H_
#define CPPFONTCONFIG_COMMON_H_





#define FCMM_DONT_CONSTRUCT( TYPE ) \
private:                        \
    template <typename T>       \
    TYPE( const T& param )      \
    {                           \
        struct TYPE##_is_an_opaque_type_use_create_method{} obj;    \
        int i = obj;            \
    }                           \


namespace fontconfig
{


const unsigned int MAJOR    = 2;
const unsigned int MINOR    = 10;
const unsigned int REVISION = 0;
const unsigned int VERSION  = ((MAJOR * 10000) + (MINOR * 100) + (REVISION));

/**
 * Current font cache file format version
 * This is appended to the cache files so that multiple
 * versions of the library will peacefully coexist
 *
 * Change this value whenever the disk format for the cache file
 * changes in any non-compatible way.  Try to avoid such changes as
 * it means multiple copies of the font information.
 */
const char* const CACHE_VERSION = "3";


typedef unsigned char   Char8_t;
typedef unsigned short  Char16_t;
typedef unsigned int    Char32_t;
typedef int             Bool_t;
typedef int             Object_t;

/// property keys
const char* const FAMILY          = "family";         ///< String
const char* const STYLE           = "style";          ///< String
const char* const SLANT           = "slant";          ///< Int
const char* const WEIGHT          = "weight";         ///< Int
const char* const SIZE            = "size";           ///< Double
const char* const ASPECT          = "aspect";         ///< Double
const char* const PIXEL_SIZE      = "pixelsize";      ///< Double
const char* const SPACING         = "spacing";        ///< Int
const char* const FOUNDRY         = "foundry";        ///< String
const char* const ANTIALIAS       = "antialias";      ///< Bool (depends)
const char* const HINTING         = "hinting";        ///< Bool (true)
const char* const HINT_STYLE      = "hintstyle";      ///< Int
const char* const VERTICAL_LAYOUT = "verticallayout"; ///< Bool (false)
const char* const AUTOHINT        = "autohint";       ///< Bool (false)

/* GLOBAL_ADVANCE is deprecated. this is simply ignored on freetype 2.4.5 or later */
const char* const GLOBAL_ADVANCE  = "globaladvance";  ///< Bool (true)
const char* const WIDTH           = "width";          ///< Int
const char* const FILE            = "file";           ///< String
const char* const INDEX           = "index";          ///< Int
const char* const FT_FACE         = "ftface";         ///< FT_Face
const char* const RASTERIZER      = "rasterizer";     ///< String
const char* const OUTLINE         = "outline";        ///< Bool
const char* const SCALABLE        = "scalable";       ///< Bool
const char* const SCALE           = "scale";          ///< double
const char* const DPI             = "dpi";            ///< double
const char* const RGBA            = "rgba";           ///< Int
const char* const MINSPACE        = "minspace";       ///< Bool use minimum line spacing
const char* const SOURCE          = "source";         ///< String (deprecated)
const char* const CHARSET         = "charset";        ///< CharSet
const char* const LANG            = "lang";           ///< String RFC 3066 langs
const char* const FONTVERSION     = "fontversion";    ///< Int from 'head' table
const char* const FULLNAME        = "fullname";       ///< String
const char* const FAMILYLANG      = "familylang";     ///< String RFC 3066 langs
const char* const STYLELANG       = "stylelang";      ///< String RFC 3066 langs
const char* const FULLNAMELANG    = "fullnamelang";   ///< String RFC 3066 langs
const char* const CAPABILITY      = "capability";     ///< String
const char* const FONTFORMAT      = "fontformat";     ///< String
const char* const EMBOLDEN        = "embolden";       ///<  Bool - true if emboldening needed
const char* const EMBEDDED_BITMAP = "embeddedbitmap"; ///< Bool - true to enable embedded bitmaps
const char* const DECORATIVE      = "decorative";     ///< Bool - true if style is a decorative variant
const char* const LCD_FILTER      = "lcdfilter";      ///< Int
const char* const NAMELANG        = "namelang";       ///< String RFC 3866 langs

namespace key
{

enum Key
{
    FAMILY,
    STYLE,
    SLANT,
    WEIGHT,
    SIZE,
    ASPECT,
    PIXEL_SIZE,
    SPACING,
    FOUNDRY,
    ANTIALIAS,
    HINTING,
    HINT_STYLE,
    VERTICAL_LAYOUT,
    AUTOHINT,
    GLOBAL_ADVANCE,
    WIDTH,
    FILE,
    INDEX,
    FT_FACE,
    RASTERIZER,
    OUTLINE,
    SCALABLE,
    SCALE,
    DPI,
    RGBA,
    MINSPACE,
    SOURCE,
    CHARSET,
    LANG,
    FONTVERSION,
    FULLNAME,
    FAMILYLANG,
    STYLELANG,
    FULLNAMELANG,
    CAPABILITY,
    FONTFORMAT,
    EMBOLDEN,
    EMBEDDED_BITMAP,
    DECORATIVE,
    LCD_FILTER,
    NAMELANG,
};

}

typedef key::Key Key_t;



// these "3" should probably not be written by hand, but I'm trying to
// avoid any macros... this can be done with boost::mpl
// (http://stackoverflow.com/questions/4693819/c-template-string-concatenation)
// but it seems kind of ridiculous to add that dependency to save 6 seconds
// of maintanance
const char* const CACHE_SUFFIX    = ".cache-3";
const char* const DIR_CACHE_FILE  = "fonts.cache-3";
const char* const USER_CACHE_FILE = ".fonts.cache-3";

// Adjust outline rasterizer
const char* const CHAR_WIDTH  = "charwidth";  ///< Int
const char* const CHAR_HEIGHT = "charheight"; ///< Int
const char* const MATRIX      = "matrix";     ///< FcMatrix



namespace weight
{

const unsigned int THIN         = 0;
const unsigned int EXTRALIGHT   = 40;
const unsigned int ULTRALIGHT   = EXTRALIGHT;
const unsigned int LIGHT        = 50;
const unsigned int BOOK         = 75;
const unsigned int REGULAR      = 80;
const unsigned int NORMAL       = REGULAR;
const unsigned int MEDIUM       = 100;
const unsigned int DEMIBOLD     = 180;
const unsigned int SEMIBOLD     = DEMIBOLD;
const unsigned int BOLD         = 200;
const unsigned int EXTRABOLD    = 205;
const unsigned int ULTRABOLD    = EXTRABOLD;
const unsigned int BLACK        = 210;
const unsigned int HEAVY        = BLACK;
const unsigned int EXTRABLACK   = 215;
const unsigned int ULTRABLACK   = EXTRABLACK;

}


namespace slant
{

const unsigned int ROMAN    = 0;
const unsigned int ITALIC   = 100;
const unsigned int OBLIQUE  = 110;

}

namespace width
{

const unsigned int ULTRACONDENSED   = 50;
const unsigned int EXTRACONDENSED   = 63;
const unsigned int CONDENSED        = 75;
const unsigned int SEMICONDENSED    = 87;
const unsigned int NORMAL           = 100;
const unsigned int SEMIEXPANDED     = 113;
const unsigned int EXPANDED         = 125;
const unsigned int EXTRAEXPANDED    = 150;
const unsigned int ULTRAEXPANDED    = 200;

}

namespace rgba
{

const unsigned int UNKNOWN  = 0;
const unsigned int RGB      = 1;
const unsigned int BGR      = 2;
const unsigned int VRGB     = 3;
const unsigned int VBGR     = 4;
const unsigned int NONE     = 5;

}

namespace hint
{

const unsigned int NONE     = 0;
const unsigned int SLIGHT   = 1;
const unsigned int MEDIUM   = 2;
const unsigned int FULL     = 3;

}

namespace lcd
{

const unsigned int NONE     = 0;
const unsigned int DEFAULT  = 1;
const unsigned int LIGHT    = 2;
const unsigned int LEGACY   = 3;

}

namespace type
{

enum Type {
    Void,
    Integer,
    Double,
    String,
    Bool,
    Matrix,
    CharSet,
    FTFace,
    LangSet
};

}

typedef type::Type Type_t;

namespace match
{

enum MatchKind
{
    Pattern,
    Font,
    Scan
};

}

typedef match::MatchKind MatchKind_t;

namespace qual
{

enum Qual
{
    Any,
    All,
    First,
    NotFirst
};


}

typedef qual::Qual Qual_t;


namespace op
{

enum Op
{
    Integer, Double, String, Matrix, Range, Bool, CharSet, LangSet,
    Nil,
    Field, Const,
    Assign, AssignReplace,
    PrependFirst, Prepend, Append, AppendLast,
    Quest,
    Or, And, Equal, NotEqual,
    Contains, Listing, NotContains,
    Less, LessEqual, More, MoreEqual,
    Plus, Minus, Times, Divide,
    Not, Comma, Floor, Ceil, Round, Trunc,
    Invalid
};

}

typedef op::Op Op_t;


namespace lang
{

enum Result
{
    Equal                 = 0,
    DifferentCountry      = 1,
    DifferentTerritory    = 1,
    DifferentLang         = 2
};

}

typedef lang::Result LangResult_t;


namespace result
{

enum Result
{
    Match,
    NoMatch,
    TypeMismatch,
    NoId,
    OutOfMemory
};

}

typedef result::Result Result_t;


namespace setname
{

enum SetName
{
    System      = 0,
    Application = 1
};

}

typedef setname::SetName SetName_t;

namespace endian
{
    enum Endian
    {
        Big,
        Little
    };
}

typedef endian::Endian Endian_t;

}













#endif // COMMON_H_
