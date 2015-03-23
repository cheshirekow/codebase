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
 *  @file   src/TypeMap.cpp
 *
 *  \date   Jul 23, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <cpp_fontconfig/TypeMap.h>

#define FCMM_INIT_KEY(KEY)  \
const char* const TypeMap<key::KEY>::object = KEY;   \

namespace fontconfig
{

FCMM_INIT_KEY(FAMILY);
FCMM_INIT_KEY(STYLE);
FCMM_INIT_KEY(SLANT);
FCMM_INIT_KEY(WEIGHT);
FCMM_INIT_KEY(SIZE);
FCMM_INIT_KEY(ASPECT);
FCMM_INIT_KEY(PIXEL_SIZE);
FCMM_INIT_KEY(SPACING);
FCMM_INIT_KEY(FOUNDRY);
FCMM_INIT_KEY(ANTIALIAS);
FCMM_INIT_KEY(HINTING);
FCMM_INIT_KEY(HINT_STYLE);
FCMM_INIT_KEY(VERTICAL_LAYOUT);
FCMM_INIT_KEY(AUTOHINT);
FCMM_INIT_KEY(GLOBAL_ADVANCE);
FCMM_INIT_KEY(WIDTH);
FCMM_INIT_KEY(FILE);
FCMM_INIT_KEY(INDEX);
//FCMM_INIT_KEY(FT_FACE);
FCMM_INIT_KEY(RASTERIZER);
FCMM_INIT_KEY(OUTLINE);
FCMM_INIT_KEY(SCALABLE);
FCMM_INIT_KEY(SCALE);
FCMM_INIT_KEY(DPI);
FCMM_INIT_KEY(RGBA);
FCMM_INIT_KEY(MINSPACE);
FCMM_INIT_KEY(SOURCE);
FCMM_INIT_KEY(CHARSET);
FCMM_INIT_KEY(LANG);
FCMM_INIT_KEY(FONTVERSION);
FCMM_INIT_KEY(FULLNAME);
FCMM_INIT_KEY(FAMILYLANG);
FCMM_INIT_KEY(STYLELANG);
FCMM_INIT_KEY(FULLNAMELANG);
FCMM_INIT_KEY(CAPABILITY);
FCMM_INIT_KEY(FONTFORMAT);
FCMM_INIT_KEY(EMBOLDEN);
FCMM_INIT_KEY(EMBEDDED_BITMAP);
FCMM_INIT_KEY(DECORATIVE);
FCMM_INIT_KEY(LCD_FILTER);
FCMM_INIT_KEY(NAMELANG);

}






