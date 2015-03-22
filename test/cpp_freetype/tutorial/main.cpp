/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of cppfreetype.
 *
 *  cppfreetype is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  cppfreetype is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Fontconfigmm.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  \file   main.cpp
 *
 *  \date   Aug 1, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */


#include <cpp_freetype/cpp_freetype.h>
#include <sigc++/sigc++.h>
#include <iostream>

int main( int argc, char** argv )
{

    using namespace freetype;

    if(argc < 2 )
    {
        std::cerr << "Provide font file path as parameter please" << std::endl;
        return 1;
    }

    Error result=1;
    RefPtr<Library> freetype;
    (freetype,result) = init_e();

    if(result)
    {
        std::cerr << "Failed to initialize freetype: "
                  << result
                  << std::endl;
        return 1;
    }

    // artificial scope... we dont want a dangling Face pointer hanging
    // around trying to free itself after we call ft::done
    {
        RefPtr<Face> face;
        (face,result) = freetype->new_face_e( argv[1], 0 );

        if(result)
        {
            std::cerr << "Freetype failed to read " << argv[1]
                      << " as a font file " << std::endl;
            return 1;
        }

        std::cout << "Some info about the font: "
          << "\n      filepath: " << argv[1]
          << "\n        family: " << face->family_name()
          << "\n         style: " << face->style_name()
          << "\n  n fixed size: " << face->num_fixed_sizes()
          << "\n    n charmaps: " << face->num_charmaps()
          << "\n      scalable: " << (face->is_scalable() ? "yes" : "no")
          << "\n      n glyphs: " << face->num_glyphs()
          << "\n  units per EM: " << face->units_per_EM()
          << "\n      charmaps: ";

        for(int i=0; i < face->num_charmaps(); i++)
        {
            Untag untag = (*face)->charmaps[i]->encoding;
            std::cout
          << "\n                " << untag;
        }

        std::cout
          << "\n"
          << std::endl;


        result = face->select_charmap( encoding::UNICODE );
        std::cout << "Set charmap to index: "
                  << FT_Get_Charmap_Index( (*face)->charmap )
                  << std::endl;

        result = face->set_char_size(
                    0,      // char_width in 1/64th of points
                    16*64,  // char height in 1/64th of points
                    300,    // horizontal device resolution
                    300 );  // vertical device resolution

        char theChar   = 'A';
        UInt charIndex = face->get_char_index(theChar);
        result = face->load_char(theChar, load::NO_BITMAP | load::NO_SCALE);
        ULong glyphFormat = face->glyph()->format();

        std::cout << "for char " << theChar << " :"
            << "\n    ascii: " << (int)theChar
            << "\n    index: " << charIndex
            << "\n   format: " << (char)( (glyphFormat >> 24 ) & 0xff )
                               << (char)( (glyphFormat >> 16 ) & 0xff )
                               << (char)( (glyphFormat >> 8  ) & 0xff )
                               << (char)( (glyphFormat >> 0  ) & 0xff )
            << "\n contours: " << face->glyph()->outline()->n_contours()
            << "\n   points: " << face->glyph()->outline()->n_points()
            << std::endl;


        ContourIterator ic = face->glyph()->outline()->begin();
        PointIterator   ip;

        int i=0;
        for(; !ic.done(); ++ic)
        {
            std::cout << "\n\nContour: " << i++ << "\n";
            for( ip = ic->begin(); ip != ic->end(); ++ip )
            {
                std::cout << "\n   ("
                          << ip->x() << ","
                          << ip->y() << ")  "
                          << ( ip->on() ? "on" : "off") << "  ";
                if( ip->off() )
                    std::cout << (ip->cubic() ? "cubic" : "quadradic");

            }
        }
    }

    result = done(freetype);
    return 0;
}



