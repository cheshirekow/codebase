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
 *  \file   tutorial.cpp
 *
 *  \date   Jul 23, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <cpp_fontconfig/cpp_fontconfig.h>
#include <iostream>

int main( int argc, char** argv )
{
    namespace fc=fontconfig;
    using namespace fontconfig;

    // we'll use the second argument as the font name
    if( argc < 2 )
    {
        std::cerr << "usage: tutorial [Font Name]" << std::endl;
        return 1;
    }

    // initialize font config
    init();

    // we put this in a separate block because we want the pattern objects
    // to destruct before we call fini... otherwise (and I'm not sure but)
    // it might be possible that the patterns will try to free memory that
    // fontconfig free's on fini
    {
        // create a pattern to search for
        RefPtr<Pattern> pat = Pattern::create();

        // type safe version, but will not give compiler error if the
        // argument is the wrong type for the key
        //pat.add(FAMILY, (Char8_t*)argv[1]);

        // type safe but inextensible version that works only on built in
        // types, will give a compiler error if the parameter is not the
        // right type for the key
        pat->addBuiltIn<key::FAMILY>( (const Char8_t*)argv[1] );

        // get a pointer to the default configuration
        RefPtr<Config> config = Config::getCurrent();

        // perform substitutions
        pat->substitute(match::Pattern);
        pat->defaultSubstitute();

        // get the match
        Result_t result;
        RefPtr<Pattern> match = config->fontMatch(pat, result);

        // get the closest matching font file
        Char8_t*    file;
        int         index;

        // we should have a better get/add interface... this isn't very
        // c++-y
        match->get( fc::FILE, 0, file);
        match->get( fc::INDEX, 0, index);

        // at this point, we probably want to use freetype to get a face
        // that we can use in our application, but since this is just a
        // demo, we'll print the font file and quit
        std::cout << "Font found for query [" << argv[1] << "] at "
                  << file << "\n" << std::endl;

        // demo the interface for ObjectTypeList
        ObjectTypeList oList = ObjectTypeList::create()
            ("ObjectA", type::Integer )
            ("ObjectB", type::String  )
            ("ObjectC", type::Double  )
            ();

        std::cout << "number of items in the list: "
                  << oList.get_nItems() << std::endl;

        // demo the interface for ConstantList
        ConstantList cList = ConstantList::create()
            ((const Char8_t*)"NameA", "ObjectA", 1001  )
            ((const Char8_t*)"NameB", "ObjectB", 1002  )
            ((const Char8_t*)"NameC", "ObjectC", 1003  )
            ();

        std::cout << "number of items in the list: "
                  << cList.get_nItems() << std::endl;

        // no need to do cleanup, pattern and match will free their memory
        // when their destructors are called
    }

    // unload fontconfig
    fini();

    return 0;
}




