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
 *  @file   src/init.cpp
 *
 *  \date   Jul 23, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <cpp_fontconfig/init.h>
#include <fontconfig/fontconfig.h>

namespace fontconfig
{


RefPtr<Config> initLoadConfig()
{
    return FcInitLoadConfig();
}

RefPtr<Config> initLoadConfigAndFonts()
{
    return FcInitLoadConfigAndFonts();
}

bool init()
{
    return FcInit();
}

void fini()
{
    FcFini();
}

int getVersion()
{
    return FcGetVersion();
}

bool initReinitialize()
{
    return FcInitReinitialize();
}

bool initBringUptoDate()
{
    return FcInitBringUptoDate();
}


}








