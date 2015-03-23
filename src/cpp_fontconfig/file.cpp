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
 *  @file   src/file.cpp
 *
 *  \date   Jul 23, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <cpp_fontconfig/file.h>
#include <fontconfig/fontconfig.h>


namespace fontconfig
{

bool fileScan(RefPtr<FontSet> set, RefPtr<StrSet> dirs, RefPtr<Blanks>blanks,
        const Char8_t* file, bool force)
{
    return FcFileScan(
                set.subvert(),
                dirs.subvert(),
                0,
                blanks.subvert(),
                file,
                force ? FcTrue : FcFalse );
}

bool fileIsDir(const Char8_t* file)
{
    return FcFileIsDir(file);
}

bool dirScan(RefPtr<FontSet> set, RefPtr<StrSet> dirs, RefPtr<Blanks> blanks,
        const Char8_t* dir, bool force)
{
    return FcDirScan(
                set.subvert(),
                dirs.subvert(),
                (FcFileCache*) 0,
                blanks.subvert(),
                dir,
                force ? FcTrue : FcFalse );
}



}









