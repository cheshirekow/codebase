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
 *  @file   include/cppfontconfig/file.h
 *
 *  \date   Jul 23, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef CPPFONTCONFIG_FILE_H_
#define CPPFONTCONFIG_FILE_H_

#include <cpp_fontconfig/common.h>
#include <cpp_fontconfig/RefPtr.h>
#include <cpp_fontconfig/FontSet.h>
#include <cpp_fontconfig/StrSet.h>
#include <cpp_fontconfig/Blanks.h>



namespace fontconfig
{

///  scan a font file
/**
 *  Scans a single file and adds all fonts found to set. If force is FcTrue,
 *  then the file is scanned even if associated information is found in cache.
 *  If file is a directory, it is added to dirs. Whether fonts are found
 *  depends on fontconfig policy as well as the current configuration.
 *  Internally, fontconfig will ignore BDF and PCF fonts which are not in
 *  Unicode (or the effectively equivalent ISO Latin-1) encoding as those are
 *  not usable by Unicode-based applications. The configuration can ignore
 *  fonts based on filename or contents of the font file itself. Returns
 *  FcFalse if any of the fonts cannot be added (due to allocation failure).
 *  Otherwise returns FcTrue.
 */
bool fileScan( RefPtr<FontSet> set, RefPtr<StrSet> dirs,
                RefPtr<Blanks> blanks, const Char8_t* file, bool force );

/// check whether a file is a directory
/**
 *  Returns FcTrue if file is a directory, otherwise returns FcFalse.
 */
bool fileIsDir( const Char8_t* file);

/// scan a font directory without caching it
/**
 *  If cache is not zero or if force is FcFalse, this function currently
 *  returns FcFalse. Otherwise, it scans an entire directory and adds all
 *  fonts found to set. Any subdirectories found are added to dirs. Calling
 *  this function does not create any cache files. Use FcDirCacheRead() if
 *  caching is desired.
 */
bool dirScan( RefPtr<FontSet> set, RefPtr<StrSet> dirs, RefPtr<Blanks> blanks,
                const Char8_t *dir, bool force);



}













#endif // FILE_H_
