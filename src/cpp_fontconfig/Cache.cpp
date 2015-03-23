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
 *  @file   src/Cache.cpp
 *
 *  \date   Jul 20, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <cpp_fontconfig/Cache.h>
#include <fontconfig/fontconfig.h>

namespace fontconfig
{


const Char8_t* CacheDelegate::dir()
{
    return FcCacheDir( m_ptr);
}

const Char8_t* CacheDelegate::subdir(int i)
{
    return FcCacheSubdir(m_ptr, i);
}

int CacheDelegate::numSubdir()
{
    return FcCacheNumSubdir(m_ptr);
}

int CacheDelegate::numFont()
{
    return FcCacheNumFont(m_ptr);
}



void CacheDelegate::unload()
{
    FcDirCacheUnload(m_ptr);
}


RefPtr<Cache> Cache::load(const Char8_t* dir, RefPtr<Config> config, Char8_t** cache_file)
{
    return RefPtr<Cache>(
            FcDirCacheLoad(dir, config.subvert(), cache_file)
    );
}

RefPtr<Cache> Cache::read(const Char8_t* dir, bool force, RefPtr<Config> config)
{
    return RefPtr<Cache>(
            FcDirCacheRead(dir, force ? FcTrue : FcFalse, config.subvert() )
    );
}

RefPtr<Cache> Cache::loadFile(const Char8_t* cache_file, struct stat* file_stat)
{
    return RefPtr<Cache>(
            FcDirCacheLoadFile(cache_file, file_stat )
    );
}

bool Cache::dirClean(const Char8_t* cache_dir, bool verbose)
{
    return FcDirCacheClean(cache_dir, verbose ? FcTrue: FcFalse );
}

bool Cache::dirValid(const Char8_t* dir)
{
    return FcDirCacheValid(dir);
}





} // namespace fontconfig
