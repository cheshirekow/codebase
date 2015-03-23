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
 *  @file   include/cppfontconfig/Cache.h
 *
 *  \date   Jul 20, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef CPPFONTCONFIG_CACHE_H_
#define CPPFONTCONFIG_CACHE_H_

#include <cpp_fontconfig/common.h>
#include <cpp_fontconfig/RefPtr.h>
#include <cpp_fontconfig/Config.h>
#include <unistd.h>
#include <sys/stat.h>


namespace fontconfig
{

class Cache;

/// Holds information about the fonts contained in a single directory.
/**
 *  Normal
 *  applications need not worry about this as caches for font access are
 *  automatically managed by the library. Applications dealing with cache
 *  management may want to use some of these objects in their work,
 *  however the included 'fc-cache' program generally suffices for all of that.
 *
 *  @see FcCache
 */
class CacheDelegate
{
    private:
        FcCache* m_ptr;

        /// wrap constructor
        /**
         *  wraps the pointer with this interface, does nothing else, only
         *  called by RefPtr<Atomic>
         */
        explicit CacheDelegate(FcCache* ptr):
            m_ptr(ptr)
        {}

        /// not copy-constructable
        CacheDelegate( const CacheDelegate& other );

        /// not assignable
        CacheDelegate& operator=( const CacheDelegate& other );

    public:
        friend class RefPtr<Cache>;

        CacheDelegate* operator->(){ return this; }
        const CacheDelegate* operator->() const { return this; }


        /// Return directory of cache
        /**
         *  This function returns the directory from which the cache was
         *  constructed.
         */
        const Char8_t* dir();

        /// Returns a copy of the fontset from cache
        /**
         *  The returned fontset contains each of the font patterns from cache.
         *  This fontset may be modified, but the patterns from the cache are
         *  read-only.
         */
        //FcFontSet * copySet();

        /// Return the i'th subdirectory.
        /**
         *  The set of subdirectories stored in a cache file are indexed by
         *  this function, i should range from 0 to n-1, where n is the return
         *  value from FcCacheNumSubdir.
         */
        const Char8_t * subdir (int i);

        /// Return the number of subdirectories in cache.
        /**
         *  This returns the total number of subdirectories in the cache.
         */
        int numSubdir ();

        /// Returns the number of fonts in cache.
        /**
         *  This returns the number of fonts which would be included in the
         *  return from FcCacheCopySet.
         */
        int numFont ();

        /// unload a cache file
        /**
         *  This function dereferences cache. When no other references to it
         *  remain, all memory associated with the cache will be freed.
         */
        void unload();
};


/// traits class for FcCache
/// Holds information about the fonts contained in a single directory.
/**
 *  Normal
 *  applications need not worry about this as caches for font access are
 *  automatically managed by the library. Applications dealing with cache
 *  management may want to use some of these objects in their work,
 *  however the included 'fc-cache' program generally suffices for all of that.
 *
 *  @see FcCache
 */
struct Cache
{
    typedef CacheDelegate Delegate;
    typedef FcCache*      Storage;
    typedef FcCache*      cobjptr;

    /// load a directory cache
    /**
     *  Loads the cache related to dir. If no cache file exists,
     *  returns NULL. The name of the cache file is returned in cache_file,
     *  unless that is NULL. See also FcDirCacheRead.
     */
    static RefPtr<Cache> load( const Char8_t* dir,
                        RefPtr<Config> config,
                        Char8_t** cache_file );

    /// read or construct a directory cache
    /**
     *  This returns a cache for dir. If force is FcFalse, then an
     *  existing, valid cache file will be used. Otherwise, a new cache
     *  will be created by scanning the directory and that returned.
     */
    static RefPtr<Cache> read( const Char8_t* dir,
                        bool force,
                        RefPtr<Config> config );

    /// load a cache file
    /**
     *  This function loads a directory cache from cache_file. If
     *  file_stat is non-NULL, it will be filled with the results of
     *  stat(2) on the cache file.
     */
    static RefPtr<Cache> loadFile( const Char8_t* cache_file,
                            struct stat* file_stat );

    ///  This tries to clean up the cache directory of cache_dir. This
    /// returns FcTrue if the operation is successfully complete.
    /// otherwise FcFalse.
    static bool dirClean (const Char8_t *cache_dir, bool verbose);

    /// checks directory cache
    /**
     *  Returns FcTrue if dir has an associated valid cache file, else
     *  returns FcFalse
     */
    static bool dirValid( const Char8_t* dir );

};





} // namespace fontconfig 

#endif // CACHE_H_
