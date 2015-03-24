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
 *  @file   include/cppfontconfig/Config.h
 *
 *  \date   Jul 22, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef CPPFONTCONFIG_CONFIG_H_
#define CPPFONTCONFIG_CONFIG_H_

#include <fontconfig/fontconfig.h>
#include <cpp_fontconfig/common.h>
#include <cpp_fontconfig/RefPtr.h>
#include <cpp_fontconfig/StrList.h>
#include <cpp_fontconfig/Blanks.h>
#include <cpp_fontconfig/Pattern.h>
#include <cpp_fontconfig/FontSet.h>
#include <cpp_fontconfig/RefPtr.h>

namespace fontconfig
{

class Config;

/// holds a complete configuration of the library;
/**
 *  there is one default configuration, other can be constructed from XML data
 *  structures. All public entry points that need global data can take an
 *  optional FcConfig* argument; passing 0 uses the default configuration.
 *  FcConfig objects hold two sets of fonts, the first contains those specified
 *  by the configuration, the second set holds those added by the application
 *  at run-time. Interfaces that need to reference a particular set use one of
 *  the FcSetName enumerated values.
 *
 *  An FcConfig object holds the internal representation of a configuration.
 *  There is a default configuration which applications may use by passing 0
 *  to any function using the data within an FcConfig.
 */
class ConfigDelegate
{
    private:
        FcConfig* m_ptr;

        /// wrap constructor
        /**
         *  wraps the pointer with this interface, does nothing else, only
         *  called by RefPtr<Atomic>
         */
        explicit ConfigDelegate(FcConfig* ptr):
            m_ptr(ptr)
        {}

        /// not copy-constructable
        ConfigDelegate( const ConfigDelegate& other );

        /// not assignable
        ConfigDelegate& operator=( const ConfigDelegate& other );

    public:
        friend class RefPtr<Config>;

        ConfigDelegate* operator->(){ return this; }
        const ConfigDelegate* operator->() const { return this; }


        /// Set configuration as default
        /**
         *  Sets the current default configuration to config. Implicitly calls
         *  FcConfigBuildFonts if necessary, returning FcFalse if that call
         *  fails.
         */
        bool setCurrent();


        /// Check timestamps on config files
        /**
         *  Checks all of the files related to config and returns whether any
         *  of them has been modified since the configuration was created. If
         *  config is NULL, the current configuration is used.
         */
        bool upToDate();


        /// Build font database
        /**
         *  Builds the set of available fonts for the given configuration. Note
         *  that any changes to the configuration after this call have
         *  indeterminate effects. Returns FcFalse if this operation runs out
         *  of memory. If config is NULL, the current configuration is used.
         */
        bool buildFonts();


        /// Get font directories
        /**
         *  Returns the list of font directories specified in the configuration
         *  files for config. Does not include any subdirectories. If config
         *  is NULL, the current configuration is used.
         */
        RefPtr<StrList> getFontDirs();


        /// Get config directories
        /**
         *
         */
        RefPtr<StrList> getConfigDirs();


        /// Get config files
        /**
         *  Returns the list of known configuration files used to generate
         *  config. If config is NULL, the current configuration is used.
         */
        RefPtr<StrList> getConfigFiles();


        /// DEPRECATED used to return per-user cache filename
        /**
         *  With fontconfig no longer using per-user cache files, this function
         *  now simply returns NULL to indicate that no per-user file exists.
         */
        // Char8_t* GetCache();


        /// Get config blanks
        /**
         *  Returns the FcBlanks object associated with the given
         *  configuration, if no blanks were present in the configuration,
         *  this function will return 0. The returned FcBlanks object if not
         *  NULL, is valid as long as the owning FcConfig is alive. If config
         *  is NULL, the current configuration is used.
         */
        RefPtr<Blanks> getBlanks();


        /// return the list of directories searched for cache files
        /**
         *  FcConfigGetCacheDirs returns a string list containing all of the
         *  directories that fontconfig will search when attempting to load a
         *  cache file for a font directory. If config is NULL, the current
         *  configuration is used.
         */
        RefPtr<StrList> getCacheDirs() const;


        /// Get config rescan interval
        /**
         *  Returns the interval between automatic checks of the configuration
         *  (in seconds) specified in config. The configuration is checked
         *  during a call to FcFontList when this interval has passed since
         *  the last check. An interval setting of zero disables automatic
         *  checks. If config is NULL, the current configuration is used.
         */
        int getRescanInterval ();


        /// Set config rescan interval
        /**
         *  Sets the rescan interval. Returns FcFalse if the interval cannot
         *  be set (due to allocation failure). Otherwise returns FcTrue. An
         *  interval setting of zero disables automatic checks. If config is
         *  NULL, the current configuration is used.
         */
        bool setRescanInterval (int rescanInterval);


        /// Get config font set
        /**
         *  Returns one of the two sets of fonts from the configuration as
         *  specified by set. This font set is owned by the library and must
         *  not be modified or freed. If config is NULL, the current
         *  configuration is used.
         */
        RefPtr<FontSet> getFonts (SetName_t set);


        /// Add font file to font database
        /**
         *  Adds an application-specific font to the configuration. Returns
         *  FcFalse if the fonts cannot be added (due to allocation failure).
         *  Otherwise returns FcTrue. If config is NULL, the current
         *  configuration is used.
         */
        bool appFontAddFile (const Char8_t *file);


        /// Add fonts from directory to font database
        /**
         *  Scans the specified directory for fonts, adding each one found to
         *  the application-specific set of fonts. Returns FcFalse if the
         *  fonts cannot be added (due to allocation failure). Otherwise
         *  returns FcTrue. If config is NULL, the current configuration is
         *  used.
         */
        bool appFontAddDir (const Char8_t  *dir);


        /// Remove all app fonts from font database
        /**
         *  Clears the set of application-specific fonts. If config is NULL,
         *  the current configuration is used.
         */
        void appFontClear ();


        /// Execute substitutions
        /**
         *  Performs the sequence of pattern modification operations, if kind
         *  is FcMatchPattern, then those tagged as pattern operations are
         *  applied, else if kind is FcMatchFont, those tagged as font
         *  operations are applied and p_pat is used for test elements with
         *  target=pattern. Returns FcFalse if the substitution cannot be
         *  performed (due to allocation failure). Otherwise returns FcTrue.
         *  If config is NULL, the current configuration is used.
         */
        bool substituteWithPat (
                       RefPtr<Pattern> p,
                       RefPtr<Pattern> p_pat,
                       MatchKind_t  kind);


        /// Execute substitutions
        /**
         *  Calls FcConfigSubstituteWithPat setting p_pat to NULL. Returns
         *  FcFalse if the substitution cannot be performed (due to allocation
         *  failure). Otherwise returns FcTrue. If config is NULL, the current
         *  configuration is used.
         */
        bool substitute (
                    RefPtr<Pattern> p,
                    MatchKind_t     kind);


        /// Return best font
        /**
         *  Finds the font in sets most closely matching pattern and returns
         *  the result of FcFontRenderPrepare for that font and the provided
         *  pattern. This function should be called only after
         *  FcConfigSubstitute and FcDefaultSubstitute have been called for p;
         *  otherwise the results will not be correct. If config is NULL, the
         *  current configuration is used.
         */
        RefPtr<Pattern> fontMatch (
                    RefPtr<Pattern> p,
                    Result_t&   result);


        /// Return the best font from a set of font sets
        /**
         *  Finds the font in sets most closely matching pattern and returns
         *  the result of FcFontRenderPrepare for that font and the provided
         *  pattern. This function should be called only after
         *  FcConfigSubstitute and FcDefaultSubstitute have been called for
         *  pattern; otherwise the results will not be correct. If config is
         *  NULL, the current configuration is used. Returns NULL if an error
         *  occurs during this process.
         */
        RefPtr<Pattern> fontSetMatch(
                    RefPtr<FontSet>*    sets,
                    int         nsets,
                    RefPtr<Pattern> pattern,
                    Result_t&   result);


        /// Prepare pattern for loading font file
        /**
         *  Creates a new pattern consisting of elements of font not appearing
         *  in pat, elements of pat not appearing in font and the best
         *  matching value from pat for elements appearing in both. The result
         *  is passed to FcConfigSubstituteWithPat with kind FcMatchFont and
         *  then returned.
         */
        RefPtr<Pattern> renderPrepare (
                    RefPtr<Pattern> pat,
                    RefPtr<Pattern> font);


        ///  Return list of matching fonts
        /**
         *  Returns the list of fonts from sets sorted by closeness to pattern.
         *  If trim is FcTrue, elements in the list which don't include
         *  Unicode coverage not provided by earlier elements in the list are
         *  elided. The union of Unicode coverage of all of the fonts is
         *  returned in csp, if csp is not NULL. This function should be
         *  called only after FcConfigSubstitute and FcDefaultSubstitute have
         *  been called for p; otherwise the results will not be correct.
         *
         *  The returned FcFontSet references FcPattern structures which may
         *  be shared by the return value from multiple FcFontSort calls,
         *  applications cannot modify these patterns. Instead, they should be
         *  passed, along with pattern to FcFontRenderPrepare which combines
         *  them into a complete pattern.
         *
         *  The FcFontSet returned by FcFontSetSort is destroyed by calling
         *  FcFontSetDestroy.
         */
        RefPtr<FontSet> fontSetSort (
                    RefPtr<FontSet>*    sets,
                    int         nsets,
                    RefPtr<Pattern> p,
                    bool        trim,
                    RefPtr<CharSet>* csp,
                    Result_t&        result);


        ///  Return list of matching fonts
        /**
         *  Returns the list of fonts sorted by closeness to p. If trim is
         *  FcTrue, elements in the list which don't include Unicode coverage
         *  not provided by earlier elements in the list are elided. The union
         *  of Unicode coverage of all of the fonts is returned in csp, if csp
         *  is not NULL. This function should be called only after
         *  FcConfigSubstitute and FcDefaultSubstitute have been called for p;
         *  otherwise the results will not be correct.
         *
         *  The returned FcFontSet references FcPattern structures which may
         *  be shared by the return value from multiple FcFontSort calls,
         *  applications must not modify these patterns. Instead, they should
         *  be passed, along with p to FcFontRenderPrepare which combines them
         *  into a complete pattern.
         *
         *  The FcFontSet returned by FcFontSort is destroyed by calling
         *  FcFontSetDestroy. If config is NULL, the current configuration is
         *  used.
         */
        RefPtr<FontSet> fontSort (
                    RefPtr<Pattern> p,
                    bool                trim,
                    RefPtr<CharSet>*    csp,
                    Result_t&           result);


        /// List fonts from a set of font sets
        /**
         *  Selects fonts matching pattern from sets, creates patterns from
         *  those fonts containing only the objects in object_set and returns
         *  the set of unique such patterns. If config is NULL, the default
         *  configuration is checked to be up to date, and used.
         */
        RefPtr<FontSet> fontSetList (
                    RefPtr<FontSet>*  sets,
                    int          nsets,
                    RefPtr<Pattern> p,
                    RefPtr<ObjectSet>    os);


        /// List fonts
        /**
         *  Selects fonts matching p, creates patterns from those fonts
         *  containing only the objects in os and returns the set of unique
         *  such patterns. If config is NULL, the default configuration is
         *  checked to be up to date, and used.
         */
        RefPtr<FontSet> fontList (
                    RefPtr<Pattern> p,
                    RefPtr<ObjectSet>   os);


        /// load a configuration file
        /**
         *  Walks the configuration in 'file' and constructs the internal
         *  representation in 'config'. Any include files referenced from
         *  within 'file' will be loaded and parsed. If 'complain' is FcFalse,
         *  no warning will be displayed if 'file' does not exist. Error and
         *  warning messages will be output to stderr. Returns FcFalse if some
         *  error occurred while loading the file, either a parse error,
         *  semantic error or allocation failure. Otherwise returns FcTrue.
         */
        bool parseAndLoad (
                    const Char8_t   *file,
                    bool            complain);

        /// Remove all caches related to \p dir
        /**
         *  Scans the cache directories in config, removing any instances of
         *  the cache file for dir. Returns FcFalse when some internal error
         *  occurs (out of memory, etc). Errors actually unlinking any files
         *  are ignored.
         */
        bool unlink (const Char8_t *dir);

        /// Create CACHEDIR.TAG at cache directory
        /**
         *  This tries to create CACHEDIR.TAG file at the cache directory
         *  registered to config.
         */
        void createTagFile() const;
};


struct Config
{
    typedef ConfigDelegate Delegate;
    typedef FcConfig*      Storage;
    typedef FcConfig*      cobjptr;

    /// return the current home directory.
    /**
     *  Return the current user's home directory, if it is available, and
     *  if using it is enabled, and NULL otherwise. See also
     *  FcConfigEnableHome).
     */
    static Char8_t* home(void);


    /// controls use of the home directory.
    /**
     *  If enable is FcTrue, then Fontconfig will use various files which
     *  are specified relative to the user's home directory (using the ~
     *  notation in the configuration). When enable is FcFalse, then all
     *  use of the home directory in these contexts will be disabled. The
     *  previous setting of the value is returned.
     */
    static bool     enableHome(bool enable);


    ///  Find a config file
    /**
     *  Given the specified external entity name, return the associated
     *  filename. This provides applications a way to convert various
     *  configuration file references into filename form.
     *
     *  A null or empty name indicates that the default configuration file
     *  should be used; which file this references can be overridden with
     *  the FONTCONFIG_FILE environment variable. Next, if the name starts
     *  with ~, it refers to a file in the current users home directory.
     *  Otherwise if the name doesn't start with '/', it refers to a file
     *  in the default configuration directory; the built-in default
     *  directory can be overridden with the FONTCONFIG_PATH environment
     *  variable.
     */
    static Char8_t* filename(const Char8_t* url );


    /// Create a configuration
    /**
     *  Creates an empty configuration.
     */
    static RefPtr<Config>  create();

    /// Return current configuration
    /**
     *  Returns the current default configuration.
     */
    static RefPtr<Config> getCurrent();
};


/// Configs are reference counted
template <> void RefPtr<Config>::reference();

/// Configs are reference counted
template <> void RefPtr<Config>::dereference();

} // namespace fontconfig 

#endif // CONFIG_H_
