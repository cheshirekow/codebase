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
 *  @file   include/cppfontconfig/init.h
 *
 *  \date   Jul 23, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef CPPFONTCONFIG_INIT_H_
#define CPPFONTCONFIG_INIT_H_

#include <cpp_fontconfig/common.h>
#include <cpp_fontconfig/Config.h>

namespace fontconfig
{
    /// load configuration
    /**
     *  Loads the default configuration file and returns the resulting
     *  configuration. Does not load any font information.
     */
    RefPtr<Config> initLoadConfig();

    /// load configuration and font data
    /**
     *  Loads the default configuration file and builds information about the
     *  available fonts. Returns the resulting configuration.
     */
    RefPtr<Config> initLoadConfigAndFonts();

    /// initialize fontconfig library
    /**
     *  Loads the default configuration file and the fonts referenced therein
     *  and sets the default configuration to that result. Returns whether
     *  this process succeeded or not. If the default configuration has already
     *  been loaded, this routine does nothing and returns FcTrue.
     */
    bool    init();

    /// finalize fontconfig library
    /**
     *  Frees all data structures allocated by previous calls to fontconfig
     *  functions. Fontconfig returns to an uninitialized state, requiring a
     *  new call to one of the FcInit functions before any other fontconfig
     *  function may be called.
     */
    void    fini();

    /// library version number
    /**
     *  Returns the version number of the library.
     */
    int     getVersion();

    /// re-initialize library
    /**
     *  Forces the default configuration file to be reloaded and resets the
     *  default configuration. Returns FcFalse if the configuration cannot be
     *  reloaded (due to configuration file errors, allocation failures or
     *  other issues) and leaves the existing configuration unchanged.
     *  Otherwise returns FcTrue.
     */
    bool    initReinitialize();

    /// reload configuration files if needed
    /**
     *  Checks the rescan interval in the default configuration, checking the
     *  configuration if the interval has passed and reloading the
     *  configuration if when any changes are detected. Returns FcFalse if the
     *  configuration cannot be reloaded (see FcInitReinitialize). Otherwise
     *  returns FcTrue.
     */
    bool    initBringUptoDate();
}














#endif // INIT_H_
