/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of gltk.
 *
 *  gltk is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  gltk is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with gltk.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @date   May 1, 2015
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief
 */
#ifndef GLTK_XLIB_TRANSLATE_EVENT_H_
#define GLTK_XLIB_TRANSLATE_EVENT_H_

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <memory>
#include <gltk/events.h>

namespace gltk {
namespace xlib {

NotifyEvent TranslateEvent(const XMapEvent& e);
NotifyEvent TranslateEvent(const XUnmapEvent& e);
WindowConfigureEvent TranslateEvent(const XConfigureEvent& e);
ButtonEvent TranslateEvent(const XButtonEvent& e);
KeyEvent TranslateEvent(const XKeyEvent& e);

}  // namespace xlib
}  // namespace gltk

#endif  // GLTK_XLIB_TRANSLATE_EVENT_H_
