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
 *  @date   Apr 15, 2015
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief
 */

#ifndef GLTK_PIPELINE_H_
#define GLTK_PIPELINE_H_

#include <gltk/events.h>

namespace gltk {

/// Encapsulates the gltk event dispatching and rendering pipline
/**
 *  GLTK processes events in multiple phases:
 *  1.  Process user input events like pointer-motions and key-presses, updating
 *      any widgets that are affected by these input events
 *  2.  Redraw any textures which are first-ancestors to any changed widgets
 *      (recursively)
 *  3.  Render the final scene
 */
class Pipeline {
 public:
  /// Dispatch the specified input event to the current event handler
  /**
   * This should be called by the native event handling system after
   * translating event messages to gltk message types. The event is handled
   * immediately, but may modify state affecting later passes of the pipeline
   */
  template <class Event>
  void PushEvent(const Event& event);
};

}  // namespace gltk

#endif  // GLTK_PIPELINE_H_
