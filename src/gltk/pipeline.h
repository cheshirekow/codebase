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

#include <memory>
#include <gltk/events.h>
#include <gltk/widget.h>

namespace gltk {

/// Passed to event handlers along with new events
class HandlerActions {
 public:
  /// Called if the handler wishes to relinquish focus for the current
  /// event type
  void PopFocus();

  /// Called if the handler wishes to give focus to a child widget
  void PushFocus(Widget* widget);

  /// Called if the handler wishes for the given widget to be redrawn
  void MarkDirty(Widget* widget);
};

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
  /// Push new events onto the event queue
  /**
   * This should be called by the native event handling system after
   * translating event messages to gltk message types.
   */
  void PushEvent(const std::unique_ptr<Event>& event);

  /// Performs each pass of the pipeline and renders to the screen if
  /// required.
  void DoFrame();

 private:
  /// Dispatch events to focus handlers
  void ProcessEvents();

  /// Render widgets to texture
  void RenderTextures();

  /// Render the scene
  void RenderScene();
};

}  // namespace gltk

#endif  // GLTK_PIPELINE_H_
