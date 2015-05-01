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
#include <gltk/pipeline.h>

namespace gltk {

void Pipeline::PushEvent(const std::unique_ptr<Event>& event) {

}

void Pipeline::DoFrame() {
  ProcessEvents();
  RenderTextures();
  RenderScene();
}

void Pipeline::ProcessEvents() {

}

void Pipeline::RenderTextures() {

}

void Pipeline::RenderScene() {

}

}  // namespace gltk