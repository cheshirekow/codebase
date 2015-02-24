/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
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
 *  @file   /home/josh/Codes/cpp/gltk/src/FrameBuffer.h
 *
 *  @date   Feb 3, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef GLTK_FRAMEBUFFER_H_
#define GLTK_FRAMEBUFFER_H_

#include <GL/glew.h>
#include <GL/glfw.h>
#include <gltk/GLuintRef.h>
#include <gltk/RefPtr.h>

namespace gltk {

class FrameBuffer:
    public GLuintRef
{
    private:
        /// calls glGenFramebuffers to instanciate a new framebuffer
        FrameBuffer();

    public:
        /// calls glDestroyFramebuffers to destroy the framebuffer
        ~FrameBuffer();

        /// binds the frame buffer so that future calls work with this
        /// buffer
        void bind();

        /// create a frame buffer
        static RefPtr<FrameBuffer> create();
};


} // namespace gltk



#endif // FRAMEBUFFER_H_
