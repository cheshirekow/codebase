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

#ifndef GLTK_EVENTS_H_
#define GLTK_EVENTS_H_

#include <bitset>

namespace gltk {

struct StateBits {
  unsigned int shift: 1;
  unsigned int lock: 1;
  unsigned int control: 1;
  unsigned int mod0: 1;
  unsigned int mod1: 1;
  unsigned int mod2: 1;
  unsigned int mod3: 1;
  unsigned int mod4: 1;

  unsigned int button0: 1;
  unsigned int button1: 1;
  unsigned int button2: 1;
  unsigned int button3: 1;
  unsigned int button4: 1;
};

struct ButtonEvent {
  int x, y;  ///< x,y coordinates of pointer in window
  int button;
  StateBits state;
};

struct KeyEvent {
  int keycode;
  StateBits state;
};

struct MotionEvent {
  int x, y;
  StateBits state;
};

struct TouchEvent {
  int x, y;
  int touch_id;
  StateBits state;
};

}  // namespace gltk

#endif  // GLTK_EVENTS_H_
