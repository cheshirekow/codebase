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
#include <list>

namespace gltk {

struct StateBits {
  unsigned int shift : 1;
  unsigned int lock : 1;
  unsigned int control : 1;
  unsigned int mod0 : 1;
  unsigned int mod1 : 1;
  unsigned int mod2 : 1;
  unsigned int mod3 : 1;
  unsigned int mod4 : 1;

  unsigned int button0 : 1;
  unsigned int button1 : 1;
  unsigned int button2 : 1;
  unsigned int button3 : 1;
  unsigned int button4 : 1;
};

enum NotifyKeys {
  kWindowMapped = 0,  // NOFORMAT
  kWindowUnmapped,
  kInvalidNotify
};

enum EventType {
  kNotifyEvent,
  kWindowConfigureEvent,
  kButtonEvent,
  kKeyEvent,
  kMotionEvent,
  kTouchEvent,
  kInvalidEvent
};

struct Event {
  int event_type;

 protected:
  Event(int event_type_in) : event_type(event_type_in) {}
};

struct NotifyEvent : public Event {
  NotifyEvent() : Event(kNotifyEvent) {}
  int notification;
};

struct WindowConfigureEvent : public Event {
  WindowConfigureEvent() : Event(kWindowConfigureEvent) {}
  int width;
  int height;
};

struct ButtonEvent : public Event {
  ButtonEvent() : Event(kButtonEvent) {}
  int x, y;  ///< x,y coordinates of pointer in window
  int button;
  StateBits state;
};

struct KeyEvent : public Event {
  KeyEvent() : Event(kKeyEvent) {}
  int keycode;
  StateBits state;
};

struct MotionEvent : public Event {
  MotionEvent() : Event(kMotionEvent) {}
  int x, y;
  StateBits state;
};

struct TouchEvent : public Event {
  TouchEvent() : Event(kTouchEvent){}
  int x, y;
  int touch_id;
  StateBits state;
};

typedef std::list<Event> EventQueue;

}  // namespace gltk

#endif  // GLTK_EVENTS_H_
