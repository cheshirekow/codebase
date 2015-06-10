/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of ck_gtk.
 *
 *  ck_gtk is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  ck_gtk is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with ck_gtk.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef CK_GTK_SIMPLE_VIEW_H_
#define CK_GTK_SIMPLE_VIEW_H_

#include <gtkmm.h>

namespace ck_gtk {

/// a very simple drawing area which just rescales the area to (0,0),(1,1) and
/// emits the context via a signal so that things can draw themselves
class SimpleView : public Gtk::DrawingArea {
 public:
  /// signal through which we emit the context
  sigc::signal<void, const Cairo::RefPtr<Cairo::Context>&> sig_draw;

  /// signal through which we emit mouse press events
  sigc::signal<void, double, double> sig_press;

  /// signal through which we emit mouse press events
  sigc::signal<void, double, double> sig_release;

  /// signal through which we emit mouse motion
  sigc::signal<void, double, double> sig_motion;

  /// needs a vtable
  virtual ~SimpleView() {}

  /// subscribes to event messages
  SimpleView();

  bool on_motion(GdkEventMotion* evt);
  bool on_press(GdkEventButton* evt);
  bool on_release(GdkEventButton* evt);

  /// overload which does the drawing, just scales ctx and then emits
  /// it over sig_draw
  virtual bool on_draw(const Cairo::RefPtr<Cairo::Context>& ctx);
};

}  // namespace ck_gtk

#endif  // CK_GTK_SIMPLE_VIEW_H_
