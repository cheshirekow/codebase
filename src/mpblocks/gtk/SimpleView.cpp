/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of openbook.
 *
 *  openbook is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  openbook is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with openbook.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @date   Feb 28, 2013
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief  
 */


#include <mpblocks/gtk/SimpleView.h>

namespace mpblocks {
namespace      gtk {

SimpleView::SimpleView() {
  add_events(Gdk::POINTER_MOTION_MASK | Gdk::BUTTON_PRESS_MASK |
             Gdk::BUTTON_RELEASE_MASK);

  signal_motion_notify_event().connect(
      sigc::mem_fun(*this, &SimpleView::on_motion));
  signal_button_press_event().connect(
      sigc::mem_fun(*this, &SimpleView::on_press));
  signal_button_release_event().connect(
      sigc::mem_fun(*this, &SimpleView::on_release));
}

bool SimpleView::on_motion(GdkEventMotion* evt) {
  sig_motion(evt->x / get_allocated_width(),
             1.0 - evt->y / get_allocated_height());
  return false;
}

bool SimpleView::on_press(GdkEventButton* evt) {
  sig_press(evt->x / get_allocated_width(),
            1.0 - evt->y / get_allocated_height());
  return false;
}

bool SimpleView::on_release(GdkEventButton* evt) {
  sig_release(evt->x / get_allocated_width(),
              1.0 - evt->y / get_allocated_height());
  return false;
}

/// a very simple drawing area which just rescales the area to (0,0),(1,1) and
/// emits the context via a signal
bool SimpleView::on_draw(const Cairo::RefPtr<Cairo::Context>& ctx) {
  ctx->rectangle(0, 0, get_allocated_width(), get_allocated_height());

  // white pane with black border
  ctx->set_source_rgb(1, 1, 1);
  ctx->fill_preserve();
  ctx->set_source_rgb(0, 0, 0);
  ctx->stroke();

  ctx->save();
  ctx->scale(get_allocated_width(), -get_allocated_height());
  ctx->translate(0, -1);
  sig_draw(ctx);
  ctx->restore();

  return true;
}

} // gtk
} // mpblocks
