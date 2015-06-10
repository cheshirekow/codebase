C++ Utilities for GTK {#ck_gtk_page}
==========

ck_gtk provides a couple of minor utilities for using GTK in C++:
  * *LayoutMap*: A class which provides a map from widget identity strings to
    widget or object pointers loaded from a glade file. It also provides the
    ability to save and restore widget model values. 
  * *EigenCairo*: A simple wrapper class for cairo contexts providing many of
    the cairo context API methods taking eigen types as parameterss.
  * *SimpleView*: A GtkDrawingArea which translates drawing coordinates to a
    bottom-left coordinate system, and emits it's cairo context through a 
    callback.
  * *PanZoomView*: A GtkDrawingArea with built-in mouse handlers for pan and
    zoom.


