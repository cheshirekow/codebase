#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/epoll.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <pangomm.h>    // Note: must preceed X11 headers
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <csignal>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>

#undef Success // defined by X.h, uh oh
#include <cairomm/cairomm.h>
#include <cairomm/xlib_surface.h>
#include <Eigen/Dense>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pangomm/init.h>
#include "util.h"

#ifndef CAIRO_HAS_XLIB_SURFACE
#error "You must have cairo with the xlib backend"
#endif

bool g_should_quit = false;  ///< Set to true when the process should exit
bool g_wm_detected = false;

// Command line flags:
DEFINE_string(
    display, "",
    "The X display server to use, overrides DISPLAY environment variable.");
DEFINE_int32(
    screen, 0,
    "Which X screen to manage, if there are multiple X screens in play.");

/// Opaque handle to our X server connections
Display* g_display = nullptr;

/// The root window of our screen
Window g_root_win = 0;
Visual* g_visual = nullptr;

bool g_vinfo_matched = false;
XVisualInfo g_vinfo_truecolor;

struct FrameGadget {
  Cairo::RefPtr<Cairo::XlibSurface> cairo_surface;
  Glib::RefPtr<Pango::Layout> pango_layout;
};

// Maps client top-level windows to the frame window we created around them
std::map<Window, Window> frame_of_client_;
std::map<Window, Window> client_of_frame_;
std::map<Window, FrameGadget> frame_gadgets_;

// The cursor position at the start of a window move/resize.
Eigen::Vector2i drag_start_pos_;

// The position of the affected window at the start of a window
// move/resize.
Eigen::Vector2i drag_start_frame_pos_;

// Atom constants.
Atom WM_PROTOCOLS;
Atom WM_DELETE_WINDOW;
Atom WM_NAME;

const unsigned int BORDER_WIDTH = 0;
const unsigned long BORDER_COLOR = 0x000000;
const unsigned long BG_COLOR = 0x0000ff;
const unsigned int TITLE_HEIGHT = 14;

struct Theme {
  Pango::FontDescription font;
  Cairo::RefPtr<Cairo::Pattern> bg_pattern;
  Cairo::RefPtr<Cairo::Pattern> txt_pattern;
  int radius;

  Theme()
      : font("Sans 8"),
        radius(15) {
    bg_pattern = Cairo::SolidPattern::create_rgb(0.2, 0.2, 0.3);
    txt_pattern = Cairo::SolidPattern::create_rgb(1.0, 1.0, 1.0);
  }
};

std::unique_ptr<Theme> g_theme;

/// Add's an X window managed by this process as a parent of @p w so that
/// we can draw a border around w
void Frame(Window w) {
  CHECK(!frame_of_client_.count(w));

  XWindowAttributes x_window_attrs;
  CHECK(XGetWindowAttributes(g_display, w, &x_window_attrs));
  XTextProperty text_prop_return;
  std::string window_name = "tabwm decorator";
  if (XGetWMName(g_display, w, &text_prop_return)) {
    window_name = std::string(reinterpret_cast<char*>(text_prop_return.value),
                              text_prop_return.nitems);
  }

  Window frame;
  Visual* frame_visual = g_visual;
  if (g_vinfo_matched) {
    frame_visual = g_vinfo_truecolor.visual;
    XSetWindowAttributes attr;
    attr.colormap = XCreateColormap(g_display, g_root_win,
                                    g_vinfo_truecolor.visual, AllocNone);
    attr.border_pixel = 0;
    attr.background_pixel = 0;

    frame = XCreateWindow(g_display, g_root_win, x_window_attrs.x,
                          x_window_attrs.y, x_window_attrs.width,
                          x_window_attrs.height + TITLE_HEIGHT, 0,
                          g_vinfo_truecolor.depth,
                          InputOutput,
                          g_vinfo_truecolor.visual,
                          CWColormap | CWBorderPixel | CWBackPixel,
                          &attr);
    LOG(WARNING)<< "Created frame window id: " << frame;
  } else {
    frame = XCreateSimpleWindow(g_display, g_root_win, x_window_attrs.x,
        x_window_attrs.y, x_window_attrs.width,
        x_window_attrs.height + TITLE_HEIGHT, BORDER_WIDTH, BORDER_COLOR,
        BG_COLOR);
  }

  XSetWMProtocols(g_display, frame, &WM_DELETE_WINDOW, 1);
  FrameGadget& gadget = frame_gadgets_[frame];
  gadget.cairo_surface = Cairo::XlibSurface::create(
      g_display, frame, frame_visual, x_window_attrs.width,
      x_window_attrs.height + TITLE_HEIGHT);
  gadget.pango_layout = Pango::Layout::create(
      Cairo::Context::create(gadget.cairo_surface));
  gadget.pango_layout->set_font_description(g_theme->font);
  gadget.pango_layout->set_text(window_name);

  XSelectInput(g_display, frame,
  SubstructureRedirectMask | SubstructureNotifyMask |
  StructureNotifyMask | ExposureMask | ButtonPressMask | Button1MotionMask);
  XSelectInput(g_display, w, PropertyChangeMask);

  // Add client to save set, notifying X that if we crash, it should restore
  // the window to a child of the root and not destroy it.
  XAddToSaveSet(g_display, w);

  // Reparent the client window to our frame window. The last two params are
  // the positional offset within the parent.
  XReparentWindow(g_display, w, frame, 0, 0 + TITLE_HEIGHT);

  // Note we do not yet map the window, we wait until the now child window
  // requests it
  frame_of_client_[w] = frame;
  client_of_frame_[frame] = w;

  LOG(INFO)<< "Framed window " << w << " [" << frame << "]";
}

/// Removes our X window "wrapper" frame from w
//void Unframe(Window w) {
//  CHECK(frame_of_client_.count(w));
//  CHECK(client_of_frame_.count(frame_of_client_[w]));
//  const Window frame = frame_of_client_[w];
//  frame_of_client_.erase(w);
//  client_of_frame_.erase(frame);
//
//  XUnmapWindow(g_display, frame);
//  XReparentWindow(g_display, w, g_root_win, 0, 0);
//  XRemoveFromSaveSet(g_display, w);
//  XDestroyWindow(g_display, frame);
//  LOG(INFO) << "Unframed window " << w << " [" << frame << "]";
//}

void OnCreateNotify(const XCreateWindowEvent& e) {
  // Don't try to frame a frame
  if (!client_of_frame_.count(e.window)) {
    Frame(e.window);
  }
}

void OnDestroyNotify(const XDestroyWindowEvent& e) {
  // If a program has destroyed it's own window then we can unframe it
  if (frame_of_client_.count(e.window)) {
    Window frame = frame_of_client_[e.window];
    XDestroyWindow(g_display, frame);
    client_of_frame_.erase(frame);
    frame_of_client_.erase(e.window);
    frame_gadgets_.erase(frame);
  }
}

/// When a window wants to be shown we map it and the frame
void OnMapRequest(const XMapRequestEvent& e) {
  XMapWindow(g_display, e.window);
  XMapWindow(g_display, frame_of_client_[e.window]);
  XMapWindow(g_display, e.window);

}

// Note that we do not grant the child's request immediately, but we wait
// until the frame window's request has been processed. This is in case the
// window manager disagrees about the size and position, and gives us something
// other than what we asked for.
void OnConfigureNotify(const XConfigureEvent& e) {
  if (client_of_frame_.count(e.window)) {
    frame_gadgets_[e.window].cairo_surface->set_size(e.width, e.height);
    XWindowChanges changes;
    changes.width = e.width;
    changes.height = e.height - TITLE_HEIGHT;
    XConfigureWindow(g_display, client_of_frame_[e.window], CWWidth | CWHeight,
                     &changes);
    LOG(INFO)<< FormatStr("Resize %ul to (%d,%d)", e.window, e.width, e.height);
  }
}

/// If a window wishes to changes size or location, then we make the bounding
/// frame match the new size/location, if it exists.window
void OnConfigureRequest(const XConfigureRequestEvent& e) {
  XWindowChanges changes;
  changes.x = e.x;
  changes.y = e.y;
  changes.width = e.width;
  changes.height = e.height + TITLE_HEIGHT;
  changes.border_width = e.border_width;
  changes.sibling = e.above;
  changes.stack_mode = e.detail;
  if (frame_of_client_.count(e.window)) {
    const Window frame = frame_of_client_[e.window];
    XConfigureWindow(g_display, frame, e.value_mask, &changes);
    LOG(INFO)<< FormatStr("Resize [%ul] to (%d,%d)", frame, e.width, e.height);
  }
}

/// When we receive a button press event we save the initial geometry of the
/// window with focus because it means the window is either going to be
/// resized or moved.
void OnButtonPress(const XButtonEvent& e) {
  // Save initial cursor position.
  drag_start_pos_ = Eigen::Vector2i(e.x_root, e.y_root);

  // Save initial window info.
  Window returned_root;
  int x, y;
  unsigned width, height, border_width, depth;

  CHECK(
      XGetGeometry(g_display, e.window, &returned_root, &x, &y, &width, &height,
                   &border_width, &depth));
  drag_start_frame_pos_ = Eigen::Vector2i(x, y);

  // Raise clicked window to top.
  XRaiseWindow(g_display, e.window);
}

void OnMotionNotify(const XMotionEvent& e) {
  const Eigen::Vector2i drag_pos(e.x_root, e.y_root);
  const Eigen::Vector2i delta = drag_pos - drag_start_pos_;

  const Eigen::Vector2i dest_frame_pos = drag_start_frame_pos_ + delta;
  XMoveWindow(g_display, e.window, dest_frame_pos[0], dest_frame_pos[1]);
}

void Draw(Window window) {
  if (frame_gadgets_.count(window)) {
    XClearWindow(g_display, window);
    FrameGadget& gadget = frame_gadgets_[window];
    Cairo::RefPtr < Cairo::Context > ctx = Cairo::Context::create(
        gadget.cairo_surface);
    gadget.pango_layout->update_from_cairo_context(ctx);
    gadget.pango_layout->add_to_cairo_context(ctx);
    ctx->set_source(g_theme->txt_pattern);
    ctx->fill();
  }
}

void OnExpose(const XExposeEvent& e) {
  // squash multiple expose events
  if (e.count != 0) {
    return;
  }
  // We should probably not allocate this on the expose event
  XWindowAttributes attrs;
  CHECK(XGetWindowAttributes(g_display, e.window, &attrs));

  Visual* visual = g_vinfo_matched ? g_vinfo_truecolor.visual : g_visual;
  Draw(e.window);
}

void OnPropertyChange(const XPropertyEvent& e) {
  if (frame_of_client_.count(e.window) && e.atom == WM_NAME) {
    Window frame = frame_of_client_[e.window];
    std::string window_name = "tabwm decorator";
    XTextProperty text_prop_return;
    if (XGetWMName(g_display, e.window, &text_prop_return)) {
      window_name = std::string(reinterpret_cast<char*>(text_prop_return.value),
                                text_prop_return.nitems);
    }
    if (frame_gadgets_.count(frame)) {
      FrameGadget& gadget = frame_gadgets_[frame];
      gadget.pango_layout->set_text(window_name);
    }
    Draw(frame);
  }
}

int OnWMDetected(Display* display, XErrorEvent* e) {
  // In the case of an already running window manager, the error code from
  // XSelectInput is BadAccess. We don't expect this handler to receive any
  // other errors.
  CHECK_EQ(static_cast<int>(e->error_code), BadAccess);
  // Set flag.
  g_wm_detected = true;
  // The return value is ignored.
  return 0;
}

int OnXError(Display* display, XErrorEvent* e) {
  const int MAX_ERROR_TEXT_LENGTH = 1024;
  char error_text[MAX_ERROR_TEXT_LENGTH];
  XGetErrorText(display, e->error_code, error_text, sizeof(error_text));
  LOG(ERROR)<< "Received X error:\n"
  << "    Request: " << int(e->request_code)
  << " - " << XRequestCodeToString(e->request_code) << "\n"
  << "    Error code: " << int(e->error_code)
  << " - " << error_text << "\n"
  << "    Resource ID: " << e->resourceid;
  return 0;
}

// If the process receives a signal, just dump it into the write end of a pipe
// so that the main loop can process it asynchronously.
void OnProcessSignal(int signum) {
  g_should_quit = true;
}

void DispatchEvent(XEvent* event) {
  switch (event->type) {
    case ButtonPress:
        OnButtonPress(event->xbutton);
        break;
    case CreateNotify:
      OnCreateNotify(event->xcreatewindow);
      break;
    case DestroyNotify:
      OnDestroyNotify(event->xdestroywindow);
      break;
    case ConfigureNotify:
      OnConfigureNotify(event->xconfigure);
      break;
    case MapRequest:
      OnMapRequest(event->xmaprequest);
      break;
    case MotionNotify:
      // Skip any already pending motion events.
      while (XCheckTypedWindowEvent(g_display, event->xmotion.window,
            MotionNotify,event)) {}
      OnMotionNotify(event->xmotion);
      break;
    case ConfigureRequest:
      OnConfigureRequest(event->xconfigurerequest);
      break;
    case Expose:
      OnExpose(event->xexpose);
      break;
    case PropertyNotify:
      OnPropertyChange(event->xproperty);
      break;
    default:
      LOG(WARNING)<< "Ignored event";
    }
  }

void PumpXQueue() {
  while (XPending(g_display)) {
    XEvent e;
    XNextEvent(g_display, &e);
    LOG(INFO)<< "Received event: " << XEventToString(e);
    DispatchEvent(&e);
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  Pango::init();
  g_theme.reset(new Theme());

  // Set up our signal handler
  signal(SIGINT, OnProcessSignal);
  signal(SIGHUP, SIG_IGN);

  // Open connection to  X display.server
  const char* display_c_str =
      FLAGS_display.empty() ? nullptr : FLAGS_display.c_str();
  LOG(INFO)<< "Opening X display " << FLAGS_display;
  g_display = XOpenDisplay(display_c_str);
  if (g_display == nullptr) {
    LOG(ERROR)<< "Failed to open X display " << XDisplayName(display_c_str);
    return 1;
  }

  // initialize program state
  WM_PROTOCOLS = XInternAtom(g_display, "WM_PROTOCOLS", false);
  WM_DELETE_WINDOW = XInternAtom(g_display, "WM_DELETE_WINDOW", false);
  WM_NAME = XInternAtom(g_display, "WM_NAME", false);

  // Get the window attributes for the root window on each screen
  int num_screens = ScreenCount(g_display);
  if (FLAGS_screen < 0 || FLAGS_screen >= num_screens) {
    LOG(WARNING)<< FormatStr("Selected screen number %d is outside the valid "
        "range [0,%d)",FLAGS_screen,num_screens);
    FLAGS_screen = 0;
  }
  LOG(INFO)<< FormatStr("Using X Screen number: (%d/%d)", FLAGS_screen,
      num_screens);
  g_root_win = XRootWindow(g_display, FLAGS_screen);
  g_visual = XDefaultVisual(g_display, FLAGS_screen);

  if (XMatchVisualInfo(g_display, FLAGS_screen, 32, TrueColor,
                       &g_vinfo_truecolor) > 0) {
    g_vinfo_matched = true;
  }

  // Select events on root window. We install a special error handler so that
  // we can detect that another window manager has already selecte
  // SubstructureRedirect on the root window.
  XSetErrorHandler(&OnWMDetected);
  XSelectInput(g_display, g_root_win, SubstructureNotifyMask);

  // Force a sync so that we can know whether or not our XSelectInput call
  // has succeeded.
  XSync(g_display, false);
  if (g_wm_detected) {
    LOG(ERROR)<< "Detected another window manager on display "
    << XDisplayString(g_display);
    return 1;
  }

  // Now install our regular error handlers
  XSetErrorHandler(&OnXError);

  // Grab X server to prevent windows from changing under us.
  XGrabServer(g_display);

  // Reparent existing top-level windows
  Window returned_root, returned_parent;
  Window* top_level_windows;
  unsigned int num_top_level_windows;

  CHECK(
      XQueryTree(g_display, g_root_win, &returned_root, &returned_parent,
                 &top_level_windows, &num_top_level_windows));
  CHECK_EQ(returned_root, g_root_win);
  for (unsigned int i = 0; i < num_top_level_windows; ++i) {
    Frame(top_level_windows[i]);
    XMapWindow(g_display, frame_of_client_[top_level_windows[i]]);
  }
  XFree(top_level_windows);
  XUngrabServer(g_display);

  while (!g_should_quit) {
    PumpXQueue();
  }
  LOG(INFO)<< "A clean exit";
  XCloseDisplay(g_display);
  return 0;
}
