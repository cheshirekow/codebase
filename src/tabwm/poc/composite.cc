#include <csignal>
#include <string>

#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/extensions/Xcomposite.h>
#include <X11/extensions/Xfixes.h>
#include <X11/extensions/shape.h>

bool g_signaled = false;
Display* g_dpy = nullptr;

void signal_handler(int signum){
  g_signaled = true;
  printf("Signaled\n");
}


void OnCreateNotify(XCreateWindowEvent* event) {
  printf("Window created\n");
}

void OnDestroyNotify(XDestroyWindowEvent* event) {
  printf("Window destroyed\n");
}

void OnMapRequest(XMapRequestEvent* event) {
  XMapWindow(g_dpy,event->window);
}

int main(int argc, char** argv) {
  signal(SIGINT, signal_handler);
  std::string display_name = ":0";

  char* env_display_name = getenv("DISPLAY");
  if (env_display_name) {
    display_name = env_display_name;
  }
  printf("Using display %s\n", display_name.c_str());
  g_dpy = XOpenDisplay(display_name.c_str());

  int event_base, error_base;
  if (XCompositeQueryExtension(g_dpy, &event_base, &error_base)) {
    printf("Server has composite extension\n");

    int major = 0, minor = 2;  // The highest version we support
    XCompositeQueryVersion(g_dpy, &major, &minor);
    printf("Composite extension %d.%d\n", major, minor);

    // major and minor will now contain the highest version the server supports.
    // The protocol specifies that the returned version will never be higher
    // then the one requested. Version 0.2 is the first version to have the
    // XCompositeNameWindowPixmap() request.
    if (major > 0 || minor >= 2) {
      printf("Supports named pixmaps\n");
    } else {
      fprintf(stderr, "XServer does not have named pixmaps\n");
      return 1;
    }
  } else {
    fprintf(stderr, "XServer does not have composite extension\n");
    return 1;
  }

  // Tell the X server to redirect all drawing to an offscreen buffer
  for (int i = 0; i < ScreenCount(g_dpy); i++) {
    Window root = XRootWindow(g_dpy,i);
    XCompositeRedirectSubwindows(g_dpy, root, CompositeRedirectAutomatic);
    printf("Selecting input on screen: %d\n", i);
    XSelectInput(g_dpy, root,
    //ResizeRedirectMask |
                 SubstructureRedirectMask | SubstructureNotifyMask);
    XSync(g_dpy, false);
  }

  XEvent event;
  while (!g_signaled) {
    while (XPending(g_dpy)) {
      XNextEvent(g_dpy, &event);

      switch (event.type) {
        case CreateNotify:
          OnCreateNotify(&event.xcreatewindow);
          break;
        case DestroyNotify:
          OnDestroyNotify(&event.xdestroywindow);
          break;
        case MapNotify:
          printf("Map Notify\n");
          break;
        case MapRequest:
          OnMapRequest(&event.xmaprequest);
          break;
        case ConfigureNotify:
          printf("Configure Notify\n");
          break;
        case UnmapNotify:
          printf("Unmap Notify\n");
          break;
        case ClientMessage:
          printf("Client Message\n");
          break;
        case PropertyNotify:
          printf("Property Notify\n");
          break;
        case KeyPress:
          printf("Key Press\n");
          break;
        case ButtonPress:
          printf("Button Press\n");
          break;
        case EnterNotify:
          printf("Enter Notify\n");
          break;
        case LeaveNotify:
          printf("Leaf Notify\n");
          break;
        case FocusIn:
          printf("Focus In\n");
          break;
        case KeyRelease:
          printf("Key Release\n");
          break;
        case VisibilityNotify:
          printf("Visibility Notify\n");
          break;
        case ColormapNotify:
          printf("Colormap Notify\n");
          break;
        case MappingNotify:
          printf("Mapping Notify\n");
          break;
        case MotionNotify:
          printf("Motion Notify\n");
          break;
        case SelectionNotify:
          printf("Selection Notify\n");
          break;
        default:
          break;
      }
    }
    usleep(100);
  }

  XCloseDisplay(g_dpy);
  return 0;
}
