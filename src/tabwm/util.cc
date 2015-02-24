#include "util.h"
#include <algorithm>
#include <map>
#include <sstream>
#include <vector>

// Joins a container of elements into a single string, assuming each element
// in the container has a stream operator defined.
template <typename Container>
std::string Join(const Container& container, const std::string& delimiter) {
  std::ostringstream out;
  for (auto i = container.cbegin(); i != container.cend(); ++i) {
    if (i != container.cbegin()) {
      out << delimiter;
    }
    out << *i;
  }
  return out.str();
}

static std::vector<std::string> kEventTypeNames = {
  "",
  "",
  "KeyPress",
  "KeyRelease",
  "ButtonPress",
  "ButtonRelease",
  "MotionNotify",
  "EnterNotify",
  "LeaveNotify",
  "FocusIn",
  "FocusOut",
  "KeymapNotify",
  "Expose",
  "GraphicsExpose",
  "NoExpose",
  "VisibilityNotify",
  "CreateNotify",
  "DestroyNotify",
  "UnmapNotify",
  "MapNotify",
  "MapRequest",
  "ReparentNotify",
  "ConfigureNotify",
  "ConfigureRequest",
  "GravityNotify",
  "ResizeRequest",
  "CirculateNotify",
  "CirculateRequest",
  "PropertyNotify",
  "SelectionClear",
  "SelectionRequest",
  "SelectionNotify",
  "ColormapNotify",
  "ClientMessage",
  "MappingNotify",
};

std::string XEventToString(const XEvent& e) {
  std::stringstream properties_strm;
  if (0 < e.type && e.type < kEventTypeNames.size()) {
    properties_strm << kEventTypeNames[e.type] << "\n";
  } else {
    properties_strm << FormatStr("Unknown event type [%d]\n", e.type);
  }
  switch (e.type) {
    case CreateNotify: {
      const XCreateWindowEvent& ee = e.xcreatewindow;
      properties_strm
          << FormatStr("window: %ul\n", ee.window)
          << FormatStr("parent: %ul\n", ee.parent)
          << FormatStr("size: (%d,%d)\n", ee.width, ee.height)
          << FormatStr("position: (%d,%d)\n", ee.x, ee.y)
          << FormatStr("border_width: %d\n", ee.border_width)
          << FormatStr("override_redirect: %d\n", ee.override_redirect);
      break;
    }
    case DestroyNotify: {
      const XDestroyWindowEvent& ee = e.xdestroywindow;
      properties_strm << FormatStr("window: %ul\n", ee.window);
      break;
    }
    case MapNotify: {
      const XMapEvent& ee = e.xmap;
      properties_strm
          << FormatStr("window: %ul\n", ee.window)
          << FormatStr("event: %ul", ee.event)
          << FormatStr("override_redirect: %ul\n", ee.override_redirect);
      break;
    }
    case UnmapNotify: {
      const XUnmapEvent& ee = e.xunmap;
      properties_strm << FormatStr("window: %ul\n", ee.window)
                      << FormatStr("event: %ul", ee.event)
                      << FormatStr("from_configure: %ul\n", ee.from_configure);
      break;
    }
    case ConfigureNotify: {
      const XConfigureEvent& ee = e.xconfigure;
      properties_strm
          << FormatStr("window: %ul\n", ee.window)
          << FormatStr("size: (%d,%d)\n", ee.width, ee.height)
          << FormatStr("position: (%d,%d)\n", ee.x, ee.y)
          << FormatStr("border_width: %d\n", ee.border_width)
          << FormatStr("override_redirect: %d\n", ee.override_redirect);
      break;
    }
    case ReparentNotify: {
      const XReparentEvent& ee = e.xreparent;
      properties_strm
          << FormatStr("window: %ul\n", ee.window)
          << FormatStr("parent: %ul\n", ee.parent)
          << FormatStr("position: (%d,%d)\n", ee.x, ee.y)
          << FormatStr("override_redirect: %d", ee.override_redirect);
      break;
    }
    case MapRequest: {
      const XMapRequestEvent& ee = e.xmaprequest;
      properties_strm << FormatStr("window: %ul\n", ee.window);
      break;
    }
    case ConfigureRequest: {
      const XConfigureRequestEvent& ee = e.xconfigurerequest;
      std::vector<std::string> masks;
      if (ee.value_mask & CWX) {
        masks.emplace_back("X");
      }
      if (ee.value_mask & CWY) {
        masks.emplace_back("Y");
      }
      if (ee.value_mask & CWWidth) {
        masks.emplace_back("Width");
      }
      if (ee.value_mask & CWHeight) {
        masks.emplace_back("Height");
      }
      if (ee.value_mask & CWBorderWidth) {
        masks.emplace_back("BorderWidth");
      }
      if (ee.value_mask & CWSibling) {
        masks.emplace_back("Sibling");
      }
      if (ee.value_mask & CWStackMode) {
        masks.emplace_back("StackMode");
      }

      properties_strm << FormatStr("window: %ul\n", ee.window)
                      << FormatStr("parent; %ul\n", ee.parent)
                      << FormatStr("value_mask: %s\n", Join(masks, "|"))
                      << FormatStr("position: (%d,%d)\n", ee.x, ee.y)
                      << FormatStr("size: (%d,%d)\n", ee.width, ee.height)
                      << FormatStr("border_width: %d\n", ee.border_width);
      break;
    }
    case ButtonPress:
    case ButtonRelease: {
      const XButtonEvent& ee = e.xbutton;
      properties_strm
          << FormatStr("window: %ul\n", ee.window)
          << FormatStr("button: %ul\n", ee.button)
          << FormatStr("position_root: (%d,%d)\n", ee.x_root, ee.y_root);
      break;
    }
    case MotionNotify: {
      const XMotionEvent& ee = e.xmotion;
      properties_strm
          << FormatStr("window: %ul\n", ee.window)
          << FormatStr("position_root: (%d,%d)\n", ee.x_root, ee.y_root)
          << FormatStr("state: %ul\n", ee.state)
          << FormatStr("time: %ul\n", ee.time);
      break;
    }
    case KeyPress:
    case KeyRelease: {
      const XKeyEvent& ee = e.xkey;
      properties_strm << FormatStr("window: %ul\n", ee.window)
                      << FormatStr("state: %ul\n", ee.state)
                      << FormatStr("keycode: %ul\n", ee.keycode);
      break;
    }
    case Expose: {
      const XExposeEvent& ee = e.xexpose;
      properties_strm << FormatStr("window: %ul\n", ee.window)
                      << FormatStr("offset: (%d,%d)\n", ee.x, ee.y)
                      << FormatStr("size: (%d,%d)\n", ee.width, ee.height);
      break;
    }
    default:
      break;
  }

  return properties_strm.str();
}

static std::vector<std::string> kRequestCodeNames = {
  "",
  "CreateWindow",
  "ChangeWindowAttributes",
  "GetWindowAttributes",
  "DestroyWindow",
  "DestroySubwindows",
  "ChangeSaveSet",
  "ReparentWindow",
  "MapWindow",
  "MapSubwindows",
  "UnmapWindow",
  "UnmapSubwindows",
  "ConfigureWindow",
  "CirculateWindow",
  "GetGeometry",
  "QueryTree",
  "InternAtom",
  "GetAtomName",
  "ChangeProperty",
  "DeleteProperty",
  "GetProperty",
  "ListProperties",
  "SetSelectionOwner",
  "GetSelectionOwner",
  "ConvertSelection",
  "SendEvent",
  "GrabPointer",
  "UngrabPointer",
  "GrabButton",
  "UngrabButton",
  "ChangeActivePointerGrab",
  "GrabKeyboard",
  "UngrabKeyboard",
  "GrabKey",
  "UngrabKey",
  "AllowEvents",
  "GrabServer",
  "UngrabServer",
  "QueryPointer",
  "GetMotionEvents",
  "TranslateCoords",
  "WarpPointer",
  "SetInputFocus",
  "GetInputFocus",
  "QueryKeymap",
  "OpenFont",
  "CloseFont",
  "QueryFont",
  "QueryTextExtents",
  "ListFonts",
  "ListFontsWithInfo",
  "SetFontPath",
  "GetFontPath",
  "CreatePixmap",
  "FreePixmap",
  "CreateGC",
  "ChangeGC",
  "CopyGC",
  "SetDashes",
  "SetClipRectangles",
  "FreeGC",
  "ClearArea",
  "CopyArea",
  "CopyPlane",
  "PolyPoint",
  "PolyLine",
  "PolySegment",
  "PolyRectangle",
  "PolyArc",
  "FillPoly",
  "PolyFillRectangle",
  "PolyFillArc",
  "PutImage",
  "GetImage",
  "PolyText8",
  "PolyText16",
  "ImageText8",
  "ImageText16",
  "CreateColormap",
  "FreeColormap",
  "CopyColormapAndFree",
  "InstallColormap",
  "UninstallColormap",
  "ListInstalledColormaps",
  "AllocColor",
  "AllocNamedColor",
  "AllocColorCells",
  "AllocColorPlanes",
  "FreeColors",
  "StoreColors",
  "StoreNamedColor",
  "QueryColors",
  "LookupColor",
  "CreateCursor",
  "CreateGlyphCursor",
  "FreeCursor",
  "RecolorCursor",
  "QueryBestSize",
  "QueryExtension",
  "ListExtensions",
  "ChangeKeyboardMapping",
  "GetKeyboardMapping",
  "ChangeKeyboardControl",
  "GetKeyboardControl",
  "Bell",
  "ChangePointerControl",
  "GetPointerControl",
  "SetScreenSaver",
  "GetScreenSaver",
  "ChangeHosts",
  "ListHosts",
  "SetAccessControl",
  "SetCloseDownMode",
  "KillClient",
  "RotateProperties",
  "ForceScreenSaver",
  "SetPointerMapping",
  "GetPointerMapping",
  "SetModifierMapping",
  "GetModifierMapping",
  "NoOperation",
};

std::string XRequestCodeToString(unsigned char request_code) {
  if (0 < request_code && request_code < kRequestCodeNames.size()) {
    return kRequestCodeNames[request_code];
  } else {
    return "Unknown request code";
  }
}
