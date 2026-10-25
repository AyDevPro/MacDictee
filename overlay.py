"""Pastille flottante en bas de l'écran pendant la dictée (comme Wispr Flow).

Fenêtre « panel » non activable : elle ne vole jamais le focus du champ où l'on dicte, ignore la souris
et apparaît sur tous les bureaux, y compris par-dessus une app en plein écran.
"""
from collections import deque

import objc
import Quartz  # noqa: F401 (déclare le type CGColor pour layer.setBackgroundColor_)
from AppKit import (NSBackingStoreBuffered, NSBezierPath, NSColor, NSEvent, NSFont, NSMakeRect, NSPanel,
                    NSScreen, NSStatusWindowLevel, NSTextField, NSView, NSVisualEffectView,
                    NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorFullScreenAuxiliary,
                    NSWindowCollectionBehaviorStationary, NSWindowStyleMaskBorderless,
                    NSWindowStyleMaskNonactivatingPanel)
from Foundation import NSObject, NSTimer

WIDTH, HEIGHT = 230, 44
BARS = 14
MARGIN_BOTTOM = 28


class _Meter(NSView):
    """Vu-mètre : barres verticales qui défilent avec le niveau du micro."""

    def initWithFrame_(self, frame):
        self = objc.super(_Meter, self).initWithFrame_(frame)
        if self is not None:
            self.levels = deque([0.0] * BARS, maxlen=BARS)
            self.color = NSColor.whiteColor()
        return self

    def push_(self, level):
        self.levels.append(level)
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        w, h = self.bounds().size.width, self.bounds().size.height
        step = w / BARS
        self.color.setFill()
        for i, level in enumerate(self.levels):
            bar_h = max(3.0, min(h, h * level))
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(i * step + step * 0.2, (h - bar_h) / 2, step * 0.6, bar_h), 1.5, 1.5).fill()


class _Ticker(NSObject):
    def initWithCallback_(self, callback):
        self = objc.super(_Ticker, self).init()
        if self is not None:
            self.callback = callback
        return self

    def tick_(self, timer):
        self.callback()


class Overlay:
    """show(mode) / hide(), à appeler sur le thread principal. level() renvoie le niveau du micro (0..1)."""

    TEXTS = {"ecoute": "J'écoute…", "mainslibres": "Mains libres", "transcription": "Transcription…"}

    def __init__(self, level):
        self.level = level
        self.panel = None
        self.timer = None
        self.mode = None

    def _build(self):
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, WIDTH, HEIGHT), style, NSBackingStoreBuffered, False)
        self.panel.setLevel_(NSStatusWindowLevel)
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(NSColor.clearColor())
        self.panel.setHasShadow_(True)
        self.panel.setIgnoresMouseEvents_(True)
        self.panel.setHidesOnDeactivate_(False)
        self.panel.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces
                                          | NSWindowCollectionBehaviorFullScreenAuxiliary
                                          | NSWindowCollectionBehaviorStationary)

        bg = NSVisualEffectView.alloc().initWithFrame_(NSMakeRect(0, 0, WIDTH, HEIGHT))
        bg.setMaterial_(13)  # NSVisualEffectMaterialHUDWindow
        bg.setBlendingMode_(0)  # derrière la fenêtre
        bg.setState_(1)  # toujours actif
        bg.setWantsLayer_(True)
        bg.layer().setCornerRadius_(HEIGHT / 2)
        bg.layer().setMasksToBounds_(True)
        self.panel.setContentView_(bg)

        self.dot = NSView.alloc().initWithFrame_(NSMakeRect(18, (HEIGHT - 10) / 2, 10, 10))
        self.dot.setWantsLayer_(True)
        self.dot.layer().setCornerRadius_(5)
        bg.addSubview_(self.dot)

        self.label = NSTextField.labelWithString_("")
        self.label.setFont_(NSFont.systemFontOfSize_weight_(13, 0.3))
        self.label.setTextColor_(NSColor.whiteColor())
        self.label.setFrame_(NSMakeRect(36, (HEIGHT - 18) / 2, 110, 18))
        bg.addSubview_(self.label)

        self.meter = _Meter.alloc().initWithFrame_(NSMakeRect(WIDTH - 18 - 66, (HEIGHT - 22) / 2, 66, 22))
        bg.addSubview_(self.meter)

        self._ticker = _Ticker.alloc().initWithCallback_(self._tick)

    def _place(self):
        mouse = NSEvent.mouseLocation()
        screen = next((s for s in NSScreen.screens() if _contains(s.frame(), mouse)), NSScreen.mainScreen())
        vf = screen.visibleFrame()  # au-dessus du Dock
        x = vf.origin.x + (vf.size.width - WIDTH) / 2
        self.panel.setFrameOrigin_((x, vf.origin.y + MARGIN_BOTTOM))

    def show(self, mode):
        if self.panel is None:
            self._build()
        if self.mode is None:
            self._place()
        self.mode = mode
        self.label.setStringValue_(self.TEXTS[mode])
        color = NSColor.systemOrangeColor() if mode == "transcription" else NSColor.systemRedColor()
        self.dot.layer().setBackgroundColor_(color.CGColor())
        self.meter.setHidden_(mode == "transcription")
        self.panel.orderFrontRegardless()
        if self.timer is None and mode != "transcription":
            self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1 / 20, self._ticker, "tick:", None, True)
        elif mode == "transcription":
            self._stop_timer()

    def hide(self):
        self.mode = None
        self._stop_timer()
        if self.panel is not None:
            self.panel.orderOut_(None)

    def _stop_timer(self):
        if self.timer is not None:
            self.timer.invalidate()
            self.timer = None

    def _tick(self):
        # la voix normale tourne autour de 0,02-0,1 en RMS : on étire pour que les barres bougent bien
        self.meter.push_(min(1.0, (self.level() * 12) ** 0.7))


def _contains(frame, point):
    return (frame.origin.x <= point.x <= frame.origin.x + frame.size.width
            and frame.origin.y <= point.y <= frame.origin.y + frame.size.height)
