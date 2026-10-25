"""Fenêtre principale de MacDictée : une page HTML (ui/index.html) dans un WKWebView.

Le JavaScript appelle Python via `window.webkit.messageHandlers.md.postMessage(json)` ;
Python répond avec `md._reply(id, ok, data)` et pousse des événements avec `md.onEvent(nom, data)`.
"""
import json
import os
from pathlib import Path

import objc
from AppKit import (NSApp, NSBackingStoreBuffered, NSMakeRect, NSMenu, NSMenuItem, NSWindow,
                    NSWindowStyleMaskClosable, NSWindowStyleMaskMiniaturizable, NSWindowStyleMaskResizable,
                    NSWindowStyleMaskTitled)
from Foundation import NSURL, NSObject
from PyObjCTools.AppHelper import callAfter
from WebKit import WKUserContentController, WKWebView, WKWebViewConfiguration

# app autonome (py2app) : ui/ est copié dans Contents/Resources ; sinon, à côté de ce fichier
UI_DIR = Path(os.environ.get("RESOURCEPATH", Path(__file__).resolve().parent)) / "ui"
if not UI_DIR.exists():
    UI_DIR = Path(__file__).resolve().parent / "ui"


class _MessageHandler(NSObject, protocols=[objc.protocolNamed("WKScriptMessageHandler")]):
    def initWithCallback_(self, callback):
        self = objc.super(_MessageHandler, self).init()
        if self is not None:
            self.callback = callback
        return self

    def userContentController_didReceiveScriptMessage_(self, controller, message):
        self.callback(str(message.body()))


def install_edit_menu():
    """Une app de barre de menus n'a pas de menu Édition : sans lui, Cmd+C / Cmd+V ne marchent pas dans la fenêtre."""
    main = NSMenu.alloc().init()
    app_item = NSMenuItem.alloc().init()
    main.addItem_(app_item)
    app_menu = NSMenu.alloc().initWithTitle_("MacDictée")
    app_menu.addItemWithTitle_action_keyEquivalent_("Fermer la fenêtre", "performClose:", "w")
    app_menu.addItemWithTitle_action_keyEquivalent_("Quitter MacDictée", "terminate:", "q")
    app_item.setSubmenu_(app_menu)
    edit_item = NSMenuItem.alloc().init()
    main.addItem_(edit_item)
    edit = NSMenu.alloc().initWithTitle_("Édition")
    for title, action, key in [("Annuler", "undo:", "z"), ("Rétablir", "redo:", "Z"), ("Couper", "cut:", "x"),
                               ("Copier", "copy:", "c"), ("Coller", "paste:", "v"),
                               ("Tout sélectionner", "selectAll:", "a")]:
        edit.addItemWithTitle_action_keyEquivalent_(title, action, key)
    edit_item.setSubmenu_(edit)
    NSApp.setMainMenu_(main)


class Window:
    def __init__(self, handle):
        self.handle = handle  # handle(action, args) -> données JSON-sérialisables
        self.window = None
        self.webview = None

    def _build(self):
        config = WKWebViewConfiguration.alloc().init()
        controller = WKUserContentController.alloc().init()
        self._handler = _MessageHandler.alloc().initWithCallback_(self._on_message)
        controller.addScriptMessageHandler_name_(self._handler, "md")
        config.setUserContentController_(controller)

        style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
                 | NSWindowStyleMaskResizable)
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 900, 620), style, NSBackingStoreBuffered, False)
        self.window.setTitle_("MacDictée")
        self.window.setMinSize_((720, 480))
        self.window.setReleasedWhenClosed_(False)  # fermer = cacher, on la rouvre depuis le menu
        self.window.center()
        self.window.setFrameAutosaveName_("MacDicteeMain")

        self.webview = WKWebView.alloc().initWithFrame_configuration_(self.window.contentView().bounds(), config)
        self.webview.setAutoresizingMask_(18)  # largeur + hauteur suivent la fenêtre
        self.window.contentView().addSubview_(self.webview)
        index = UI_DIR / "index.html"
        self.webview.loadFileURL_allowingReadAccessToURL_(NSURL.fileURLWithPath_(str(index)),
                                                          NSURL.fileURLWithPath_(str(UI_DIR)))

    def show(self):
        if self.window is None:
            self._build()
        self.window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)
        self._js("window.refresh && refresh()")  # rattrape ce qui a changé pendant que la fenêtre était fermée

    def visible(self):
        return self.window is not None and self.window.isVisible()

    def _on_message(self, raw):
        msg = json.loads(raw)
        try:
            data, ok = self.handle(msg["action"], msg.get("args") or {}), True
        except Exception as e:
            data, ok = str(e), False
        self._js(f"md._reply({json.dumps(msg.get('id'))}, {json.dumps(ok)}, {json.dumps(data, ensure_ascii=False)})")

    def emit(self, name, data=None):
        """Utilisable depuis n'importe quel thread."""
        self._js(f"md.onEvent({json.dumps(name)}, {json.dumps(data, ensure_ascii=False)})")

    def _js(self, code):
        if self.webview is not None:
            callAfter(self.webview.evaluateJavaScript_completionHandler_, code, None)
