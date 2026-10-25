"""Génère assets/MacDictee.icns (icône de l'app) : fond dégradé + micro blanc + ondes.

Usage : .venv/bin/python tools/make_icon.py
"""
import subprocess
import tempfile
from pathlib import Path

from AppKit import (NSBezierPath, NSBitmapImageRep, NSColor, NSCompositingOperationSourceOver, NSGradient,
                    NSGraphicsContext, NSImage, NSMakeRect, NSPNGFileType)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "MacDictee.icns"


def rgb(r, g, b, a=1.0):
    return NSColor.colorWithSRGBRed_green_blue_alpha_(r / 255, g / 255, b / 255, a)


def draw(size):
    image = NSImage.alloc().initWithSize_((size, size))
    image.lockFocus()
    NSGraphicsContext.currentContext().setImageInterpolation_(3)
    s = size / 1024

    # plaque arrondie (marges et rayon des icônes macOS)
    plate = NSMakeRect(100 * s, 100 * s, 824 * s, 824 * s)
    path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(plate, 185 * s, 185 * s)
    NSGradient.alloc().initWithStartingColor_endingColor_(rgb(255, 94, 98), rgb(122, 60, 255)).drawInBezierPath_angle_(path, -60)

    white = rgb(255, 255, 255)
    soft = rgb(255, 255, 255, 0.55)
    # corps du micro
    white.setFill()
    NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(422 * s, 430 * s, 180 * s, 330 * s), 90 * s, 90 * s).fill()
    # arceau
    white.setStroke()
    arc = NSBezierPath.bezierPath()
    arc.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_((512 * s, 560 * s), 150 * s, 200, 340, False)
    arc.setLineWidth_(36 * s)
    arc.setLineCapStyle_(1)
    arc.stroke()
    # pied
    stem = NSBezierPath.bezierPath()
    stem.moveToPoint_((512 * s, 410 * s))
    stem.lineToPoint_((512 * s, 300 * s))
    stem.moveToPoint_((430 * s, 290 * s))
    stem.lineToPoint_((594 * s, 290 * s))
    stem.setLineWidth_(36 * s)
    stem.setLineCapStyle_(1)
    stem.stroke()
    # ondes sonores de part et d'autre
    soft.setStroke()
    for dx, r in ((0, 235), (0, 315)):
        for start, end in ((150, 210), (-30, 30)):
            wave = NSBezierPath.bezierPath()
            wave.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_((512 * s, 595 * s), r * s, start, end, False)
            wave.setLineWidth_(30 * s)
            wave.setLineCapStyle_(1)
            wave.stroke()
    image.unlockFocus()
    return image


def save_png(image, size, path):
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, size, size, 8, 4, True, False, "NSDeviceRGBColorSpace", 0, 0)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))
    image.drawInRect_fromRect_operation_fraction_(NSMakeRect(0, 0, size, size), NSMakeRect(0, 0, 0, 0),
                                                  NSCompositingOperationSourceOver, 1.0)
    NSGraphicsContext.restoreGraphicsState()
    rep.representationUsingType_properties_(NSPNGFileType, None).writeToFile_atomically_(str(path), True)


def main():
    OUT.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "MacDictee.iconset"
        iconset.mkdir()
        for base in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                px = base * scale
                name = f"icon_{base}x{base}{'@2x' if scale == 2 else ''}.png"
                save_png(draw(px), px, iconset / name)
        save_png(draw(1024), 1024, OUT.with_suffix(".png"))
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(OUT)], check=True)
    print(f"Icône écrite : {OUT}")


if __name__ == "__main__":
    main()
