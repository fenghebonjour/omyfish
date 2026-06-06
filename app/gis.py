from typing import Optional
from PIL import Image


def extract_exif_gps(image: Image.Image) -> Optional[tuple]:
    """Return (lat, lon) from image EXIF GPS tags, or None if unavailable."""
    try:
        import piexif
        exif_bytes = image.info.get("exif")
        if not exif_bytes:
            return None
        exif = piexif.load(exif_bytes)
        gps = exif.get("GPS", {})
        lat_tag = piexif.GPSIFD.GPSLatitude
        lon_tag = piexif.GPSIFD.GPSLongitude
        if lat_tag not in gps or lon_tag not in gps:
            return None

        def to_decimal(dms, ref):
            d = dms[0][0] / dms[0][1]
            m = dms[1][0] / dms[1][1]
            s = dms[2][0] / dms[2][1]
            v = d + m / 60 + s / 3600
            return -v if ref in (b"S", b"W") else v

        lat = to_decimal(gps[lat_tag], gps[piexif.GPSIFD.GPSLatitudeRef])
        lon = to_decimal(gps[lon_tag], gps[piexif.GPSIFD.GPSLongitudeRef])
        return lat, lon
    except Exception:
        return None
