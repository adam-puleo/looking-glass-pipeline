"""
apple_depth_to_lookingglass.py

Converts an Apple iPhone portrait or depth-enabled photo to an MP4 that
Looking Glass Studio can import directly as an RGB-D video.

The RGB-D format is a side-by-side video: color on the left half, grayscale
depth on the right half.  Looking Glass Studio recognizes this when you import
the file and select "RGB-D Photo/Video".

Supports:
  - JPEG with MPF-embedded depth (portrait mode, "Most Compatible" setting)
  - HEIC with auxiliary depth images (requires pillow-heif)

Usage:
  python apple_depth_to_lookingglass.py input.jpg output.mp4
  python apple_depth_to_lookingglass.py input.heic output.mp4
  python apple_depth_to_lookingglass.py input.jpg output.mp4 --duration 30
  python apple_depth_to_lookingglass.py input.jpg output.mp4 --codec h265
  python apple_depth_to_lookingglass.py input.jpg output.mp4 --max-width 1920
  python apple_depth_to_lookingglass.py input.jpg output.mp4 --invert-depth

Requirements:
  pip install Pillow numpy
  pip install pillow-heif          # only needed for HEIC input
  ffmpeg must be installed and on PATH
"""

import argparse
import concurrent.futures
import io
import logging
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import PIL.Image
import numpy as np
from PIL import Image, ImageCms, ImageFilter, ImageOps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JPEG / MPF parsing
# ---------------------------------------------------------------------------

JPEG_SOI  = b'\xff\xd8'
JPEG_APP1 = b'\xff\xe1'
JPEG_APP2 = b'\xff\xe2'
MPF_MAGIC = b'MPF\x00'

# XMP namespace tag that identifies an auxiliary depth image
XMP_DEPTH_TYPE = b'GDepth:MimeType'
XMP_DISPARITY_TYPE = b'GFocus:BlurAtInfinity'
APPLE_DISPARITY_XMP = b'apple:depthData'   # appears in Apple disparity XMP


def _read_be16(data, offset):
    return struct.unpack_from('>H', data, offset)[0]


def _read_be32(data, offset):
    return struct.unpack_from('>I', data, offset)[0]


def find_jpeg_markers(data):
    """Walk a JPEG byte string and return a list of (marker, offset, length)."""
    markers = []
    i = 0
    n = len(data)
    while i < n - 1:
        if data[i] != 0xff:
            break
        marker = data[i:i+2]
        if marker == b'\xff\xd8':          # SOI – no length field
            markers.append((marker, i, 0))
            i += 2
            continue
        if marker in (b'\xff\xd9', b'\xff\xda'):  # EOI / SOS
            markers.append((marker, i, 0))
            break
        if i + 3 >= n:
            break
        length = _read_be16(data, i + 2)   # includes the 2-byte length itself
        markers.append((marker, i, length))
        i += 2 + length
    return markers


def parse_mpf(data, app2_offset, app2_length):
    """
    Parse the MPF (Multi-Picture Format) APP2 block.
    Returns a list of dicts: {index, offset, length, flags}
    Offsets are absolute positions within `data` (the whole JPEG).
    """
    # APP2 payload starts after 0xFFE2 (2) + length field (2) + 'MPF\0' (4)
    base = app2_offset + 2 + 2 + 4          # absolute base for IFD offsets
    payload_start = app2_offset + 2 + 2 + 4

    byte_order = data[payload_start:payload_start+2]
    big_endian = (byte_order == b'MM')

    def r16(off):
        raw = data[base + off: base + off + 2]
        return struct.unpack('>H' if big_endian else '<H', raw)[0]

    def r32(off):
        raw = data[base + off: base + off + 4]
        return struct.unpack('>I' if big_endian else '<I', raw)[0]

    ifd_offset = r32(4)          # offset to IFD from base
    entry_count = r16(ifd_offset)

    entries = {}
    for e in range(entry_count):
        entry_off = ifd_offset + 2 + e * 12
        tag   = r16(entry_off)
        # type  = r16(entry_off + 2)
        # count = r32(entry_off + 4)
        value_off = entry_off + 8
        entries[tag] = value_off

    # Tag 0xB002 = MP Entry – 16 bytes per image
    if 0xB002 not in entries:
        return []

    mp_entry_offset = entries[0xB002]
    # For >4 bytes the value field holds an offset to the data
    mp_data_ptr = r32(mp_entry_offset)

    images = []
    img_offset_in_ifd = ifd_offset + 2 + entry_count * 12 + 4  # skip next IFD ptr

    # The actual MP Entries live at mp_data_ptr from base
    ptr = mp_data_ptr
    idx = 0
    while True:
        # Each MP entry is 16 bytes
        raw = data[base + ptr: base + ptr + 16]
        if len(raw) < 16:
            break

        if big_endian:
            attrs, size, img_off, dep1, dep2 = struct.unpack('>IIIHH', raw)
        else:
            attrs, size, img_off, dep1, dep2 = struct.unpack('<IIIHH', raw)

        # Image offset for the first image is always 0 (= start of JPEG SOI)
        abs_offset = img_off if idx > 0 else 0
        if idx > 0:
            # Offsets are relative to the end of the APP2 marker+length field
            abs_offset = img_off + app2_offset + 2 + 2

        images.append({
            'index': idx,
            'offset': abs_offset,
            'length': size,
            'attrs': attrs,
        })

        ptr += 16
        idx += 1

        # Stop when we've read as many entries as implied by the tag count
        if idx >= 10:   # safety cap
            break
        # Heuristic stop: if offset/size look invalid
        if size == 0 or (idx > 1 and img_off == 0):
            break

    return images


def xmp_suggests_depth(xmp_bytes):
    """
    Return True if the XMP block looks like it describes a depth/disparity image.
    Checks for known Apple and Google depth XMP markers.
    """
    markers = [
        b'apple:depthData',
        b'apple:focusPoint',
        b'GDepth:',
        b'GFocus:',
        b'IsDepthFiltered',
        b'CalibrationData',
        b'DepthMeasureType',
    ]
    for m in markers:
        if m in xmp_bytes:
            return True
    return False


def extract_sub_jpeg_xmp(sub_data):
    """Find and return the XMP payload from a sub-JPEG byte string."""
    i = 2   # skip SOI
    while i < len(sub_data) - 3:
        if sub_data[i] != 0xff:
            break
        marker = sub_data[i:i+2]
        if marker == b'\xff\xda':
            break
        length = _read_be16(sub_data, i + 2)
        if marker == b'\xff\xe1':
            payload = sub_data[i+4: i+2+length]
            if payload.startswith(b'http://ns.adobe.com/xap'):
                return payload
        i += 2 + length
    return b''


def extract_depth_jpeg_mpf(jpeg_data, tag: str):
    """
    Extract (color_image, depth_image) from a JPEG with MPF depth.

    Strategy:
      1. Find the APP2/MPF block.
      2. Parse sub-image offsets.
      3. For each sub-image after index 0, inspect its XMP to find the
         disparity/depth image. Fall back to MPImage2 if XMP is absent.

    Returns (PIL.Image color, PIL.Image depth_gray) or raises ValueError.
    """
    markers = find_jpeg_markers(jpeg_data)
    app2_info = None
    for marker, off, length in markers:
        if marker == JPEG_APP2:
            payload = jpeg_data[off+4: off+2+length]
            if payload.startswith(MPF_MAGIC):
                app2_info = (off, length)
                break

    if app2_info is None:
        raise ValueError("No MPF block found in JPEG – not an Apple depth photo?")

    mp_images = parse_mpf(jpeg_data, app2_info[0], app2_info[1])

    if len(mp_images) < 2:
        raise ValueError(
            f"MPF block found but only {len(mp_images)} image(s) listed – "
            "depth map is missing."
        )

    logger.info(f"{tag}   MPF: found {len(mp_images)} sub-image(s)")

    # Image 0 is always the main color photo
    color_data = jpeg_data  # whole file *is* image 0
    color_raw = Image.open(io.BytesIO(color_data))
    icc_profile = color_raw.info.get('icc_profile')
    color_img = color_raw.convert('RGB')

    # Search sub-images for depth/disparity, starting at index 1
    depth_img = None
    for entry in mp_images[1:]:
        off = entry['offset']
        length = entry['length']
        if length == 0:
            # length=0 means "to end of file"
            sub = jpeg_data[off:]
        else:
            sub = jpeg_data[off: off + length]

        if not sub.startswith(JPEG_SOI):
            logger.warning(f"{tag}   Sub-image {entry['index']}: does not start with JPEG SOI, skipping")
            continue

        xmp = extract_sub_jpeg_xmp(sub)
        is_depth = xmp_suggests_depth(xmp) if xmp else False

        try:
            candidate = Image.open(io.BytesIO(sub))
            mode = candidate.mode
        except Exception as exc:
            logger.warning(f"{tag}   Sub-image {entry['index']}: cannot open ({exc}), skipping")
            continue

        logger.info(f"{tag}   Sub-image {entry['index']}: mode={mode}, size={candidate.size}, "
                    f"xmp_depth={is_depth}")

        # Prefer an image that XMP confirms as depth, OR that is grayscale
        if is_depth or mode in ('L', 'I', 'F'):
            depth_img = candidate.convert('L')
            logger.info(f"{tag}   → Selected sub-image {entry['index']} as depth map")
            break

    if depth_img is None:
        # Last resort: use whatever sub-image 1 is (the old MPImage2 assumption)
        logger.warning(f"{tag}   No XMP-confirmed depth found; falling back to MPImage2")
        entry = mp_images[1]
        off = entry['offset']
        length = entry['length'] or (len(jpeg_data) - off)
        sub = jpeg_data[off: off + length]
        depth_img = Image.open(io.BytesIO(sub)).convert('L')

    return color_img, depth_img, icc_profile


# ---------------------------------------------------------------------------
# HEIC parsing (via pillow-heif)
# ---------------------------------------------------------------------------

def extract_depth_heic(heic_path, tag: str):
    """
    Extract (color_image, depth_image) from a HEIC file using pillow-heif.
    """
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        raise ImportError(
            "pillow-heif is required for HEIC files. "
            "Install with: pip install pillow-heif"
        )

    img_raw = Image.open(heic_path)
    icc_profile = img_raw.info.get('icc_profile')
    # if icc_profile:
    #     try:
    #         src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
    #         icc_profile = ImageCms.createProfile('sRGB')
    #         color_img = ImageCms.profileToProfile(
    #             img_raw.convert('RGB'),
    #             src_profile,
    #             icc_profile,
    #             renderingIntent=ImageCms.Intent.PERCEPTUAL,
    #             outputMode='RGB',
    #         )
    #     except Exception as e:
    #         print(f"  WARNING: ICC conversion failed ({e}), using raw pixel values")
    #         color_img = img_raw.convert('RGB')
    # else:
    #     color_img = img_raw.convert('RGB')

    depth_images = img_raw.info.get('depth_images', [])
    if not depth_images:
        raise ValueError(
            "No depth images found in HEIC file. "
            "Make sure this is a Portrait mode photo."
        )

    logger.info(f"{tag}   HEIC: found {len(depth_images)} depth image(s)")
    depth_raw = depth_images[0]

    # pillow-heif depth images may be 16-bit; normalize to 8-bit L
    depth_arr = np.array(depth_raw)
    logger.info(f"{tag}   Depth array dtype={depth_arr.dtype}, shape={depth_arr.shape}, "
                f"min={depth_arr.min()}, max={depth_arr.max()}")

    if depth_arr.dtype in (np.float16, np.float32, np.float64):
        # Disparity values – higher = closer. Normalize 0-255.
        finite = depth_arr[np.isfinite(depth_arr)]
        if len(finite) == 0:
            raise ValueError("Depth map contains only non-finite values.")
        lo, hi = finite.min(), finite.max()
        if hi == lo:
            raise ValueError("Depth map is uniform – no depth information.")
        norm = ((depth_arr - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
    elif depth_arr.dtype == np.uint16:
        norm = (depth_arr / 256).astype(np.uint8)
    else:
        norm = depth_arr.astype(np.uint8)

    depth_img = Image.fromarray(norm, mode='L')
    # return color_img, depth_img, icc_profile
    return img_raw, depth_img, icc_profile


# ---------------------------------------------------------------------------
# Depth map processing
# ---------------------------------------------------------------------------

def process_depth(depth_img: PIL.Image.Image, color_size, invert=False, blur_radius=2):
    """
    Prepare the depth map for Looking Glass RGB-D:
      - Resize to match the color image
      - Optionally invert (Apple stores disparity: bright=near;
        some Looking Glass tools expect dark=near)
      - Light smoothing to reduce JPEG compression artifacts at edges
      - Convert to RGB (Looking Glass RGB-D expects color image and color depth
        side by side; a depth channel is encoded as gray in all three channels)
    """
    # Upscale with high-quality resampling.
    # BICUBIC is used rather than LANCZOS because it avoids ringing artifacts
    # at hard depth edges and works correctly across all Pillow versions.
    depth_resized = depth_img.resize(color_size, Image.Resampling.BICUBIC)

    if invert:
        depth_resized = ImageOps.invert(depth_resized)

    if blur_radius > 0:
        depth_resized = depth_resized.filter(
            ImageFilter.GaussianBlur(radius=blur_radius)
        )

    # Convert to RGB so both halves of the side-by-side are the same mode
    depth_rgb = depth_resized.convert('RGB')
    return depth_rgb


def scale_image(img: PIL.Image.Image, max_width: int, max_height: int):
    """
    Downscale img so that it fits within max_width x max_height,
    preserving the aspect ratio.  Returns the image unchanged if it
    already fits.
    """
    w, h = img.size
    if w <= max_width and h <= max_height:
        return img
    scale = min(max_width / w, max_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.Resampling.BICUBIC)


def enforce_even(img: PIL.Image.Image):
    """
    Crop the image to even pixel dimensions if necessary.
    yuv420p (required by H.264 and H.265) demands both width and height be even.
    Cropping one pixel from the bottom/right is unnoticeable in practice.
    """
    w, h = img.size
    new_w = w if w % 2 == 0 else w - 1
    new_h = h if h % 2 == 0 else h - 1
    if (new_w, new_h) != (w, h):
        img = img.crop((0, 0, new_w, new_h))
    return img


def build_rgbd(color_img: PIL.Image.Image, depth_rgb: PIL.Image.Image):
    """
    Concatenate color (left) and depth (right) side by side and ensure the
    resulting canvas has even dimensions (required for yuv420p encoding).
    Both halves must be the same size on entry.
    """
    # Enforce even dimensions on each half before compositing so both halves
    # are always identical in size regardless of rounding.
    color_img = enforce_even(color_img)
    depth_rgb  = enforce_even(depth_rgb)

    w, h = color_img.size
    rgbd = Image.new('RGB', (w * 2, h))
    rgbd.paste(color_img, (0, 0))
    rgbd.paste(depth_rgb, (w, 0))
    return rgbd


# ---------------------------------------------------------------------------
# FFmpeg encode
# ---------------------------------------------------------------------------

# Pixel count threshold above which H.265 is chosen automatically.
# 1920 * 1080 * 4 = ~8 MP total canvas (color half would be ~4 MP / 2.7K).
_H265_PIXEL_THRESHOLD = 1920 * 1080 * 4


def icc_to_ffmpeg_color_params(icc_bytes):
    """
    Map an ICC profile (raw bytes) to FFmpeg colorspace/color_primaries/color_trc strings.
    Falls back to bt709 for unknown or missing profiles.
    """
    if not icc_bytes:
        return 'bt709', 'bt709', 'bt709'
    try:
        profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        desc = ImageCms.getProfileDescription(profile).strip().lower()
    except Exception:
        return 'bt709', 'bt709', 'bt709'

    if 'display p3' in desc or 'p3-d65' in desc:
        return 'bt709', 'smpte432', 'bt709'
    if 'bt.2020' in desc or 'bt2020' in desc:
        return 'bt2020nc', 'bt2020', 'bt2020-10'
    # sRGB, BT.709, or anything else
    return 'bt709', 'bt709', 'bt709'


def encode_mp4(
        rgbd_img: PIL.Image.Image,
        output_file: Path,
        tag: str,
        duration=10,
        codec='auto',
        crf: Optional[int]=None,
        colorspace='bt709',
        color_primaries='bt709',
        color_trc='bt709',
        extra_ffmpeg_args=None,
):
    """
    Encode the RGB-D PIL image as an MP4 that Looking Glass Studio can import
    directly as an RGB-D video.

    Args:
        rgbd_img:          PIL.Image — side-by-side RGB-D canvas.
        output_file:       File, including a path, to write output in.
        tag:               Name of the file currently being processed. Used
                           for generating per-file log messages.
        duration:          Length of the output video in seconds (default 10).
                           Looking Glass Studio loops playlist items, so even a
                           short duration works fine for still photos.
        codec:             'h264', 'h265', or 'auto'.  'auto' chooses H.264 for
                           canvases up to ~8 MP and H.265 above that threshold.
        crf:               H.264/H.265 quality (0=lossless, 51=worst; defaults
                           to 18 for H.264 and 22 for H.265). For other codecs
                           it is unused.
        extra_ffmpeg_args: Optional list of additional FFmpeg flags appended
                           before the output path.

    The output is always yuv420p, 30 fps, with the -movflags +faststart flag
    set, so the file is streamable and compatible with all Looking Glass players.
    """
    w, h = rgbd_img.size
    total_pixels = w * h

    ext = os.path.splitext(output_file)[1].lower()
    is_video = ext == '.mp4'

    if is_video:
        # Resolve codec choice
        if codec == 'auto':
            codec = 'h265' if total_pixels >= _H265_PIXEL_THRESHOLD else 'h264'

        match codec:
            case 'h264':
                encoder = 'libx264'
                crf = 18 if crf is None else crf
                vcodec_params = [
                    '-vcodec',    encoder,
                    '-profile:v', 'baseline',
                    '-level',     '3.1',
                    '-crf',       str(crf),
                ]
            case 'h265':
                encoder = 'libx265'
                crf = 22 if crf is None else crf
                vcodec_params = [
                    '-vcodec', encoder,
                    '-profile:v', 'baseline',
                    '-level',     '3.1',
                    '-crf',    str(crf),
                ]
            case 'hevc_videotoolbox':
                encoder = 'hevc_videotoolbox'  # Apple HW encoder
                bitrate = '35511k'  # VideoToolbox uses target bitrate, not CRF
                vcodec_params = [
                    '-vcodec', encoder,
                    # '-b:v',    bitrate,
                    '-q:v',    '5',  # 1 = best, 31 = worst
                    '-tag:v',  'hev1',  # Don't add the Apple tag.
                ]
            case _:
                raise ValueError(f"Unknown codec: {codec}")
        logger.info(f"{tag}   Canvas: {w}x{h} px  →  encoding as {encoder}, duration={duration}s, crf={crf}")
    else:
        logger.info(f"{tag}   Canvas: {w}x{h} px")

    # Write the RGB-D canvas to a temporary lossless PNG so FFmpeg gets the
    # exact pixels without any intermediate JPEG re-compression loss.
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        rgbd_img.save(tmp_path, format='PNG')

    try:
        cmd = [
            'ffmpeg', '-y',
        ]
        if is_video:
            cmd += [
                '-stream_loop', '-1',
                '-framerate',   '12',
            ]


        cmd += [
            '-colorspace',      colorspace,
            '-color_primaries', color_primaries,
            '-color_trc',       color_trc,
            '-i', tmp_path,
        ]

        if is_video:
            cmd += vcodec_params
            cmd += [
                '-color_range',     'tv',
                # '-pix_fmt',         'yuv420p',  # universal compatibility; required by LKG
                # '-vf',              'setsar=1/2',  # tells the player to display 3072px wide as 1536px wide
                '-vf',              'scale=in_range=full:out_range=limited,setsar=1/1,fps=30,format=yuv420p',
                '-movflags',        '+faststart',
                '-r',               '12',
                # Encode for exactly `duration` seconds
                '-t', str(duration),
            ]
        else:
            cmd += ['-q:v', '2']  # high-quality JPEG

        if extra_ffmpeg_args:
            cmd += extra_ffmpeg_args

        cmd.append(str(output_file))

        logger.info(f"{tag}   Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"{tag}   FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed with return code {result.returncode}")

        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"{tag}   FFmpeg finished → {output_file}  ({size_mb:.1f} MB)")

    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def detect_format(path: Path):
    """Sniff the first few bytes to detect JPEG vs. HEIC."""
    with open(path, 'rb') as f:
        header = f.read(12)
    if header[:2] == b'\xff\xd8':
        return 'jpeg'
    # HEIC/HEIF: ftyp box at offset 4 with 'heic', 'heix', 'heif', 'mif1', etc.
    if header[4:8] == b'ftyp' and header[8:12] in (
        b'heic', b'heix', b'heif', b'mif1', b'msf1', b'hevx'
    ):
        return 'heic'
    return 'unknown'


def process_file(input_file: Path, output_file: Path, params: dict) -> None:
    """
    Convert a single Apple depth photo to a Looking Glass RGB-D MP4.

    This function is the unit of work submitted to ProcessPoolExecutor.  It
    must be a top-level function (not a lambda or nested function) so that
    Python's multiprocessing pickler can serialize it for the worker process.

    All arguments are plain picklable types: Path objects and a dict of
    primitive values.  No PIL Images or other non-picklable objects are passed
    across the process boundary.

    Args:
        input_file:  Path to the source JPEG or HEIC file.
        output_file: Path where the output MP4 will be written.
        params:      Dict of processing options (see main() for keys).
    """
    # Use a per-file prefix on every print, so interleaved output from concurrent
    # workers is easy to attribute to the right file.
    tag = f"[{input_file.name}]"

    fmt = detect_format(input_file)
    logger.info(f"{tag} Detected format: {fmt.upper()}")

    # --- Extract color image and depth map ---
    if fmt == 'jpeg':
        with open(input_file, 'rb') as f:
            jpeg_data = f.read()
        color_img, depth_img, icc_profile = extract_depth_jpeg_mpf(jpeg_data, tag=tag)

    elif fmt == 'heic':
        color_img, depth_img, icc_profile = extract_depth_heic(input_file, tag=tag)

    else:
        ext = input_file.suffix.lower()
        if ext in ('.heic', '.heif'):
            color_img, depth_img, icc_profile = extract_depth_heic(input_file, tag=tag)
        else:
            raise ValueError(
                f"Unrecognised format for {input_file} — "
                "expected JPEG or HEIC with embedded depth data."
            )

    logger.info(f"{tag}   Color: {color_img.size} {color_img.mode}")
    logger.info(f"{tag}   Depth: {depth_img.size} {depth_img.mode}")

    # --- Optional downscale ---
    if params['max_width'] or params['max_height'] > 0:
        color_img = scale_image(color_img, params['max_width'], params['max_height'])
        logger.info(f"{tag}   Scaled color to: {color_img.size}")

    # --- Retrieve colorspace and color_primaries from ICC profile ---
    colorspace, color_primaries, color_trc = icc_to_ffmpeg_color_params(icc_profile)
    logger.info(f"{tag}   Color params from ICC: colorspace={colorspace} primaries={color_primaries} trc={color_trc}")

    # --- Resize / smooth / invert depth map ---
    depth_rgb = process_depth(
        depth_img,
        color_size=color_img.size,
        invert=params['invert_depth'],
        blur_radius=params['blur'],
    )
    logger.info(f"{tag}   Resized depth to: {depth_rgb.size}")

    # --- Save intermediates if requested ---
    if params['save_intermediates']:
        base = output_file.with_suffix('')
        color_img.save(str(base) + '_color.png')
        depth_rgb.save(str(base) + '_depth.png')
        logger.info(f"{tag}   Saved intermediates: {base}_color.png, {base}_depth.png")

    # --- Build side-by-side RGB-D canvas ---
    rgbd = build_rgbd(color_img, depth_rgb)
    logger.info(f"{tag}   RGB-D canvas: {rgbd.size}")

    # --- Encode to MP4 ---
    encode_mp4(
        rgbd_img=rgbd,
        output_file=output_file,
        tag=tag,
        duration=params['duration'],
        codec=params['codec'],
        extra_ffmpeg_args=params['ffmpeg_args'],
        colorspace=colorspace,
        color_primaries=color_primaries,
        color_trc=color_trc,
    )

    logger.info(f"{tag} Done → {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Convert an Apple iPhone portrait or depth-enabled photo to an MP4 '
            'that Looking Glass Studio can import directly as an RGB-D video.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'After encoding, import into Looking Glass Studio by dragging the\n'
            'MP4 onto the app and selecting "RGB-D Photo/Video" when prompted.\n'
            '\n'
            'If the 3D effect looks inside-out, re-run with --invert-depth.\n'
            'If the file is too large for your Looking Glass, reduce --max-width.'
        )
    )
    parser.add_argument(
        'input',
        type=Path,
        help='Input file or directory with JPEG or HEIC portrait(s) with depth data.',
    )
    parser.add_argument(
        'output',
        type=Path,
        help='Directory where the output MP4 will be written.',
    )

    parser.add_argument(
        '--duration', type=float, default=10.0, metavar='SECONDS',
        help=(
            'Length of the output video in seconds (default: 10.0). '
            'Looking Glass Studio loops playlist items, so a short duration '
            'works fine for a still photo.'
        )
    )
    parser.add_argument(
        '--codec', choices=['auto', 'h264', 'h265', 'hevc_videotoolbox'], default='auto',
        help=(
            'Video codec. "auto" (default) uses H.264 for canvases up to ~8 MP '
            'and H.265 above that. H.265 gives smaller files but encodes slower. '
            'Both are supported by Looking Glass Studio.'
        )
    )
    parser.add_argument(
        '--max-width', type=int, default=1536, metavar='PX',
        help=(
            'Scale the color image so its width does not exceed this value '
            'before building the RGB-D canvas. 0 = no limit. '
            'The Looking Glass Portrait display is 1536 px wide, so '
            '--max-width 1536 produces a well-matched output.'
        )
    )
    parser.add_argument(
        '--max-height', type=int, default=2048, metavar='PX',
        help=(
            'Scale the color image so its height does not exceed this value '
            'before building the RGB-D canvas. 0 = no limit. '
            'The Looking Glass Portrait display is 2048 px high, so '
            '--max-height 2048 produces a well-matched output.'
        )
    )
    parser.add_argument(
        '--invert-depth', action='store_true',
        help='Invert the depth map. Use this if the 3D effect looks reversed.'
    )
    parser.add_argument(
        '--blur', type=float, default=2.0, metavar='RADIUS',
        help='Gaussian blur radius for depth map edge smoothing (default: 2.0, 0=off)'
    )
    parser.add_argument(
        '--crf', type=int, default=None, metavar='N',
        help=(
            'Override the CRF quality value passed to FFmpeg '
            '(default: 18 for H.264, 22 for H.265; lower = better quality / larger file)'
        )
    )
    parser.add_argument(
        '--save-intermediates', action='store_true',
        help='Save <output>_color.png and <output>_depth.png for inspection'
    )
    num_cpus = os.cpu_count() or 1
    # Leave one CPU free for OS and overhead.
    # Each FFMPEG process seems to use up to three CPUs.
    def_workers = int(max(1, num_cpus - 1) / 3)
    parser.add_argument(
        '--workers', type=int,
        default=def_workers,
        metavar='N',
        help=(
            'Number of files to encode concurrently using separate worker '
            'processes (default: CPU count minus 1 divided by 3, currently '
            f'{def_workers}). Each FFMPEG process uses up to three CPUs.'
            'Each worker runs the full pipeline — image extraction, depth '
            'processing, and FFmpeg encoding — for one file independently. '
            'Set to 1 to disable concurrency.'
        )
    )
    parser.add_argument(
        "--ffmpeg-args",
        dest="ffmpeg_args",
        action="append",
        default=[],
        help="Extra arguments passed directly to FFmpeg. This must be the final option.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)
        
    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        file_iter = (
            path for path in args.input.iterdir()
            if path.is_file() and path.name[0] != '.'
        )
    else:
        file_iter = (args.input,)

    # Build the list of (input_path, output_path) pairs up front so we know
    # the total count before submitting any work.
    pairs = [
        (input_file, (args.output / input_file.name).with_suffix('.jpeg'))
        # (input_file, (args.output / (input_file.name + '__lkgrec.mp4')))
        for input_file in file_iter
    ]

    if not pairs:
        logger.warning("No input files found.")
        sys.exit(0)

    # Collect encoding parameters into a plain dict so it can be pickled and
    # sent to worker processes by ProcessPoolExecutor.
    params = {
        'max_width':        args.max_width,
        'max_height':       args.max_height,
        'invert_depth':     args.invert_depth,
        'blur':             args.blur,
        'save_intermediates': args.save_intermediates,
        'duration':         args.duration,
        'codec':            args.codec,
        'crf_h264':         args.crf if args.crf is not None else 18,
        'crf_h265':         args.crf if args.crf is not None else 22,
        'ffmpeg_args':      args.ffmpeg_args,
    }

    n_workers = min(args.workers, def_workers)
    total = len(pairs)

    logger.info(f"Processing {total} file(s) with {n_workers} worker process(es).")

    n_ok = 0
    n_err = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        # Map each (input, output) pair to a Future, keeping the path for
        # error reporting once the future completes.
        future_to_input = {
            pool.submit(process_file, input_file, output_file, params): input_file
            for input_file, output_file in pairs
        }

        for future in concurrent.futures.as_completed(future_to_input):
            input_file = future_to_input[future]
            try:
                future.result()   # re-raises any exception from the worker
                n_ok += 1
            except Exception as exc:
                n_err += 1
                # Log the full failure reason so the specific cause is clear.
                logger.error(f"[{input_file.name}]   {'='*30}")
                logger.error(f"[{input_file.name}]   FAILED: {input_file}")
                logger.error(f"[{input_file.name}]   Reason: {exc}")
                logger.error(f"[{input_file.name}]   {'='*30}")

    logger.info(f"Done — {n_ok} succeeded, {n_err} failed.")
    if n_err:
        sys.exit(1)


if __name__ == '__main__':
    main()
