
import streamlit as st
import duckdb

import logging
from pathlib import Path
from html import escape
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from base64 import b64encode
from io import BytesIO
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent

logger = logging.getLogger("footmarket.media")


def _debug(*args):
    """Debug output goes to the logger (silent unless DEBUG is enabled)."""
    logger.debug(" ".join(str(a) for a in args))


# Normalise player and club IDs
def normalise_id(value):

    try:
        return str(int(value))

    except (ValueError, TypeError, OverflowError):
        return str(value).strip()


# Load player photograph URLs
@st.cache_data
def load_player_images():

    database_path = (
        ROOT / "data/deployment/footmarket.duckdb"
    )

    with duckdb.connect(
        str(database_path),
        read_only=True
    ) as conn:

        columns = conn.execute(
            "DESCRIBE players"
        ).df()["column_name"].tolist()

        if "image_url" not in columns:

            _debug("[Media] players.image_url is missing")

            return {}

        players = conn.execute("""
            SELECT player_id, image_url
            FROM players
            WHERE image_url IS NOT NULL
        """).df()

    images = {
        normalise_id(player_id): image_url
        for player_id, image_url in zip(
            players["player_id"],
            players["image_url"]
        )
    }

    _debug("[Media] Image URLs loaded:", len(images))

    return images


# Validate image URL
def allowed_url(url):

    if not isinstance(url, str):
        return False

    parsed = urlparse(url)

    host = parsed.hostname or ""

    domains = (
        "transfermarkt.de",
        "transfermarkt.com",
        "transfermarkt.technology",
        "transfermarkt.co.uk"
    )

    return parsed.scheme == "https" and (
        host == "tmssl.akamaized.net"
        or any(
            host == domain
            or host.endswith("." + domain)
            for domain in domains
        )
    )


# Convert Pillow image into PNG data
def png_data(image):

    buffer = BytesIO()

    image.save(buffer, format="PNG")

    encoded = b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return "data:image/png;base64," + encoded


# Download and process image
@st.cache_data(ttl=3600, max_entries=500)
def fetch_image(url):

    if not isinstance(url, str) or not url.strip():

        _debug("[Media] No image URL")

        return None

    url = url.strip()

    if not allowed_url(url):

        _debug("[Media] Unsupported image URL:", url)

        return None

    try:

        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urlopen(request, timeout=15) as response:

            _debug("[Media] HTTP status:", response.status)

            _debug(
                "[Media] Content type:",
                response.headers.get_content_type()
            )

            data = response.read(5_000_001)

        if not data:

            raise ValueError("Empty image")

        if len(data) > 5_000_000:

            raise ValueError("Image exceeds 5 MB")

        with Image.open(BytesIO(data)) as image:

            if image.width * image.height > 20_000_000:

                raise ValueError("Image dimensions too large")

            _debug("[Media] Original image size:", image.size)

            image.thumbnail((500, 500))

            result = png_data(
                image.convert("RGBA")
            )

        _debug("[Media] Download successful:", url)

        return result

    except Exception as error:

        _debug("[Media] Download failed:", url)

        _debug("[Media] Error:", repr(error))

        return None


# Retrieve player photograph
def get_player_photo(player_id):

    player_id = normalise_id(player_id)

    images = load_player_images()

    url = images.get(player_id)

    _debug("\n[Photo] Player ID:", player_id)

    _debug("[Photo] Image URL:", url)

    if not isinstance(url, str) or not url.strip():

        _debug("[Photo] URL not found")

        return None

    url = url.strip()

    _debug(
        "[Photo] URL allowed:",
        allowed_url(url)
    )

    photo = fetch_image(url)

    _debug(
        "[Photo] Retrieved:",
        photo is not None
    )

    return photo


# Retrieve football club badge
def get_club_badge(club_id):

    club_id = normalise_id(club_id)

    if not club_id.isdigit():

        _debug("[Media] Invalid club ID:", club_id)

        return None

    url = (
        "https://tmssl.akamaized.net/"
        f"images/wappen/head/{club_id}.png"
    )

    return fetch_image(url)


# Default player photograph
def player_placeholder():

    image = Image.new(
        "RGBA",
        (160, 180),
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(image)

    draw.ellipse(
        (48, 16, 112, 80),
        fill="#526579"
    )

    draw.rounded_rectangle(
        (18, 95, 142, 190),
        radius=42,
        fill="#526579"
    )

    return (
        f'<img src="{png_data(image)}" '
        'alt="Player photograph unavailable" '
        'class="fm-photo-placeholder">'
    )


# Default club shield
def badge_placeholder():

    image = Image.new(
        "RGBA",
        (48, 48),
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(image)

    draw.polygon(
        [
            (24, 3),
            (42, 10),
            (42, 25),
            (35, 37),
            (24, 45),
            (13, 37),
            (6, 25),
            (6, 10)
        ],
        fill="#304056",
        outline="#94A3B8"
    )

    return (
        f'<img src="{png_data(image)}" '
        'alt="Club badge unavailable" '
        'class="fm-club-badge">'
    )


# Generate player photograph HTML
def player_photo_html(player_id, player_name):

    photo = get_player_photo(player_id)

    if photo is None:

        return player_placeholder()

    return (
        f'<img src="{escape(photo, quote=True)}" '
        f'alt="{escape(str(player_name), quote=True)}" '
        'class="fm-player-photo">'
    )


# Generate club badge HTML
def club_badge_html(club_id, club_name):

    badge = get_club_badge(club_id)

    if badge is None:

        return badge_placeholder()

    return (
        f'<img src="{escape(badge, quote=True)}" '
        f'alt="{escape(str(club_name), quote=True)}" '
        'class="fm-club-badge">'
    )
