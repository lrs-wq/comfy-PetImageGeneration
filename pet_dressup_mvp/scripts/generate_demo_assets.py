from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
PET_IMAGE = ROOT / "assets" / "pets" / "cat_pose_01.png"
OVERLAY_DIR = ROOT / "assets" / "overlays"
ITEM_DIR = ROOT / "assets" / "items"


def rgba(size: tuple[int, int]) -> Image.Image:
    return Image.new("RGBA", size, (0, 0, 0, 0))


def save_overlay(name: str, mask: Image.Image) -> None:
    pet = Image.open(PET_IMAGE).convert("RGBA")
    alpha = pet.getchannel("A")
    mask = ImageChops.multiply(alpha, mask)
    out = pet.copy()
    out.putalpha(mask)
    out.save(OVERLAY_DIR / name)


def rounded_polygon(draw: ImageDraw.ImageDraw, points, radius: int, fill, outline=None, width=1) -> None:
    # Pillow has no native rounded polygon, so this softens hard seams with a blur pass.
    layer = rgba(draw.im.size)
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.polygon(points, fill=fill)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=radius))
    draw.bitmap((0, 0), layer.getchannel("A"), fill=fill)
    if outline:
        draw.line(points + [points[0]], fill=outline, width=width, joint="curve")


def make_overlays() -> None:
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    pet = Image.open(PET_IMAGE).convert("RGBA")
    w, h = pet.size

    head = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(head)
    d.ellipse((72, 24, 476, 425), fill=255)
    d.polygon([(56, 132), (109, 5), (210, 232)], fill=255)
    d.polygon([(492, 132), (439, 5), (338, 232)], fill=255)
    head = head.filter(ImageFilter.GaussianBlur(2))
    save_overlay("cat_pose_01_head_overlay.png", head)

    paws = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(paws)
    d.ellipse((128, 658, 264, 846), fill=255)
    d.ellipse((285, 658, 424, 846), fill=255)
    d.ellipse((64, 684, 178, 825), fill=180)
    d.ellipse((374, 684, 486, 825), fill=180)
    paws = paws.filter(ImageFilter.GaussianBlur(2))
    save_overlay("cat_pose_01_front_paws_overlay.png", paws)


def hoodie() -> None:
    im = rgba((430, 390))
    d = ImageDraw.Draw(im)
    rounded_polygon(
        d,
        [(74, 78), (356, 78), (414, 366), (16, 366)],
        1,
        (220, 63, 73, 255),
        (116, 35, 43, 255),
        6,
    )
    d.polygon([(64, 98), (8, 280), (80, 328), (132, 124)], fill=(188, 48, 58, 255))
    d.polygon([(366, 98), (422, 280), (350, 328), (298, 124)], fill=(188, 48, 58, 255))
    d.rounded_rectangle((84, 32, 346, 152), radius=72, fill=(235, 75, 84, 255), outline=(116, 35, 43, 255), width=5)
    d.ellipse((140, 42, 290, 168), fill=(248, 237, 224, 240), outline=(116, 35, 43, 255), width=5)
    d.line((176, 148, 146, 212), fill=(254, 217, 128, 255), width=5)
    d.line((254, 148, 284, 212), fill=(254, 217, 128, 255), width=5)
    d.ellipse((138, 206, 154, 222), fill=(254, 217, 128, 255))
    d.ellipse((276, 206, 292, 222), fill=(254, 217, 128, 255))
    d.rounded_rectangle((182, 230, 248, 330), radius=22, fill=(176, 42, 51, 255))
    d.line((215, 160, 215, 364), fill=(154, 38, 46, 210), width=4)
    im.save(ITEM_DIR / "top_red_hoodie.png")


def sailor() -> None:
    im = rgba((430, 350))
    d = ImageDraw.Draw(im)
    d.polygon([(76, 64), (354, 64), (402, 330), (28, 330)], fill=(245, 249, 248, 255))
    d.line([(76, 64), (354, 64), (402, 330), (28, 330), (76, 64)], fill=(65, 83, 114, 255), width=5)
    d.polygon([(92, 70), (200, 178), (215, 80), (230, 178), (338, 70), (292, 166), (215, 230), (138, 166)], fill=(40, 92, 150, 255))
    d.line((112, 92, 196, 174), fill=(255, 246, 188, 255), width=5)
    d.line((318, 92, 234, 174), fill=(255, 246, 188, 255), width=5)
    d.polygon([(205, 156), (170, 300), (215, 260), (260, 300), (225, 156)], fill=(229, 64, 68, 255))
    d.rectangle((68, 296, 362, 320), fill=(40, 92, 150, 255))
    im.save(ITEM_DIR / "top_sailor.png")


def utility_vest() -> None:
    im = rgba((390, 330))
    d = ImageDraw.Draw(im)
    d.polygon([(58, 48), (166, 60), (186, 300), (28, 304)], fill=(48, 139, 117, 255))
    d.polygon([(224, 60), (332, 48), (362, 304), (204, 300)], fill=(48, 139, 117, 255))
    d.line([(58, 48), (166, 60), (186, 300), (28, 304), (58, 48)], fill=(25, 85, 76, 255), width=5)
    d.line([(224, 60), (332, 48), (362, 304), (204, 300), (224, 60)], fill=(25, 85, 76, 255), width=5)
    for x in (72, 238):
        d.rounded_rectangle((x, 178, x + 86, 232), radius=10, fill=(255, 198, 92, 255), outline=(25, 85, 76, 255), width=4)
        d.line((x + 10, 200, x + 76, 200), fill=(174, 122, 44, 255), width=3)
    d.line((195, 70, 195, 302), fill=(255, 235, 180, 255), width=5)
    im.save(ITEM_DIR / "top_utility_vest.png")


def cape() -> None:
    im = rgba((540, 560))
    d = ImageDraw.Draw(im)
    d.polygon([(112, 76), (428, 76), (522, 532), (18, 532)], fill=(79, 71, 151, 230))
    d.line([(112, 76), (428, 76), (522, 532), (18, 532), (112, 76)], fill=(36, 35, 94, 255), width=5)
    d.pieslice((78, 16, 462, 196), 0, 180, fill=(254, 206, 90, 255), outline=(36, 35, 94, 255), width=5)
    d.line((270, 90, 168, 520), fill=(109, 96, 181, 180), width=4)
    d.line((270, 90, 372, 520), fill=(109, 96, 181, 180), width=4)
    im.save(ITEM_DIR / "back_royal_cape.png")


def collars() -> None:
    bell = rgba((370, 116))
    d = ImageDraw.Draw(bell)
    d.rounded_rectangle((26, 38, 344, 74), radius=18, fill=(39, 129, 157, 255), outline=(24, 75, 96, 255), width=4)
    d.ellipse((156, 56, 214, 114), fill=(246, 190, 56, 255), outline=(126, 89, 27, 255), width=4)
    d.rectangle((179, 96, 191, 112), fill=(126, 89, 27, 255))
    d.ellipse((176, 76, 194, 92), fill=(255, 233, 142, 255))
    bell.save(ITEM_DIR / "collar_bell.png")

    bow = rgba((360, 160))
    d = ImageDraw.Draw(bow)
    d.polygon([(170, 70), (20, 16), (52, 132)], fill=(231, 83, 118, 255))
    d.polygon([(190, 70), (340, 16), (308, 132)], fill=(231, 83, 118, 255))
    d.line([(170, 70), (20, 16), (52, 132), (170, 70)], fill=(131, 39, 69, 255), width=5)
    d.line([(190, 70), (340, 16), (308, 132), (190, 70)], fill=(131, 39, 69, 255), width=5)
    d.rounded_rectangle((146, 44, 214, 104), radius=16, fill=(255, 186, 74, 255), outline=(131, 39, 69, 255), width=5)
    bow.save(ITEM_DIR / "collar_bow.png")


def glasses() -> None:
    round_glasses = rgba((280, 110))
    d = ImageDraw.Draw(round_glasses)
    d.ellipse((18, 20, 112, 100), outline=(45, 50, 61, 255), width=8)
    d.ellipse((168, 20, 262, 100), outline=(45, 50, 61, 255), width=8)
    d.arc((108, 38, 172, 82), 180, 360, fill=(45, 50, 61, 255), width=6)
    d.line((18, 54, 0, 44), fill=(45, 50, 61, 255), width=6)
    d.line((262, 54, 280, 44), fill=(45, 50, 61, 255), width=6)
    round_glasses.save(ITEM_DIR / "eyes_round_glasses.png")

    shades = rgba((300, 120))
    d = ImageDraw.Draw(shades)
    d.rounded_rectangle((14, 24, 126, 96), radius=20, fill=(26, 32, 40, 235), outline=(10, 12, 16, 255), width=5)
    d.rounded_rectangle((174, 24, 286, 96), radius=20, fill=(26, 32, 40, 235), outline=(10, 12, 16, 255), width=5)
    d.line((126, 58, 174, 58), fill=(10, 12, 16, 255), width=6)
    d.line((38, 34, 102, 88), fill=(102, 214, 225, 110), width=5)
    d.line((198, 34, 262, 88), fill=(102, 214, 225, 110), width=5)
    shades.save(ITEM_DIR / "eyes_sunglasses.png")


def hats() -> None:
    crown = rgba((320, 210))
    d = ImageDraw.Draw(crown)
    d.polygon([(42, 152), (74, 58), (138, 120), (160, 36), (182, 120), (246, 58), (278, 152)], fill=(246, 190, 56, 255))
    d.rounded_rectangle((42, 142, 278, 188), radius=12, fill=(246, 190, 56, 255), outline=(135, 89, 24, 255), width=5)
    for x, c in [(74, (229, 64, 68, 255)), (160, (59, 130, 246, 255)), (246, (48, 139, 117, 255))]:
        d.ellipse((x - 12, 140, x + 12, 164), fill=c)
    crown.save(ITEM_DIR / "hat_crown.png")

    beanie = rgba((360, 190))
    d = ImageDraw.Draw(beanie)
    d.pieslice((28, 12, 332, 222), 180, 360, fill=(38, 129, 157, 255), outline=(16, 76, 98, 255), width=5)
    d.rounded_rectangle((28, 128, 332, 174), radius=16, fill=(246, 190, 56, 255), outline=(16, 76, 98, 255), width=5)
    d.ellipse((150, 0, 210, 54), fill=(238, 84, 91, 255), outline=(122, 42, 48, 255), width=4)
    for x in range(70, 310, 52):
        d.line((x, 46, x - 28, 130), fill=(105, 185, 200, 190), width=4)
    beanie.save(ITEM_DIR / "hat_beanie.png")

    party = rgba((230, 300))
    d = ImageDraw.Draw(party)
    d.polygon([(115, 16), (30, 268), (200, 268)], fill=(99, 102, 241, 255), outline=(44, 48, 139, 255))
    d.line((68, 156, 174, 122), fill=(255, 235, 120, 255), width=14)
    d.line((52, 212, 190, 172), fill=(238, 84, 91, 255), width=14)
    d.ellipse((96, 0, 134, 38), fill=(246, 190, 56, 255), outline=(135, 89, 24, 255), width=3)
    party.save(ITEM_DIR / "hat_party.png")


def main() -> None:
    ITEM_DIR.mkdir(parents=True, exist_ok=True)
    make_overlays()
    hoodie()
    sailor()
    utility_vest()
    cape()
    collars()
    glasses()
    hats()
    print(f"Generated assets in {ITEM_DIR}")


if __name__ == "__main__":
    main()
