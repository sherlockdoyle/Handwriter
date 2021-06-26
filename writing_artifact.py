from typing import Tuple, Union, List
from collections import namedtuple
import os
import argparse
import numpy as np
import cv2

image = imgRGB = imgGray = np.ndarray
"""imgRGB and imgGray refers to 3 and 2 dimension arrays respectively. image is either."""


def imshow(*img: image, scale: float = 2.5):
    """Utility method to display one or more images."""
    for i, im in enumerate(img):
        cv2.imshow(
            f'image{i}',
            cv2.resize(im, (int(im.shape[1]/scale), int(im.shape[0]/scale)))
        )
    while cv2.waitKey(0) != 27:
        continue
    cv2.destroyAllWindows()


def remove_holes(img: image, size: int = 5) -> image:
    """Removes artifacts/noises from image."""
    kernel = np.ones((size, size), np.uint8)
    dilated = cv2.dilate(img, kernel)
    return cv2.erode(dilated, kernel)


def flood_fill(img: image, fill_color: Union[int, Tuple[int, int, int]] = None) -> image:
    """Flood fill areas in an image."""
    if fill_color is None:
        fill_color = 255 if img.ndim == 2 else (255, 255, 255)
    copy = img.copy()
    cv2.floodFill(copy, None, (0, 0), fill_color)
    return cv2.bitwise_or(
        img,
        cv2.bitwise_not(copy)
    )


def extract_mask(img: imgRGB, mask_hue_range: Tuple[float, float] = (-10, 10)) -> Tuple[imgGray, imgRGB]:
    """Extract masks from image and return the mask and original image without mask."""
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv_img, (mask_hue_range[0], 200, 200), (mask_hue_range[1], 255, 255))
    if mask_hue_range[0] < 0:  # special case, negative range
        color_mask |= cv2.inRange(hsv_img, (180+mask_hue_range[0], 200, 200), (180, 255, 255))
    mask = cv2.dilate(color_mask, np.ones((2, 2), np.uint8))
    orig = cv2.bitwise_or(img, cv2.merge((mask, mask, mask)))
    return flood_fill(remove_holes(mask)), orig


def preprocess(img: imgRGB) -> imgRGB:
    """Preprocess (thicken text) the image."""
    eroded = cv2.erode(img, np.ones((2, 2), np.uint8))
    return cv2.addWeighted(img, 0.5, eroded, 0.5, 0)


def mask_image(img: imgRGB, mask: imgGray) -> imgRGB:
    """Apply mask to an image and extract text."""
    mask = cv2.merge((mask, mask, mask))
    return cv2.bitwise_and(
        cv2.bitwise_not(
            cv2.bitwise_and(img, mask)
        ),
        mask
    )


def extract_contours(img: image) -> List[np.ndarray]:
    """Extract contours from image."""
    contours, _ = cv2.findContours(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    return contours


def dilate_contours(contours: List[np.ndarray], w: int, h: int) -> List[np.ndarray]:
    """Dilate contours of image."""
    black = np.zeros((h, w), np.uint8)
    cv2.drawContours(black, contours, -1, 255, -1)
    contours, _ = cv2.findContours(
        cv2.dilate(black, np.ones((7, 7), np.uint8)),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    return contours


# def erode_contours(contours: List[np.ndarray], w: int, h: int) -> List[np.ndarray]:
#     """Erode contours of image."""
#     black = np.zeros((h, w), np.uint8)
#     cv2.drawContours(black, contours, -1, 255, -1)
#     contours, _ = cv2.findContours(
#         cv2.erode(black, np.ones((2, 2), np.uint8)),
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_NONE
#     )
#     return contours


def get_hull_and_rect(contours: List[np.ndarray]) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """Return the convex hull and bounding box of the counters."""
    hull = [cv2.convexHull(cnt) for cnt in contours]
    rect = [cv2.boundingRect(cnt) for cnt in hull]
    return hull, rect


def get_avg_color(img: imgRGB) -> Tuple[int, int, int]:
    """Get an image's average color."""
    mask = cv2.threshold(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        225, 255, cv2.THRESH_BINARY
    )[1]
    return cv2.mean(img, mask=mask)[:3]


def get_strikes(rect: List[Tuple[int, int, int, int]], w: int, h: int, refImg: imgRGB) -> imgRGB:
    """Get the strikes to be put on top of wrong text. refImg is used to extract color."""
    strikes = np.zeros((h, w, 3), np.uint8)
    if not rect:
        return strikes
    max_width = max(w for _, _, w, _ in rect)
    min_width = min(w for _, _, w, _ in rect)
    for x, y, w, h in rect:
        strike_color = get_avg_color(refImg[y:y+h, x:x+w])
        # One line in the middle
        cv2.line(strikes, (x, y + h // 2), (x + w, y + h // 2), strike_color, 2)
        d = max_width - min_width
        d = 2 if d == 0 else 3 * (w - min_width) / d + 2
        n = np.random.randint(1, d)
        for i in range(n):
            l = (x, np.random.randint(y, y + h))
            r = (x + w, np.random.randint(y, y + h))
            cv2.line(strikes, l, r, strike_color, 2)
    return cv2.blur(strikes, (2, 2))


def put_strikes(img: imgRGB, strike: imgRGB, hull: List[np.ndarray]) -> imgRGB:
    """Put the strikes on top of image."""
    black = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(black, hull, -1, 255, -1)
    return cv2.bitwise_and(
        img,
        cv2.bitwise_not(
            cv2.bitwise_and(
                strike,
                cv2.merge((black, black, black))
            )
        )
    )


def perlin(shape, res=(64, 64)) -> imgGray:
    """Generate a perlin noise image."""
    # TODO: Where did I find this code from?
    orig_shape = shape
    shape = np.ceil(shape[0] / res[0]) * res[0], np.ceil(shape[1] / res[1]) * res[1]

    d0, d1 = shape[0] // res[0], shape[1] // res[1]
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    grad = np.dstack((np.cos(angles), np.sin(angles)))
    grid = np.mgrid[:res[0]:res[0] / shape[0], :res[1]:res[1] / shape[1]].transpose(1, 2, 0) % 1

    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * grad[:-1, :-1].repeat(d0, 0).repeat(d1, 1), 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * grad[1:, :-1].repeat(d0, 0).repeat(d1, 1), 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * grad[:-1, 1:].repeat(d0, 0).repeat(d1, 1), 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * grad[1:, 1:].repeat(d0, 0).repeat(d1, 1), 2)

    t = 6 * grid ** 5 - 15 * grid ** 4 + 10 * grid ** 3
    n0 = (1 - t[:, :, 0]) * n00 + t[:, :, 0] * n10
    n1 = (1 - t[:, :, 0]) * n01 + t[:, :, 0] * n11
    return (
        np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)
    )[:orig_shape[0], :orig_shape[1]].astype(np.float32)


def displace_image(img: imgRGB, mapx: imgGray, mapy: imgGray, fill: Tuple[int, int, int] = (255, 255, 255)) -> imgRGB:
    """Apply displacement map to an image."""
    gridx, gridy = np.meshgrid(np.arange(img.shape[1], dtype=np.float32),
                               np.arange(img.shape[0], dtype=np.float32))
    if mapx is None:
        mapx = gridx
    else:
        mapx += gridx
    if mapy is None:
        mapy = gridy
    else:
        mapy += gridy

    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def get_white_rows(img: imgRGB) -> Tuple[List[int], imgGray]:
    """Get rows which are a boundary between white rows and text lines. Also return the internal binary image."""
    rows = [0]  # Initial header white section
    is_white = True
    bin_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 55, 255, cv2.THRESH_BINARY)[1]
    row_sum = bin_img.shape[1] * 255
    for i, row in enumerate(bin_img):
        new_iswhite = row.sum() >= row_sum
        if new_iswhite != is_white:
            rows.append(i)
            is_white = new_iswhite
    return rows, bin_img


def get_n_shortest_line_idx(img: imgGray, white_rows: List[int], n: int) -> List[int]:
    """Return the index of the n shortest lines in img. First two lines are ignored."""
    widths = []
    l = len(white_rows)
    for i in range(5, l, 2):  # ignore first two
        seg = img[white_rows[i]:white_rows[i+1]]
        top_height = white_rows[i] - white_rows[i-1]
        down_height = (white_rows[i+2] if i+2 < l else img.shape[0]) - white_rows[i+1]
        widths.append((
            cv2.boundingRect(
                np.concatenate(extract_contours(255 - seg))
            )[2] * top_height / down_height,
            i // 2
        ))
    widths.sort()
    breaks = sorted(i for e, i in widths[:n])
    if breaks[-1] != len(widths) + 1:  # last line shouldn't be the start
        breaks.append(len(widths) + 1)
    return breaks


def draw_rows(img: imgRGB, rows: List[int], small_lines: List[int] = None) -> imgRGB:
    """Utility method to draw the text rows on the image."""
    w = img.shape[1]
    rowed_img = img.copy()
    for r in rows:
        cv2.line(rowed_img, (0, r), (w, r), (0, 0, 255), 1)
    if small_lines is not None:
        for l in small_lines:
            r = rows[l*2+2]
            cv2.line(rowed_img, (0, r), (w, r), (255, 0, 0), 2)
    return rowed_img


def perform_moves(img: imgRGB, w: int, rows: List[int], f: float = 1) -> imgRGB:
    """Move each line vertically and horizontally."""
    white = np.ones_like(img)*255
    l = len(rows)
    white[rows[1]:rows[2]] = img[rows[1]:rows[2]]
    white[rows[-2]:rows[-1]] = img[rows[-2]:rows[-1]]
    for i in range(3, l-2, 2):  # ignore first and last line
        shiftY = 0
        t = np.random.randint(3)
        if t == 1:
            shiftY = -np.random.randint((rows[i]-rows[i-1])/2)
        elif t == 2:
            shiftY = np.random.randint((rows[i+2]-rows[i+1])/2)
        shiftY = round(shiftY * f)
        shiftX = np.random.randint(-w/163, w/163+1)
        if shiftX > 0:
            white[rows[i]+shiftY:rows[i+1]+shiftY, :-shiftX] &= img[rows[i]:rows[i+1], shiftX:]
        elif shiftX < 0:
            white[rows[i]+shiftY:rows[i+1]+shiftY, -shiftX:] &= img[rows[i]:rows[i+1], :shiftX]
        else:
            white[rows[i]+shiftY:rows[i+1]+shiftY] &= img[rows[i]:rows[i+1]]
        rows[i] += shiftY
        rows[i+1] += shiftY
    return white


def slant_block(img: imgRGB, row1: int, row2: int, shift: int, dst: imgRGB):
    """Slant row1 to row2 upwards in img and put in dst."""
    w = img.shape[1]
    h = row2 - row1
    seg = img[row1:row2]
    matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]),
                                         np.float32([[0, shift], [w, 0], [w, h], [0, h + shift]]))
    slanted = cv2.warpPerspective(seg, matrix, (w, h + shift),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    dst[row1 - shift:row2, 0:w, :] &= slanted


def slant_pers(img: imgRGB, row1: int, row2: int, shift: int, dst: imgRGB):
    """Slant row1 to row2 with fake perspective, ie. only compresses the right side."""
    w = img.shape[1]
    h = row2 - row1
    seg = img[row1:row2]
    disp = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            s = j/w*shift
            disp[i, j] = s*i/h
    dst[row1:row2] &= displace_image(seg, None, disp)


def slant_lines(img: imgRGB, idx1: int, idx2: int, rows: List[int], shift: int, dst: imgRGB):
    """Slant rows from index idx1 to idx2, with each row slanted a little more than the one above."""
    iw = idx2-idx1
    for j in range(iw):
        slant_block(img, rows[(idx1+j)*2+1], rows[(idx1+j+1)*2], round(j/(iw-1)*shift), dst)


def perform_slants(img: imgRGB, lines: List[int], rows: List[int], f: float = 1) -> imgRGB:
    """Slant blocks of lines of the image."""
    white = np.ones_like(img)*255
    start = 0
    for i in lines:
        idx1 = rows[start*2+1]
        idx2 = rows[i*2+2]
        idx1_1 = rows[start*2]
        idx2_1 = rows[i*2+1]
        idx2_2 = rows[i*2]
        prob = np.array([1, 2, 3, 1])
        t = 4 if i-start <= 2 else np.random.choice(list(range(4)), p=prob/prob.sum())
        if t == 0:
            slant_block(img, idx1, idx2, round(min(idx1-idx1_1, idx2-idx2_1)*f), white)
        elif t == 1:
            slant_pers(img, idx1, idx2, round((idx2-(idx2_1+idx2_2)//2)*f), white)
        elif t == 2:
            slant_lines(img, start, i+1, rows, round((idx2-idx2_2)*f), white)
        else:
            white[idx1:idx2] &= img[idx1:idx2]
        start = i+1
    return white


def put_fading(img: imgRGB, fade: imgGray, f: float = 0.5) -> imgRGB:
    fade -= fade.min()
    fade /= fade.max()
    # fade = 1-(1-fade)**2
    fade += (1-fade) * f
    return (255 - (255-img) * fade.reshape((fade.shape[0], fade.shape[1], 1))).astype(np.uint8)


background_code = namedtuple('background_code', ['path_idx', 'merges', 'resize', 'rotate', 'flip'])


def get_background_codes(n: int, back_dir: str) -> Tuple[List[str], List[background_code]]:
    """Return a list of background paths and n unique background codes."""
    paths = list(map(
        lambda p: os.path.join(back_dir, p),
        os.listdir(back_dir)
    ))
    backgrounds = set()
    while len(backgrounds) < n:
        backgrounds.add(background_code(
            path_idx=np.random.randint(len(paths)),  # index of path
            merges=np.random.randint(len(paths)),  # number of other backgrounds to merge with
            resize=np.random.rand() < 0.5,  # resize the background from (W, H) to (H, W)
            rotate=np.random.rand() < 0.5,  # rotate the background by 180 degrees
            flip=np.random.rand() < 0.5  # flip the background
        ))
    return paths, list(backgrounds)


def get_back(code: background_code, paths: List[str], size: Tuple[int, int]) -> imgRGB:
    """Return a background for the code of given size."""
    back = cv2.resize(
        cv2.imread(paths[code.path_idx]),
        size
    )
    if code.merges:
        back2 = get_back(background_code(
            path_idx=np.random.randint(len(paths)),
            merges=code.merges-1,  # one less merge
            resize=np.random.rand() < 0.5,
            rotate=np.random.rand() < 0.5,
            flip=np.random.rand() < 0.5
        ), paths, size)
        h, s, v = cv2.split(
            cv2.cvtColor((back * (back2/255)).astype(np.uint8), cv2.COLOR_BGR2HSV)
        )
        x, y = v.min(), v.max()
        val = np.random.randint(50, 128)
        back = cv2.cvtColor(
            cv2.merge((h, s, ((v-x)/(y-x)*(255-val-x)+x+val).astype(np.uint8))),
            cv2.COLOR_HSV2BGR
        )
    if code.resize:
        back = cv2.rotate(cv2.resize(back, back.shape[:2]), cv2.ROTATE_90_CLOCKWISE)
    if code.rotate:
        back = cv2.rotate(back, cv2.ROTATE_180)
    if code.flip:
        back = cv2.flip(back, 0)
    return back


def do_artifact(img: imgRGB, back: imgRGB, *,
                text_shift_scale: int = 64,
                text_shift_factor: float = 5.5,
                line_slant_factor: float = 1,
                line_move_factor: float = 1,
                text_fade_factor: float = 0.5
                ) -> imgRGB:
    """Add the handwritten text artifacts."""
    H, W, _ = img.shape
    mask, orig = extract_mask(img)
    orig = preprocess(orig)
    img_dispx = perlin((H, W), (text_shift_scale, text_shift_scale))
    img_dispy = perlin((H, W), (text_shift_scale, text_shift_scale))
    disp_img = displace_image(orig, -0.363636364*text_shift_factor*img_dispx, text_shift_factor*img_dispy)
    mistake_masked = mask_image(disp_img, mask)
    contours = extract_contours(mistake_masked)
    big_contours = dilate_contours(contours, W, H)
    hull, rect = get_hull_and_rect(big_contours)
    strikes = get_strikes(rect, W, H, mistake_masked)
    disp_strikes = displace_image(strikes, None, perlin((H, W), (16, 16))*7, (0, 0, 0))
    striked_img = put_strikes(disp_img, disp_strikes, hull)
    rows, bin_img = get_white_rows(striked_img)
    print('Found', len(rows)//2, 'lines in image.')
    small_lines = get_n_shortest_line_idx(bin_img, rows, np.random.randint(
        max(len(rows)//12-3, 1),
        max(len(rows)//12+1, 3)
    ))
    # imshow(draw_rows(striked_img, rows, small_lines))
    moved_img = perform_moves(striked_img, W, rows, line_move_factor)
    slanted_img = perform_slants(moved_img, small_lines, rows, line_slant_factor)
    faded_img = put_fading(slanted_img, perlin((H, W), (text_shift_scale, text_shift_scale)), text_fade_factor)
    norm_back = cv2.normalize(
        cv2.cvtColor(back, cv2.COLOR_BGR2GRAY),
        None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    page_morphed_img = displace_image(faded_img, None, 40-60*norm_back)
    on_page_img = cv2.normalize(
        (back * (page_morphed_img/255)).astype(np.uint8),
        None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    return on_page_img


parser = argparse.ArgumentParser(description='Generate handwritten like text.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('images', nargs='+', help='path to images or a directory of images of text')
parser.add_argument('-o', '--out', default='./out', help='path to output directory', metavar='DIR')
parser.add_argument('-f', '--output-format', help='format (extension) of output images', metavar='EXT')
parser.add_argument('-b', '--background', default='./background',
                    help='path to directory with background images', metavar='DIR')
parser.add_argument('--seed', type=int, help='seed for random number generator, used if specified', metavar='VAL')
parser.add_argument('-s', default=64, type=int, help='scale of text shift', metavar='VAL')
parser.add_argument('-r', default=5.5, type=float, help='amount to shift the text randomly', metavar='VAL')
parser.add_argument('-k', default=1, type=float,
                    help='amount to slant lines, 0 means no slant, positive means upwards, negative means downwards', metavar='VAL')
parser.add_argument('-t', default=1, type=float, help='amount to move lines up or down', metavar='VAL')
parser.add_argument('-a', default=0.5, type=float, help='lowest opacity for fading text', metavar='VAL')
args = parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)

image_paths = []
for path in args.images:
    if os.path.isdir(path):
        file_names = os.listdir(path)
        for fn in file_names:
            file_path = os.path.join(path, fn)
            if not os.path.isdir(file_path):
                image_paths.append(file_path)
    else:
        image_paths.append(path)

num_images = len(image_paths)
print(f'Will process {num_images} images.')
background_paths, background_codes = get_background_codes(len(image_paths), args.background)
if not os.path.exists(args.out):
    os.mkdir(args.out)

for i in range(num_images):
    path = image_paths[i]
    try:
        print(f'Processing image {i+1}...')
        img = cv2.imread(path)
        H, W, _ = img.shape
        back = get_back(background_codes[i], background_paths, (W, H))
        edited = do_artifact(img, back,
                             text_shift_scale=args.s,
                             text_shift_factor=args.r,
                             line_slant_factor=args.k,
                             line_move_factor=args.t,
                             text_fade_factor=args.a
                             )

        save_path, ext = os.path.splitext(os.path.basename(path))
        if args.output_format is not None:
            ext = '.'+args.output_format
        cv2.imwrite(
            os.path.join(args.out, '_edited'.join((save_path, ext))),
            edited
        )
    except Exception as e:
        print(f"Could not process image '{path}'")
        print(e)
