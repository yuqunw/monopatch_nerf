import os 
import cv2 
try:
    # for backward compatibility
    import imageio.v2 as imageio
except ModuleNotFoundError:
    import imageio
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse

def extract_frames():
    parser = argparse.ArgumentParser()
    parser.add_argument('-video', help='path to video')
    parser.add_argument('-outdir', help='output folder')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    vidcap = cv2.VideoCapture(args.video)
    i = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        path = os.path.join(args.outdir, '{:0>5d}.png'.format(i))
        cv2.imwrite(path, image)
        i += 1

def read_video_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, image = vidcap.read()
        if not success:
            break
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image)
    if len(frames) == 0:
        print('ERROR: {} does not exist'.format(video_path))
    return frames

def is_image_name(name):
    valid_names = ['.jpg', '.png', '.JPG', '.PNG']
    for v in valid_names:
        if name.endswith(v):
            return True
    return False

def generate_video():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir')
    parser.add_argument('-out')
    parser.add_argument('-fps', type=int, default=30)
    args = parser.parse_args()

    imgs = sorted([os.path.join(args.dir, img) for img in os.listdir(args.dir) if is_image_name(img)])
    imgs = [np.array(Image.open(img)) for img in imgs]
    imgs += imgs[::-1]
    imageio.mimsave(args.out, imgs, fps=args.fps, macro_block_size=1)

    # imgs = [Image.open(img) for img in imgs]
    # imgs += imgs[::-1]
    # imgs[0].save(args.out, format='GIF', append_images=imgs,
    #      save_all=True, duration=10, loop=0)

def add_text():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_in')
    parser.add_argument('--video_out')
    parser.add_argument('--text')
    parser.add_argument('--font_size', type=float, default=2.0)
    parser.add_argument('--font_thickness', type=int, default=4)
    parser.add_argument('--right', dest='right', action='store_true')
    parser.add_argument('--bottom', dest='bottom', action='store_true')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=-1)
    parser.add_argument('--buffer', type=int, default=30)
    args = parser.parse_args()

    background_color = (96, 96, 96)  # black color in BGR format
    text = args.text
    font = cv2.FONT_HERSHEY_COMPLEX
    font_size = args.font_size
    font_color = (192, 192, 192)  # White color in BGR format
    thickness = args.font_thickness
    opacity = 0.4

    frames_in = read_video_frames(args.video_in)
    h, w, _ = frames_in[0].shape

    text_size = cv2.getTextSize(text, font, font_size, thickness)[0]
    border = 10 # to image 
    buffer = args.buffer # inside box
    x, y = border, border # Position of the text
    x2, y2 = x + text_size[0] + buffer, y + text_size[1] + buffer
    if args.right:
        x2 = w - border
        x  = x2 - text_size[0] - buffer
    if args.bottom:
        y2 = h - border
        y  = y2 - text_size[1] - buffer
    
    text_shift_x = 15/2 * args.font_size
    text_shift_y = 55/2 * args.font_size
    frames_out = []
    for frame in frames_in:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x2, y2), background_color, -1)
        
        # Blend the overlay with the frame
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        
        # Draw the text on the frame
        cv2.putText(frame, text, (int(x + text_shift_x), int(y + text_shift_y)), font, font_size, font_color, thickness)
        
        frames_out.append(frame)

    imageio.mimsave(args.video_out,
                    frames_out[args.start_index:args.end_index],
                    fps=args.fps, macro_block_size=1)

def switch_video():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_in', nargs='+')
    parser.add_argument('--video_out')
    parser.add_argument('--mid', type=int)
    parser.add_argument('--slope', type=float, default=1.0)
    parser.add_argument('--window', type=int, default=30)
    parser.add_argument('--linewidth', type=int, default=0)
    parser.add_argument('--flip', dest='flip', action='store_true')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    vdo_path_0, vdo_path_1  = args.video_in
    frames_0 = read_video_frames(vdo_path_0)
    frames_1 = read_video_frames(vdo_path_1)
    h, w, _ = frames_0[0].shape

    # to be removed
    frames_0 = frames_0
    frames_1 = frames_1
    
    v_start = 0 
    v_end   = (w-1) + (h-1) * args.slope
    v_slope = (v_end - v_start) / args.window
    if args.flip:
        v_slope *= -1
    v_const = (v_end+v_start)/2 - args.mid * v_slope

    grid_y, grid_x = np.meshgrid(np.arange(w), np.arange(h))
    grid_value = grid_y + grid_x * args.slope

    frames_out = []
    vdo_len = len(frames_0)
    for i in tqdm(range(vdo_len)):
        v_threshold = i * v_slope + v_const
        mask = grid_value > v_threshold

        f_0 = frames_0[i]
        f_1 = frames_1[i]
        f_out = np.zeros_like(f_0)
        f_out[mask] = f_0[mask]
        f_out[~mask] = f_1[~mask]
        frames_out.append(f_out)
    
    imageio.mimsave(args.video_out,
                    frames_out,
                    fps=args.fps, macro_block_size=1)
    
def combine_switch_videos():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_in', nargs='+')
    parser.add_argument('--video_out')
    parser.add_argument('--mid', type=int, default=0)
    parser.add_argument('--slope', type=float, default=1.0)
    parser.add_argument('--window', type=int, default=30)
    parser.add_argument('--linewidth', type=int, default=0)
    parser.add_argument('--flip', dest='flip', action='store_true')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    vdo_path_0, vdo_path_1  = args.video_in
    frames_front = read_video_frames(vdo_path_0)[200:]
    frames_back = read_video_frames(vdo_path_1)
    h, w, _ = frames_front[0].shape

    # # to be removed
    # frames_0 = frames_0
    # frames_1 = frames_1
    print(len(frames_front), len(frames_back))
    frames_0 = frames_front + frames_back[100:]
    frames_1 = frames_front[:-100] + frames_back
    
    v_start = 0 
    v_end   = (w-1) + (h-1) * args.slope
    v_slope = (v_end - v_start) / args.window
    if args.flip:
        v_slope *= -1
    v_const = (v_end+v_start)/2 - (len(frames_front)-60) * v_slope

    grid_y, grid_x = np.meshgrid(np.arange(w), np.arange(h))
    grid_value = grid_y + grid_x * args.slope

    frames_out = []
    vdo_len = len(frames_0)
    for i in tqdm(range(vdo_len)):
        v_threshold = i * v_slope + v_const
        mask = grid_value > v_threshold

        f_0 = frames_0[i]
        f_1 = frames_1[i]
        f_out = np.zeros_like(f_0)
        f_out[mask] = f_0[mask]
        f_out[~mask] = f_1[~mask]
        frames_out.append(f_out)
    
    imageio.mimsave(args.video_out,
                    frames_out,
                    fps=args.fps, macro_block_size=1)    

def merge_video():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first')
    parser.add_argument('--second')
    parser.add_argument('--out')
    parser.add_argument('--axis', type=int, default=0)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    frames_first  = read_video_frames(args.first)
    frames_second = read_video_frames(args.second)
    frames_out   = []

    frame_len = min(len(frames_first), len(frames_second))
    for i in range(frame_len):
        left  = frames_first[i]
        right = frames_second[i]
        out = np.concatenate([left, right], axis=args.axis)
        frames_out.append(out)

    imageio.mimsave(args.out,
                    frames_out,
                    fps=args.fps, macro_block_size=1)
    print('Export video:', args.out)

def loop():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_in')
    parser.add_argument('--video_out')
    parser.add_argument('-fps', type=int, default=30)
    args = parser.parse_args()

    frames = read_video_frames(args.video_in)
    frames += frames[::-1]
    imageio.mimsave(args.video_out,
                    frames,
                    fps=args.fps, macro_block_size=1)



if __name__ == '__main__':
    # extract_frames()
    # generate_video()
    merge_video()
    # add_text()
    # switch_video()
    # combine_switch_videos()
    # loop()
    # horizontal_merge_videos()