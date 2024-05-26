import os
import sys
module_path = os.path.realpath(os.path.join(__file__ , '..', '..', '..'))
sys.path.append(module_path)
import numpy as np
import open3d as o3d

import torch
from pathlib import Path
import math

import tkinter as tk
from PIL import ImageTk, Image
import torchvision.transforms.functional as F

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500

class Viewer:
    def __init__(self, camera, model, grid):
        self.camera = camera
        self.grid = grid
        self.model = model
        self.cx = 0
        self.cy = 0

        window = tk.Tk()
        #creates the canvas
        canvas = tk.Canvas(window, width = WINDOW_WIDTH, height = WINDOW_HEIGHT)

        pil_image = F.to_pil_image(torch.zeros(3, 128, 128))
        pil_image = F.resize(pil_image, (WINDOW_HEIGHT, WINDOW_WIDTH))
        self.photo_image = ImageTk.PhotoImage(pil_image)

        image_container =canvas.create_image(0,0, anchor="nw",image=self.photo_image)
        canvas.pack(side=tk.BOTTOM)
        canvas.bind('<Button-1>', self.press)
        canvas.bind('<Button-3>', self.press)
        canvas.bind('<B1-Motion>', self.drag)
        canvas.bind('<B3-Motion>', self.pan)
        canvas.bind('<Button-4>', self.wheelup)
        canvas.bind('<Button-5>', self.wheeldown)

        self.window = window
        self.canvas = canvas
        self.image_container = image_container
        self.closing = False
        def close():
            self.closing = True

        window.protocol('WM_DELETE_WINDOW', close)
        self.render()



    def render(self):
        rays = self.camera.get_rays()
        H, W = rays.shape[:2]
        # rays = rays.view(H*W, -1).half()
        rays = rays.view(H*W, -1)
        all_rgbs = torch.zeros((H*W, 3)).float().to(rays)

        batch_size = 1024
        for bs in range(0, H*W, batch_size):
            be = bs + batch_size

            ray_o = rays[bs:be, 0:3]
            ray_d = rays[bs:be, 3:6]
            near = rays[bs:be, 6:7]
            far = rays[bs:be, 7:8]
            mask = rays[bs:be, 8] == 1
            rgbs = all_rgbs[bs:be]


            # filter with mask
            ray_o = ray_o[mask]
            ray_d = ray_d[mask]
            near = near[mask]
            far = far[mask]

            if ray_o.shape[0] > 0:
                with torch.cuda.amp.autocast(enabled=False):
                    with torch.no_grad():
                        ray_i, t_starts, t_ends = self.grid.sample(ray_o, ray_d)
                        n_rendering_samples = len(t_starts)
                        if n_rendering_samples  == 0:
                            return None
                        res = self.model.render(ray_o, ray_d, ray_i, t_starts, t_ends)
                        rgb = res['rgb']
                        rgbs[mask] = rgb
        # convert to PIL
        pil_image = F.to_pil_image(torch.fliplr(all_rgbs.view(H, W, 3)).permute(2, 0, 1))
        pil_image = F.resize(pil_image, (WINDOW_HEIGHT, WINDOW_WIDTH))
        self.photo_image = ImageTk.PhotoImage(pil_image)
        self.canvas.itemconfig(self.image_container,image=self.photo_image)
        return
    def press(self, input):
        self.cx = input.x
        self.cy = input.y
        # self.render()
    def drag(self, input):
        dx = (input.x - self.cx) / WINDOW_WIDTH
        dy = (input.y - self.cy) / WINDOW_HEIGHT
        self.camera.rotate(dx, dy)
        self.cx = input.x
        self.cy = input.y
        # self.render()

    def pan(self, input):
        dx = (input.x - self.cx) / WINDOW_WIDTH
        dy = (input.y - self.cy) / WINDOW_HEIGHT
        self.camera.pan(dx, dy)
        self.cx = input.x
        self.cy = input.y
        # self.render()

    def wheelup(self, input):
        self.camera.zoom(1)
        # self.render()

    def wheeldown(self, input):
        self.camera.zoom(-1)
        # self.render()
    

    def loop(self):
        # self.window.mainloop()
        while True:
            if self.closing:
                print('closing!')
                break
            self.window.update_idletasks()
            self.window.update()
            self.render()
