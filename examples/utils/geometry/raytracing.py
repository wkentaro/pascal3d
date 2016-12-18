#!/usr/bin/env python

import time

import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import PIL.ImageDraw

import pascal3d


def main():
    dataset = pascal3d.dataset.Pascal3DDataset('val')
    data = dataset.get_data(0)
    img = data['img']
    class_cads = data['class_cads']
    cls, obj = data['objects'][1]

    cad = class_cads[cls][obj['cad_index']]
    vertices = cad['vertices']
    vertices_camframe = pascal3d.utils.transform_to_camera_frame(
        vertices,
        obj['viewpoint']['azimuth'],
        obj['viewpoint']['elevation'],
        obj['viewpoint']['distance'],
    )
    vertices_2d = pascal3d.utils.project_points_3d_to_2d(
        vertices, **obj['viewpoint'])
    faces = cad['faces'] - 1

    height, width = img.shape[:2]

    mask_pil = PIL.Image.new('L', (width, height), 0)
    vertices_2d = vertices_2d.astype(int)
    for face in faces:
        xy = vertices_2d[face].flatten().tolist()
        PIL.ImageDraw.Draw(mask_pil).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask_pil)

    im_xy = np.vstack(np.where(mask)).T
    im_xy = im_xy.astype(np.float64)

    pt_camera_origin = np.array([0, 0, 0], dtype=np.float64)
    pts_camera_frame = pascal3d.utils.project_points_2d_to_3d(
        im_xy,
        obj['viewpoint']['theta'],
        obj['viewpoint']['focal'],
        obj['viewpoint']['principal'],
        obj['viewpoint']['viewport'],
    )

    # select triangles with sorting by distance from camera
    # (n_triangles, n_points_tri=3, xyz=3)
    triangles = vertices_camframe[faces]
    indices = np.argsort(np.abs(triangles[:, :, 2]).max(axis=-1))
    mask_pil = PIL.Image.new('L', (width, height), 0)
    faces_sorted = faces[indices]
    for i, face in enumerate(faces_sorted):
        xy = vertices_2d[face].flatten().tolist()
        PIL.ImageDraw.Draw(mask_pil).polygon(xy=xy, outline=1, fill=1)
        mask2 = np.array(mask_pil)
        if mask2.sum() == mask.sum():
            break
    print('faces are subtracted: {} -> {}'.format(len(faces), i+1))

    pts_tri0 = vertices_camframe[faces_sorted[:i][:, 0]]
    pts_tri1 = vertices_camframe[faces_sorted[:i][:, 1]]
    pts_tri2 = vertices_camframe[faces_sorted[:i][:, 2]]

    print('raytracing...: rays: {}, triangles: {}'
          .format(len(pts_camera_frame), i+1))
    t_start = time.time()
    min_depth, max_depth = pascal3d.utils.raytrace_camera_frame_on_triangles(
        pt_camera_origin, pts_camera_frame, pts_tri0, pts_tri1, pts_tri2)
    print('elapsed_time: {} [s]'.format(time.time() - t_start))

    min_depth[min_depth < 1e-6] = 0
    max_depth[max_depth < 1e-6] = 0

    depth = np.zeros((height, width), dtype=np.float32)
    depth[mask == 1] = min_depth

    backdepth = np.zeros((height, width), dtype=np.float32)
    backdepth[mask == 1] = max_depth

    # visualize
    plt.subplot(131)
    plt.imshow(img)

    max_imvalue = max_depth.max()
    min_imvalue = min_depth[min_depth > 0].min()

    # static scaling
    depth = (depth - min_imvalue) / (max_imvalue - min_imvalue)
    depth[depth > 1.0] = 1.0
    depth[depth < 0.0] = 0.0
    depth_viz = matplotlib.cm.jet(depth)[:, :, :3]
    depth_viz = (depth_viz * 255).astype(np.uint8)
    depth_viz[depth == 0] = [0, 0, 0]
    plt.subplot(132)
    plt.imshow(depth_viz)

    backdepth = (backdepth - min_imvalue) / (max_imvalue - min_imvalue)
    backdepth[backdepth > 1.0] = 1.0
    backdepth[backdepth < 0.0] = 0.0
    backdepth_viz = matplotlib.cm.jet(backdepth)[:, :, :3]
    backdepth_viz = (backdepth_viz * 255).astype(np.uint8)
    backdepth_viz[depth == 0] = [0, 0, 0]
    plt.subplot(133)
    plt.imshow(backdepth_viz)

    plt.show()


if __name__ == '__main__':
    main()
