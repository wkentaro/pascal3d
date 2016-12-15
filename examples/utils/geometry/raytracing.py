#!/usr/bin/env python

import time

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
    depth = pascal3d.utils.raytrace_camera_frame_on_triangles(
        pt_camera_origin, pts_camera_frame, pts_tri0, pts_tri1, pts_tri2)
    print('elapsed_time: {} [s]'.format(time.time() - t_start))

    depth_img = np.zeros((height, width), dtype=np.float32)
    depth_img[...] = np.nan
    depth_img[mask == 1] = depth

    # visualize
    depth_colorized = matplotlib.cm.jet(depth_img / depth_img.max())[:, :, :3]
    depth_colorized[mask != 1] = [0, 0, 0]
    depth_colorized = (depth_colorized * 255).astype(np.uint8)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(depth_colorized)
    plt.show()


if __name__ == '__main__':
    main()
