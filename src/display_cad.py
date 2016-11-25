#!/usr/bin/env python

import os.path as osp

import chainer
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *


def load_off(filename):
    with open(filename, 'r') as f:
        if 'OFF' != f.readline().strip():
            raise ValueError('Not a valid OFF header')
        n_verts, n_faces, _ = map(int, f.readline().strip().split(' '))
        vertices = []
        for i_vert in range(n_verts):
            vertex = map(float, f.readline().strip().split(' '))
            vertices.append(vertex)
        faces = []
        for i_face in range(n_faces):
            face = map(int, f.readline().strip().split(' ')[1:])
            faces.append(face)
        return vertices, faces


def redraw():
    for surface in surfaces:
        glBegin(GL_POLYGON)
        for i_vertex in surface:
            glColor4fv((1, 1, 1, 0.4))
            glVertex3fv(vertices[i_vertex])
        glEnd()

    glBegin(GL_LINES)
    for vertex in vertices:
        glVertex3fv(vertex)
    glEnd()


def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    glTranslatef(0.0, 0.0, -1)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    glTranslatef(-0.5,0,0)
                if event.key == pygame.K_RIGHT:
                    glTranslatef(0.5,0,0)

                if event.key == pygame.K_UP:
                    glTranslatef(0,1,0)
                if event.key == pygame.K_DOWN:
                    glTranslatef(0,-1,0)

                if event.key == pygame.K_l:
                    glRotatef(1, 3, 0, 0);
                if event.key == pygame.K_h:
                    glRotatef(1, -3, 0, 0);

        glRotatef(5, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        redraw()
        pygame.display.flip()
        pygame.time.wait(10)


if __name__ == '__main__':
    dataset_dir = chainer.dataset.get_dataset_directory(
        'pascal3d/PASCAL3D+_release1.1')
    off_fname = osp.join(dataset_dir, 'Anchor/aeroplane/01.off')
    vertices, surfaces = load_off(off_fname)
    colors = (
        (1,0,0),
        (0,1,0),
        (0,0,1),
        (0,1,0),
        (1,1,1),
        (0,1,1),
        (1,0,0),
        (0,1,0),
        (0,0,1),
        (1,0,0),
        (1,1,1),
        (0,1,1),
    )
    main()
