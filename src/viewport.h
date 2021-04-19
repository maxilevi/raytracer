/*
 * Created by Maximiliano Levi on 4/19/2021.
 */

#ifndef RAYTRACER_VIEWPORT_H
#define RAYTRACER_VIEWPORT_H


struct Viewport {
    Vector3 view_port_lower_left_corner;
    Vector3 origin;
    Vector3 horizontal;
    Vector3 vertical;
    uint32_t width;
    uint32_t height;
    uint32_t viewport_height;
    uint32_t viewport_width;
};


#endif //RAYTRACER_VIEWPORT_H
