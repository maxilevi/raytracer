# raytracer

wip raytracer written from scratch in C++. the program processes a defined scene then rasterizes it and outputs a `.tga` file. future versions will use SSE.

# eye-candy

![](screenshots/diffuse_big.png)

# how it works

## overview

a `Scene` object contains a collection of volumes.

a `Camera` can draw a given scene. when drawing, the camera casts rays to the screen in order to rasterize the volumes in the scene. after each ray the result is save into a pixel of a `Vector3` buffer of size `width` x `height`. This buffer is then dumped into a `.tga` file so it can be visualized.

current supported volumes are `Triangle`s (i know its not technicaly a volume) and `Sphere`s. `Triangle` support allows us to load custom models.

## models

current version supports the loading of 3d `.ply` models. model loading is done via a simple parser written in C++ (see `src/io/ply.h`)`, from it a group of triangles are extracted and these are used to raycast against. to calculate the intersection between a triangle and a ray the engine uses the [Möller–Trumbore intersection algorithm](https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm)

## anti aliasing

to avoid image "rough edges" on the rendering we cast multiple rays per pixel with a slight randomized offset on each ray. then we average the results and use that as the pixel value. 

## lighting

### diffuse

in order to model diffuse lighting we simulate the way light rays works. whenever a ray hits something depeding on it's reflectivity, it can either be absorbed or bounce of it. repeat this enough times and the light models itself

## effects

# other

## c++ version

i am using C++ 17 to take advantage of `std::execution`

## code style

it tries to follow [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html)

## model credits

["The Jennings Dog"](https://skfb.ly/OrYs) by The British Museum is licensed under [CC Attribution-NonCommercial-ShareAlike](http://creativecommons.org/licenses/by-nc-sa/4.0/).
