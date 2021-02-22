#include <iostream>
#include "Vector3.h"
#include "Camera.h"

int main()
{
    Camera camera(200, 100);

    camera.Draw();
    camera.GetFrame();

    return 0;
}