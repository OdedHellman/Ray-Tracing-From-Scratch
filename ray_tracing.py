from helper_classes import *
import matplotlib.pyplot as plt
import numpy as np


def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    """
    :param camera: The position of the camera
    :param ambient: Color of the Ambient light
    :param lights: list of all LightSource objects
    :param objects: list of all Object3D objects
    :param screen_size: tuple of the height and width
    :param max_depth: recursion level for each ray

    :return: image
    """
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            pixel = np.array([x, y, 0])
            color = np.zeros(3)
            direction = normalize(pixel - camera)
            origin = camera
            reflection = 1

            for k in range(max_depth):
                ray = Ray(origin=origin, direction=direction)
                # Gets nearest object for a given ray
                nearest_object, min_distance = ray.nearest_intersected_object(objects=objects)

                if nearest_object is None:
                    break

                intersection = origin + min_distance * direction

                if isinstance(nearest_object, Triangle) or isinstance(nearest_object, Plane):
                    normal_to_surface = nearest_object.normal
                else:
                    normal_to_surface = normalize(intersection - nearest_object.center)

                sum_of_lights = np.zeros(3)
                # fix to the intersection point
                shifted_point = intersection + 1e-5 * normal_to_surface

                for light_src in lights:

                    light_illumination = np.zeros(3)
                    ray = light_src.get_light_ray(intersection)
                    ray_direction = ray.direction
                    shadow_ray = Ray(intersection, ray_direction)
                    if nearest_object in objects:
                        objects.remove(nearest_object)
                        _, shadow_distance = shadow_ray.nearest_intersected_object(objects=objects)
                        objects.append(nearest_object)
                    else:
                        _, shadow_distance = shadow_ray.nearest_intersected_object(objects=objects)
                    if shadow_distance < min_distance:
                        break

                    light_illumination += nearest_object.diffuse * light_src.specular * np.dot(normal_to_surface,
                                                                                               ray_direction)
                    intersection_to_camera = normalize(camera - intersection)
                    H = normalize(ray_direction + intersection_to_camera)
                    light_illumination += nearest_object.specular * light_src.specular * np.dot(normal_to_surface, H) \
                                          ** nearest_object.shininess
                    sum_of_lights += light_illumination * light_src.get_intensity(intersection=intersection)

                color += (nearest_object.ambient * ambient + sum_of_lights) * reflection
                reflection *= nearest_object.reflection
                origin = shifted_point
                direction = reflected(direction, normal_to_surface)

            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)

    return image
