import numpy as np
from abc import ABC, abstractmethod


# This function gets a vector and returns its normalized form.
def normalize(vector: np.array):
    return vector / np.linalg.norm(vector)

# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, normal):
    return vector - (2 * (vector.dot(normal)) * normal)


class LightSource(ABC):

    def __init__(self, intensity):
        self.intensity = intensity
        self.specular = np.ones(3)

    @abstractmethod
    def get_light_ray(self, intersection):
        pass

    @abstractmethod
    def get_distance_from_light(self, intersection):
        pass

    @abstractmethod
    def get_intensity(self, intersection):
        pass

class DirectionalLight(LightSource):

    def __init__(self, intensity: np.array, direction: np.array):
        super().__init__(intensity)
        self.direction = normalize(direction)

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection, self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.inf

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity

class PointLight(LightSource):

    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl * d + self.kq * (d ** 2))


class SpotLight(LightSource):

    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.direction = normalize(np.array(direction))
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        ray = self.get_light_ray(intersection)
        return self.intensity * np.dot(self.direction, ray.direction) / (self.kc + self.kl * d + self.kq * (d ** 2))

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        nearest_object = None
        min_distance = np.inf

        for obj in objects:
            distance, object_returned = obj.intersect(self)
            if float and isinstance(distance, float) and distance < min_distance:
                min_distance = distance
                nearest_object = object_returned

        return nearest_object, min_distance

class Object3D:

    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection

class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = (np.dot(v, self.normal) / np.dot(self.normal, ray.direction))
        if t > 0:
            return t, self
        else:
            return None, None

class Triangle(Object3D):
    # Triangle gets 3 points as arguments
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal()

    def compute_normal(self):
        return np.cross(self.b - self.a, self.c - self.a)

    def intersect(self, ray):
        t, _ = Plane(self.normal, self.b).intersect(ray)
        if t:
            p = ray.origin + t * ray.direction

            v = self.b - self.a
            u = p - self.a
            if np.dot(self.normal, np.cross(v, u)) < 0:
                return None, None

            v = self.c - self.b
            u = p - self.b
            if np.dot(self.normal, np.cross(v, u)) < 0:
                return None, None

            v = self.a - self.c
            u = p - self.c
            if np.dot(self.normal, np.cross(v, u)) < 0:
                return None, None

            return t, self

        return None, None

class Sphere(Object3D):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        b = 2 * np.dot(ray.direction, ray.origin - self.center)
        delta = (b ** 2) - 4 * (np.linalg.norm(ray.origin - self.center) ** 2 - self.radius ** 2)

        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                return min(t1, t2), self

        return None, None

class Mesh(Object3D):
    # Mesh are defined by a list of vertices, and a list of faces.
    # The faces are triplets of vertices by their index number.
    def __init__(self, v_list, f_list):
        self.v_list = v_list
        self.f_list = f_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        return [Triangle(self.v_list[a], self.v_list[b], self.v_list[c]) for a, b, c in self.f_list]

    def apply_materials_to_triangles(self):
        for t in self.triangle_list:
            t.set_material(self.ambient, self.diffuse, self.specular, self.shininess, self.reflection)

    def intersect(self, ray: Ray):
        nearest_object, min_dist = ray.nearest_intersected_object(self.triangle_list)
        return min_dist, nearest_object
