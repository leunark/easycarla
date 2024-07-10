import carla
import pygame
import numpy as np

class BoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_camera_bounding_boxes(vehicles: list, image: carla.Image):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [BoundingBoxes.get_bounding_box(vehicle, image) for vehicle in vehicles]

        return bounding_boxes
    
    @staticmethod
    def get_depth_at_point(depth_image_array, point):
        """Retrieve depth value at the specified point from the depth image array."""
        x, y = int(point[0, 0]), int(point[0, 1])
        
        if 0 <= x < depth_image_array.shape[1] and 0 <= y < depth_image_array.shape[0]:
            # Convert the depth image value to meters
            depth_value = int(depth_image_array[y, x][0]) + int(depth_image_array[y, x][1]) * 256 + int(depth_image_array[y, x][2]) * 256 * 256
            normalized_depth = depth_value / (256 ** 3 - 1)  # Normalize to range [0, 1]
            depth_in_meters = normalized_depth * 1000.0  # Assuming 0-1 maps to 0-1000 meters
            
            return depth_in_meters
        return None

    @staticmethod
    def is_occluded(bounding_box, depth_image_array):
        """Determine if the bounding box is occluded based on depth image array."""
        num_occluded_points = 0
        for point in bounding_box:
            x, y, depth = int(point[0, 0]), int(point[0, 1]), point[0, 2]
            depth_at_point = depth_image_array[y, x]
            if depth >= depth_at_point + 0.1:
                num_occluded_points += 1
        if num_occluded_points > 6:
            return True
        return False

    # Assuming `image` is the depth image you obtained from Carla
    @staticmethod
    def decode_depth_image(image: carla.Image):
        """
        Decodes a depth image from Carla to create a depth matrix.
        
        Args:
            image (carla.Image): The depth image obtained from Carla.
            
        Returns:
            np.ndarray: The decoded depth matrix.
        """
        # Convert the image raw data to a numpy array
        image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
        
        # Reshape array into (height, width, 4) where the last dimension is RGBA
        image_data = np.reshape(image_data, (image.height, image.width, 4))
        
        # Extract the R, G, and B channels (ignore A)
        R = image_data[:, :, 2].astype(np.float32)
        G = image_data[:, :, 1].astype(np.float32)
        B = image_data[:, :, 0].astype(np.float32)
        
        # Calculate the normalized depth
        normalized_depth = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1)
        
        # Convert normalized depth to meters
        depth_in_meters = 1000.0 * normalized_depth
        
        return depth_in_meters


    @staticmethod
    def filter_occluded(bounding_boxes: list, image_depth: carla.Image):
        """
        Filters out occluded bounding boxes based on depth image.
        
        :param bounding_boxes: List of bounding boxes, each represented by an 8x3 matrix in image space.
        :param image_depth: Depth image from Carla.
        :return: Filtered list of bounding boxes.
        """
        # Convert Carla depth image to numpy array once
        depth_image_array = BoundingBoxes.decode_depth_image(image_depth)
        
        bounding_boxes = [bb for bb in bounding_boxes if np.all(bb[:, 2] > 0)]
        bounding_boxes = [bb for bb in bounding_boxes if np.all((bb[:, 0] >= 0) & (bb[:, 0] < image_depth.width))]
        bounding_boxes = [bb for bb in bounding_boxes if np.all((bb[:, 1] >= 0) & (bb[:, 1] < image_depth.height))]
        bounding_boxes = [bb for bb in bounding_boxes if not BoundingBoxes.is_occluded(bb, depth_image_array)]

        return bounding_boxes

    @staticmethod
    def get_calibration(image: carla.Image):
        calibration = np.identity(3)
        calibration[0, 2] = image.width / 2.0
        calibration[1, 2] = image.height / 2.0
        calibration[0, 0] = calibration[1, 1] = image.width / (2.0 * np.tan(image.fov * np.pi / 360.0))
        return calibration


    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes, image: carla.Image):
        """
        Draws bounding boxes on pygame display.
        """
        bb_surface = pygame.Surface((image.width, image.height))
        bb_surface.set_colorkey((0, 0, 0))
        bb_color = (248, 64, 24)
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            pygame.draw.line(bb_surface, bb_color, points[0], points[1])
            pygame.draw.line(bb_surface, bb_color, points[1], points[2])
            pygame.draw.line(bb_surface, bb_color, points[2], points[3])
            pygame.draw.line(bb_surface, bb_color, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, bb_color, points[4], points[5])
            pygame.draw.line(bb_surface, bb_color, points[5], points[6])
            pygame.draw.line(bb_surface, bb_color, points[6], points[7])
            pygame.draw.line(bb_surface, bb_color, points[7], points[4])
            # base-top
            pygame.draw.line(bb_surface, bb_color, points[0], points[4])
            pygame.draw.line(bb_surface, bb_color, points[1], points[5])
            pygame.draw.line(bb_surface, bb_color, points[2], points[6])
            pygame.draw.line(bb_surface, bb_color, points[3], points[7])
        display.blit(bb_surface, (0, 0))

    @staticmethod
    def draw_bounding_boxes_in_world(world: carla.World, vehicles: list[carla.Vehicle]):
        """
        Draws 3D bounding boxes in the world using the Carla debug drawing API.
        """
        for vehicle in vehicles:
            # Get the bounding box points in vehicle local space
            bb_local = BoundingBoxes._create_bb_points(vehicle)
            
            # Transform bounding box points to world coordinates
            bb_world = BoundingBoxes._vehicle_to_world(bb_local, vehicle)
            bb_world = np.array(bb_world).T

            color = carla.Color(1, 100, 20, 30)
            lifetime = 0.2
            for i in range(4):
                world.debug.draw_line(
                    carla.Location(x=bb_world[i, 0], y=bb_world[i, 1], z=bb_world[i, 2]),
                    carla.Location(x=bb_world[(i + 1) % 4, 0], y=bb_world[(i + 1) % 4, 1], z=bb_world[(i + 1) % 4, 2]),
                    life_time=lifetime, color=color)
                world.debug.draw_line(
                    carla.Location(x=bb_world[i + 4, 0], y=bb_world[i + 4, 1], z=bb_world[i + 4, 2]),
                    carla.Location(x=bb_world[(i + 1) % 4 + 4, 0], y=bb_world[(i + 1) % 4 + 4, 1], z=bb_world[(i + 1) % 4 + 4, 2]),
                    life_time=lifetime, color=color)
                world.debug.draw_line(
                    carla.Location(x=bb_world[i, 0], y=bb_world[i, 1], z=bb_world[i, 2]),
                    carla.Location(x=bb_world[i + 4, 0], y=bb_world[i + 4, 1], z=bb_world[i + 4, 2]),
                    life_time=lifetime, color=color)

                
    @staticmethod
    def get_bounding_box(vehicle: carla.Vehicle, image: carla.Image):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """
        calibration = BoundingBoxes.get_calibration(image)
        bb_cords = BoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = BoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, image)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle: carla.Vehicle, sensor_data: carla.SensorData):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = BoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = BoundingBoxes._world_to_sensor(world_cord, sensor_data)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle: carla.Vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = BoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = BoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor_data: carla.SensorData):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = BoundingBoxes.get_matrix(sensor_data.transform)
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform: carla.Transform):
        """
        Creates matrix from carla transform.
        """
        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix