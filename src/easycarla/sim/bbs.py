import carla
import numpy as np

class BBS:
    
    @staticmethod
    def get_vehicles_and_pedestrians_bbs(world):
        # Define the types you want to retrieve
        bb_types = [
            carla.CityObjectLabel.Pedestrians,
            carla.CityObjectLabel.Car,
            carla.CityObjectLabel.Bus,
            carla.CityObjectLabel.Truck,
            carla.CityObjectLabel.Motorcycle,
            carla.CityObjectLabel.Bicycle,
            carla.CityObjectLabel.Rider]
        # Retrieve the bounding boxes for the specified types
        bbs = [world.get_level_bbs(bb_type) for bb_type in bb_types]
        bbs = [bb for sublist in bbs for bb in sublist]
        return bbs
    
    @staticmethod
    def get_calibration(image: carla.Image):
        calibration = np.identity(3)
        calibration[0, 2] = image.width / 2.0
        calibration[1, 2] = image.height / 2.0
        calibration[0, 0] = calibration[1, 1] = image.width / (2.0 * np.tan(image.fov * np.pi / 360.0))
        return calibration
    
    @staticmethod
    def build_projection_matrix(w: float, h: float, fov: float):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K
    
    @staticmethod
    def get_image_point(loc: carla.Location, K: np.ndarray, w2c: np.ndarray):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]