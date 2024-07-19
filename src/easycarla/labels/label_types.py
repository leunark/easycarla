from enum import Enum
import carla

class ObjectType(Enum):
    CAR = "Car"
    VAN = "Van"
    TRUCK = "Truck"
    PEDESTRIAN = "Pedestrian"
    PERSON_SITTING = "Person_sitting"
    CYCLIST = "Cyclist"
    TRAM = "Tram"
    MISC = "Misc"
    DONT_CARE = "DontCare"

def map_carla_to_kitti(carla_label: carla.CityObjectLabel) -> ObjectType:
    mapping = {
        carla.CityObjectLabel.Buildings: ObjectType.DONT_CARE,
        carla.CityObjectLabel.Fences: ObjectType.DONT_CARE,
        carla.CityObjectLabel.Other: ObjectType.MISC,
        carla.CityObjectLabel.Pedestrians: ObjectType.PEDESTRIAN,
        carla.CityObjectLabel.Poles: ObjectType.DONT_CARE,
        carla.CityObjectLabel.RoadLines: ObjectType.DONT_CARE,
        carla.CityObjectLabel.Roads: ObjectType.DONT_CARE,
        carla.CityObjectLabel.Sidewalks: ObjectType.DONT_CARE,
        carla.CityObjectLabel.TrafficSigns: ObjectType.DONT_CARE,
        carla.CityObjectLabel.Vegetation: ObjectType.DONT_CARE,
        carla.CityObjectLabel.Car: ObjectType.CAR,
        carla.CityObjectLabel.Bus: ObjectType.VAN,  # Could be TRUCK depending on size
        carla.CityObjectLabel.Truck: ObjectType.TRUCK,
        carla.CityObjectLabel.Motorcycle: ObjectType.CYCLIST,
        carla.CityObjectLabel.Bicycle: ObjectType.CYCLIST,
        carla.CityObjectLabel.Rider: ObjectType.CYCLIST,
        carla.CityObjectLabel.Train: ObjectType.TRAM,
        carla.CityObjectLabel.Walls: ObjectType.DONT_CARE,
        carla.CityObjectLabel.Sky: ObjectType.DONT_CARE,
        carla.CityObjectLabel.Ground: ObjectType.DONT_CARE,
        carla.CityObjectLabel.Bridge: ObjectType.DONT_CARE,
        carla.CityObjectLabel.RailTrack: ObjectType.DONT_CARE,
        carla.CityObjectLabel.GuardRail: ObjectType.DONT_CARE,
        carla.CityObjectLabel.TrafficLight: ObjectType.DONT_CARE,
        carla.CityObjectLabel.Static: ObjectType.MISC,
        carla.CityObjectLabel.Dynamic: ObjectType.MISC,
        carla.CityObjectLabel.Water: ObjectType.DONT_CARE,
        carla.CityObjectLabel.Terrain: ObjectType.DONT_CARE,
        carla.CityObjectLabel.Any: ObjectType.MISC,
    }
    return mapping.get(carla_label, ObjectType.DONT_CARE)

def map_kitti_to_carla(kitti_type: ObjectType) -> carla.CityObjectLabel:
    mapping = {
        ObjectType.CAR: carla.CityObjectLabel.Car,
        ObjectType.VAN: carla.CityObjectLabel.Bus,  # Assuming Van maps to Bus in Carla
        ObjectType.TRUCK: carla.CityObjectLabel.Truck,
        ObjectType.PEDESTRIAN: carla.CityObjectLabel.Pedestrians,
        ObjectType.PERSON_SITTING: carla.CityObjectLabel.Pedestrians,  # No direct equivalent, mapping to Pedestrians
        ObjectType.CYCLIST: carla.CityObjectLabel.Bicycle,  # Could be Bicycle, Motorcycle, or Rider
        ObjectType.TRAM: carla.CityObjectLabel.Train,
        ObjectType.MISC: carla.CityObjectLabel.Other,
        ObjectType.DONT_CARE: carla.CityObjectLabel.Any,
    }
    return mapping.get(kitti_type, carla.CityObjectLabel.Other)