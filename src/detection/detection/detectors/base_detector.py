from abc import ABC, abstractmethod

class BaseDetector(ABC):
    """Abstract base class for food detection"""
    
    def __init__(self, node=None):
        """
        Initialize detector
        
        Args:
            node: ROS node instance for logging (optional)
        """
        self.node = node
    
    @abstractmethod
    def detect_food(self, frame):
        """
        Detect food in frame and return detection result
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            Detection result (format depends on detector implementation)
            None if no food detected
        """
        pass
    
    @abstractmethod
    def get_current_item(self):
        """
        Get the current detected food item name
        
        Returns:
            str: Current food item name
        """
        pass
    
    @abstractmethod
    def is_single_bite(self):
        """
        Check if current item is single bite
        
        Returns:
            bool: True if single bite, False if multi-bite
        """
        pass