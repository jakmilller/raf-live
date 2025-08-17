#!/usr/bin/env python3

import cv2
import base64
import google.generativeai as genai
from .base_detector import BaseDetector


class GeminiDetector(BaseDetector):
    """Gemini-based food detection"""
    
    def __init__(self, node=None, prompt="", current_food_target=None):
        """
        Initialize Gemini detector
        
        Args:
            node: ROS node instance for logging
            prompt: Detection prompt text
            current_food_target: Specific food target (optional)
        """
        super().__init__(node)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        self.prompt = prompt
        self.current_food_target = current_food_target
        self.current_item = ""
        self.single_bite = True
    
    def detect_food(self, frame):
        """
        Use Gemini to detect food and return center point coordinates
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            tuple: (x, y) coordinates if food found, None otherwise
        """
        try:
            # Get image dimensions
            height, width = frame.shape[:2]
            if self.node:
                self.node.get_logger().info(f"Frame dimensions sent to Gemini: {width}x{height}")

            # Convert frame to RGB and encode as base64
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', frame_rgb)
            base64_image = base64.b64encode(buffer).decode('utf-8')

            # Modify the loaded prompt with image dimensions and target food
            prompt_text = self.prompt

            # Add specific food target if we have one
            if self.current_food_target:
                prompt_text = prompt_text.replace(
                    "identify the food item you see", 
                    f"find the {self.current_food_target}"
                )
                prompt_text = prompt_text.replace(
                    "cannot find any food", 
                    f"cannot find a {self.current_food_target}"
                )

            # Generate response
            response = self.model.generate_content([
                prompt_text,
                {"mime_type": "image/jpeg", "data": base64_image}
            ])

            response_text = response.text.strip()
            if self.node:
                self.node.get_logger().info(f"Gemini response: {response_text}")

            # Parse response
            if "FOUND:" in response_text:
                try:
                    coords_part = response_text.split("FOUND:")[1].strip()
                    coords_part = coords_part.split('\n')[0].split(',')

                    if len(coords_part) >= 2:
                        x = int(float(int(coords_part[0].strip())/1000)*width)
                        y = int(float(int(coords_part[1].strip())/1000)*height)

                        # Ensure coordinates are within image bounds
                        x = max(0, min(x, width - 1))
                        y = max(0, min(y, height - 1))

                        if self.node:
                            self.node.get_logger().info(f"Parsed coordinates: ({x}, {y})")
                        
                        # Parse bite information from Gemini response
                        self._parse_item_info(response_text)
                        
                        return (x, y)

                except (ValueError, IndexError) as e:
                    if self.node:
                        self.node.get_logger().error(f"Failed to parse coordinates: {e}")

            elif "NOT_FOUND" in response_text:
                if self.node:
                    self.node.get_logger().warn("Gemini could not find food item")
            else:
                if self.node:
                    self.node.get_logger().warn(f"Unexpected response format: {response_text}")

            return None

        except Exception as e:
            if self.node:
                self.node.get_logger().error(f"Error with Gemini detection: {e}")
            return None
    
    def _parse_item_info(self, response_text):
        """Parse item name and bite information from Gemini response"""
        if "FOUND:" in response_text:
            # Extract item name and bite info if present
            lines = response_text.split('\n')
            for line in lines:
                if 'item:' in line.lower() or 'food:' in line.lower():
                    item_text = line.split(':')[-1].strip()
                    parts = item_text.rsplit(' ', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        self.current_item = parts[0]
                        bite_number = int(parts[1])
                    else:
                        self.current_item = item_text
                        bite_number = 1
                    self.single_bite = bite_number <= 1
                    break
    
    def get_current_item(self):
        """Get the current detected food item name"""
        return self.current_item
    
    def is_single_bite(self):
        """Check if current item is single bite"""
        return self.single_bite
    
    def set_food_target(self, target):
        """Set specific food target"""
        self.current_food_target = target
