#!/usr/bin/env python3

import cv2
import base64
import tempfile
import os
import requests
import numpy as np
import random
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task
from .base_detector import BaseDetector


class DinoxDetector(BaseDetector):
    """DINOX + ChatGPT based food detection"""
    
    def __init__(self, node=None, dinox_api_key="", openai_api_key="", prompt=""):
        """
        Initialize DINOX detector
        
        Args:
            node: ROS node instance for logging
            dinox_api_key: DINOX API key
            openai_api_key: OpenAI API key
            prompt: ChatGPT identification prompt
        """
        super().__init__(node)
        self.dinox_api_key = dinox_api_key
        self.openai_api_key = openai_api_key
        self.prompt = prompt
        self.current_item = ""
        self.single_bite = True
        
        # Setup DINOX client
        self._setup_dinox()
        
        # Setup OpenAI headers
        self.openai_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
    
    def _setup_dinox(self):
        """Initialize DINOX client"""
        try:
            config = Config(self.dinox_api_key)
            self.dinox_client = Client(config)
            if self.node:
                self.node.get_logger().info('DINOX client initialized successfully')
        except Exception as e:
            if self.node:
                self.node.get_logger().error(f'Failed to initialize DINOX: {e}')
            raise e
    
    def detect_food(self, frame):
        """
        Use ChatGPT + DINOX stack to detect food and return highest confidence bounding box
        
        Args:
            frame: Input image frame (BGR format)
            
        Returns:
            list: [x1, y1, x2, y2] bounding box if food found, None otherwise
        """
        try:
            # Save frame to temporary file for DINOX
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
                temp_path = tmpfile.name
            cv2.imwrite(temp_path, frame)
            
            # Step 1: Identify food items with ChatGPT
            if self.node:
                self.node.get_logger().info("Identifying food items with ChatGPT...")
            identified_objects = self._identify_with_chatgpt(frame)
            
            if not identified_objects:
                if self.node:
                    self.node.get_logger().error("ChatGPT failed to identify food items")
                os.remove(temp_path)
                return None
            
            if self.node:
                self.node.get_logger().info(f"ChatGPT identified: {identified_objects}")
            
            # Step 2: Randomly select one item
            selected_item = random.choice(identified_objects)
            if self.node:
                self.node.get_logger().info(f"Randomly selected: {selected_item}")
            
            # Parse bite information
            self._parse_item_info(selected_item)
            
            # Step 3: Create DINOX prompt
            text_prompt = self.current_item + " ."
            
            # Step 4: Detect with DINOX
            if self.node:
                self.node.get_logger().info(f"Detecting with DINOX using prompt: '{text_prompt}'")
            input_boxes, confidences, class_names, class_ids = self._detect_with_dinox(temp_path, text_prompt)
            
            # Cleanup temp file
            os.remove(temp_path)
            
            if input_boxes is None or len(input_boxes) == 0:
                if self.node:
                    self.node.get_logger().warn("DINOX could not detect the selected food item")
                return None
            
            # Step 5: Get highest confidence detection
            highest_idx = np.argmax(confidences)
            highest_bbox = input_boxes[highest_idx]
            highest_confidence = confidences[highest_idx]
            
            if self.node:
                self.node.get_logger().info(f"Highest confidence detection: {class_names[highest_idx]} "
                                         f"with confidence {highest_confidence:.3f}")
                self.node.get_logger().info(f"Bounding box: {highest_bbox}")
            
            return highest_bbox
            
        except Exception as e:
            if self.node:
                self.node.get_logger().error(f"Error with DINOX stack detection: {e}")
            return None
    
    def _identify_with_chatgpt(self, frame):
        """Identify food items using ChatGPT"""
        try:
            # Convert frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }],
                "max_tokens": 300
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                   headers=self.openai_headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content'].strip()
                if ',' in content:
                    return [item.strip() for item in content.split(',')]
                else:
                    return [content]
            return None
        except Exception as e:
            if self.node:
                self.node.get_logger().error(f"ChatGPT identification failed: {e}")
            return None
    
    def _detect_with_dinox(self, image_path, text_prompt):
        """Detect objects using DINOX and return bounding boxes"""
        try:
            image_url = self.dinox_client.upload_file(image_path)
            
            task = V2Task(
                api_path="/v2/task/dinox/detection",
                api_body={
                    "model": "DINO-X-1.0",
                    "image": image_url,
                    "prompt": {"type": "text", "text": text_prompt},
                    "targets": ["bbox"],
                    "bbox_threshold": 0.25,
                    "iou_threshold": 0.8,
                }
            )
            
            self.dinox_client.run_task(task)
            result = task.result
            
            if not result or "objects" not in result or len(result["objects"]) == 0:
                return None, None, None, None
            
            objects = result["objects"]
            input_boxes = []
            confidences = []
            class_names = []
            class_ids = []
            
            classes = [x.strip().lower() for x in text_prompt.split('.') if x.strip()]
            class_name_to_id = {name: id for id, name in enumerate(classes)}
            
            for obj in objects:
                input_boxes.append(obj["bbox"])
                confidences.append(obj["score"])
                category = obj["category"].lower().strip()
                class_names.append(category)
                class_ids.append(class_name_to_id.get(category, 0))

            return np.array(input_boxes), confidences, class_names, np.array(class_ids)
        except Exception as e:
            if self.node:
                self.node.get_logger().error(f"DINOX detection failed: {e}")
            return None, None, None, None
    
    def _parse_item_info(self, selected_item):
        """Parse bite information from selected item"""
        parts = selected_item.rsplit(' ', 1)
        if len(parts) == 2 and parts[1].isdigit():
            self.current_item = parts[0]
            bite_number = int(parts[1])
        else:
            self.current_item = selected_item
            bite_number = 1
            print("No bite number found, defaulting to 1")
        
        # Set single/multi bite based on number
        self.single_bite = bite_number <= 1
    
    def get_current_item(self):
        """Get the current detected food item name"""
        return self.current_item
    
    def is_single_bite(self):
        """Check if current item is single bite"""
        return self.single_bite