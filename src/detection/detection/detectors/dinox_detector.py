#!/usr/bin/env python3

import cv2
import base64
import tempfile
import os
import requests
import numpy as np
import random
from collections import deque
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task
from .base_detector import BaseDetector
from std_msgs.msg import String


class DinoxDetector(BaseDetector):
    """DINOX + ChatGPT based food detection with voice command support"""
    
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
        
        # Voice command queue system
        self.voice_command_queue = deque(maxlen=10)
        
        # Setup DINOX client
        self._setup_dinox()
        
        # Setup OpenAI headers
        self.openai_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        
        # Subscribe to voice commands if node is provided
        if self.node:
            # Check if voice is enabled in config
            voice_enabled = getattr(self.node, 'config', {}).get('feeding', {}).get('voice', False)
            if voice_enabled:
                self.voice_command_sub = self.node.create_subscription(
                    self._get_string_msg_type(), '/voice_commands', 
                    self.voice_command_callback, 10
                )
                self.currently_serving_pub = self.node.create_publisher(
                    self._get_string_msg_type(), '/currently_serving', 10)
                
                self.voice_command_queue_pub = self.node.create_publisher(
                    self._get_string_msg_type(), '/voice_command_queue', 10)
                
                self.node.get_logger().info('Voice command subscription created')

            

    
    def _get_string_msg_type(self):
        """Get String message type (helper for ROS import)"""
        from std_msgs.msg import String
        return String
    
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
    
    def voice_command_callback(self, msg):
        """Add voice commands to the queue"""
        # Split the commands up by comma
        commands = [cmd.strip() for cmd in msg.data.split(',')]

        for command in commands:
            if command == 'clear':
                self.voice_command_queue.clear()
                if self.node:
                    self.node.get_logger().info("Voice command queue cleared!")
                    self.voice_command_queue_pub.publish(self.queue2string(self.voice_command_queue))
                
            else:
                # Add to queue (deque automatically handles maxlen=10)
                self.voice_command_queue.append(command)
                if self.node:
                    self.node.get_logger().info(f"Added voice command to queue: '{command}' "
                                               f"(Queue size: {len(self.voice_command_queue)})")
                    self.voice_command_queue_pub.publish(self.queue2string(self.voice_command_queue))

        self.voice_command_queue_pub.publish(self.queue2string(self.voice_command_queue))
        

    def get_voice_command(self):
        """Get the next command from queue or return None if empty"""
        if len(self.voice_command_queue) > 0:
            command = self.voice_command_queue.popleft()
            self.voice_command_queue_pub.publish(self.queue2string(self.voice_command_queue))

            if self.node:
                self.node.get_logger().info(f"Processing voice command: '{command}' "
                                           f"(Remaining in queue: {len(self.voice_command_queue)})")
            return command
        return None
    
    def detect_food(self, frame):
        """
        Use voice commands or ChatGPT + DINOX stack to detect food and return highest confidence bounding box
        
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
            
            # Step 1: Check for voice commands first
            voice_command = self.get_voice_command()
            
            if voice_command:
                # Use voice command and parse it properly
                if self.node:
                    self.node.get_logger().info(f"Using voice command: '{voice_command}'")
                
                # Parse voice command for bite information - THIS IS THE CORRECT PARSING
                parts = voice_command.rsplit(' ', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    item_name = parts[0]
                    bite_number = int(parts[1])
                    self.single_bite = bite_number <= 1  # 1 = single bite, 2+ = multi bite
                    self.current_item = item_name
                else:
                    item_name = voice_command
                    bite_number = 1  # Default to single bite
                    self.single_bite = True
                    self.current_item = item_name
                
                selected_item = item_name  # Use just the item name for DINOX
                
            else:
                # Fall back to ChatGPT identification
                if self.node:
                    self.node.get_logger().info("No voice commands in queue, using ChatGPT identification...")
                
                identified_objects = self._identify_with_chatgpt(frame)
                if not identified_objects:
                    if self.node:
                        self.node.get_logger().error("ChatGPT failed to identify food items")
                    os.remove(temp_path)
                    return None
                
                if self.node:
                    self.node.get_logger().info(f"ChatGPT identified: {identified_objects}")
                
                # Check for voice command interrupt after ChatGPT returns
                interrupt_command = self.get_voice_command()
                if interrupt_command:
                    if self.node:
                        self.node.get_logger().info(f"ChatGPT overwritten with voice command: '{interrupt_command}'")
                    
                    # Parse interrupt command the same way
                    parts = interrupt_command.rsplit(' ', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        item_name = parts[0]
                        bite_number = int(parts[1])
                        self.single_bite = bite_number <= 1
                        self.current_item = item_name
                    else:
                        item_name = interrupt_command
                        bite_number = 1
                        self.single_bite = True
                        self.current_item = item_name
                    
                    selected_item = item_name
                else:
                    # Use ChatGPT result and parse it properly
                    selected_chatgpt_item = random.choice(identified_objects)
                    if self.node:
                        self.node.get_logger().info(f"Randomly selected: {selected_chatgpt_item}")
                    
                    # Parse ChatGPT result the same way as voice commands
                    parts = selected_chatgpt_item.rsplit(' ', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        item_name = parts[0]
                        bite_number = int(parts[1])
                        self.single_bite = bite_number <= 1
                        self.current_item = item_name
                    else:
                        item_name = selected_chatgpt_item
                        bite_number = 1
                        self.single_bite = True
                        self.current_item = item_name
                    
                    selected_item = item_name

            # Log the final parsing result for debugging
            if self.node:
                self.node.get_logger().info(f"Final parsing: item='{self.current_item}', single_bite={self.single_bite}")

            self.currently_serving_pub.publish(String(data=self.current_item))
            
            # Step 2: Create DINOX prompt using just the item name (no number)
            text_prompt = selected_item + " ."
            
            # Step 3: Detect with DINOX
            if self.node:
                self.node.get_logger().info(f"Detecting with DINOX using prompt: '{text_prompt}'")
            input_boxes, confidences, class_names, class_ids = self._detect_with_dinox(temp_path, text_prompt)
            
            # Cleanup temp file
            os.remove(temp_path)
            
            if input_boxes is None or len(input_boxes) == 0:
                if self.node:
                    self.node.get_logger().warn("DINOX could not detect the selected food item")
                return None
            
            # Step 4: Get highest confidence detection
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
    
    def get_current_item(self):
        """Get the current detected food item name"""
        return self.current_item
    
    def is_single_bite(self):
        """Check if current item is single bite"""
        return self.single_bite
    
    def queue2string(self, queue):
        # Remove trailing numbers from each command before publishing
        queue_str = ", ".join([cmd.rsplit(' ', 1)[0] if cmd.rsplit(' ', 1)[-1].isdigit() else cmd for cmd in queue])
        queue_msg = String()
        queue_msg.data = queue_str
        return queue_msg