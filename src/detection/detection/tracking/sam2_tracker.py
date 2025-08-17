#!/usr/bin/env python3

import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_camera_predictor


class SAM2Tracker:
    def __init__(self, node=None):
        """
        Initialize SAM2 tracker
        
        Args:
            node: ROS node instance for logging (optional)
        """
        self.node = node
        self.predictor = None
        self.tracking_initialized = False
        self.tracking_active = False
        
        self._setup_sam2()
    
    def _setup_sam2(self):
        """Initialize SAM2 for real-time tracking"""
        try:
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            sam2_checkpoint = "/home/mcrr-lab/raf-live/SAM2_streaming/checkpoints/sam2.1/sam2.1_hiera_tiny.pt"
            model_cfg = "sam2.1/sam2.1_hiera_t.yaml"
            self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
            
            if self.node:
                self.node.get_logger().info('SAM2 initialized successfully')
        except Exception as e:
            if self.node:
                self.node.get_logger().error(f'Failed to initialize SAM2: {e}')
            raise e
    
    def initialize_tracking(self, frame, detection_input, detection_type='gemini'):
        """
        Initialize SAM2 tracking with either point or bounding box
        
        Args:
            frame: Input image frame
            detection_input: Either (x, y) point for gemini or [x1, y1, x2, y2] bbox for dinox
            detection_type: 'gemini' or 'dinox'
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.predictor.load_first_frame(frame)
            ann_frame_idx = 0
            ann_obj_id = (1,)
            
            if detection_type == 'dinox':
                # Use bounding box for DINOX stack
                bbox = np.array([[detection_input[0], detection_input[1]], 
                               [detection_input[2], detection_input[3]]], dtype=np.float32)
                
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                    frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
                )
                if self.node:
                    self.node.get_logger().info(f"SAM2 tracking initialized with bounding box: {detection_input}")
            else:
                # Use point for Gemini
                labels = np.array([1], dtype=np.int32)
                points = np.array([detection_input], dtype=np.float32)

                _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                    frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
                )
                if self.node:
                    self.node.get_logger().info(f"SAM2 tracking initialized with point: {detection_input}")

            self.tracking_initialized = True
            self.tracking_active = True
            return True

        except Exception as e:
            if self.node:
                self.node.get_logger().error(f"Failed to initialize SAM2 tracking: {e}")
            return False
    
    def update_tracking(self, frame):
        """
        Update SAM2 tracking and return mask and centroid
        
        Args:
            frame: Input image frame
            
        Returns:
            tuple: (mask_2d, centroid) or (None, None) if tracking failed
        """
        if not self.tracking_initialized or not self.tracking_active:
            return None, None
            
        try:
            # Track object
            out_obj_ids, out_mask_logits = self.predictor.track(frame)
            
            if len(out_obj_ids) == 0:
                if self.node:
                    self.node.get_logger().warn("Tracking lost")
                return None, None
            
            # Get mask and convert to proper format
            out_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy()
            mask_2d = (out_mask.squeeze() * 255).astype(np.uint8)
            
            # Calculate centroid
            centroid = self._get_mask_centroid(mask_2d)
            if centroid is None:
                if self.node:
                    self.node.get_logger().warn("No centroid found in mask")
                return None, None
            
            return mask_2d, centroid
            
        except Exception as e:
            if self.node:
                self.node.get_logger().error(f"Error in SAM2 tracking: {e}")
            return None, None
    
    def _get_mask_centroid(self, mask):
        """Find the centroid of a binary mask"""
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] == 0:
            return None 
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)
    
    def reset_tracking(self):
        """Reset tracking state"""
        self.tracking_initialized = False
        self.tracking_active = False
    
    def is_tracking_active(self):
        """Check if tracking is currently active"""
        return self.tracking_active and self.tracking_initialized