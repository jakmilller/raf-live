#!/usr/bin/env python3

import asyncio
import os
import sys
import traceback
from dotenv import load_dotenv
import pyaudio
from google.genai import types
from google import genai
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pygame
import time
import yaml
import threading
import queue

# Handle older Python versions if necessary
if sys.version_info < (3, 11, 0):
    import taskgroup
    import exceptiongroup
    from exceptiongroup import ExceptionGroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

# --- Configuration ---
# Audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK_SIZE = 1600  # 100ms chunks for lower latency

# Gemini
MODEL = "models/gemini-2.0-flash-exp"

class AudioProcessor:
    def __init__(self, ros_queue):
        self.session = None
        self.audio_stream = None
        self.out_queue = asyncio.Queue(maxsize=10)
        self.ros_queue = ros_queue

    async def stream_audio(self):
        """Continuously captures audio and puts it on the async queue for Gemini."""
        pya = pyaudio.PyAudio()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        print("✅ Voice control active. Listening for commands...")
        
        while True:
            try:
                data = await asyncio.to_thread(
                    self.audio_stream.read, CHUNK_SIZE, exception_on_overflow=False
                )
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error reading audio stream: {e}")
                break

    async def send_to_gemini(self):
        """Sends audio chunks from the queue to the Gemini session."""
        while True:
            try:
                msg = await self.out_queue.get()
                if self.session:
                    await self.session.send_realtime_input(audio=msg)
                self.out_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error sending to Gemini: {e}")
                break

    async def receive_from_gemini(self):
        """Receives and processes text responses from Gemini."""
        while True:
            try:
                turn = self.session.receive()
                complete_text = ""

                async for response in turn:
                    if text := getattr(response, 'text', ''):
                        complete_text += text
                
                final_response = complete_text.strip()

                if final_response and final_response != "None":
                    print(f"✅ Identified: '{final_response}'")
                    self.ros_queue.put(final_response)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error receiving from Gemini: {e}")
                traceback.print_exc()
                break

    async def run(self, client):
        """Main execution loop for the audio processing."""
        # Gemini Configuration
        CONFIG = {
            "system_instruction": types.Content(
                parts=[types.Part(text="""
                You are a food item identifier for a robot. Your ONLY task is to listen to the user and identify which food item they are asking for.

                        IMPORTANT: Only respond when you hear a COMPLETE food request. Do not respond to partial words or incomplete sentences.

                        Your response MUST be ONLY the name of the food item they want in lowercase, combined with a number. This number will be either 1 (if the food is typically eaten in a single bite) or 2 (if the food is typically eaten in multiple bites.
                        You should only return a food item if the context of the conversation shows that the user would like to be fed that item in the current moment. Passively talking about food should not trigger a response.
                        For example, if the user says "I would like some fruit gummies, please", you MUST output "fruit gummy 1".
                        If the user says "Can I have the pretzel rods?", you MUST output "pretzel rod 2". It is ok to include an adjective to describe the specific food that the user wants.
                        For example, if the user says "I would like green grapes, the output should be "green grape 1". Notice that you should always avoid using plurals in your output. Even if the users says "I want peppers", you should output "pepper".
                        If the user does not provide a complete food item request or if you receive incomplete audio, you MUST output "None".
                        Do not add any other words, explanations, or punctuation.
                                  
                        Also be responsive to when the user wants to clear the queue. If the users says "clear the queue", you must output "clear". Phrases like "clear queue, "please reset commands", or anything similar should result in an output of "clear".

                                     
                        The last case is if the user asks for multiple items in the same command. When this happens you must output each item and number, seperated by commas. If the user says "Could I have a red gummy, then an orange gummy?", you must output "red gummy 1, orange gummy 1".
                        If the user specifies the number of items they want, include that many items in the output. If the user says "I want two red gummies and a grape", you must output "red gummy 1, red gummy 1, grape 1".
                        If the user says "I want three french fries and a chocolate bar", you must output "french fry 2, french fry 2, french fry 2, chocolate bar 2".
                """)]
            ),
            "response_modalities": ["TEXT"],
        }

        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                tg.create_task(self.stream_audio())
                tg.create_task(self.send_to_gemini())
                tg.create_task(self.receive_from_gemini())
        except asyncio.CancelledError:
            print("Audio processing cancelled.")
        except Exception:
            print("An error occurred in the main audio processing loop:")
            traceback.print_exc()
        finally:
            if self.audio_stream and self.audio_stream.is_active():
                self.audio_stream.stop_stream()
                self.audio_stream.close()


class VoiceNode(Node):
    def __init__(self):
        super().__init__('voice_node')
        
        # Load environment variables
        load_dotenv(os.path.expanduser('~/raf-live/.env'))
        
        # Initialize Gemini client
        try:
            self.client = genai.Client(api_key=os.getenv('google_api_key'))
        except KeyError:
            self.get_logger().error("Error: google_api_key environment variable not set.")
            sys.exit(1)

        # Load sound config
        config_path = os.path.expanduser('~/raf-live/config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Check if voice is enabled
        self.voice_enabled = config['feeding']['voice']
        if not self.voice_enabled:
            self.get_logger().info("Voice control is disabled in config")
            return
            
        self.confirmation_sound = config['feeding']['confirmation']
        
        # Initialize audio playback
        pygame.mixer.init()

        # Voice command queue and processing
        self.voice_queue = queue.Queue()
        self.audio_processor = AudioProcessor(self.voice_queue)
        
        # ROS2 Publishers
        self.voice_command_pub = self.create_publisher(String, '/voice_commands', 10)
        
        # Timer to check queue
        self.timer = self.create_timer(0.1, self.check_queue)
        
        if self.voice_enabled:
            # Start audio processing in separate thread
            self.audio_thread = threading.Thread(
                target=lambda: asyncio.run(self.audio_processor.run(self.client)), 
                daemon=True
            )
            self.audio_thread.start()
            
            self.get_logger().info("🚀 Voice Node initialized and listening for commands")
        else:
            self.get_logger().info("Voice Node initialized but voice control is disabled")

    def play_confirmation(self):
        """Play confirmation sound"""
        try:
            pygame.mixer.music.load(self.confirmation_sound)
            pygame.mixer.music.play()
        except Exception as e:
            self.get_logger().warn(f"Could not play confirmation sound: {e}")

    def check_queue(self):
        """Check for new voice commands and publish them"""
        if not self.voice_enabled:
            return
            
        try:
            result = self.voice_queue.get_nowait()
            items = [item.strip() for item in result.split(',') if item.strip()]

            # Handle clear command
            if "clear" in items:
                # For clear command, we publish it directly so the detector can handle it
                msg = String(data="clear")
                self.voice_command_pub.publish(msg)
                self.get_logger().info("Published clear command")
                self.play_confirmation()
                return

            # Filter out non-food items and publish food items
            food_items = [item for item in items if item not in ["clear"]]
            
            if food_items:
                msg_data = ', '.join(food_items)
                msg = String(data=msg_data)
                self.voice_command_pub.publish(msg)
                self.get_logger().info(f"Published voice commands: '{msg_data}'")
                
                # Play confirmation for each food item
                for _ in food_items:
                    self.play_confirmation()
                    time.sleep(0.2)

        except queue.Empty:
            pass
        except Exception as e:
            self.get_logger().error(f"Error in queue processing: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VoiceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()