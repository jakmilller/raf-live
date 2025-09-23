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
# --- NEW: Import multiprocessing instead of threading ---
import multiprocessing
import queue # Still need this for queue.Empty exception

# Handle older Python versions if necessary
if sys.version_info < (3, 11, 0):
    import taskgroup
    import exceptiongroup
    from exceptiongroup import ExceptionGroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

# --- Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK_SIZE = 1600
MODEL = "models/gemini-2.0-flash-exp"

# --- This class remains the same, but will be used inside the separate process ---
class AudioProcessor:
    def __init__(self, process_queue):
        self.session = None
        self.audio_stream = None
        self.pya = None
        self.out_queue = asyncio.Queue(maxsize=10)
        self.process_queue = process_queue
        self._audio_initialized = False

    async def init_audio(self):
        if not self._audio_initialized:
            try:
                self.pya = pyaudio.PyAudio()
                self._audio_initialized = True
                print("✅ PyAudio initialized")
            except Exception as e:
                print(f"Error initializing PyAudio: {e}")
                raise

    async def stream_audio(self):
        try:
            await self.init_audio()
            self.audio_stream = await asyncio.to_thread(
                self.pya.open,
                format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
                input=True, frames_per_buffer=CHUNK_SIZE,
            )
            print("✅ Voice control active. Listening for commands...")
            while True:
                try:
                    if not self.audio_stream or not self.audio_stream.is_active():
                        print("❌ Audio stream is not active")
                        break
                    data = await asyncio.to_thread(
                        self.audio_stream.read, CHUNK_SIZE, exception_on_overflow=False
                    )
                    await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
                except asyncio.CancelledError: break
                except Exception as e:
                    print(f"Error reading audio stream: {e}")
                    break
        except Exception as e:
            print(f"Error in audio streaming: {e}")
            raise
        finally:
            self.close_stream()

    def close_stream(self):
        if self.audio_stream:
            try:
                if self.audio_stream.is_active(): self.audio_stream.stop_stream()
                self.audio_stream.close()
            except Exception as e: print(f"Error closing audio stream: {e}")
            finally: self.audio_stream = None

    def terminate_audio(self):
        self.close_stream()
        if self.pya and self._audio_initialized:
            try:
                self.pya.terminate()
                print("✅ PyAudio terminated")
            except Exception as e: print(f"Error terminating PyAudio: {e}")
            finally:
                self.pya = None
                self._audio_initialized = False

    async def send_to_gemini(self):
        while True:
            try:
                msg = await self.out_queue.get()
                if self.session: await self.session.send_realtime_input(audio=msg)
                self.out_queue.task_done()
            except asyncio.CancelledError: break
            except Exception as e:
                print(f"Error sending to Gemini: {e}")
                raise

    async def receive_from_gemini(self):
        while True:
            try:
                turn = self.session.receive()
                complete_text = ""
                async for response in turn:
                    if text := getattr(response, 'text', ''): complete_text += text
                final_response = complete_text.strip()
                if final_response and final_response != "None":
                    print(f"✅ Identified: '{final_response}'")
                    self.process_queue.put(final_response)
            except asyncio.CancelledError: break
            except Exception as e:
                print(f"Error receiving from Gemini: {e}")
                traceback.print_exc()
                raise

    async def run(self, client):
        CONFIG = {
            "system_instruction": types.Content(parts=[types.Part(text="""
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
            """)]), "response_modalities": ["TEXT"],
        }
        try:
            print("🔄 Connecting to Gemini...")
            client = genai.Client(api_key=os.getenv('google_api_key'))
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                print("✅ Connected to Gemini successfully")
                tg.create_task(self.stream_audio())
                tg.create_task(self.send_to_gemini())
                tg.create_task(self.receive_from_gemini())
        except asyncio.CancelledError: print("Audio processing cancelled.")
        except Exception as e:
            print(f"❌ Session failed: {e}. The process will now terminate.")
        finally:
            print("🧹 Cleaning up audio processor resources...")
            self.terminate_audio()

# --- NEW: This function is the entry point for the separate process ---
def run_audio_process(process_queue):
    """
    This function runs in a separate process to isolate it from the main ROS node.
    """
    try:
        load_dotenv(os.path.expanduser('~/raf-live/.env'))
        client = genai.Client(api_key=os.getenv('google_api_key'))
        processor = AudioProcessor(process_queue)
        asyncio.run(processor.run(client))
    except Exception as e:
        print(f"FATAL ERROR in audio process: {e}")
        traceback.print_exc()

class VoiceNode(Node):
    def __init__(self):
        super().__init__('voice_node')
        load_dotenv(os.path.expanduser('~/raf-live/.env'))
        
        # NOTE: The genai.Client is no longer needed here, it's created in the child process
        
        config_path = os.path.expanduser('~/raf-live/config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self.voice_enabled = config['feeding']['voice']
        if not self.voice_enabled:
            self.get_logger().info("Voice control is disabled in config")
            return
            
        self.confirmation_sound = config['feeding']['confirmation']
        pygame.mixer.init()

        # --- MODIFIED: Use a multiprocessing queue for inter-process communication ---
        self.voice_queue = multiprocessing.Queue()
        self.audio_process = None
        
        self.voice_command_pub = self.create_publisher(String, '/voice_commands', 10)
        
        self.queue_timer = self.create_timer(0.1, self.check_queue)
        self.monitor_timer = self.create_timer(5.0, self.monitor_audio_process)
        
        if self.voice_enabled:
            self.start_audio_process()
            self.get_logger().info("🚀 Voice Node initialized and listening for commands")
        else:
            self.get_logger().info("Voice Node initialized but voice control is disabled")

    def start_audio_process(self):
        if self.audio_process and self.audio_process.is_alive():
            self.get_logger().info("Audio process is already running.")
            return

        self.get_logger().info("Starting new audio processing process...")
        self.audio_process = multiprocessing.Process(
            target=run_audio_process,
            args=(self.voice_queue,),
            daemon=True
        )
        self.audio_process.start()

    def monitor_audio_process(self):
        if not self.voice_enabled: return
        
        if not self.audio_process or not self.audio_process.is_alive():
            self.get_logger().warn("Audio processing process has died. Attempting to restart...")
            self.start_audio_process()

    def shutdown_hook(self):
        self.get_logger().info("Shutting down voice node...")
        if self.audio_process and self.audio_process.is_alive():
            self.get_logger().info("Terminating audio process...")
            self.audio_process.terminate()
            self.audio_process.join()

    def play_confirmation(self):
        try:
            pygame.mixer.music.load(self.confirmation_sound)
            pygame.mixer.music.play()
        except Exception as e: self.get_logger().warn(f"Could not play confirmation sound: {e}")

    def check_queue(self):
        if not self.voice_enabled: return
        try:
            result = self.voice_queue.get_nowait()
            items = [item.strip() for item in result.split(',') if item.strip()]
            if "clear" in items:
                msg = String(data="clear")
                self.voice_command_pub.publish(msg)
                self.get_logger().info("Published clear command")
                self.play_confirmation()
                return
            food_items = [item for item in items if item not in ["clear"]]
            if food_items:
                msg_data = ', '.join(food_items)
                msg = String(data=msg_data)
                self.voice_command_pub.publish(msg)
                self.get_logger().info(f"Published voice commands: '{msg_data}'")
                for _ in food_items:
                    self.play_confirmation()
                    time.sleep(0.2)
        except queue.Empty: pass
        except Exception as e: self.get_logger().error(f"Error in queue processing: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VoiceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if node:
            node.shutdown_hook()
        rclpy.shutdown()

if __name__ == "__main__":
    main()