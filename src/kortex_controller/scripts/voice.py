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
from std_msgs.msg import String, Bool
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


# Load environment variables
load_dotenv(os.path.expanduser('~/raf-deploy/.env'))
try:
    client = genai.Client(api_key=os.getenv('google_api_key'))
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    sys.exit(1)

# Load sound config
config_path = os.path.expanduser('~/raf-deploy/config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
confirmation_sound = config['feeding']['confirmation']

# Initialize audio playback
pygame.mixer.init()

# --- Gemini Configuration ---
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
                          
                There is another case you should consider, and that is where the user wants a drink. If you detect speech which requests a drink, you MUST output "drink". Do not output a number with this response.
                For example, if the user says "Can I have some water?", you MUST output "drink". If the user says "Pass me the cup." you MUST output "drink". Asking for any beverage should result in you outputting "drink".
                You should also listen for when the user is done drinking. If the user indicates they are finished drinking (phrases like "done", "finished", "I'm done", "I'm finished drinking", "that's enough", "stop", etc.), you MUST output "drinking_complete". 
                          
                Also be responsive to when the user wants to clear the queue. If the users says "clear the queue", you must output "clear". Phrases like "clear queue, "please reset commands", or anything similar should result in an output of "clear".

                             
                The last case is if the user asks for multiple items in the same command. When this happens you must output each item and number, seperated by commas. If the user says "Could I have a red gummy, then an orange gummy?", you must output "red gummy 1, orange gummy 1".
                If the user says "Give me a drink then a pretzel rod" you must output "drink, pretzel rod 2". If the user specifies the number of items they want, include that many items in the output. If the user says "I want two red gummies and a grape", you must output "red gummy 1, red gummy 1, grape 1".
                If the user says "I want three french fries, a drink, and a chocolate bar", you must output "french fry 2, french fry 2, french fry 2, drink, chocolate bar 2".
        """)]
    ),
    "response_modalities": ["TEXT"],
}

pya = pyaudio.PyAudio()

class AudioProcessor:
    def __init__(self):
        self.session = None
        self.audio_stream = None
        self.out_queue = asyncio.Queue(maxsize=10)
        self.ros_queue = queue.Queue()

    async def stream_audio(self):
        """Continuously captures audio and puts it on the async queue for Gemini."""
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
                complete_text = "" # Initialize an empty string to accumulate text

                # The async for loop iterates through all parts of the turn's response
                async for response in turn:
                    if text := getattr(response, 'text', ''):
                        complete_text += text
                
                # After the loop, process the final, accumulated text
                final_response = complete_text.strip()

                if final_response and final_response != "None":
                    print(f"✅ Identified: '{final_response}'")
                    self.ros_queue.put(final_response)
                else:
                    # This case handles "None" or empty responses from the model.
                    pass 
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error receiving from Gemini: {e}")
                traceback.print_exc()
                break

    async def run(self):
        """Main execution loop for the audio processing."""
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

class VoicePublisher(Node):
    def __init__(self, voice_queue):
        super().__init__('voice_command_node')
        self.publisher = self.create_publisher(String, 'voice_commands', 10)
        self.drinking_complete_pub = self.create_publisher(Bool, '/drinking_complete', 10)
        self.robot_state_sub = self.create_subscription(String, '/robot_state', self.robot_state_callback, 10)
        
        self.queue = voice_queue
        self.robot_state = 'Unknown'
        self.timer = self.create_timer(0.1, self.check_queue)

    def robot_state_callback(self, msg):
        if self.robot_state != msg.data:
            self.robot_state = msg.data
            self.get_logger().info(f'Robot state updated: {self.robot_state}')

    def play_confirmation(self):
        pygame.mixer.music.load(confirmation_sound)
        pygame.mixer.music.play()

    def check_queue(self):
        try:
            result = self.queue.get_nowait()
            items = [item.strip() for item in result.split(',') if item.strip()]

            if "drinking_complete" in items and self.robot_state == 'sipping':
                self.drinking_complete_pub.publish(Bool(data=True))
                self.get_logger().info('Published: drinking_complete=True')
                self.play_confirmation()

            food_items = [item for item in items if item not in ["drink", "drinking_complete"]]
            
            if food_items:
                msg_data = ', '.join(food_items)
                msg = String(data=msg_data)
                self.publisher.publish(msg)
                self.get_logger().info(f"Published to /voice_commands: '{msg_data}'")
                for _ in food_items:
                    self.play_confirmation()
                    time.sleep(0.2)
            
            if "drink" in items:
                drink_msg = String(data="drink")
                self.publisher.publish(drink_msg)
                self.get_logger().info("Published to /voice_commands: 'drink'")
                self.play_confirmation()

        except queue.Empty:
            pass
        except Exception as e:
            self.get_logger().error(f"Error in queue processing: {e}")

def main():
    rclpy.init()
    
    audio_processor = AudioProcessor()
    ros_node = VoicePublisher(audio_processor.ros_queue)
    
    audio_thread = threading.Thread(target=lambda: asyncio.run(audio_processor.run()), daemon=True)
    audio_thread.start()
    
    print("🚀 ROS2 node spinning. Voice control is ready.")
    try:
        rclpy.spin(ros_node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()