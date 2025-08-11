# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
## Setup

To install the dependencies for this script, run:

```
pip install google-generativeai pyaudio
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

## Run

To run the script:

```
python multimodal.py
```
The script will listen to your microphone. When you say a food item from the list,
it will print the identified item to the console.
"""

import asyncio
import os
import sys
import traceback
from dotenv import load_dotenv
import pyaudio
import struct
import math

from google.genai import types
from google import genai

# Handle older Python versions for asyncio TaskGroups
if sys.version_info < (3, 11, 0):
    import taskgroup
    import exceptiongroup
    from exceptiongroup import ExceptionGroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

# --- Audio Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

# this makes sure that gemini doesn't cut off speech (Voice Activity Detection)
SPEECH_THRESHOLD = 2000 # basically how sensitive the mic is to detecting speech
SILENCE_CHUNKS = 12 # 16000 / chunk size ~= 8 chunks
MODEL = "models/gemini-2.0-flash-exp" 


# --- Gemini API Client ---
load_dotenv(os.path.expanduser('~/raf-deploy/.env'))

try:
    client = genai.Client(api_key=os.getenv('google_api_key'), http_options={"api_version": "v1alpha"})
except KeyError:
    print("Error: The GOOGLE_API_KEY environment variable is not set.")
    print("Please set it to your Google AI Studio API key.")
    sys.exit(1)


# --- System Instructions and Configuration for Gemini ---
CONFIG = {
    "system_instruction": types.Content(
        parts=[
            types.Part(
                text="""You are a food item identifier for a robot. Your ONLY task is to listen to the user and identify which food item they are asking for.

                IMPORTANT: Only respond when you hear a COMPLETE food request. Do not respond to partial words or incomplete sentences.

                Your response MUST be ONLY the name of the food item they want, in lowercase. You should only return a food item if the context of the conversation
                shows that the user would like to be fed that item in the current moment. Passively talking about food should not trigger a response.
                For example, if the user says "I would like some fruit gummies, please", you MUST output "fruit gummy".
                If the user says "Can I have the pretzel rods?", you MUST output "pretzel rod". It is ok to include an adjective to describe the specific food that the user wants.
                For example, if the user says "I would like green grapes, the output should be "green grape". Notice that you should always avoid using plurals in your output. Even if the users says "I want peppers", you should output "pepper".
                If the user does not provide a complete food item request or if you receive incomplete audio, you MUST output "None".
                Do not add any other words, explanations, or punctuation. """
            )
        ]
    ),
    "response_modalities": ["TEXT"],
}

pya = pyaudio.PyAudio()

class AudioFoodIdentifier:
    def __init__(self):
        self.out_queue = None
        self.session = None
        self.audio_stream = None

        # for VAD
        self.is_speaking = False
        self.audio_buffer = []
        self.silence_chunks_counted = 0

    def _calculate_rms(self, data):
        """Calculates the Root Mean Square (energy) of an audio chunk."""
        # Unpack the audio data into a sequence of 16-bit integers
        count = len(data) // 2
        shorts = struct.unpack(f'{count}h', data)
        sum_squares = sum(s**2 for s in shorts)
        return math.sqrt(sum_squares / count)

    async def listen_and_detect_speech(self):
        """
        Listens to the microphone, implements VAD, and sends complete utterances
        to Gemini for processing.
        """
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        
        print("Listening... Speak a food item from the list and then pause.")

        while True:
            audio_chunk = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, exception_on_overflow=False)
            
            rms = self._calculate_rms(audio_chunk)

            if rms > SPEECH_THRESHOLD:
                # If speech is detected
                if not self.is_speaking:
                    # This is the start of a new utterance
                    print("Speech detected...")
                    self.is_speaking = True
                
                # Add the chunk to the buffer and reset silence counter
                self.audio_buffer.append(audio_chunk)
                self.silence_chunks_counted = 0
            
            elif self.is_speaking:
                # If we were speaking, but this chunk is silent
                self.audio_buffer.append(audio_chunk) # Buffer the silence too
                self.silence_chunks_counted += 1

                if self.silence_chunks_counted >= SILENCE_CHUNKS:
                    # If silence duration is met, the utterance is over
                    print("Processing utterance...")
                    full_utterance = b''.join(self.audio_buffer)
                    
                    # Send the complete utterance for processing
                    await self.session.send(input={"data": full_utterance, "mime_type": "audio/pcm"}, end_of_turn=True)
                    
                    # Reset for the next turn
                    self.is_speaking = False
                    self.audio_buffer = []
                    self.silence_chunks_counted = 0


    async def get_text_response(self):
        """Receives and prints the final text response from the Gemini session."""
        while True:
            turn = self.session.receive()
            async for response in turn:
                if text := response.text:
                    print(f"Identified Food Item: {text.strip()}", end='\n\n')
                    print("Listening... Speak a food item from the list and then pause.")

    async def run(self):
        """Main execution loop that sets up the Gemini session and runs the async tasks."""
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                # Create and run the concurrent tasks.
                # The listen_and_detect_speech task now also handles sending data.
                tg.create_task(self.listen_and_detect_speech())
                tg.create_task(self.get_text_response())

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.audio_stream:
                self.audio_stream.close()
            print("An error occurred:")
            traceback.print_exception(e)


if __name__ == "__main__":
    main = AudioFoodIdentifier()
    try:
        asyncio.run(main.run())
    except KeyboardInterrupt:
        print("\nExiting program.")
    finally:
        pya.terminate()