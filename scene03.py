import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch
import re
import sys
import numpy as np
import sounddevice as sd
import socket
import asyncio
import pyaudio
from time import sleep
from pythonosc import udp_client
from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from sys import platform
from scipy.io import wavfile
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server

p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
print(info)
microExt = False

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        name = p.get_device_info_by_host_api_device_index(0, 2).get('name')
        print("Input Device id ", i, " - ", name)
        if name == 'External Microphone (Realtek(R)':
            microExt = True
        else:
            microExt = False

if microExt == True:
    pass
else:
    print("Error: no esta conectado el microfono externo, desconectar USB")
    #sys.exit()l

os.chdir(r'C:\Users\Usuario\Documents\EscenasPythonScripts\Scene\Scene03')

#Configuration
IP = '148.220.164.180'
IP_lap = socket.gethostbyname(socket.gethostname()) #IP as a server
PORT0 = 8001    #ReceivePort
PORT1 = 6002        #word
PORT2 = 7002        #size
PORT3 = 8002        #lifetime

dispatcher = Dispatcher()

minsize = 0.0 
maxsize = 2.0

minlife = 1.0
maxlife = 5.0

minvol = 0.0
maxvol = 50000

#Parser argument for whisper
parserWhisper = argparse.ArgumentParser()
parserWhisper.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
parserWhisper.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
parserWhisper.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
parserWhisper.add_argument("--record_timeout", default=0.8,
                        help="How real time the recording is in seconds.", type=float)
parserWhisper.add_argument("--phrase_timeout", default=0.1,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)  
argsWhisper = parserWhisper.parse_args()

#Parser argument for OSC with the word
parserOSCword = argparse.ArgumentParser()
parserOSCword.add_argument("--ip", default=IP, help="The ip of the OSC server")
parserOSCword.add_argument("--port", type=str, default=PORT1, help="The port the OSC server is listening ON")
argsOSCword = parserOSCword.parse_args()

#Parser argument for OSC with the size of the word
parserOSCsize = argparse.ArgumentParser()
parserOSCsize.add_argument("--ip", default=IP, help="The ip of the OSC server")
parserOSCsize.add_argument("--port", type=str, default=PORT2, help="The port the OSC server is listening ON")
argsOSCsize = parserOSCsize.parse_args()

#Parser argument for OSC of the lifetime of the word
parserOSClifetime = argparse.ArgumentParser()
parserOSClifetime.add_argument("--ip", default=IP, help="The ip of the OSC server")
parserOSClifetime.add_argument("--port", type=str, default=PORT3, help="The port the OSC server is listening ON")
argsOSClifetime = parserOSClifetime.parse_args()

data_queue = Queue()
last_sample = bytes()
phrase_time = None
recorder = sr.Recognizer()
recorder.energy_threshold = argsWhisper.energy_threshold
recorder.dynamic_energy_threshold =False
source = sr.Microphone(sample_rate=16000)

#Load model
model = argsWhisper.model
if argsWhisper.model != "large" and not argsWhisper.non_english:
    model = model #+ ".en"
audio_model = whisper.load_model(model)
record_timeout = argsWhisper.record_timeout
phrase_timeout = argsWhisper.phrase_timeout
temp_file = NamedTemporaryFile().name
transcription = ['']
selectedlist = []

with source:
    recorder.adjust_for_ambient_noise(source)
    
def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
print("Model loaded.\n")

def osc_word(word):
    client = udp_client.SimpleUDPClient(argsOSCword.ip,argsOSCword.port)
    client.send_message("/word",word)

def osc_size(size):
    client = udp_client.SimpleUDPClient(argsOSCsize.ip, argsOSCsize.port)
    client.send_message("/size", size)

def osc_lifetime(lifetime):
    client = udp_client.SimpleUDPClient(argsOSClifetime.ip, argsOSClifetime.port)
    client.send_message("/lifetime", lifetime)
    
def sound(indata,outdata,frames,time,status):
    volume_norm = np.linalg.norm(indata)*10
    print(volume_norm)

def data_save(address: str,*args):
    global save
    save = args[0]
    return save

def data_subject(address: str,*args):
    global subject
    subject = args[0]
    return subject

def data_scene(address: str,*args):
    global scene
    scene = args[0]
    return scene

def data_date(address: str,*args):
    global date
    date = args[0]
    return date

def data_hour(address: str,*args):
    global hour
    hour = args[0]
    return hour

def data_emotion(address: str,*args):
    global emotion
    emotion = args[0]
    return emotion

save = dispatcher.map("/Save", data_save)
emotion = dispatcher.map("/Selected Emotion", data_emotion)
subject = dispatcher.map("/Subject", data_subject)
scene = dispatcher.map("/Scene", data_scene)
date = dispatcher.map("/Date", data_date)
hour = dispatcher.map("/Hour", data_hour)

async def loop():
    counter=0
    while save is not True:
        print(save,emotion,subject,scene,date,hour,counter)
        counter += 1
        await asyncio.sleep(0.5)

async def init_main():
    server = AsyncIOOSCUDPServer((IP_lap, PORT0), dispatcher, asyncio.get_event_loop())
    transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving
    await loop()  # Enter main loop of program
    transport.close()  # Clean up serve endpoint

asyncio.run(init_main())

while True:  
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data


                
                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())
                rate, data = wavfile.read(wav_data)
                vol = max(data)
                print(vol)

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available(), language="es",temperature=0.2, word_timestamps=True)
                text = result['text'].strip()
                
                text = re.sub(r'[^\w\s]','',text)
                text=text.lower()
                print(text)
            
                osc_word(text)
                osc_size(maxsize/maxvol*vol)
                osc_lifetime(maxlife/maxvol*vol)
            
                  
                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # Clear the console to reprint the updated transcription.
                #os.system('cls' if os.name=='nt' else 'clear')
                #for line in transcription:
                #    print(line)
                # Flush stdout.
                #print('', end='', flush=True)

                # Infinite loops are bad for processors, must sleep.
                sleep(0.01)
                
        except KeyboardInterrupt:
            break

print("\n\nTranscription:")
for line in transcription:
        print(line)
        





