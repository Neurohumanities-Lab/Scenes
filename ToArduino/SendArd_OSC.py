from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
import asyncio
import serial
import time

portard = "com3"

def fear_level(address: str,*args):
    global fearlevel
    fearlevel = args[0]
    return fearlevel

dispatcher = Dispatcher()
fearlevel = dispatcher.map("/real-emotion", fear_level)

ip = "192.168.0.158"
port = 1337

# Configuration Arduino
try:
  arduino = serial.Serial(portard, 115200)
  print("Port is open")

except serial.SerialException:
  serial.Serial(portard, 115200).close()
  print("Port is closed")
  arduino = serial.Serial(portard,115200)
  print("Port is open again")

print("Ready to use")

def write_read(x):
    arduino.write(bytes(str(x), 'utf-8'))
    time.sleep(0.05)


async def loop():
    
    while (True):
      
      delayed = -1
      if fearlevel == "No Fear":
         delayed = 0
      if fearlevel == "Low Fear":
         delayed = 1
      if fearlevel == "Medium Fear":
         delayed = 2
      if fearlevel == "High Fear":
         delayed = 3
      write_read(delayed)
      await asyncio.sleep(2.5)
      print(delayed)


async def init_main():
    server = AsyncIOOSCUDPServer((ip, port), dispatcher, asyncio.get_event_loop())
    transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving

    await loop()  # Enter main loop of program

    transport.close()  # Clean up serve endpoint


asyncio.run(init_main())