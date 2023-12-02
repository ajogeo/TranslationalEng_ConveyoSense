
import asyncio
import websockets
import json
import sqlite3

from datetime import datetime

import readDB as rdb

"""
import BlynkLib  # BlynkLib v1.1
# Initialize Blynk with your Blynk authentication token
BLYNK_AUTH = 'SWsXT9VssZqxHBWF54ncNyeMvXV1sq9L'
blynk = BlynkLib.Blynk(BLYNK_AUTH)
"""

# Initialize variables to store the data
accelX = 0.0
accelY = 0.0
accelZ = 0.0
microphoneData = 0
timestamp = 0

# Create a connection to the local SQLite database
conn = sqlite3.connect('local_data.db')
cursor = conn.cursor()

# Create a table to store data
cursor.execute('''
    CREATE TABLE IF NOT EXISTS sensor_data (
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        accelX REAL,
        accelY REAL,
        accelZ REAL,
        microphoneData INTEGER
    )
''')
conn.commit()

async def data_analysis():
    while True:
        await asyncio.sleep(10)  # Sleep for 10 minutes
        # Add your data analysis logic here
        print("Performing data analysis...")
        # Example: Fetch data from the database and print the result
        analysis_result = rdb.analyse_database()
        #analysis_result = 1

        if analysis_result == 0:
            print("XXXXX------DEFECT------XXXXX")
        elif analysis_result == 1:
            print("ALL GOOD")
        else:
            print("Error: analysis done wrong")

async def receive_data(websocket, path):
    global accelX, accelY, accelZ, microphoneData, timestamp
    while True:
        try:
            data = await websocket.recv()
            start_datetime = datetime.now()
            ts = start_datetime.strftime("%Y-%m-%d_%H-%M-%S-%f")

            data = json.loads(data)
            data["timestamp"] = ts
            print("Received Data:", data)

            # Store the values in variables
            timestamp = data.get('timestamp', timestamp)
            accelX = data.get('accelX', accelX)
            accelY = data.get('accelY', accelY)
            accelZ = data.get('accelZ', accelZ)
            microphoneData = data.get('microphoneData', microphoneData)

            

            # Insert the data into the local database
            cursor.execute("INSERT INTO sensor_data (timestamp, accelX, accelY, accelZ, microphoneData) VALUES (?, ?, ?, ?, ?)",
                           (timestamp, accelX, accelY, accelZ, microphoneData))
            conn.commit()
            """
            #send sensor data to virtual pin V1, V2, V3, V4 of Blynk app
            blynk.virtual_write(1, accelX)  # Virtual Pin V1 for X-axis
            blynk.virtual_write(2, accelY)  # Virtual Pin V2 for Y-axis
            blynk.virtual_write(3, accelZ)  # Virtual Pin V3 for Z-axis
            blynk.virtual_write(4, microphoneData)  # Virtual Pin V4 for Microphone data
            threshold_x = 10
            threshold_y = 10
            threshold_z = 10
            # Check for faults
            if accelX > threshold_x or accelY > threshold_y or accelZ > threshold_z:
                print("Fault detected: Acceleration exceeded threshold.")
            # blynk.notify('Fault detected: Acceleration exceeded threshold') # send notification to mobile app

            #send notification to virtual pin associated with your terminal widget
            blynk.virtual_write(5,'Fault detected: Acceleration exceeded threshold')

            #Run Blynk communication
            blynk.run()
            """

        except websockets.ConnectionClosed:
            break

async def start_server():
    server_host = "0.0.0.0"  # Listen on all available network interfaces
    server_port = 8765  # Replace with the port your server should listen on
    start_server = websockets.serve(receive_data, server_host, server_port)
    await start_server

#if __name__ == "__main__":
#    loop = asyncio.new_event_loop()
#    loop.create_task(start_server())
#    loop.run_forever()
