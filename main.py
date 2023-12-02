
import asyncio
import sensorRead as srd

def main():
    loop = asyncio.new_event_loop()

    # Schedule the data analysis task
    analysis_task = loop.create_task(srd.data_analysis())

    # Start the WebSocket server
    start_server_task = loop.create_task(srd.start_server())
    
    # Run both tasks concurrently
    loop.run_until_complete(asyncio.gather(analysis_task, start_server_task))



if __name__ == "__main__":
    main()