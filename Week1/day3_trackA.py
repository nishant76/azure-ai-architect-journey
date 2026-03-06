import asyncio
import httpx
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

#-----Basi async func _-----
async def fetch_data(id: int) -> str:
    logger.info(f"Starting task {id}")
    await asyncio.sleep(1)
    logger.info(f"Finished task {id}")
    return f"Result from task {id}"

#---------Sequential slow ------------
async def run_sequential():
    print("\\n=====Sequential slow ===")

    start = time.time()

    r1 = await fetch_data(1)
    r2 = await fetch_data(2)
    r3 = await fetch_data(3)

    print(f"Results: {r1}, {r2}, {r3}")

    print(f"Time taken: {time.time() - start:.2f}s")

#--------concurrent - fast------------
async def run_concurrent():
    print("\\n====Concurrent fast ====")
    start = time.time()

    results = await asyncio.gather(fetch_data(1), fetch_data(2), fetch_data(3))

    for r in results:
        print(r)

    print(f"Time taken: {time.time() - start:.2f}s")

#----------Real async HTTP Calls -----
async def fetch_url(url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return f"Status: {response.status_code} from {url}"
    
async def run_http():
    print("\\n==== Async http calls ====")
    start = time.time()

    results = await asyncio.gather(fetch_url("https://httpbin.org/get"),
                                   fetch_url("https://httpbin.org/ip"),
                                   fetch_url("https://httpbin.org/uuid"))
    
    for r in results:
        print(r)

    print(f"Time taken: {time.time() - start:.2f}s")

#------Main --------------------

async def main():
    await run_sequential()
    await run_concurrent()
    await run_http()

asyncio.run(main())